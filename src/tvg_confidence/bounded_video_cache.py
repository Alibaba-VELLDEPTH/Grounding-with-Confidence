"""Node-shared video cache: reserve bytes under a short lock, copy concurrently."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


class BoundedVideoCache:
    def __init__(self, root, max_bytes=8 * 1024**3, min_free_bytes=8 * 1024**3,
                 wait_seconds=300, report_path=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in (".locks", ".reservations"):
            (self.root / name).mkdir(exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.wait_seconds = float(wait_seconds)
        self.report_path = Path(report_path) if report_path else None
        self.last_metrics = {}
        if self.max_bytes <= 0 or self.min_free_bytes < 0:
            raise ValueError("invalid video cache byte limits")

    @contextmanager
    def _guard(self):
        start = time.monotonic()
        with (self.root / ".cache.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.last_metrics["cache_lock_wait_s"] += time.monotonic() - start
            yield

    def _files(self):
        return [p for p in self.root.iterdir() if not p.name.startswith(".") and p.is_file()]

    def _reservation_usage(self):
        reserved = remaining = 0
        for entry in (self.root / ".reservations").glob("*.json"):
            data = json.loads(entry.read_text())
            partial = self.root / ("." + data["target"] + ".partial")
            reserved += data["size"]
            copied = partial.stat().st_size if partial.exists() else 0
            remaining += max(0, data["size"] - copied)
        return reserved, remaining

    def _reap_abandoned(self):
        # A live writer holds its file lease even while outside the node lock.
        for entry in (self.root / ".reservations").glob("*.json"):
            data = json.loads(entry.read_text())
            with (self.root / ".locks" / data["target"]).open("a+b") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                (self.root / ("." + data["target"] + ".partial")).unlink(missing_ok=True)
                entry.unlink()

    def _publish(self, hit=None, evictions=0):
        path = self.root / ".state.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "hits": 0, "misses": 0, "evictions": 0, "peak_bytes": 0,
            "peak_committed_and_reserved_bytes": 0,
        }
        files = self._files()
        used = sum(p.stat().st_size for p in files)
        reserved, remaining = self._reservation_usage()
        total = used + reserved
        if total > self.max_bytes:
            raise RuntimeError("video cache exceeded committed plus reserved byte budget")
        state.update(bytes=used, reserved_bytes=reserved, files=len(files), max_bytes=self.max_bytes,
                     min_free_bytes=self.min_free_bytes, free_bytes=shutil.disk_usage(self.root).free,
                     peak_bytes=max(state["peak_bytes"], used), updated_at=time.time(),
                     peak_committed_and_reserved_bytes=max(state.get("peak_committed_and_reserved_bytes", 0), total),
                     outstanding_copy_bytes=remaining)
        if hit is not None:
            state["hits" if hit else "misses"] += 1
        state["evictions"] += evictions
        temporary = self.root / ".state.partial"
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return state

    def _make_room(self, size):
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime_ns)
        used = sum(p.stat().st_size for p in files)
        reserved, remaining = self._reservation_usage()
        evictions = 0

        def fits():
            return (used + reserved + size <= self.max_bytes
                    and shutil.disk_usage(self.root).free >= remaining + size + self.min_free_bytes)

        for path in files:
            if fits():
                break
            with (self.root / ".locks" / path.name).open("a+b") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                used -= path.stat().st_size
                path.unlink()
                evictions += 1
        return fits(), evictions

    def _copy(self, source, partial):
        with source.open("rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)

    @contextmanager
    def localize(self, source):
        start = time.monotonic()
        self.last_metrics = {"cache_lock_wait_s": 0.0, "cache_copy_s": 0.0,
                             "cache_copy_bytes": 0, "cache_hit": 0, "cache_evictions": 0}
        source = Path(source)
        size = source.stat().st_size
        if not 0 < size <= self.max_bytes:
            raise RuntimeError(f"training video size {size} is outside cache budget {self.max_bytes}: {source}")
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:24]
        target = self.root / (digest + (source.suffix or ".mp4"))
        partial = self.root / ("." + target.name + ".partial")
        reservation = self.root / ".reservations" / (target.name + ".json")
        lease = (self.root / ".locks" / target.name).open("a+b")
        deadline = start + self.wait_seconds
        owns_reservation = False
        state = None
        try:
            while True:
                copy_needed = hit = False
                with self._guard():
                    self._reap_abandoned()
                    if target.is_file() and target.stat().st_size == size:
                        try:
                            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
                            hit = True
                        except BlockingIOError:
                            pass
                        if hit:
                            os.utime(target, None)
                            state = self._publish(hit=True)
                    else:
                        try:
                            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            locked = True
                        except BlockingIOError:
                            locked = False
                        if locked:
                            target.unlink(missing_ok=True)
                            copy_needed, removed = self._make_room(size)
                            self.last_metrics["cache_evictions"] += removed
                            if copy_needed:
                                temporary = reservation.with_suffix(".tmp")
                                temporary.write_text(json.dumps({"target": target.name, "size": size}))
                                os.replace(temporary, reservation)
                                owns_reservation = True
                            else:
                                fcntl.flock(lease, fcntl.LOCK_UN)
                            self._publish(evictions=removed)
                if hit:
                    self.last_metrics["cache_hit"] = 1
                    break
                if copy_needed:
                    copy_start = time.monotonic()
                    self._copy(source, partial)  # No node lock held during remote I/O.
                    self.last_metrics["cache_copy_s"] = time.monotonic() - copy_start
                    self.last_metrics["cache_copy_bytes"] = size
                    if partial.stat().st_size != size:
                        raise RuntimeError(f"localized video size mismatch: {source}")
                    with self._guard():
                        os.replace(partial, target)
                        reservation.unlink()
                        owns_reservation = False
                        fcntl.flock(lease, fcntl.LOCK_SH)
                        state = self._publish(hit=False)
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("video cache cannot reserve space; active decodes or disk headroom exhausted")
                time.sleep(0.05)
            self.last_metrics["cache_localize_s"] = time.monotonic() - start
            if self.report_path and (state["hits"] + state["misses"]) % 64 == 0:
                # Never hold the cache transaction lock across an OSS/FUSE report write.
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.report_path.with_name(self.report_path.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
                os.replace(temporary, self.report_path)
            yield str(target)  # Shared lease pins the file until decoding completes.
        finally:
            if owns_reservation:
                with self._guard():
                    partial.unlink(missing_ok=True)
                    reservation.unlink(missing_ok=True)
                    self._publish()
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()


def distributed_cache_preflight(root, output_path, rank, world_size):
    """Check concurrent writers, byte reservations, eviction and decode leases."""
    import torch.distributed as dist
    import uuid

    run_id = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(run_id, src=0)
    root = Path(root) / run_id[0]
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    class ConcurrentProbe(BoundedVideoCache):
        def _copy(self, source, partial):
            # All ranks must enter copy together; serializing copy deadlocks here
            # and the process-group timeout turns that regression into a failure.
            dist.barrier()
            super()._copy(source, partial)

    concurrent = ConcurrentProbe(root / "concurrent", max_bytes=world_size * 65536, min_free_bytes=0)
    source = sources / f"concurrent-{rank}.mp4"
    source.write_bytes(bytes([rank]) * 65536)
    with concurrent.localize(source) as localized:
        assert Path(localized).read_bytes() == source.read_bytes()
    dist.barrier()
    cache = BoundedVideoCache(root / "cache", max_bytes=3 * 65536, min_free_bytes=0)
    for index in range(6):
        source = sources / f"{rank}-{index}.mp4"
        expected = bytes([rank + index]) * 65536
        source.write_bytes(expected)
        with cache.localize(source) as localized:
            assert Path(localized).read_bytes() == expected
            time.sleep(0.02)
            assert Path(localized).read_bytes() == expected
        source.unlink()
    dist.barrier()
    report = json.loads((cache.root / ".state.json").read_text())
    assert report["peak_committed_and_reserved_bytes"] <= cache.max_bytes and report["evictions"] > 0
    report.update(status="passed", ranks=world_size, checked_reads=world_size * 13,
                  concurrent_writers=world_size,
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if rank == 0:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps({"stage": "bounded_video_cache_preflight", **report}), flush=True)
    dist.barrier()
    return report
