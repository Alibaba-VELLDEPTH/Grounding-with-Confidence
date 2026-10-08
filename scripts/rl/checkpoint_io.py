"""Bounded parallel checkpoint hashing with byte counts and visible progress."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def wait_for_checkpoint_size(path, *, expected_bytes=None, timeout=120, interval=2):
    """Wait for shared filesystem metadata visibility; byte and SHA checks remain mandatory."""
    path = Path(path)
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = None
        elapsed = time.monotonic() - started
        if size is not None and size > 0 and (expected_bytes is None or size == expected_bytes):
            if attempts > 1:
                print(json.dumps({"stage": "checkpoint_metadata_visible", "file": str(path),
                    "bytes": size, "attempts": attempts, "seconds": elapsed}), flush=True)
            return size
        if elapsed >= timeout:
            raise RuntimeError(f"checkpoint file size mismatch after {elapsed:.1f}s: {path}; "
                f"observed={size}, expected={expected_bytes}")
        if attempts == 1 or attempts % 15 == 0:
            print(json.dumps({"stage": "checkpoint_metadata_wait", "file": str(path),
                "observed_bytes": size, "expected_bytes": expected_bytes,
                "attempts": attempts, "seconds": elapsed}), flush=True)
        time.sleep(min(interval, timeout - elapsed))


def hash_checkpoint_files(directory, names, *, expected=None, phase="verify"):
    directory = Path(directory)
    names = sorted(names)
    sizes = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("invalid checkpoint receipt path")
        sizes[name] = wait_for_checkpoint_size(directory / name,
            expected_bytes=None if expected is None else expected[name]["bytes"])
    if not names:
        raise RuntimeError("empty checkpoint file set")
    workers = min(4, len(names))
    started = time.monotonic()
    lock = threading.Lock()
    state = {name: {"bytes": 0, "status": "pending"} for name in names}
    results = {}

    def progress(status):
        with lock:
            observed = {name: dict(value) for name, value in state.items()}
        elapsed = time.monotonic() - started
        count = sum(value["bytes"] for value in observed.values())
        record = {"stage": "checkpoint_io", "phase": phase, "status": status,
            "checkpoint": str(directory), "workers": workers, "total_files": len(names),
            "completed_files": sum(value["status"] == "verified" for value in observed.values()),
            "total_bytes": sum(sizes.values()), "bytes_read": count,
            "seconds": elapsed, "mib_per_second": count / max(elapsed, 1e-6) / 1024**2,
            "active_files": {name: value["bytes"] for name, value in observed.items()
                             if value["status"] == "reading"}, "time": time.time()}
        print(json.dumps(record), flush=True)
        if os.environ.get("TVG_RL_OUTPUT"):
            output = Path(os.environ["TVG_RL_OUTPUT"])
            temporary = output / "checkpoint-io.json.tmp"
            temporary.write_text(json.dumps(record) + "\n")
            os.replace(temporary, output / "checkpoint-io.json")

    def one(name):
        path = directory / name
        digest = hashlib.sha256()
        counted = 0
        with lock:
            state[name]["status"] = "reading"
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                counted += len(block)
                digest.update(block)
                with lock:
                    state[name]["bytes"] = counted
        actual = {"bytes": counted, "sha256": digest.hexdigest()}
        if counted != sizes[name]:
            raise RuntimeError("checkpoint changed or short read: " + name)
        wait_for_checkpoint_size(path, expected_bytes=sizes[name])
        if expected is not None and actual != expected[name]:
            raise RuntimeError("resume checkpoint checksum failed: " + name)
        with lock:
            state[name]["status"] = "verified"
        return name, actual

    progress("started")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, name) for name in names}
            last_report = started
            while pending:
                completed, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in completed:
                    name, actual = future.result()
                    results[name] = actual
                if time.monotonic() - last_report >= 30:
                    progress("running")
                    last_report = time.monotonic()
    except Exception:
        progress("failed")
        raise
    progress("passed")
    return {name: results[name] for name in names}
