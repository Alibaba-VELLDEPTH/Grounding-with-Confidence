"""Small dependency-free assignment helpers for temporal set metrics."""

from __future__ import annotations

from collections.abc import Sequence


def maximum_weight_assignment(
    weights: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    """Return a maximum-weight one-to-one assignment for a rectangular matrix.

    The implementation is the O(n^3) Hungarian algorithm. It assigns every row
    of the smaller matrix dimension, matching SciPy's rectangular assignment
    cardinality without adding dummy pairs to the returned result.
    """

    matrix = [list(map(float, row)) for row in weights]
    if not matrix:
        return []
    column_count = len(matrix[0])
    if column_count == 0:
        return []
    if any(len(row) != column_count for row in matrix):
        raise ValueError("assignment matrix is ragged")

    transposed = len(matrix) > column_count
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
    row_count = len(matrix)
    column_count = len(matrix[0])

    # Hungarian minimization with costs negated for maximum-weight matching.
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        p[0] = row_index
        min_value = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            current_row = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced_cost = -matrix[current_row - 1][column - 1] - u[current_row] - v[column]
                if reduced_cost < min_value[column]:
                    min_value[column] = reduced_cost
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    pairs = [(p[column] - 1, column - 1) for column in range(1, column_count + 1) if p[column] != 0]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)
