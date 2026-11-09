"""Occupied integer matrix cells counted and merged on Metal, with original indices."""

from typing import cast

from mlxradiomics.numerics.protocols import IndexedAddition
from mlxradiomics.numerics.runtime import evaluate, mx


def coalesce(keys: mx.array, weights: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Sort and sum integer cells without truncating unique keys or transferring data."""
    if not keys.size:
        return mx.zeros((1,), dtype=mx.int64), mx.zeros((1,), dtype=mx.int32)
    values, inverse, occurrences = cast(
        tuple[mx.array, mx.array, mx.array],
        mx.unique(
            keys,
            keys.size,
            return_inverse=True,
            return_counts=True,
            fill_value=-1,
        ),
    )
    if weights is None:
        counts = occurrences
    else:
        storage = mx.zeros((keys.size,), dtype=mx.int32)
        counts = cast(IndexedAddition, storage.at[inverse]).add(weights.astype(mx.int32))
    # The -1 sentinel sorts first. All padded keys also have -1 but zero counts.
    occupied = mx.sum(((values >= 0) & (occurrences > 0)).astype(mx.int32))
    first = (values[0] < 0).astype(mx.int32)
    evaluate(occupied, first, values, counts)
    size, start = int(cast(int, occupied.item())), int(cast(int, first.item()))
    if not size:
        return mx.zeros((1,), dtype=mx.int64), mx.zeros((1,), dtype=mx.int32)
    return values[start : start + size], counts[start : start + size]


def _pad_rows(rows: list[tuple[mx.array, mx.array]]) -> tuple[mx.array, mx.array]:
    """Stack per-row ``(keys, counts)`` into ``(batch, width)`` arrays, zero-padding the tail.

    A padded entry has key 0 and count 0, so it contributes nothing to a count-weighted sum.
    """
    width = max(keys.size for keys, _ in rows)
    return (
        mx.stack([mx.pad(keys, [(0, width - keys.size)]) for keys, _ in rows]),
        mx.stack([mx.pad(counts, [(0, width - counts.size)]) for _, counts in rows]),
    )


def coordinate_counts(
    left: mx.array,
    right: mx.array,
    valid: mx.array,
    columns: int,
    *,
    symmetric: bool = False,
) -> tuple[mx.array, mx.array]:
    """Return padded ``(batch, occupied cells)`` keys and exact integer counts."""
    i, j = left.astype(mx.int64) - 1, right.astype(mx.int64) - 1
    keys = mx.where(valid, i * columns + j, -1)
    if symmetric:
        keys = mx.concatenate([keys, mx.where(valid, j * columns + i, -1)], axis=1)
    return _pad_rows([coalesce(row) for row in keys])


def merge_counts(
    keys: list[mx.array],
    counts: list[mx.array],
) -> tuple[mx.array, mx.array]:
    """Merge coordinate counts before nonlinear feature evaluation."""
    all_keys, all_counts = mx.concatenate(keys, axis=1), mx.concatenate(counts, axis=1)
    if bool(mx.any(mx.sum(all_counts.astype(mx.int64), axis=1) >= 2**31).item()):
        raise ValueError("merged texture count must be below 2**31")
    return _pad_rows(
        [
            coalesce(mx.where(weights > 0, row, -1), weights)
            for row, weights in zip(all_keys, all_counts, strict=True)
        ]
    )
