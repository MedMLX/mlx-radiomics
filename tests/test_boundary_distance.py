"""Exact Manhattan distances, including image borders and independent planes."""

from collections import deque
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics.features.matrices import boundary_distances
from mlxradiomics.numerics.runtime import array_from_bool, mx


def _breadth_first_distances(
    mask: NDArray[np.bool_], excluded_axis: int | None,
) -> NDArray[np.int32]:
    """Independent shortest-path oracle seeded by holes and exterior neighbors."""
    result = np.full(mask.shape, mask.size + 1, dtype=np.int32)
    queue: deque[tuple[int, ...]] = deque()
    steps = tuple(
        tuple(sign if i == axis else 0 for i in range(mask.ndim))
        for axis in range(mask.ndim) if axis != excluded_axis for sign in (-1, 1)
    )
    for point in product(*(range(size) for size in mask.shape)):
        if not mask[point]:
            result[point] = 0
            queue.append(point)
        elif any(point[i] in (0, size - 1) for i, size in enumerate(mask.shape)
                 if i != excluded_axis):
            result[point] = 1
            queue.append(point)
    while queue:
        point = queue.popleft()
        for step in steps:
            neighbor = tuple(a + b for a, b in zip(point, step, strict=True))
            if all(0 <= v < size for v, size in zip(neighbor, mask.shape, strict=True)):
                distance = int(result[point]) + 1
                if distance < result[neighbor]:
                    result[neighbor] = distance
                    queue.append(neighbor)
    return result


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (4, 5, 6), (1, 6, 1), (9, 1, 7)])
@pytest.mark.parametrize("pattern", ["full", "holes", "empty"])
def test_boundary_distances_match_shortest_paths(
    shape: tuple[int, ...], pattern: str,
) -> None:
    rng = np.random.default_rng(111)
    masks = rng.random((2, *shape)) > 0.2
    if pattern != "holes":
        masks[:] = pattern == "full"
    for axis in (None, *range(len(shape))):
        expected = np.stack([_breadth_first_distances(mask, axis) for mask in masks])
        with mx.stream(mx.gpu):
            actual = boundary_distances(array_from_bool(masks.reshape(2, -1)), shape, axis)
            assert actual.dtype == mx.int32
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.int32).reshape(masks.shape), expected,
            )
