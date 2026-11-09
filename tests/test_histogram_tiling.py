"""Integer histogram kernels against independently enumerated co-occurrences."""

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics.numerics.histogram import histogram
from mlxradiomics.numerics.runtime import array_from_bool, mx


def _reference(
    left: NDArray[np.int32], right: NDArray[np.int32], valid: NDArray[np.bool_], ng: int,
    columns: int | None = None,
) -> NDArray[np.uint32]:
    result = np.zeros((left.shape[0], ng, ng if columns is None else columns), dtype=np.uint32)
    for row in range(left.shape[0]):
        pairs = zip(left[row, valid[row]], right[row, valid[row]], strict=True)
        for first, second in pairs:
            result[row, int(first) - 1, int(second) - 1] += 1
    return result


@pytest.mark.parametrize("width", [0, 1, 4095, 4096, 4097, 8193])
@pytest.mark.parametrize("ng", [1, 7, 32, 33])
@pytest.mark.parametrize("columns", [None, 40])
def test_histogram_rows_tiles_and_tail(width: int, ng: int, columns: int | None) -> None:
    """Compare both dispatch regimes, including row boundaries and partial tiles.

    Size matrices use a second axis wider than ``ng``; 7 x 40 cells still tile.
    """
    rng = np.random.default_rng(762)
    left = rng.integers(1, ng + 1, size=(3, width), dtype=np.int32)
    right = rng.integers(1, (ng if columns is None else columns) + 1, size=left.shape,
                         dtype=np.int32)
    valid = rng.random(left.shape) > 0.27
    # False pairs may hold zero bins and must never reach an atomic destination.
    left[~valid] = 0
    right[~valid] = 0
    valid[1] = False
    expected = _reference(left, right, valid, ng, columns)
    for method in ("metal", "mlx"):
        result = histogram(
            mx.array(left), mx.array(right), array_from_bool(valid), ng, method=method,
            columns=columns,
        )
        assert result.dtype == mx.uint32
        actual = np.asarray(result, dtype=np.uint32)
        np.testing.assert_array_equal(actual, expected)


def test_hot_bin_histogram_does_not_narrow_integer_counts() -> None:
    width = 131_077
    left = np.ones((2, width), dtype=np.int32)
    right = np.full(left.shape, 5, dtype=np.int32)
    valid = np.ones(left.shape, dtype=np.bool_)
    valid[1, ::3] = False
    actual = np.asarray(histogram(
        mx.array(left), mx.array(right), array_from_bool(valid), 8, method="metal",
    ), dtype=np.uint32)
    expected = np.zeros((2, 8, 8), dtype=np.uint32)
    expected[:, 0, 4] = np.count_nonzero(valid, axis=1).astype(np.uint32)
    np.testing.assert_array_equal(actual, expected)


def test_tiled_histogram_accepts_strided_inputs_and_preserves_rows() -> None:
    rng = np.random.default_rng(39)
    left = rng.integers(1, 17, size=(4, 16_390), dtype=np.int32)
    right = rng.integers(1, 17, size=left.shape, dtype=np.int32)
    valid = rng.random(left.shape) > 0.3
    device_left = mx.array(left)[::2, ::2]
    device_right = mx.array(right)[::2, ::2]
    device_valid = array_from_bool(valid)[::2, ::2]
    actual = np.asarray(histogram(
        device_left, device_right, device_valid, 16, method="metal",
    ), dtype=np.uint32)
    np.testing.assert_array_equal(
        actual, _reference(left[::2, ::2], right[::2, ::2], valid[::2, ::2], 16),
    )
    for row in range(2):
        separate = np.asarray(histogram(
            device_left[row:row + 1], device_right[row:row + 1],
            device_valid[row:row + 1], 16, method="metal",
        ), dtype=np.uint32)
        np.testing.assert_array_equal(separate[0], actual[row])

