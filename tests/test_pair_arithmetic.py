"""Source-pair subtraction contracts used by the first-order calculation graph."""

import numpy as np
import pytest

from mlxradiomics.numerics.pair_arithmetic import subtract_pairs
from mlxradiomics.numerics.runtime import mx


def test_broadcast_centers_keep_residual_scale_differences() -> None:
    # All high words are equal. The result must retain the differences between
    # residuals, including the (1,N) minus (N,1) broadcast used for source centers.
    high, low = subtract_pairs(
        mx.full((1, 4), 1e8, dtype=mx.float32),
        mx.array([[1, 2, 3, 4]], dtype=mx.float32),
        mx.full((3, 1), 1e8, dtype=mx.float32),
        mx.array([[1], [2], [3]], dtype=mx.float32),
    )
    expected = np.array([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(high, dtype=np.float32), expected)
    np.testing.assert_array_equal(np.asarray(low, dtype=np.float32), np.zeros_like(expected))


def test_finite_residuals_recover_an_overflowing_high_word_span() -> None:
    maximum = float(np.finfo(np.float32).max)
    high, low = subtract_pairs(
        mx.array([maximum], dtype=mx.float32),
        mx.array([-maximum / 2], dtype=mx.float32),
        mx.array([-maximum], dtype=mx.float32),
        mx.array([maximum / 2], dtype=mx.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(high, dtype=np.float32), np.array([maximum], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(low, dtype=np.float32), np.array([0], dtype=np.float32),
    )


def test_true_overflow_has_an_infinite_high_word_and_zero_low_word() -> None:
    maximum = float(np.finfo(np.float32).max)
    high, low = subtract_pairs(
        mx.array([maximum, -maximum], dtype=mx.float32),
        mx.array(0, dtype=mx.float32),
        mx.array([-maximum, maximum], dtype=mx.float32),
        mx.array(0, dtype=mx.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(high, dtype=np.float32), np.array([np.inf, -np.inf], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(low, dtype=np.float32), np.array([0, 0], dtype=np.float32),
    )


def test_non_float32_operands_are_refused() -> None:
    with pytest.raises(ValueError, match="requires float32"):
        subtract_pairs(
            mx.array(1, dtype=mx.int32), mx.array(0, dtype=mx.float32),
            mx.array(0, dtype=mx.float32), mx.array(0, dtype=mx.float32),
        )
