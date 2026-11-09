"""Analytic IBSI intensity-histogram features.

Expected values are derived from a four-value histogram by hand.  Percentiles use
linear order statistics, kurtosis is excess kurtosis, and the histogram-gradient
endpoints are one-sided.  The test does not use a radiomics oracle.
"""

import math

import numpy as np
import pytest

from mlxradiomics.features.intensity_histogram import histogram_statistics
from mlxradiomics.numerics.runtime import mx

EXPECTED = {
    "Mean": 2.0,
    "Variance": 1.5,
    "Skewness": 1.5 / 1.5**1.5,
    "Kurtosis": -1.0,
    "Median": 1.5,
    "Minimum": 1.0,
    "10Percentile": 1.0,
    "90Percentile": 3.4,
    "Maximum": 4.0,
    "Mode": 1.0,
    "InterquartileRange": 1.5,
    "Range": 3.0,
    "MeanAbsoluteDeviation": 1.0,
    "RobustMeanAbsoluteDeviation": 4 / 9,
    "MedianAbsoluteDeviation": 1.0,
    "CoefficientOfVariation": math.sqrt(1.5) / 2,
    "QuartileCoefficientOfDispersion": 3 / 7,
    "Entropy": 1.5,
    "Uniformity": 0.375,
    "MaximumGradient": 1.0,
    "MaximumGradientGrayLevel": 4.0,
    "MinimumGradient": -1.0,
    "MinimumGradientGrayLevel": 1.0,
}


def _value(values: dict[str, mx.array], name: str) -> float:
    return float(np.asarray(values[name], dtype=np.float32).item())


def test_histogram_statistics_all_twenty_three_outputs() -> None:
    # Counts [2, 1, 0, 1] over levels 1..4; every expected value is derived directly.
    bins = mx.array([[1, 1, 2, 4]], dtype=mx.int32)
    valid = mx.array([[True, True, True, True]])
    actual = histogram_statistics(bins, valid, 4)
    assert set(actual) == set(EXPECTED)
    for name, target in EXPECTED.items():
        assert _value(actual, name) == pytest.approx(target, rel=1e-6, abs=1e-6), name


def test_histogram_batch_constant_row_and_invalid_values() -> None:
    # Row 0 is the analytic four-value histogram.  Row 1 is a constant level-3 row;
    # levels 4, 5 and 6 carry out-of-range dummy bins in invalid positions.  Global
    # N_g is four, so row 1's empty boundary bins still define its gradients.
    bins = mx.array([[1, 1, 2, 4, 0, 0], [3, 3, 3, 4, 5, 6]], dtype=mx.int32)
    valid = mx.array([[True, True, True, True, False, False],
                      [True, True, True, False, False, False]])
    actual = histogram_statistics(bins, valid, 4)
    for name, target in EXPECTED.items():
        values = np.asarray(actual[name], dtype=np.float32)
        assert values[0] == pytest.approx(target, rel=1e-6, abs=1e-6), name

    constant = {
        "Mean": 3.0, "Variance": 0.0, "Skewness": 0.0, "Kurtosis": 0.0,
        "Median": 3.0, "Minimum": 3.0, "10Percentile": 3.0, "90Percentile": 3.0,
        "Maximum": 3.0, "Mode": 3.0, "InterquartileRange": 0.0, "Range": 0.0,
        "MeanAbsoluteDeviation": 0.0, "RobustMeanAbsoluteDeviation": 0.0,
        "MedianAbsoluteDeviation": 0.0, "CoefficientOfVariation": 0.0,
        "QuartileCoefficientOfDispersion": 0.0, "Entropy": 0.0, "Uniformity": 1.0,
        # Counts [0, 0, 3, 0] give H' = [0, 1.5, 0, -3].
        "MaximumGradient": 1.5, "MaximumGradientGrayLevel": 2.0,
        "MinimumGradient": -3.0, "MinimumGradientGrayLevel": 4.0,
    }
    for name, target in constant.items():
        values = np.asarray(actual[name], dtype=np.float32)
        assert values[1] == pytest.approx(target, rel=1e-6, abs=1e-6), name


@pytest.mark.parametrize("row,expected_range", [((0, 0), 0.0), ((-1, 0), 1.0)])
def test_histogram_range_is_exact_above_float32_integer_limit(
    row: tuple[int, int], expected_range: float,
) -> None:
    # Above 2**24 the endpoints round individually; their difference must not.
    ng = 2**24 + 2
    bins = mx.array([[ng + offset for offset in row]], dtype=mx.int32)
    actual = histogram_statistics(bins, mx.array([[True, True]]), ng)
    assert actual["Range"].item() == expected_range


def test_histogram_mode_tie_breaks_by_distance_then_left_bin() -> None:
    # Row 0: counts [2, 2] at levels 1 and 2, mean 1.5, so the equidistant tie chooses
    # the left bin.  Row 1: counts [2, 2, 1], mean 1.8, so level 2 is closer than level 1.
    bins = mx.array([[1, 1, 2, 2, 0], [1, 1, 2, 2, 3]], dtype=mx.int32)
    valid = mx.array([[True, True, True, True, False],
                      [True, True, True, True, True]])
    actual = histogram_statistics(bins, valid, 3)
    mode = np.asarray(actual["Mode"], dtype=np.float32)
    assert mode[0] == 1.0
    assert mode[1] == 2.0


def test_histogram_percentiles_use_linear_order_statistics() -> None:
    # Five equally frequent levels: positions are 0.4, 1.0, 2.0, 3.0 and 3.6.
    bins = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    valid = mx.array([[True, True, True, True, True]])
    actual = histogram_statistics(bins, valid, 5)
    assert _value(actual, "10Percentile") == pytest.approx(1.4, rel=1e-6)
    assert _value(actual, "Median") == 3.0
    assert _value(actual, "90Percentile") == pytest.approx(4.6, rel=1e-6)
    assert _value(actual, "InterquartileRange") == 2.0


def test_histogram_single_level_has_zero_gradient() -> None:
    bins = mx.array([[1, 1]], dtype=mx.int32)
    valid = mx.array([[True, True]])
    actual = histogram_statistics(bins, valid, 1)
    assert _value(actual, "MaximumGradient") == 0.0
    assert _value(actual, "MinimumGradient") == 0.0
    assert _value(actual, "MaximumGradientGrayLevel") == 1.0
    assert _value(actual, "MinimumGradientGrayLevel") == 1.0


def test_histogram_mode_uses_exact_mean_to_resolve_near_tie() -> None:
    # Equal modal counts at 255 and 257. One observation at 258 shifts the
    # exact mean above 256, while its float32 rounding is still exactly 256.
    bins = np.concatenate([
        np.full(131072, 255, dtype=np.int32),
        np.full(131072, 257, dtype=np.int32),
        np.array([258], dtype=np.int32),
    ])[None, :]
    actual = histogram_statistics(mx.array(bins), mx.ones(bins.shape, dtype=mx.bool_), 258)
    assert _value(actual, "Mode") == 257.0


def test_histogram_large_index_robust_deviation_keeps_rational_center() -> None:
    # Counts [20, 10] at levels [1048576, 1048577]: the retained mean is 1048576 + 1/3 and
    # the mean absolute deviation is 4/9. Rounding the absolute mean to float32 first
    # (spacing 0.125 there) produced 0.45833334.
    levels = np.concatenate([
        np.full(20, 1048576, dtype=np.int32),
        np.full(10, 1048577, dtype=np.int32),
    ])[None, :]
    actual = histogram_statistics(
        mx.array(levels), mx.ones(levels.shape, dtype=mx.bool_), 1048577
    )
    assert _value(actual, "RobustMeanAbsoluteDeviation") == pytest.approx(4 / 9, rel=1e-6)
    assert _value(actual, "MedianAbsoluteDeviation") == pytest.approx(1 / 3, rel=1e-6)


def test_histogram_large_index_half_level_survives_interquartile_range() -> None:
    # Two adjacent levels at 2**22. The exact quartiles are 4194304.25 and 4194304.75,
    # i.e. an interquartile range of exactly 0.5. Reported as float32 they round to
    # 4194304 and 4194305, so subtracting the reported values would give 1.0; the exact
    # integer-plus-fraction pair must form the spread.
    level = 2**22
    bins = np.array([[level, level + 1]], dtype=np.int32)
    actual = histogram_statistics(
        mx.array(bins), mx.ones(bins.shape, dtype=mx.bool_), level + 1
    )
    assert _value(actual, "Median") == level + 0.5
    assert _value(actual, "InterquartileRange") == 0.5
    assert _value(actual, "QuartileCoefficientOfDispersion") == pytest.approx(
        0.5 / (2 * level + 1), rel=1e-6
    )


def test_histogram_moments_scale_by_occupied_levels_not_empty_lattice() -> None:
    # One voxel at level ng - 1 and N at ng, with every lower level empty (as an anchored
    # fixed-width lattice leaves them). This is a Bernoulli histogram with p = 1/(N + 1):
    # variance pq, skewness (p - q)/sqrt(pq) and excess kurtosis (1 - 6pq)/(pq), computed
    # here in float64. Scaling offsets by the empty lattice span (about 2**21) made the
    # scaled second moment about 2**-64, whose square flushes to zero on Metal, so
    # kurtosis was reported as inf.
    ng, n = 2**21, 2**22
    bins = np.full((1, n + 1), ng, dtype=np.int32)
    bins[0, 0] = ng - 1
    actual = histogram_statistics(mx.array(bins), mx.ones(bins.shape, dtype=mx.bool_), ng)
    p = 1 / (n + 1)
    q = 1 - p
    assert _value(actual, "Variance") == pytest.approx(p * q, rel=1e-5)
    assert _value(actual, "Skewness") == pytest.approx((p - q) / math.sqrt(p * q), rel=1e-5)
    assert _value(actual, "Kurtosis") == pytest.approx((1 - 6 * p * q) / (p * q), rel=1e-5)

