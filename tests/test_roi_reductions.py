"""ROI reductions preserve directional cancellation and independently summed levels."""

import math

import numpy as np
import pytest

from mlxradiomics.features.texture import neighborhood_totals
from mlxradiomics.numerics.reduction import compensated_sum_rows, directional_mean
from mlxradiomics.numerics.runtime import array_from_bool, mx


@pytest.mark.parametrize("compensated", [False, True])
def test_directional_mean_retains_order_and_ignores_inactive_nan(compensated: bool) -> None:
    values = mx.array([
        [2**24, float("nan"), 1, -2**24],
        [float("inf"), float("nan"), 1, 0],
        [float("inf"), float("nan"), -float("inf"), 0],
        [1, float("nan"), 2, 6],
    ], dtype=mx.float32)
    active = mx.array([True, False, True, True])
    expected = np.array([np.float32(1 / 3) if compensated else 0, np.inf, np.nan, 3],
                        dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(directional_mean(values, active, compensated=compensated), dtype=np.float32),
        expected,
    )
    np.testing.assert_array_equal(
        np.asarray(directional_mean(values, mx.zeros((4,), dtype=mx.bool_),
                                    compensated=compensated), dtype=np.float32),
        np.full((4,), np.nan, dtype=np.float32),
    )


@pytest.mark.parametrize("width", [65, 513, 1025, 4097])
def test_roi_ngtdm_totals_match_independent_counts_and_sums(width: int) -> None:
    # Cross reduction sizes and partial tails; an unoccupied level must stay zero.
    rng = np.random.default_rng(846)
    bins = rng.integers(1, 5, size=(1, width), dtype=np.int32)
    eligible = rng.random((1, width)) > 0.2
    difference = (rng.random((1, width)) * 4).astype(np.float32)
    selected = [eligible & (bins == level) for level in range(1, 6)]
    expected_n = [np.count_nonzero(mask) for mask in selected]
    expected_s = [math.fsum(float(value) for value in difference[mask]) for mask in selected]
    # Ineligible samples must not contribute invalid labels or NaNs to the sums.
    bins[~eligible] = 0
    difference[~eligible] = np.nan
    with mx.stream(mx.gpu):
        n, s = neighborhood_totals(
            mx.array(bins), array_from_bool(eligible), mx.array(difference), 5, queued=True,
        )
        actual_n = np.asarray(n, dtype=np.int32)[0]
        actual_s = np.asarray(s, dtype=np.float32)[0]
    np.testing.assert_array_equal(actual_n, expected_n)
    # Bound relative reduction error to two float32 epsilons without prescribing
    # an addition tree. Zero reference sums still require exact zero (atol=0).
    np.testing.assert_allclose(actual_s, expected_s, rtol=2 * np.finfo(np.float32).eps, atol=0)


@pytest.mark.parametrize("width", [1025, 2**20 + 3])
def test_compensated_sum_recovers_absorbed_units_across_tree_passes(width: int) -> None:
    # Units added beside +-1e8 round away in float32 partial sums (ulp 8) and return only
    # through the two-sum errors. Above 1024 leaves the tree spans several Metal passes;
    # 2**20 + 3 leaves need 21 levels, more than one error-accumulation link.
    rng = np.random.default_rng(width)
    values = np.zeros((2, width), dtype=np.float32)
    values[:, 0], values[:, -1] = 1e8, -1e8
    units = rng.choice(np.arange(1, width - 1), size=37, replace=False)
    values[0, units], values[1, units] = 1.0, -0.5
    expected = np.array([math.fsum(row.astype(np.float64)) for row in values], np.float32)
    observed = np.asarray(compensated_sum_rows(mx.array(values)), dtype=np.float32)
    np.testing.assert_array_equal(observed, expected)
