"""IBSI intensity-histogram features from 1-based gray-level bins.

The histogram family operates on an already discretised intensity image.  Counts are
kept as integers until entropy, uniformity and percentile reductions, so a dominant
state does not lose its small positive entropy contribution to a probability that rounds
to one.  All reported values are float32 arrays with a leading batch axis.
"""

from collections.abc import Collection
from typing import cast

from mlxradiomics.config.settings import INTENSITY_HISTOGRAM
from mlxradiomics.numerics.count_statistics import count_entropy
from mlxradiomics.numerics.protocols import IndexedAddition
from mlxradiomics.numerics.reduction import sum_rows
from mlxradiomics.numerics.runtime import mx


def integer_histogram(bins: mx.array, valid: mx.array, ng: int) -> mx.array:
    """Count 1-based bins exactly, with uint32 indexed addition and int64 output.

    The source grid is already validated by preprocessing. Invalid entries may carry an
    arbitrary level; clamping keeps their zero-weight indexed addition in bounds.
    """
    batch = bins.shape[0]
    rows = mx.arange(batch, dtype=mx.int32)[:, None]
    clamped = mx.minimum(mx.maximum(bins - 1, 0), ng - 1)
    index = rows * ng + clamped
    storage = mx.zeros((batch * ng,), dtype=mx.uint32)
    counts = cast(IndexedAddition, storage.at[mx.reshape(index, (-1,))]).add(
        mx.reshape(valid.astype(mx.uint32), (-1,))
    )
    return mx.reshape(counts, (batch, ng)).astype(mx.int64)


def histogram_order_values(
    counts: mx.array, total: mx.array, orders: mx.array,
) -> mx.array:
    """Find integer order statistics with one search of a cumulative integer lattice.

    Row histograms are flattened before the inclusive scan, so their cumulative counts
    form a single sorted sequence. Adding each row's preceding population to its ranks
    makes a native ``searchsorted(..., side="right")`` return the same 1-based levels as
    counting every cumulative value <= each rank. Integer scans and searches are exact;
    empty bins, including empty row boundaries, remain in their declared lattice.
    """
    batch, ng = counts.shape
    population_before = mx.cumsum(total, dtype=mx.int64) - total
    cumulative = mx.cumsum(mx.reshape(counts, (-1,)), dtype=mx.int64)
    positions = mx.searchsorted(
        cumulative, population_before[:, None] + orders, side="right",
    ).astype(mx.int64)
    row_start = mx.arange(batch, dtype=mx.int64)[:, None] * ng
    # The statistics caller excludes empty sets. Keeping their level at Ng+1 also
    # gives an empty histogram the level Ng + 1.
    return mx.minimum(positions, row_start + ng) - row_start + 1


def _linear_percentiles(
    counts: mx.array, total: mx.array, fractions: tuple[tuple[int, int], ...],
) -> tuple[mx.array, mx.array]:
    """Interpolate integer ranks in parallel, retaining the integer/fraction pair.

    Position ``(n - 1) * numerator / denominator`` stays integer through rank selection.
    All lower/upper ranks are searched together; no percentile scans an Ng-wide mask.
    """
    numerator = mx.array([part[0] for part in fractions], dtype=mx.int64)[None, :]
    denominator = mx.array([part[1] for part in fractions], dtype=mx.int64)[None, :]
    n1 = mx.maximum(total - 1, 0)[:, None]
    whole = mx.floor_divide(n1, denominator)
    remainder = mx.remainder(n1, denominator)
    lower = whole * numerator + mx.floor_divide(remainder * numerator, denominator)
    fraction_remainder = mx.remainder(remainder * numerator, denominator)
    upper = mx.minimum(lower + 1, n1)
    values = histogram_order_values(counts, total, mx.concatenate([lower, upper], axis=1))
    lower_value, upper_value = values[:, :len(fractions)], values[:, len(fractions):]
    step = (upper_value - lower_value) * fraction_remainder
    return (
        lower_value + mx.floor_divide(step, denominator),
        mx.remainder(step, denominator).astype(mx.float32) / denominator.astype(mx.float32),
    )


def _interpolated_level(level: mx.array, fraction: mx.array, split: int) -> mx.array:
    """Round an integer-plus-fraction gray level to float32 once.

    The integer part can exceed ``2**24`` and then is not exactly float32-representable.
    Splitting it at a power-of-two boundary keeps the high part exact (at most 24
    significant bits) and adds the small remainder and the fraction before the final
    rounding instead of rounding the integer level first.
    """
    if split == 0:
        return level.astype(mx.float32) + fraction
    bucket = 2**split
    base = mx.floor_divide(level, bucket) * bucket
    return base.astype(mx.float32) + ((level - base).astype(mx.float32) + fraction)


def histogram_statistics(
    bins: mx.array, valid: mx.array, ng: int, *,
    selected: Collection[str] | None = None,
) -> dict[str, mx.array]:
    """Compute the 23 IBSI intensity-histogram features for a batch of voxel sets.

    Args:
        bins: Integer array ``(batch, n)`` of 1-based gray levels.  Values outside the valid
            mask may be zero and are not counted.
        valid: Boolean array of the same shape; voxels that belong to each set.
        ng: Number of gray levels (``N_g``).  Levels absent from a row remain in the
            histogram with zero counts.
        selected: Optional feature names; omitted outputs do not build their calculation graphs.

    Returns:
        Feature name mapped to a ``(batch,)`` float32 array.  Names shared with the raw
        intensity statistics use the same spellings (``Mean``, ``Variance``, ``Skewness``,
        ``Kurtosis``, ``Median``, ``Minimum``, ``10Percentile``, ``90Percentile``,
        ``Maximum``, ``InterquartileRange``, ``Range``, ``MeanAbsoluteDeviation``,
        ``RobustMeanAbsoluteDeviation``, ``MedianAbsoluteDeviation``,
        ``CoefficientOfVariation``, ``QuartileCoefficientOfDispersion``, ``Entropy`` and
        ``Uniformity``).  The histogram-only outputs are ``Mode``, ``MaximumGradient``,
        ``MaximumGradientGrayLevel``, ``MinimumGradient`` and
        ``MinimumGradientGrayLevel``.

    Notes:
        Skewness and kurtosis follow the IBSI intensity-histogram definitions: kurtosis
        is excess kurtosis (three is subtracted) and both are zero for a constant
        histogram.  Percentiles use linear interpolation at position ``(n - 1) q`` with
        exact integer rank arithmetic. The integer level and its residual fraction stay
        separate until each reported value is rounded once; the interquartile range,
        quartile coefficient of dispersion and the absolute deviations use that pair
        directly, so a half level survives at gray-level indices whose absolute value is
        not float32-representable.  The histogram gradient is
        ``H'_1 = n_2 - n_1``, ``H'_i = (n_{i+1} - n_{i-1}) / 2`` for interior bins and
        ``H'_{Ng} = n_{Ng} - n_{Ng-1}``; endpoints are one-sided and a single-bin
        histogram has gradient zero.  Gradient extrema and the mode choose the smallest
        gray level on a tie.  Mode ties are otherwise resolved by the bin closest to the
        mean, then the left bin, as IBSI specifies.  Callers never pass a row with no
        valid voxel; the result for one is undefined.
    """
    if ng < 1:
        raise ValueError("ng must be a positive integer")
    if bins.shape != valid.shape or bins.ndim != 2:
        raise ValueError("bins and valid must share a two-dimensional shape")
    requested = frozenset(INTENSITY_HISTOGRAM if selected is None else selected)
    unknown = requested - set(INTENSITY_HISTOGRAM)
    if unknown:
        raise ValueError(f"unknown intensity-histogram statistics: {', '.join(sorted(unknown))}")
    if not requested:
        return {}
    counts = integer_histogram(bins, valid, ng)
    total = mx.sum(counts, axis=1)
    total_f = mx.maximum(total, 1).astype(mx.float32)
    counts_f = counts.astype(mx.float32)
    levels_i = mx.arange(1, ng + 1, dtype=mx.int64)[None, :]
    output: dict[str, mx.array] = {}

    weighted_total: mx.array | None = None
    mean: mx.array | None = None
    if requested & {"Mean", "CoefficientOfVariation", "Mode"}:
        weighted_total = mx.sum(counts * levels_i, axis=1)
        if requested & {"Mean", "CoefficientOfVariation"}:
            mean = weighted_total.astype(mx.float32) / total_f
            if "Mean" in requested:
                output["Mean"] = mean

    moments = {"Variance", "Skewness", "Kurtosis", "MeanAbsoluteDeviation",
               "CoefficientOfVariation"}
    reference: mx.array | None = None
    offsets: mx.array | None = None
    offsets_i: mx.array | None = None
    if requested & (moments | {"RobustMeanAbsoluteDeviation"}):
        # Center on the most frequent occupied level before rounding offsets.
        reference = mx.argmax(counts, axis=1).astype(mx.int32)
        offsets = mx.arange(ng, dtype=mx.int32)[None, :] - reference[:, None]
        offsets_i = offsets.astype(mx.int64)
    if requested & moments:
        assert offsets is not None and offsets_i is not None
        mean_offset = mx.sum(counts * offsets_i, axis=1).astype(mx.float32) / total_f
        offset_f = offsets.astype(mx.float32)
        # Empty lattice levels must not shrink the scaled moments into subnormals.
        amplitude = mx.max(mx.where(counts > 0, mx.abs(offset_f), 0), axis=1)
        amplitude_scale = mx.where(amplitude > 0, amplitude, 1)
        scaled = offset_f / amplitude_scale[:, None]
        mean_scaled = mean_offset / amplitude_scale
        centered = scaled - mean_scaled[:, None]
        if requested & {"Variance", "Skewness", "Kurtosis", "CoefficientOfVariation"}:
            second = sum_rows(counts_f * centered**2) / total_f
            if "Variance" in requested:
                output["Variance"] = (second * amplitude) * amplitude
            if "Skewness" in requested:
                third = sum_rows(counts_f * centered**3) / total_f
                output["Skewness"] = mx.where(second > 0, third / (second * mx.sqrt(second)), 0)
            if "Kurtosis" in requested:
                fourth = sum_rows(counts_f * centered**4) / total_f
                output["Kurtosis"] = mx.where(second > 0, fourth / second**2 - 3, 0)
            if "CoefficientOfVariation" in requested:
                assert mean is not None
                output["CoefficientOfVariation"] = mx.sqrt(second) * amplitude / mean
        if "MeanAbsoluteDeviation" in requested:
            output["MeanAbsoluteDeviation"] = (
                sum_rows(counts_f * mx.abs(centered)) / total_f * amplitude
            )

    percentile_names = {
        "10Percentile", "90Percentile", "Median", "InterquartileRange",
        "QuartileCoefficientOfDispersion", "RobustMeanAbsoluteDeviation",
        "MedianAbsoluteDeviation",
    }
    if requested & percentile_names:
        fractions: list[tuple[int, int]] = []
        if requested & {"10Percentile", "RobustMeanAbsoluteDeviation"}:
            fractions.append((1, 10))
        if requested & {"InterquartileRange", "QuartileCoefficientOfDispersion"}:
            fractions.extend(((1, 4), (3, 4)))
        if requested & {"Median", "MedianAbsoluteDeviation"}:
            fractions.append((1, 2))
        if requested & {"90Percentile", "RobustMeanAbsoluteDeviation"}:
            fractions.append((9, 10))
        percentile_levels, percentile_fractions = _linear_percentiles(
            counts, total, tuple(fractions),
        )
        parts = {fraction: (percentile_levels[:, column], percentile_fractions[:, column])
                 for column, fraction in enumerate(fractions)}
        split = max(0, ng.bit_length() - 24)
        for name, fraction in (("10Percentile", (1, 10)), ("Median", (1, 2)),
                               ("90Percentile", (9, 10))):
            if name in requested:
                output[name] = _interpolated_level(*parts[fraction], split)
        if requested & {"InterquartileRange", "QuartileCoefficientOfDispersion"}:
            p25_level, p25_fraction = parts[(1, 4)]
            p75_level, p75_fraction = parts[(3, 4)]
            interquartile = (
                (p75_level - p25_level).astype(mx.float32) + (p75_fraction - p25_fraction)
            )
            if "InterquartileRange" in requested:
                output["InterquartileRange"] = interquartile
            if "QuartileCoefficientOfDispersion" in requested:
                p25 = _interpolated_level(p25_level, p25_fraction, split)
                p75 = _interpolated_level(p75_level, p75_fraction, split)
                output["QuartileCoefficientOfDispersion"] = interquartile / (p75 + p25)
        if "MedianAbsoluteDeviation" in requested:
            median_level, median_fraction = parts[(1, 2)]
            median_offset = (
                (levels_i - median_level[:, None]).astype(mx.float32) - median_fraction[:, None]
            )
            output["MedianAbsoluteDeviation"] = sum_rows(counts_f * mx.abs(median_offset)) / total_f
        if "RobustMeanAbsoluteDeviation" in requested:
            assert offsets_i is not None
            p10_level, p10_fraction = parts[(1, 10)]
            p90_level, _ = parts[(9, 10)]
            inside = (
                (levels_i > p10_level[:, None])
                | ((levels_i == p10_level[:, None]) & (p10_fraction == 0)[:, None])
            ) & (levels_i <= p90_level[:, None])
            retained = mx.where(inside, counts, 0)
            retained_total = mx.sum(retained, axis=1)
            retained_total_f = retained_total.astype(mx.float32)
            retained_offsets = mx.sum(retained * offsets_i, axis=1)
            retained_divisor = mx.maximum(retained_total, 1)
            retained_mean_int = mx.floor_divide(retained_offsets, retained_divisor)
            retained_mean_fraction = (
                mx.remainder(retained_offsets, retained_divisor).astype(mx.float32)
                / mx.maximum(retained_total_f, 1)
            )
            output["RobustMeanAbsoluteDeviation"] = (
                sum_rows(mx.where(
                    inside,
                    counts_f * mx.abs(
                        (offsets_i - retained_mean_int[:, None]).astype(mx.float32)
                        - retained_mean_fraction[:, None]
                    ),
                    0,
                )) / retained_total_f
            )

    if requested & {"Minimum", "Maximum", "Range"}:
        present = (counts > 0).astype(mx.int32)
        minimum_level: mx.array | None = None
        maximum_level: mx.array | None = None
        if requested & {"Minimum", "Range"}:
            minimum_level = 1 + mx.argmax(present, axis=1).astype(mx.int64)
            if "Minimum" in requested:
                output["Minimum"] = minimum_level.astype(mx.float32)
        if requested & {"Maximum", "Range"}:
            maximum_level = ng - mx.argmax(mx.flip(present, axis=1), axis=1).astype(mx.int64)
            if "Maximum" in requested:
                output["Maximum"] = maximum_level.astype(mx.float32)
        if "Range" in requested:
            assert minimum_level is not None and maximum_level is not None
            output["Range"] = (maximum_level - minimum_level).astype(mx.float32)
    if "Mode" in requested:
        assert weighted_total is not None
        count_max = (mx.max(counts, axis=1) if reference is None or ng >= 2**31
                     else mx.take_along_axis(counts, reference[:, None], axis=1)[:, 0])
        is_mode = counts == count_max[:, None]
        distance = mx.abs(levels_i * total[:, None] - weighted_total[:, None])
        closest = mx.min(mx.where(is_mode, distance, mx.array(2**63 - 1, dtype=mx.int64)), axis=1)
        levels_f = mx.arange(1, ng + 1, dtype=mx.float32)[None, :]
        output["Mode"] = mx.min(
            mx.where(is_mode & (distance == closest[:, None]), levels_f, float(ng + 1)), axis=1,
        )
    if "Entropy" in requested:
        output["Entropy"] = count_entropy(counts, total, mx.float32)
    if "Uniformity" in requested:
        output["Uniformity"] = mx.sum(counts * counts, axis=1).astype(mx.float32) / total_f**2

    if requested & {"MaximumGradient", "MaximumGradientGrayLevel",
                    "MinimumGradient", "MinimumGradientGrayLevel"}:
        # Twice the gradient stays integer, retaining one-sided endpoint factors.
        if ng == 1:
            two_gradient = mx.zeros_like(counts)
        elif ng == 2:
            endpoint = 2 * (counts[:, 1:2] - counts[:, 0:1])
            two_gradient = mx.concatenate([endpoint, endpoint], axis=1)
        else:
            first = 2 * (counts[:, 1:2] - counts[:, 0:1])
            interior = counts[:, 2:] - counts[:, :-2]
            last = 2 * (counts[:, -1:] - counts[:, -2:-1])
            two_gradient = mx.concatenate([first, interior, last], axis=1)
        if "MaximumGradientGrayLevel" in requested:
            max_index = mx.argmax(two_gradient, axis=1).astype(mx.int64)
            output["MaximumGradientGrayLevel"] = (max_index + 1).astype(mx.float32)
            if "MaximumGradient" in requested:
                max_two = mx.take_along_axis(two_gradient, max_index[:, None], axis=1)[:, 0]
                output["MaximumGradient"] = max_two.astype(mx.float32) * 0.5
        elif "MaximumGradient" in requested:
            output["MaximumGradient"] = mx.max(two_gradient, axis=1).astype(mx.float32) * 0.5
        if "MinimumGradientGrayLevel" in requested:
            min_index = mx.argmin(two_gradient, axis=1).astype(mx.int64)
            output["MinimumGradientGrayLevel"] = (min_index + 1).astype(mx.float32)
            if "MinimumGradient" in requested:
                min_two = mx.take_along_axis(two_gradient, min_index[:, None], axis=1)[:, 0]
                output["MinimumGradient"] = min_two.astype(mx.float32) * 0.5
        elif "MinimumGradient" in requested:
            output["MinimumGradient"] = mx.min(two_gradient, axis=1).astype(mx.float32) * 0.5
    return {name: output[name] for name in INTENSITY_HISTOGRAM if name in requested}
