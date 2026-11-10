"""First-order intensity features and their source-precision guards."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import frexp, isfinite
from typing import cast

import numpy as np

from mlxradiomics.features.first_order_source import FirstOrderAdmission
from mlxradiomics.features.integer_quantiles import integer_source_quantiles
from mlxradiomics.numerics.count_statistics import count_entropy
from mlxradiomics.numerics.pair_arithmetic import folded_pair_differences, subtract_pairs
from mlxradiomics.numerics.precision import (
    pair_representation_error,
    robust_membership,
    source_order,
    source_quantiles,
)
from mlxradiomics.numerics.protocols import IndexedAddition, MetalKernel
from mlxradiomics.numerics.reduction import compensated_masked_pair_sum_rows
from mlxradiomics.numerics.reduction import sum_rows as _sum_rows
from mlxradiomics.numerics.runtime import evaluate, mx

_FIRST_ORDER_NAMES = frozenset({
    "10Percentile", "90Percentile", "CoefficientOfVariation", "Energy", "Entropy",
    "InterquartileRange", "Kurtosis", "Maximum", "Mean", "MeanAbsoluteDeviation",
    "Median", "MedianAbsoluteDeviation", "Minimum", "QuartileCoefficientOfDispersion",
    "Range", "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness", "TotalEnergy",
    "Uniformity", "Variance",
})
# Spread-dependent outputs cannot be recovered when distinct float64 source values
# collapse to one float32 value plus residual pair.  Mean, extrema, percentiles and
# the histogram outputs remain meaningful because their reported float32 value is
# unchanged by that particular loss of source resolution.
_SPREAD_FEATURES = frozenset({
    "CoefficientOfVariation", "InterquartileRange", "Kurtosis", "MeanAbsoluteDeviation",
    "MedianAbsoluteDeviation", "QuartileCoefficientOfDispersion", "Range",
    "RobustMeanAbsoluteDeviation", "Skewness", "Variance",
})
# Largest accepted ratio of a row's pair representation error to its standard deviation.
# Perturbing every sample by at most r*sigma changes the variance by at most
# (4r + 4r^2)*sigma^2 (Cauchy-Schwarz on the centered deviations), so 2**-20 keeps that
# source loss below 4e-6 relative, inside the float32 reduction error of the moments.
_PAIR_RESOLUTION = 2.0**-20
# Outputs that add the configured shift before squaring.
_SHIFT_FEATURES = frozenset({"Energy", "RootMeanSquared", "TotalEnergy"})
_ORDERED_FEATURES = frozenset({
    "10Percentile", "90Percentile", "InterquartileRange", "Maximum", "Median",
    "MedianAbsoluteDeviation", "Minimum", "QuartileCoefficientOfDispersion", "Range",
    "RobustMeanAbsoluteDeviation",
})


@dataclass(frozen=True)
class FirstOrderInputs:
    """Source buffers read by the requested arithmetic and its precision guards."""

    residual: bool
    source_bits: bool
    representation_error: bool


def first_order_inputs(
    selected: Collection[str] | None, source: FirstOrderAdmission | None, *, shift: float,
) -> FirstOrderInputs:
    """Plan optional source buffers from conservative admission, before GPU upload.

    Omitting a buffer never removes a precision guard. An unproved source retains
    its bit patterns for normal-range, spread and shifted-energy checks; ordered
    outputs retain them unless the original bounded-integer histogram route applies.
    """
    requested = _FIRST_ORDER_NAMES if selected is None else frozenset(selected)
    arithmetic = bool(requested - _BIN_FEATURES)
    integer = source is not None and source.bounded_integer_mean
    spread = bool(requested & _SPREAD_FEATURES) and not integer
    with np.errstate(over="ignore"):
        finite_shift = isfinite(float(np.float32(shift)))
    shifted = (bool(requested & _SHIFT_FEATURES) and finite_shift
               and not (integer and shift == 0))
    ordered = (bool(requested & _ORDERED_FEATURES)
               and not (source is not None and source.integer_quantiles))
    return FirstOrderInputs(
        residual=arithmetic and not (source is not None and source.zero_residual),
        source_bits=arithmetic and (
            source is None or not source.source_normal or spread or shifted or ordered
        ),
        representation_error=(spread or shifted)
        and not (source is not None and source.exact_pair),
    )


# Float64 bit pattern of the smallest normal float32 magnitude, 2**-126. Metal flushes
# float32 subnormals to zero, so nonzero sources below it cannot enter the arithmetic.
_FLOAT32_TINY_BITS = int(np.array(np.finfo(np.float32).tiny, dtype=np.float64).view(np.uint64))
_BIN_FEATURES = frozenset({"Entropy", "Uniformity"})


def first_order_histogram(
    bins: mx.array, valid: mx.array, ng: int, *, dtype: mx.Dtype,
    selected: Collection[str] | None = None, count: mx.array | None = None,
) -> dict[str, mx.array]:
    """Compute Entropy/Uniformity from bins without a source-intensity representation.

    Counts, their reduction tree and the entropy evaluator match :func:`first_order`.
    These outputs depend only on discretized bins, so source subnormals remain admitted.
    ``count`` can reuse a caller's valid-voxel count for each batch row.
    """
    requested = _BIN_FEATURES if selected is None else frozenset(selected)
    unknown = requested - _BIN_FEATURES
    if unknown:
        raise ValueError(f"unknown first-order histogram statistics: {', '.join(sorted(unknown))}")
    if not requested:
        return {}
    if count is None:
        count = _sum_rows(valid.astype(mx.int32))
    rows = mx.arange(bins.shape[0], dtype=mx.int32)[:, None]
    index = rows * ng + mx.maximum(bins - 1, 0)
    level_counts = mx.zeros((bins.shape[0] * ng,), dtype=mx.uint32)
    level_counts = cast(IndexedAddition, level_counts.at[mx.reshape(index, (-1,))]).add(
        mx.reshape(valid.astype(mx.uint32), (-1,))
    )
    level_counts = mx.reshape(level_counts, (bins.shape[0], ng))
    output: dict[str, mx.array] = {}
    if "Entropy" in requested:
        output["Entropy"] = count_entropy(level_counts, count, dtype)
    if "Uniformity" in requested:
        p = level_counts.astype(dtype) / count[:, None]
        output["Uniformity"] = _sum_rows(p**2)
    return output


@lru_cache(maxsize=16)
def _sample_kernel(expression: str, inputs: tuple[str, ...]) -> MetalKernel:
    """Compile a Metal kernel evaluating ``expression`` once per sample.

    ``row`` is defined only when the expression uses it. Kernels are cached by
    expression and input names.
    """
    row = "\n        uint row = t / width;" if "row" in expression else ""
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_first_order_samples", input_names=list(inputs),
        output_names=["out"],
        source=f"""
        uint t = thread_position_in_grid.x;
        uint width = {inputs[0]}_shape[1];
        if (t >= {inputs[0]}_shape[0] * width) return;{row}
        out[t] = {expression};
        """,
    ))


def _per_sample(expression: str, dtype: mx.Dtype, **inputs: mx.array) -> mx.array:
    """Evaluate one exact per-sample predicate or selection over ``(rows, n)`` inputs.

    The first input fixes the shape; ``row`` indexes per-row ``(rows,)`` inputs. Each use
    holds only integer tests, float comparisons, ``fabs`` or a selection, which round
    nothing, so it equals the corresponding MLX array expression without forming its
    intermediate arrays.
    """
    first = next(iter(inputs.values()))
    return _sample_kernel(expression, tuple(inputs))(
        inputs=list(inputs.values()), grid=(first.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[first.shape], output_dtypes=[dtype],
    )[0]


_MAGNITUDE = "0x7FFFFFFFFFFFFFFFul"


def _masked_abs(values: mx.array, mask: mx.array) -> mx.array:
    """``where(mask, abs(values), 0)``: absolute values inside ``mask``, positive zero outside."""
    return _per_sample("mask[t] ? fabs(values[t]) : 0.0f", mx.float32, values=values, mask=mask)


class _FirstOrderTerms:
    """First-order intermediates, each constructed on first use and then reused.

    MLX executes only the graph that requested outputs need, but constructing unused
    graphs still costs host time on every ROI call and voxel batch. Each property holds
    the arithmetic of the corresponding intermediate, so a requested output and its
    checks are the same operations on the same operands whatever else is requested.
    """

    def __init__(
        self, values: mx.array, valid: mx.array, *, shift: float, voxel_volume: float,
        residual: mx.array, source_bits: mx.array | None, representation_error: mx.array | None,
        kurtosis: str, zero_variance_moments: str, source: FirstOrderAdmission | None,
    ) -> None:
        """Hold one batch of samples and the host facts that select algorithms.

        ``values``/``residual`` are the float32 high/low pair of each source sample,
        ``valid`` marks samples inside the ROI or kernel, and ``source`` carries the
        host admission proofs from :mod:`mlxradiomics.features.first_order_source`.
        """
        self.values = values
        self.valid = valid
        self.shift = shift
        self.voxel_volume = voxel_volume
        self.residual = residual
        self.source_bits = source_bits
        self.representation_error = representation_error
        self.kurtosis_convention = kurtosis
        self.zero_variance_moments = zero_variance_moments
        self.source = source
        # Host admission proofs (see first_order_source) select exact integer algorithms.
        self.integer_source = source is not None and source.bounded_integer_mean
        self.integer_ranks = source is not None and source.integer_quantiles
        # Split the scalar parameter on the host; all intensity arithmetic stays on Metal.
        with np.errstate(over="ignore"):
            self.shift_high = float(np.float32(shift))
        self.shift_is_finite = isfinite(self.shift_high)

    @cached_property
    def valid_counts(self) -> mx.array:
        """``valid`` as int32 indicators."""
        return self.valid.astype(mx.int32)

    @cached_property
    def count(self) -> mx.array:
        """Number of valid samples per row."""
        return _sum_rows(self.valid_counts)

    @cached_property
    def order(self) -> mx.array:
        """Per-row sample indices sorted by exact source value."""
        assert self.source_bits is not None
        return source_order(self.source_bits, self.valid)

    @cached_property
    def flushed(self) -> mx.array:
        """Rows holding a valid nonzero source below the normal float32 range."""
        assert self.source_bits is not None
        return mx.any(_per_sample(
            f"valid[t] && (bits[t] & {_MAGNITUDE}) > 0"
            f" && (bits[t] & {_MAGNITUDE}) < {_FLOAT32_TINY_BITS}ul",
            mx.bool_, bits=self.source_bits, valid=self.valid,
        ), axis=1)

    def flushed_extrema(self, requested: frozenset[str]) -> mx.array:
        """Reject only a selected extremum that cannot be represented on Metal.

        Source ordering still uses binary64 bits. A tiny interior value therefore
        cannot invalidate a normal maximum or minimum, and a true zero remains zero.
        """
        assert self.source_bits is not None
        indices: list[mx.array] = []
        if "Minimum" in requested:
            indices.append(self.order[:, :1])
        if "Maximum" in requested:
            indices.append(mx.take_along_axis(self.order, (self.count - 1)[:, None], axis=1))
        selected = mx.concatenate(indices, axis=1)
        bits = mx.take_along_axis(self.source_bits, selected, axis=1)
        magnitude = mx.bitwise_and(bits, mx.array(2**63 - 1, dtype=mx.uint64))
        return mx.any((magnitude > 0) & (magnitude < int(_FLOAT32_TINY_BITS)), axis=1)

    @cached_property
    def center_index(self) -> mx.array:
        """Column of the first valid sample in each row, shape ``(rows, 1)``."""
        # Center on an observed valid sample. Choosing the first valid column rather than
        # an arbitrary origin keeps the centered values small for translated data.
        return mx.argmax(self.valid_counts, axis=1).astype(mx.int32)[:, None]

    @cached_property
    def source_amplitude(self) -> mx.array:
        """Largest absolute valid sample per row."""
        return mx.max(_masked_abs(self.values, self.valid), axis=1)

    @cached_property
    def centering_scale(self) -> mx.array:
        """Per-row factor 0.5 for near-overflow rows, else 1."""
        # Only extreme rows need a binary half-scale before subtraction. This preserves
        # pair precision while preventing a finite opposite-sign span from overflowing;
        # ordinary and tiny values keep their original exponent range.
        return mx.where(self.source_amplitude > float(np.finfo(np.float32).max) / 2, 0.5, 1)

    @cached_property
    def centered(self) -> tuple[mx.array, mx.array]:
        """Return ``(amplitude, centered_scaled)``: offsets from the center, normalized."""
        center_high = mx.take_along_axis(self.values, self.center_index, axis=1)[:, 0]
        center_low = mx.take_along_axis(self.residual, self.center_index, axis=1)[:, 0]
        # Fold compensated differences only after removing the shared source offset.
        offset = folded_pair_differences(
            self.values, self.residual, center_high, center_low,
            scale=self.centering_scale, mask=self.valid,
        )
        # Standardized moments are dimensionless. Remove amplitude before reductions and
        # powers; restore it only for dimensionful outputs, without first squaring the scale.
        amplitude = mx.max(mx.abs(offset), axis=1)
        scaled = offset / mx.where(amplitude > 0, amplitude, 1)[:, None]
        mean_offset = _sum_rows(scaled) / self.count
        return amplitude, mx.where(self.valid, scaled - mean_offset[:, None], 0)

    @cached_property
    def amplitude(self) -> mx.array:
        """Largest absolute centered offset per row."""
        return self.centered[0]

    @cached_property
    def second(self) -> mx.array:
        """Second central moment of the amplitude-normalized samples."""
        return _sum_rows(self.centered[1]**2) / self.count

    @cached_property
    def variance(self) -> mx.array:
        """Population variance in source intensity units squared."""
        return (self.second * self.amplitude) * self.amplitude / self.centering_scale**2

    @cached_property
    def deviation(self) -> mx.array:
        """Population standard deviation in source intensity units."""
        return mx.sqrt(self.second) * self.amplitude / self.centering_scale

    @cached_property
    def source_gap(self) -> mx.array:
        """Largest pair representation error among valid samples per row."""
        if self.source is not None and self.source.exact_pair:
            return mx.zeros((self.values.shape[0],), dtype=self.values.dtype)
        assert self.representation_error is not None
        return mx.max(mx.where(self.valid, self.representation_error, 0), axis=1)

    @cached_property
    def spread_unresolved(self) -> mx.array:
        """Rows whose computed spread cannot be trusted at float32 pair precision.

        True when distinct sources produced zero spread, or when the pair representation
        error is not negligible against a nonzero standard deviation.
        """
        assert self.source_bits is not None
        # A valid source differs from some other source iff it differs from the first
        # valid source. Signed zeros remain equal observations. This guard needs one
        # scalar gather per row and does not depend on percentile sorting.
        center_bits = mx.take_along_axis(self.source_bits, self.center_index, axis=1)[:, 0]
        distinct_sources = mx.any(_per_sample(
            f"valid[t] && bits[t] != center[row]"
            f" && ((bits[t] & {_MAGNITUDE}) != 0 || (center[row] & {_MAGNITUDE}) != 0)",
            mx.bool_, bits=self.source_bits, valid=self.valid, center=center_bits,
        ), axis=1)
        # Zero computed spread is exact only for equal sources (constant 0.1 shares one
        # inexact pair). Distinct sources with zero spread collapsed to one pair or lost
        # their differences to flushing; a nonzero spread must dominate the pair error.
        deviation = self.deviation
        return (distinct_sources & (deviation == 0)) | (
            (deviation > 0) & (self.source_gap > _PAIR_RESOLUTION * deviation)
        )

    @cached_property
    def skewness(self) -> mx.array:
        """Skewness; 0 (IBSI) or NaN (PyRadiomics) for zero-variance rows."""
        second = self.second
        third = _sum_rows(self.centered[1]**3) / self.count
        constant = 0 if self.zero_variance_moments == "zero" else float("nan")
        return mx.where(second > 0, third / (second * mx.sqrt(second)), constant)

    @cached_property
    def kurtosis(self) -> mx.array:
        """Kurtosis; IBSI reports excess kurtosis, PyRadiomics the raw moment ratio.

        Zero-variance rows give 0 (IBSI) or NaN (PyRadiomics).
        """
        second = self.second
        fourth = _sum_rows(self.centered[1]**4) / self.count
        constant = 0 if self.zero_variance_moments == "zero" else float("nan")
        offset = 3 if self.kurtosis_convention == "excess" else 0
        return mx.where(second > 0, fourth / second**2 - offset, constant)

    @cached_property
    def mean_deviation(self) -> mx.array:
        """Mean absolute deviation from the mean."""
        return (
            _sum_rows(mx.abs(self.centered[1])) / self.count * self.amplitude
        ) / self.centering_scale

    @cached_property
    def shifted(self) -> mx.array:
        """Samples plus ``voxelArrayShift``, formed from the compensated pair."""
        if self.shift == 0 and self.source is not None and self.source.zero_residual:
            return self.values
        if self.shift_is_finite:
            other_high = mx.array([-self.shift_high], dtype=mx.float32)
            other_low = mx.array([-(self.shift - self.shift_high)], dtype=mx.float32)
            # A finite source plus a finite shift can still overflow the float32 sum. The
            # high-part difference is then already +-inf, and the residual compensation
            # would turn it into NaN; keep the overflow classification instead.
            return folded_pair_differences(
                self.values, self.residual, other_high, other_low, keep_overflow=True,
            )
        # Preserve float32 overflow for shifts outside its exponent range.
        return self.values + self.shift

    @cached_property
    def shifted_amplitude(self) -> mx.array:
        """Largest absolute shifted valid sample per row."""
        return mx.max(_masked_abs(self.shifted, self.valid), axis=1)

    @cached_property
    def shift_unresolved(self) -> mx.array:
        """Rows whose shifted Energy/RMS are below the pair's resolvable precision."""
        # The shifted pair is exact only up to the source pair error plus the shift's own
        # pair error. When the shift cancels the source offset those errors can dominate
        # or hide the shifted value entirely, so a zero or inaccurate Energy/RMS is refused
        # rather than inferred from the rounded pair. Finite shifts only.
        assert self.source_bits is not None
        shift_gap = mx.array(
            float(pair_representation_error(np.array([self.shift], dtype=np.float64))[0]),
            dtype=mx.float32,
        )
        source_gap = self.source_gap + shift_gap
        wanted = int(np.float64(-self.shift).view(np.uint64))
        wanted_bits = mx.array(np.array([wanted], dtype=np.uint64))
        # Both signed zeros denote the same shifted value.
        either_zero = (f" || (bits[t] & {_MAGNITUDE}) == 0"
                       if (wanted & 0x7FFFFFFFFFFFFFFF) == 0 else "")
        exact_zero = mx.all(_per_sample(
            f"!valid[t] || bits[t] == wanted[0]{either_zero}",
            mx.bool_, bits=self.source_bits, valid=self.valid, wanted=wanted_bits,
        ), axis=1)
        shifted_amplitude = self.shifted_amplitude
        return (
            ((shifted_amplitude == 0) & ~exact_zero)
            | (
                (shifted_amplitude > 0) & mx.isfinite(shifted_amplitude)
                & (source_gap > _PAIR_RESOLUTION * shifted_amplitude)
            )
        )

    @cached_property
    def integer_order_statistics(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Quantile pairs and extrema from integer ranks of an admitted integral source."""
        source = self.source
        assert source is not None
        assert source.integer_minimum is not None and source.integer_maximum is not None
        return integer_source_quantiles(
            self.values, self.residual, self.valid, self.count,
            minimum=source.integer_minimum, maximum=source.integer_maximum,
        )

    @cached_property
    def extrema(self) -> tuple[mx.array, mx.array, mx.array]:
        """Return ``(minimum, maximum, range)`` selected and differenced in source order."""
        if self.integer_ranks:
            _, _, minimum, maximum = self.integer_order_statistics
            minimum_low = mx.zeros_like(minimum)
            maximum_low = mx.zeros_like(maximum)
        else:
            order = self.order
            minimum_index = order[:, 0]
            maximum_index = mx.take_along_axis(order, (self.count - 1)[:, None], axis=1)[:, 0]
            minimum = mx.take_along_axis(self.values, minimum_index[:, None], axis=1)[:, 0]
            maximum = mx.take_along_axis(self.values, maximum_index[:, None], axis=1)[:, 0]
            minimum_low = mx.take_along_axis(self.residual, minimum_index[:, None], axis=1)[:, 0]
            maximum_low = mx.take_along_axis(self.residual, maximum_index[:, None], axis=1)[:, 0]
        range_high, range_low = subtract_pairs(maximum, maximum_low, minimum, minimum_low)
        return minimum, maximum, range_high + range_low

    @cached_property
    def quantiles(self) -> tuple[mx.array, mx.array]:
        """``(high, low)`` pairs of the 10th, 25th, 50th, 75th and 90th percentiles."""
        if self.integer_ranks:
            high, low, _, _ = self.integer_order_statistics
            return high, low
        return source_quantiles(self.values, self.residual, self.order, self.count)

    @cached_property
    def percentile(self) -> mx.array:
        """The five percentile values, columns ordered 10, 25, 50, 75, 90."""
        high, low = self.quantiles
        return high + low

    @cached_property
    def interquartile_range(self) -> mx.array:
        """75th minus 25th percentile, differenced in pair precision."""
        high, low = self.quantiles
        iqr_high, iqr_low = subtract_pairs(high[:, 3], low[:, 3], high[:, 1], low[:, 1])
        return iqr_high + iqr_low

    @cached_property
    def median_abs_deviation(self) -> mx.array:
        """Mean absolute distance to the median (PyRadiomics ``MedianAbsoluteDeviation``)."""
        # Median absolute deviation is the mean absolute distance to the median, not the
        # median of the absolute deviations. Center on the source-precision median pair
        # and scale before the absolute reduction so an opposite-sign span cannot overflow.
        high, low = self.quantiles
        median_offset = folded_pair_differences(
            self.values, self.residual, high[:, 2], low[:, 2],
            scale=self.centering_scale, mask=self.valid,
        )
        median_amplitude = mx.max(mx.abs(median_offset), axis=1)
        median_scale = mx.where(median_amplitude > 0, median_amplitude, 1)
        return (
            _sum_rows(mx.abs(median_offset / median_scale[:, None])) / self.count
            * median_scale / self.centering_scale
        )

    @cached_property
    def quartile_coefficient(self) -> mx.array:
        """Quartile coefficient of dispersion, ``(P75 - P25) / (P75 + P25)``."""
        # Scale both quartiles before forming the numerator and denominator. The ratio is
        # dimensionless, so this prevents avoidable sum/span overflow without changing
        # its value or folding away source residuals.
        high, low = self.quantiles
        quartile_scale = mx.where(
            mx.maximum(mx.abs(high[:, 3]), mx.abs(high[:, 1]))
            > float(np.finfo(np.float32).max) / 2,
            0.5, 1.0,
        )
        upper_high = high[:, 3] * quartile_scale
        upper_low = low[:, 3] * quartile_scale
        lower_high = high[:, 1] * quartile_scale
        lower_low = low[:, 1] * quartile_scale
        difference_high, difference_low = subtract_pairs(
            upper_high, upper_low, lower_high, lower_low,
        )
        sum_high, sum_low = subtract_pairs(upper_high, upper_low, -lower_high, -lower_low)
        return (difference_high + difference_low) / (sum_high + sum_low)

    @cached_property
    def robust(self) -> tuple[mx.array, mx.array]:
        """Return ``(robust_offset_amplitude, robust_deviation)`` of the 10-90% subset."""
        high, low = self.quantiles
        robust = robust_membership(self.values, self.residual, self.valid, high, low)
        robust_integer = robust.astype(mx.int32)
        robust_count = _sum_rows(robust_integer)
        # Binary scaling before subtraction prevents an opposite-sign span overflow
        # without rounding ordinary translated source pairs. Normalize offsets separately
        # before reduction so their sum cannot overflow either.
        robust_amplitude = mx.max(_masked_abs(self.values, robust), axis=1)
        robust_scale = mx.where(
            robust_amplitude > float(np.finfo(np.float32).max) / 2, 2.0, 1.0
        )
        # An excluded outlier must not set the origin: its large offsets can erase all
        # variation inside the retained subset before the robust mean is subtracted.
        robust_index = mx.argmax(robust_integer, axis=1)[:, None]
        robust_offset = folded_pair_differences(
            self.values, self.residual,
            mx.take_along_axis(self.values, robust_index, axis=1)[:, 0],
            mx.take_along_axis(self.residual, robust_index, axis=1)[:, 0],
            scale=robust_scale, divide=True, mask=robust,
        )
        robust_offset_amplitude = mx.max(mx.abs(robust_offset), axis=1)
        robust_offset_scale = mx.where(robust_offset_amplitude > 0, robust_offset_amplitude, 1)
        robust_offset = robust_offset / robust_offset_scale[:, None]
        robust_mean_offset = _sum_rows(robust_offset) / robust_count
        robust_centered = mx.where(robust, robust_offset - robust_mean_offset[:, None], 0)
        robust_deviation = (
            _sum_rows(mx.abs(robust_centered)) / robust_count
            * robust_offset_scale * robust_scale
        )
        return robust_offset_amplitude, robust_deviation

    @cached_property
    def mean(self) -> mx.array:
        """Mean of the valid samples, exact for admitted integer sources."""
        if self.integer_source:
            # At most 2**40 absolute integer mass, over at most 31 tree levels: each
            # addition error is integral and the sum of their magnitudes is below 2**24.
            # The compensated per-level error sums and their running accumulator are
            # therefore exact integers. Folding the final error rounds the exact integer
            # total once, exactly as the int64 reduction/cast does here.
            total = mx.sum(mx.where(self.valid, self.values, 0).astype(mx.int64), axis=1)
            return total.astype(self.values.dtype) / self.count
        # Reduce the original parts for the reported mean: reconstructing it from an
        # arbitrary sample plus mean_offset causes cancellation for sign-balanced data.
        # Scale only rows whose raw sum can overflow, preserving ordinary pair precision.
        # The two-sum reduction retains the small exact total when large opposite-sign
        # values cancel, which a plain float32 tree reduces to zero.
        mean_scale = mx.where(
            self.source_amplitude > float(np.finfo(np.float32).max) / self.count,
            self.source_amplitude, 1,
        )
        # Values and residuals share one compensated tree: residuals that cancel each
        # other next to a surviving small residual (1, 2**-26, -1) must not round the
        # survivor away in a separate plain reduction.
        return compensated_masked_pair_sum_rows(
            self.values, self.residual, self.valid, mean_scale,
        ) / self.count * mean_scale

    @cached_property
    def rms_scale(self) -> mx.array:
        """Finite positive normalizer for Energy and RMS, else 1."""
        # Infinite shifted values must keep their overflow classification: inf/inf
        # would replace positive-infinite RMS with NaN. Only finite scales normalize.
        amplitude = self.shifted_amplitude
        return mx.where((amplitude > 0) & mx.isfinite(amplitude), amplitude, 1)

    @cached_property
    def scaled_energy(self) -> mx.array:
        """Sum of squared shifted samples after division by ``rms_scale``."""
        # Energy can overflow even when RMS is representable. Normalize before squaring
        # and restore the intensity scale only after the square root.
        return _sum_rows(mx.where(self.valid, (self.shifted / self.rms_scale[:, None])**2, 0))

    @cached_property
    def rms(self) -> mx.array:
        """Root mean square of the shifted samples."""
        return mx.sqrt(self.scaled_energy / self.count) * self.rms_scale

    @cached_property
    def energy(self) -> mx.array:
        """Sum of squared shifted samples (PyRadiomics ``Energy``)."""
        # A square below 2**-126 is flushed on Metal even when Energy itself is normal.
        # Only rows holding such a value restore the amplitude after the normalized
        # reduction; the others keep the direct sum, exact for small integer squares.
        shifted = self.shifted
        # 0x20000000 is the float32 bit pattern of 2**-63.
        tiny_square = mx.any(_per_sample(
            "valid[t] && shifted[t] != 0.0f && fabs(shifted[t]) < as_type<float>(0x20000000u)",
            mx.bool_, shifted=shifted, valid=self.valid,
        ), axis=1)
        return mx.where(
            tiny_square,
            (self.scaled_energy * self.rms_scale) * self.rms_scale,
            _sum_rows(mx.where(self.valid, shifted**2, 0)),
        )

    @cached_property
    def total_energy(self) -> mx.array:
        """``Energy`` multiplied by the voxel volume, without intermediate overflow."""
        # TotalEnergy has its own range: Energy * volume can be representable when Energy
        # is not (two 1e20 samples in 1e-6 mm**2 voxels). Restore the squared amplitude
        # with the voxel volume's binary exponent split across both factors.
        volume_mantissa, volume_exponent = frexp(self.voxel_volume)
        half_exponent = volume_exponent // 2
        restored_scale = self.rms_scale * 2.0**half_exponent
        return (
            (self.scaled_energy * volume_mantissa) * restored_scale
            * (restored_scale * 2.0 ** (volume_exponent - 2 * half_exponent))
        )

    @cached_property
    def nonzero_shifted(self) -> mx.array:
        """Rows holding at least one nonzero shifted valid sample."""
        return mx.any(_per_sample(
            "valid[t] && shifted[t] != 0.0f", mx.bool_, shifted=self.shifted, valid=self.valid,
        ), axis=1)


type _Term = Callable[[_FirstOrderTerms], mx.array]

# Output order of the reported mapping; histogram outputs follow the intensity outputs.
_FIRST_ORDER_OUTPUTS: dict[str, _Term] = {
    "10Percentile": lambda t: t.percentile[:, 0],
    "90Percentile": lambda t: t.percentile[:, 4],
    "Median": lambda t: t.percentile[:, 2],
    "InterquartileRange": lambda t: t.interquartile_range,
    "MeanAbsoluteDeviation": lambda t: t.mean_deviation,
    "MedianAbsoluteDeviation": lambda t: t.median_abs_deviation,
    # Coefficient of variation is dimensionless; evaluate the scaled population standard
    # deviation before dividing by the reported mean.
    "CoefficientOfVariation": lambda t: t.deviation / t.mean,
    "QuartileCoefficientOfDispersion": lambda t: t.quartile_coefficient,
    "RobustMeanAbsoluteDeviation": lambda t: t.robust[1],
    "Mean": lambda t: t.mean,
    "Minimum": lambda t: t.extrema[0],
    "Maximum": lambda t: t.extrema[1],
    "Range": lambda t: t.extrema[2],
    "Variance": lambda t: t.variance,
    "Skewness": lambda t: t.skewness,
    "Kurtosis": lambda t: t.kurtosis,
    "Energy": lambda t: t.energy,
    "TotalEnergy": lambda t: t.total_energy,
    "RootMeanSquared": lambda t: t.rms,
}
# Each condition establishes a nonzero true value from the operands, so a zero result
# can only be float32 underflow (including Metal's subnormal flush).
_UNDERFLOW_CONDITIONS: dict[str, _Term] = {
    "Variance": lambda t: (t.deviation > 0) & (t.variance == 0),
    "MeanAbsoluteDeviation": lambda t: (t.deviation > 0) & (t.mean_deviation == 0),
    "MedianAbsoluteDeviation": lambda t: (t.deviation > 0) & (t.median_abs_deviation == 0),
    "RobustMeanAbsoluteDeviation": lambda t: (t.robust[0] > 0) & (t.robust[1] == 0),
    "Range": lambda t: (t.deviation > 0) & (t.extrema[2] == 0),
    "InterquartileRange": lambda t: (
        ((t.quantiles[0][:, 3] != t.quantiles[0][:, 1])
         | (t.quantiles[1][:, 3] != t.quantiles[1][:, 1]))
        & (t.interquartile_range == 0)
    ),
    "Energy": lambda t: t.nonzero_shifted & (t.energy == 0),
    "TotalEnergy": lambda t: t.nonzero_shifted & (t.total_energy == 0),
    "RootMeanSquared": lambda t: (t.shifted_amplitude > 0) & (t.rms == 0),
}


def first_order(
    values: mx.array, valid: mx.array, bins: mx.array | None, ng: int, *,
    shift: float, voxel_volume: float, residual: mx.array | None, source_bits: mx.array | None,
    representation_error: mx.array | None, selected: Collection[str] | None = None,
    kurtosis: str = "excess", zero_variance_moments: str = "zero",
    source: FirstOrderAdmission | None = None,
) -> dict[str, mx.array]:
    """Compute first-order intensity statistics.

    Centered moments use the float32 value plus its source residual as a float-float pair.
    Each row is centered on a selected sample with a compensated pair subtraction before
    any reduction, so variance, skewness and kurtosis preserve variation and translation
    even when multiple source intensities round to the same float32 value (for example
    near ``1e8``). Moment arithmetic and reductions remain native float32; the residual
    recovers source-scale differences before those reductions and orders source values
    for the percentile-derived outputs. Moment offsets are scaled before reductions and
    powers, so standardized moments do not require representable dimensionful powers.
    Variance restores the amplitude after its reduction. Energy and RootMeanSquared add
    the configured shift as a compensated pair before squaring, preserving source variation
    when the shift cancels a large source offset. Energy retains float32 squaring and
    reduction overflow limits, but rows with a square below the normal float32 range
    reduce amplitude-normalized squares so flushed squares still count; TotalEnergy
    reduces amplitude-normalized squares and applies the squared amplitude and voxel
    volume afterwards, so it is finite and normal whenever its own value is.
    RootMeanSquared removes amplitude before squaring and restores it after the square
    root, retaining representable results when Energy overflows. Infinite shifted
    intensities retain positive-infinite RMS, and a finite source plus a finite shift
    whose exact sum exceeds the float32 range reports positive infinity rather than NaN
    from the compensation arithmetic.

    Variance, skewness and kurtosis use population (divide by n) moments.  By default
    (``kurtosis="excess"``, ``zero_variance_moments="zero"``) three is subtracted from
    kurtosis and both standardized moments are zero when the variance is zero (IBSI IPH6
    and its first-order companions).  ``kurtosis="raw"`` keeps the PyRadiomics ratio and
    ``zero_variance_moments="nan"`` the PyRadiomics NaN for zero-variance rows.
    MeanAbsoluteDeviation is the mean
    absolute distance from the mean.  MedianAbsoluteDeviation is the mean absolute
    distance from the median (not the median of the deviations).  CoefficientOfVariation
    is the population standard deviation divided by the mean, and
    QuartileCoefficientOfDispersion is ``(P75 - P25) / (P75 + P25)``.
    Minimum, Maximum, Range and the percentiles are selected and interpolated in source
    order, so Range and InterquartileRange retain residual-scale differences. Mean,
    percentiles and the other outputs are float32. Mean reduces the value and residual
    parts together in one two-sum tree that folds back each addition's rounding error,
    scaling rows whose raw sum could overflow, so a small exact total survives
    cancellation such as ``[1e8, 1, -1e8]`` in either part. Reduction
    rounding remains float32 rather than an exactly rounded float64 mean.
    Entropy and Uniformity use the gray-level histogram of ``bins``; entropy keeps integer
    histogram counts until its stable evaluation.
    RobustMeanAbsoluteDeviation is the mean absolute deviation, about their own mean, of
    the values between the 10th and 90th percentiles; that membership test is done at
    source precision (see ``mlxradiomics.numerics.precision``). Extreme retained subsets use
    binary scaling before centering and magnitude scaling before reduction, so finite
    opposite-sign spans cannot overflow a representable reported deviation. An empty
    trimmed subset, possible with two unequal observations, has undefined robust
    deviation and returns NaN.

    A float32 value plus float32 residual pair need not represent its float64 source:
    near ``0.1`` the ``nextafter`` neighbours share one pair, and near ``1e12`` the
    residual itself rounds, perturbing differences of ``1e-3`` by about ``2**-9``.
    A requested spread-dependent output raises ``RuntimeError`` when a row's computed
    standard deviation is zero although its sources differ, or when the largest pair
    representation error exceeds ``2**-20`` of that deviation (see
    ``_PAIR_RESOLUTION``). Otherwise the representation changes centered moments by at
    most about ``4e-6`` relative and Range, InterquartileRange and the absolute
    deviations by at most ``2**-19`` standard deviations absolute, so a shared pair
    inside a resolved spread is accepted. Requests containing only Mean, Minimum,
    Maximum, percentiles or histogram outputs are not checked for spread resolution.
    Energy, TotalEnergy and RootMeanSquared also validate the source-plus-shift pair:
    when the shift cancels a source offset below the pair representation resolution, the
    nonzero shifted value cannot be recovered from the rounded pair and the request raises
    rather than reporting zero.

    Metal flushes float32 subnormals to zero. Arithmetic requests raise when a valid
    nonzero source lies below ``2**-126`` in magnitude. Extrema-only requests check
    the selected extrema, so an unrelated tiny sample cannot invalidate a normal
    minimum or maximum. Entropy and Uniformity read integer bins instead.
    Variance, Range, InterquartileRange, the absolute deviations, Energy,
    TotalEnergy and RootMeanSquared raise when the result is zero although its operands
    establish a nonzero value: their true value lies below the normal float32 range and
    a reported zero would be indistinguishable from constant data. Results above the
    float32 range remain ``inf``.

    Only the requested outputs, their intermediates and their checks are constructed.
    The pair subtractions, the masked mean leaves and the source-order keys run as fused
    Metal kernels whose separately rounded operations equal the corresponding MLX array
    operations. One GPU wait completes the intensity outputs with every applicable check;
    failures are then raised in order: subnormal sources, unresolved spread, unresolved
    shifted energy, underflow. Entropy and Uniformity are returned unevaluated.

    Args:
        values: Float32 array ``(batch, n)`` of intensities, one row per voxel set (the
            ROI, or the neighborhood of one center).
        valid: Boolean array of the same shape; voxels that belong to each set.
        bins: Integer array of the same shape with 1-based gray levels; may be omitted
            when neither Entropy nor Uniformity is selected.
        ng: Number of gray levels.
        shift: ``voxel_array_shift``, added to intensities for Energy, TotalEnergy and
            RootMeanSquared only.
        voxel_volume: Volume of one voxel (product of the spacing); scales TotalEnergy.
        residual: Float32 array, ``float64 source - float32 value``; may be omitted when
            source admission proves it zero. Used for the
            compensated centering and intensity-shift arithmetic, extrema and quantiles, and
            the robust percentile membership.
        source_bits: uint64 bit patterns of the float64 source intensities, used to sort
            values in source order and to detect collapsed source distinctions. May be
            omitted only when ``first_order_inputs`` proves no selected operation reads it.
        representation_error: Float32 upper bound on ``|source - (value + residual)|``
            as evaluated on Metal, from
            :func:`mlxradiomics.numerics.precision.pair_representation_error`. May be omitted when
            admission proves zero error or the selected operations do not inspect it.
        selected: Statistic names without the ``firstorder_`` prefix, or ``None`` to
            return every statistic.  When supplied, only those names are returned and
            the source-resolution check is limited to the requested spread features.
        kurtosis: ``"excess"`` (default) or ``"raw"``, as described above.
        zero_variance_moments: ``"zero"`` (default) or ``"nan"``, as described above.
        source: Internal single-ROI source admission. Exact bounded integers admit an
            integer mean and bounded histogram ranks, and checks that such sources
            cannot fail are omitted; other sources retain their compensated reductions,
            source-order sorting and checks.

    Returns:
        Feature name without the ``firstorder_`` prefix mapped to a ``(batch,)`` float32
        array. Callers never pass a row with no valid voxel; the result for one is undefined.

    Raises:
        ValueError: If a convention is unknown, or ``selected`` contains an unknown name.
        RuntimeError: If a requested spread-dependent output cannot be resolved from the
            float32-plus-residual representation, a requested shifted output cannot be
            resolved from the source-plus-shift pair, a requested arithmetic output has a
            nonzero source below the normal float32 range, or a requested dimensional
            output underflows the float32 range.
    """
    if kurtosis not in {"excess", "raw"}:
        raise ValueError("kurtosis must be excess or raw")
    if zero_variance_moments not in {"zero", "nan"}:
        raise ValueError("zero_variance_moments must be zero or nan")
    requested = None if selected is None else frozenset(selected)
    if requested is not None:
        unknown = requested - _FIRST_ORDER_NAMES
        if unknown:
            raise ValueError(f"unknown first-order statistics: {', '.join(sorted(unknown))}")
        if not requested:
            return {}
        if requested <= _BIN_FEATURES:
            if bins is None:
                raise ValueError("first-order histogram bins are required")
            return first_order_histogram(
                bins, valid, ng, dtype=values.dtype, selected=requested,
            )

    inputs = first_order_inputs(selected, source, shift=shift)
    if residual is None:
        if inputs.residual:
            raise ValueError("first-order source residual is required")
        residual = mx.zeros_like(values)
    if source_bits is None and inputs.source_bits:
        raise ValueError("first-order source bits are required")
    if representation_error is None and inputs.representation_error:
        raise ValueError("first-order representation error is required")
    terms = _FirstOrderTerms(
        values, valid, shift=shift, voxel_volume=voxel_volume, residual=residual,
        source_bits=source_bits, representation_error=representation_error, kurtosis=kurtosis,
        zero_variance_moments=zero_variance_moments,
        source=source,
    )
    features = {
        name: output(terms) for name, output in _FIRST_ORDER_OUTPUTS.items()
        if requested is None or name in requested
    }
    if requested is None or not requested.isdisjoint(_BIN_FEATURES):
        if bins is None:
            raise ValueError("first-order histogram bins are required")
        features.update(first_order_histogram(
            bins, valid, ng, dtype=values.dtype,
            selected=None if requested is None else requested & _BIN_FEATURES,
            count=terms.count,
        ))
    # Rejected-sample counts in reporting precedence, then one flag per requested output
    # that can underflow. A proven-normal source cannot hold a flushed value; an admitted
    # integer source has an exact pair, so its spread and unshifted energy are resolved
    # and only TotalEnergy's volume scaling can underflow.
    exact_unshifted = terms.integer_source and shift == 0
    rejections: list[tuple[str, mx.array]] = []
    if source is None or not source.source_normal:
        rejections.append(("flushed", terms.flushed_extrema(requested)
                           if requested is not None
                           and requested <= _BIN_FEATURES | {"Minimum", "Maximum"}
                           else terms.flushed))
    if not terms.integer_source and (
        requested is None or not requested.isdisjoint(_SPREAD_FEATURES)
    ):
        rejections.append(("spread", terms.spread_unresolved))
    if terms.shift_is_finite and not exact_unshifted and (
        requested is None or not requested.isdisjoint(_SHIFT_FEATURES)
    ):
        rejections.append(("shift", terms.shift_unresolved))
    underflow = [
        (name, condition(terms)) for name, condition in _UNDERFLOW_CONDITIONS.items()
        if (requested is None or name in requested)
        and (not exact_unshifted or name == "TotalEnergy")
    ]
    if not rejections and not underflow:
        return features
    flags = mx.stack([
        *(mx.sum(rows.astype(mx.int32)) for _, rows in rejections),
        *(mx.any(condition).astype(mx.int32) for _, condition in underflow),
    ])
    # One wait completes the checked outputs and all checks. Rejections are then raised
    # in reporting precedence, with their sample counts and requested-feature
    # names. Entropy and Uniformity enter no check; they stay lazy for the caller's
    # evaluation, so their gray-level tables do not coexist with the intensity graph.
    evaluate(flags, *(value for name, value in features.items() if name not in _BIN_FEATURES))
    counts = [int(count) for count in np.asarray(flags).tolist()]
    for (kind, _), samples in zip(rejections, counts, strict=False):
        if samples:
            raise _rejection(kind, samples, requested)
    underflowed = sorted(
        name for (name, _), failed in zip(underflow, counts[len(rejections):], strict=True)
        if failed
    )
    if underflowed:
        raise RuntimeError(f"first-order {underflowed} underflow the float32 range")
    return features


def _rejection(kind: str, samples: int, requested: frozenset[str] | None) -> RuntimeError:
    """Return the diagnostic of an unresolved subnormal, spread or shifted-energy check."""
    if kind == "flushed":
        names = sorted(_FIRST_ORDER_NAMES - _BIN_FEATURES if requested is None
                       else requested - _BIN_FEATURES)
        return RuntimeError(
            f"first-order intensities are unresolved for {samples} sample(s) and "
            f"requested feature(s) {names}: nonzero float64 sources below 2**-126 "
            "are flushed to zero by float32 arithmetic on Metal"
        )
    if kind == "spread":
        names = sorted(_SPREAD_FEATURES if requested is None else requested & _SPREAD_FEATURES)
        return RuntimeError(
            "first-order source spread is unresolved for "
            f"{samples} sample(s) and requested feature(s) {names}: the float32 value "
            "plus residual pair misrepresents distinct float64 sources by more than "
            "2**-20 of the standard deviation"
        )
    names = sorted(_SHIFT_FEATURES if requested is None else requested & _SHIFT_FEATURES)
    return RuntimeError(
        "first-order shifted energy is unresolved for "
        f"{samples} sample(s) and requested feature(s) {names}: the float32 value "
        "plus residual pair and the represented shift lose source information "
        "relative to the shifted intensities"
    )

