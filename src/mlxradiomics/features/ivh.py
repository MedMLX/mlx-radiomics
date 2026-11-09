"""Intensity-volume histograms with an independent, explicitly calibrated lattice."""

from collections.abc import Collection
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, frexp, lcm, ldexp, trunc

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import IVH, Settings
from mlxradiomics.features.intensity_histogram import histogram_order_values, integer_histogram
from mlxradiomics.numerics.runtime import mx


@dataclass(frozen=True)
class IVHLattice:
    """One-based bin indices, first physical intensity, step, and inclusive level count."""

    bins: NDArray[np.int32]
    origin: float
    step: float
    levels: int


def _lattice_edges(low: Fraction, interval: Fraction, levels: int) -> NDArray[np.float64]:
    """Round exact rational boundaries upward to their first binary64 representative.

    Binary lattices whose common integer numerators fit 53 bits are exact in NumPy's
    float64 multiply/add, avoiding a Fraction allocation for every boundary. Other
    lattices use integer ratios and exact cross products, retaining the same ceiling
    even for cancelling origins, fractional widths and fixed-count rational steps.
    """
    denominator = lcm(low.denominator, interval.denominator)
    start = low.numerator * (denominator // low.denominator)
    stride = interval.numerator * (denominator // interval.denominator)
    last = start + levels * stride
    if (denominator & (denominator - 1) == 0
            and max(abs(start), abs(last), abs(levels * stride)) <= 2**53
            and Fraction(float(interval)) == interval):
        return (np.arange(levels + 1, dtype=np.float64) * float(interval) + float(low))
    edges = np.empty(levels + 1, dtype=np.float64)
    down = np.empty(levels + 1, dtype=np.bool_)
    numerator = start
    for index in range(levels + 1):
        edge = numerator / denominator
        edge_numerator, edge_denominator = edge.as_integer_ratio()
        edges[index] = edge
        down[index] = edge_numerator * denominator < numerator * edge_denominator
        numerator += stride
    return np.where(down, np.nextafter(edges, np.inf), edges)


def prepare_ivh(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    settings: Settings,
    *,
    sampled: NDArray[np.float64] | None = None,
) -> IVHLattice:
    """Discretize IVH inputs independently of the texture histogram.

    No IVH discretizer means calibrated integer intensities with unit steps. Fixed count
    uses levels 1..B (arbitrary units). Fixed width reports physical bin centers. An
    explicit fixed-width range is covered by ``ceil((high-low)/width)`` bins, with
    at least one bin. A partial final bin retains its nominal full-bin center; the
    effective calibrated domain ends at that center. The explicit upper edge belongs
    to the last bin, including when it is an exact width boundary. An inferred range
    includes the observed maximum in its ordinary half-open bin. Thresholds compare
    exact supplied binary64 values; no approximate edge snapping is performed.
    Absolute ``resegment_range`` bounds are sorted here as they are in resegmentation, so
    reversed bounds select the same effective interval rather than being rejected.

    Bin-center calibration follows the IBSI IVH definition. Covering a nonaligned
    explicit interval extends its final bin to a complete nominal bin. This endpoint
    policy is explicit because the IBSI formula for a supplied upper bin edge does
    not otherwise define a partial-bin center. The returned origin, step and levels
    fully specify the effective domain, including unoccupied bins.
    """
    selected = image[mask] if sampled is None else sampled
    minimum, maximum = float(selected.min()), float(selected.max())
    bounds = settings.ivh_intensity_range
    if bounds is None and settings.resegment_mode == "absolute" and settings.resegment_range:
        lower = settings.resegment_range[0]
        upper = settings.resegment_range[1] if len(settings.resegment_range) == 2 else maximum
        # Resegmentation sorts its thresholds before clipping, so reversed absolute
        # bounds such as (4, 1) are the effective (1, 4) interval. Use the same effective
        # interval here instead of rejecting a request the preprocessing already accepted.
        bounds = (min(lower, upper), max(lower, upper))
    low, high = bounds or (minimum, maximum)
    if minimum < low or maximum > high:
        raise ValueError("IVH intensity range must contain every sampled intensity")
    bins = np.zeros(image.shape, dtype=np.int32)
    width = settings.ivh_bin_width
    count = settings.ivh_bin_count
    if count is None and width is None:
        if low != floor(low) or high != floor(high) or not np.all(selected == np.rint(selected)):
            raise ValueError("continuous IVH intensities require ivh_bin_width or ivh_bin_count")
        levels = int(high - low) + 1
        if not 1 <= levels < 2**31:
            raise ValueError("IVH lattice exceeds the supported int32 index range")
        bins[mask] = (selected - low).astype(np.int32) + 1
        return IVHLattice(bins, low, 1.0, levels)
    lo, hi = Fraction(low), Fraction(high)
    if count is not None:
        if settings.ivh_intensity_range is not None:
            raise ValueError("fixed-count IVH uses the sampled range, not ivh_intensity_range")
        lo, hi = Fraction(minimum), Fraction(maximum)
        if lo == hi:
            bins[mask] = 1
            return IVHLattice(bins, 1.0, 1.0, 1)
        levels = count
        interval = (hi - lo) / count
        origin, step = 1.0, 1.0
    else:
        assert width is not None
        interval = Fraction(width)
        if bounds is not None:
            levels = max(1, ceil((hi - lo) / interval))
        else:
            levels = floor((hi - lo) / interval) + 1
        origin, step = float(lo + interval / 2), width
    if not 1 <= levels < 2**31 or (levels + 1) * 32 > settings.scratch_mib * 1024**2:
        raise ValueError("IVH lattice exceeds scratch_mib or the supported index range")
    edges = _lattice_edges(lo, interval, levels)
    bins[mask] = np.minimum(np.searchsorted(edges, selected, side="right"), levels)
    return IVHLattice(bins, origin, step, levels)


_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
# Index limbs and step pieces carry at most 12 significant bits each, so every product
# of one limb and one piece has at most 24 bits and is exact in float32.
_LIMB_BITS = 12


def _truncated_pieces(value: float, bits: int) -> list[float]:
    """Split a binary64 value exactly into leading pieces of at most ``bits`` bits.

    Each piece is the truncation of the remaining value to ``bits`` significant bits,
    so the float64 subtraction that forms the next remainder is exact. Pieces below the
    normal float32 range are dropped: Metal flushes them, and their sum is below
    ``2**-125`` absolute.
    """
    pieces: list[float] = []
    remainder = value
    while remainder != 0:
        mantissa, exponent = frexp(remainder)
        piece = ldexp(trunc(ldexp(mantissa, bits)), exponent - bits)
        if abs(piece) < _FLOAT32_TINY:
            break
        pieces.append(piece)
        remainder -= piece
    return pieces


def _lattice_exponent(origin: float, step: float, levels: int) -> int:
    """Binary exponent ``E`` that scales the lattice's largest magnitude into ``[0.5, 1)``.

    Calibration runs on ``origin * 2**E`` and ``step * 2**E``, exact binary64 rescalings,
    so tiny coefficients (a ``2**-140`` step whose products reach normal float32 values)
    are not flushed or dropped before multiplication by the index. ``_restore`` applies
    ``2**-E`` afterwards.
    """
    magnitude = max(abs(origin), abs(step) * max(levels - 1, 1))
    return 0 if magnitude == 0 else -frexp(magnitude)[1]


def _restore(values: mx.array, exponent: int) -> mx.array:
    """Multiply by ``2**-exponent`` using two normal float32 power-of-two factors."""
    first = -exponent // 2
    return values * ldexp(1.0, first) * ldexp(1.0, -exponent - first)


def _calibrated_intensities(index: mx.array, origin: float, step: float) -> mx.array:
    """Return ``origin + index * step`` rounded to float32 for 0-based lattice indexes.

    Evaluating the affine map in float32 rounds ``index * step`` and then cancels it
    against a large ``origin``: the integer lattice ``[-1e8, 0, ..., 9]`` reports 8
    instead of 9, and a bin-width lattice starting at ``-999.95`` reports
    0.5499878 instead of 0.55. Here the binary64 step is split on the host into
    12-bit pieces and the index into 12-bit limbs, so every partial product is exact in
    float32; the binary64 origin is split into float32 pieces. These exact terms are
    added on Metal with a two-sum chain whose rounding errors are summed separately and
    added back once, so the result is the exactly evaluated affine value up to second
    order rounding of the error sum, followed by one final float32 rounding.
    """
    with_index = index.astype(mx.int64)
    limbs = [
        ((with_index >> (_LIMB_BITS * limb)) & ((1 << _LIMB_BITS) - 1)).astype(mx.float32)
        * float(2 ** (_LIMB_BITS * limb))
        for limb in range(3)
    ]
    terms = [mx.full(index.shape, piece, dtype=mx.float32)
             for piece in _truncated_pieces(origin, 24)]
    for piece in _truncated_pieces(step, _LIMB_BITS):
        terms.extend(limb * piece for limb in limbs)
    total = mx.zeros(index.shape, dtype=mx.float32)
    error = mx.zeros(index.shape, dtype=mx.float32)
    for term in terms:
        rounded = total + term
        back = rounded - total
        error = error + ((total - (rounded - back)) + (term - back))
        total = rounded
    return total + error


def ivh_statistics(
    bins: mx.array,
    valid: mx.array,
    levels: int,
    origin: float,
    step: float,
    *,
    selected: Collection[str] | None = None,
) -> dict[str, mx.array]:
    """Evaluate seven IVH outputs for batched samples on a shared lattice.

    Missing levels remain in the domain. Volume thresholds use integer cross products;
    intensity thresholds use the first lattice point whose survival count is at most
    the requested volume fraction. An absent threshold within the closed domain, or a
    constant one-level domain, produces NaN. Differences are formed before restoring
    physical units, so a common intensity offset cannot erase them. Reported
    intensities evaluate the calibrated affine map from exact float32 partial products
    with compensated summation before one final rounding (see
    :func:`_calibrated_intensities`), so a large origin cannot degrade a small value.
    Origin and step are first rescaled by an exact power of two (see
    :func:`_lattice_exponent`) and physical units restored last, so coefficients below
    the normal float32 range still yield their normal threshold and difference values.

    The area under the IVH curve integrates the discrete curve over every lattice level
    ``i = 1..L``: ``nu_i`` is the fraction of valid samples with bin at least ``i`` and
    ``gamma_i = (i - 1) / (L - 1)``. The trapezoidal rule is evaluated as the equivalent
    per-sample weight ``b - 0.5`` for a one-based bin ``b < L`` and ``L - 1`` for the
    last bin, divided by ``(L - 1)`` and the sample count. A ROI with one discretised
    intensity has area zero (IBSI 9CMM); no valid samples produce NaN.
    """
    if max(abs(origin), abs(step) * max(levels - 1, 0)) > _FLOAT32_MAX:
        # The compensated calibration needs every exact partial product in float32.
        raise ValueError("IVH calibrated lattice exceeds the float32 range")
    requested = frozenset(IVH if selected is None else selected)
    unknown = requested - set(IVH)
    if unknown:
        raise ValueError(f"unknown IVH statistics: {', '.join(sorted(unknown))}")
    if not requested:
        return {}
    exponent = _lattice_exponent(origin, step, levels)
    wide = bins.astype(mx.int64)
    volume_fractions = tuple(q for q in (10, 90) if (
        f"VolumeAtIntensityFraction{q}" in requested or "VolumeFractionDifference" in requested
    ))
    intensity_fractions = tuple(q for q in (10, 90) if (
        f"IntensityAtVolumeFraction{q}" in requested or "IntensityFractionDifference" in requested
    ))
    # A short integer lattice replaces an O(N log N) voxel sort with O(N) exact
    # counts and a small cumulative search. Long sparse lattices retain the sort,
    # bounding count storage by the source width rather than the physical range.
    use_counts = bool(intensity_fractions) and 1 <= levels <= bins.shape[1] < 2**32
    counts = integer_histogram(bins, valid, levels) if use_counts else None
    count = (mx.sum(counts, axis=1) if counts is not None
             else mx.sum(valid.astype(mx.int64), axis=1))
    defined = (count > 0) & (levels > 1)
    volumes: dict[int, mx.array] = {}
    intensities: dict[int, mx.array] = {}
    admissible: dict[int, mx.array] = {}
    for fraction in volume_fractions:
        threshold = 1 + ceil(Fraction(fraction * (levels - 1), 100))
        volumes[fraction] = (
            mx.sum(counts[:, threshold - 1:], axis=1) if counts is not None
            else mx.sum((valid & (bins >= threshold)).astype(mx.int64), axis=1)
        )
    if intensity_fractions:
        fractions = mx.array(intensity_fractions, dtype=mx.int64)[None, :]
        permitted = count[:, None] * fractions // 100
        orders = mx.maximum(count[:, None] - permitted - 1, 0)
        if counts is not None:
            threshold_levels = histogram_order_values(counts, count, orders) + 1
        else:
            # The invalid sentinel lies one beyond the largest admitted int32 bin.
            sorted_bins = mx.sort(mx.where(valid, wide, levels + 1), axis=1)
            threshold_levels = mx.take_along_axis(
                sorted_bins, orders.astype(mx.int32), axis=1,
            ) + 1
        for column, fraction in enumerate(intensity_fractions):
            level = threshold_levels[:, column]
            intensities[fraction] = level
            admissible[fraction] = defined & (level <= levels)

    output: dict[str, mx.array] = {}
    for fraction in volume_fractions:
        name = f"VolumeAtIntensityFraction{fraction}"
        if name in requested:
            output[name] = mx.where(defined, volumes[fraction] / count, float("nan"))
    calibrated_fractions = tuple(q for q in intensity_fractions
                                if f"IntensityAtVolumeFraction{q}" in requested)
    if calibrated_fractions:
        indexes = mx.stack([intensities[q] - 1 for q in calibrated_fractions], axis=1)
        calibrated = _restore(_calibrated_intensities(
            indexes, ldexp(origin, exponent), ldexp(step, exponent),
        ), exponent)
        for column, fraction in enumerate(calibrated_fractions):
            output[f"IntensityAtVolumeFraction{fraction}"] = mx.where(
                admissible[fraction], calibrated[:, column], float("nan"),
            )
    if "VolumeFractionDifference" in requested:
        output["VolumeFractionDifference"] = mx.where(
            defined, (volumes[10] - volumes[90]) / count, float("nan"),
        )
    if "IntensityFractionDifference" in requested:
        output["IntensityFractionDifference"] = mx.where(
            admissible[10] & admissible[90],
            _restore(
                (intensities[10] - intensities[90]).astype(mx.float32) * ldexp(step, exponent),
                exponent,
            ),
            float("nan"),
        )
    if "AreaUnderIVHCurve" in requested:
        if levels > 1:
            if counts is not None:
                occupied_levels = mx.arange(1, levels + 1, dtype=mx.int64)[None, :]
                doubled = mx.where(occupied_levels < levels,
                                   2 * occupied_levels - 1, 2 * (levels - 1))
                total = mx.sum(counts * doubled, axis=1)
                single_intensity = mx.sum((counts > 0).astype(mx.int32), axis=1) == 1
            else:
                occupied_min = mx.min(mx.where(valid, wide, levels + 1), axis=1)
                occupied_max = mx.max(mx.where(valid, wide, 0), axis=1)
                single_intensity = occupied_min == occupied_max
                doubled = mx.where(wide < levels, 2 * wide - 1, 2 * (levels - 1))
                total = mx.sum(mx.where(valid, doubled, 0), axis=1)
            denominator = 2 * (levels - 1) * mx.maximum(count, 1)
            area = total.astype(mx.float32) / denominator.astype(mx.float32)
            area = mx.where(single_intensity, 0.0, area)
        else:
            area = mx.zeros_like(count, dtype=mx.float32)
        output["AreaUnderIVHCurve"] = mx.where(count > 0, area, float("nan"))
    return {name: output[name] for name in IVH if name in requested}
