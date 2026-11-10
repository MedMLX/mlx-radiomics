"""Host preprocessing: normalization, resegmentation and gray-level discretization.

These steps retain float64 source intensities. Fixed-width bins classify the exact
binary64 values against a zero-origin lattice, or an explicit ``Settings.bin_anchor``.
Classification uses exact rational floors, without floating-step drift or epsilon ties.
Small lattices use equivalent upward-rounded thresholds; large lattices classify only
the sampled values without materializing their empty intervals.
Only the later feature arithmetic is float32.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import floor, frexp

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.contracts.types import HostArray


@dataclass(frozen=True)
class PreparedInput:
    """Validated float64 image and the masks derived from the label.

    Attributes:
        image: Float64 copy of the input, normalized if requested; same shape as the input.
        roi: Voxels of the selected label that remain after every resegmentation filter
            (the absolute or derived range intersected with any outlier-sigma window).
        shape_roi: Mask used for shape features: the original label mask, or the fully
            filtered ``roi`` when ``resegment_shape`` is set.
        sampling_mask: Voxels used for discretization and neighborhood sampling: ``roi``,
            or the whole image in voxel mode with ``masked_kernel=False``.
    """

    image: NDArray[np.float64]
    roi: NDArray[np.bool_]
    shape_roi: NDArray[np.bool_]
    sampling_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class Prepared(PreparedInput):
    """``PreparedInput`` plus gray-level discretization.

    Attributes:
        bins: Int32 array of the image's shape, or ``None`` when no selected feature
            reads this lattice. Inside ``sampling_mask`` it holds the 1-based gray-level
            bin; elsewhere 0. Raw first-order statistics use source intensities.
        gray_levels: Highest bin index (the number of gray levels, empty ones included).
            The unused-lattice sentinel is 1 when no selected feature reads these bins:
            raw intensity statistics, shape, physical local-intensity peaks or IVH
            (which has its own intensity lattice).
        integer_source_bounds: Internal admission proof from exact integer fixed-width
            binning. It qualifies later source preparation without rescanning samples.
    """

    bins: NDArray[np.int32] | None
    gray_levels: int
    integer_source_bounds: tuple[int, int] | None = None


def round_intensities(values: HostArray) -> NDArray[np.float64]:
    """Return a float64 copy of ``values`` rounded to the nearest integer.

    Rounding ties go to the nearest even integer, as ``numpy.rint`` and the numeric
    recovery of converted NIfTI CT arrays already do. IBSI configurations that interpolate
    CT intensities prescribe nearest-integer rounding of converted source values and of
    interpolated values before re-segmentation; the returned floats hold exact integers.

    Raises:
        ValueError: If 64-bit integer values are not exactly representable in float64;
            the check runs before conversion, so rounding cannot hide the loss.
    """
    return np.rint(exact_float64(values))


def validate_exact_float64(image: HostArray) -> None:
    """Refuse 64-bit integer values that conversion to float64 would round.

    Integer magnitudes above ``2**53`` are exact in float64 only when their odd part
    (the value with trailing zero bits removed) fits in 53 bits; others would be
    silently rounded, changing differences, bins and spread before any later precision
    guard can see the source values. The test runs in uint64 so neither operand is
    coerced to float64 first. Integer types of at most 32 bits are always exact.

    This is the admission check of :func:`exact_float64` without its float64 copy, so a
    caller that converts only one window of the source can still validate every source
    value, including integers outside that window. The check runs before any conversion:
    once converted, a float64 array no longer shows which integers were merged.

    Raises:
        ValueError: If a 64-bit integer value is not exactly representable in float64.
    """
    values = np.asarray(image)
    if values.dtype.kind in "iu" and values.dtype.itemsize == 8:
        # Reinterpreting bytes is only meaningful in native order: a big-endian array
        # viewed as native uint64 tests unrelated integers. The byte-order cast stays in
        # the integer domain, so it cannot round.
        native = values.astype(values.dtype.newbyteorder("="), copy=False)
        raw: NDArray[np.uint64] = native.view(np.uint64)
        # Two's-complement magnitude of negative int64 values (sign bit set in the
        # uint64 view); int64 minimum maps to 2**63, which is exact.
        negative = (raw >= np.uint64(2**63)) & (values.dtype.kind == "i")
        magnitude = np.where(negative, ~raw + np.uint64(1), raw)
        nonzero = magnitude[magnitude != 0]
        lowest_bit = nonzero & (~nonzero + np.uint64(1))
        if np.any(nonzero // lowest_bit >= np.uint64(2**53)):
            raise ValueError(
                "image contains 64-bit integer intensities that float64 cannot represent "
                "exactly; rescale or convert the image explicitly"
            )


def exact_float64(image: HostArray) -> NDArray[np.float64]:
    """Convert ``image`` to float64, refusing integer values that the conversion rounds.

    ``validate_exact_float64`` documents the representability rule and runs before the
    conversion, so rounding cannot hide the loss. Every entry path calls this before its
    first float conversion.

    Raises:
        ValueError: If a 64-bit integer value is not exactly representable in float64.
    """
    values = np.asarray(image)
    validate_exact_float64(values)
    return values.astype(np.float64)


def normalize_values(image: HostArray, settings: Settings) -> NDArray[np.float64]:
    """Return a float64 copy of ``image``, normalized over the whole image if requested.

    Normalization is ``(x - mean) / std`` over every voxel, not just the ROI, with the sample
    standard deviation (ddof=1) as in ITK's normalize filter. If ``remove_outliers`` is set
    the result is clipped to plus or minus that many standard deviations, then multiplied
    by ``normalize_scale``.

    Raises:
        ValueError: If normalization is requested and the image has fewer than two voxels,
            non-finite values or zero variance, or if 64-bit integer intensities are not
            exactly representable in float64.
    """
    values = exact_float64(image)
    if not settings.normalize:
        return values
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("normalization requires at least two finite image intensities")
    # Center on an observed value before reducing. Identical source values then have
    # exactly zero spread, and adjacent floats keep their sub-ulp mean offset. The z-score
    # does not depend on a common scale factor, so the values are first rescaled by an
    # exact power of two to unit magnitude (the subtraction cannot overflow near 1e308),
    # and the offsets again (their squares cannot overflow or flush to zero; 1e-162
    # offsets squared otherwise underflow into a wrong but finite deviation).
    values = _unit_scaled(values)
    offsets = _unit_scaled(values - values.flat[0])
    deviation = float(np.std(offsets, ddof=1))
    if deviation == 0 or not np.isfinite(deviation):
        raise ValueError("normalization requires nonzero finite whole-image variance")
    values = (offsets - np.mean(offsets)) / deviation
    if settings.remove_outliers is not None:
        values = np.clip(values, -settings.remove_outliers, settings.remove_outliers)
    return values * settings.normalize_scale


def _unit_scaled(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale ``values`` by the exact power of two that brings their peak into [0.5, 1)."""
    peak = float(np.max(np.abs(values)))
    return np.ldexp(values, -frexp(peak)[1] if peak > 0 else 0)


def _resegment_samples(
    values: NDArray[np.float64], roi: NDArray[np.bool_], settings: Settings,
    *, keep_samples: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.float64] | None]:
    """Apply the same thresholds in ROI sample order and optionally retain samples.

    Background comparisons never affect the retained mask. Operating on the already
    gathered ROI therefore removes full-grid comparison and float64 centering scratch.
    Retained samples are a subset of the vector validated here, so masked callers can
    reuse them for binning without another image gather or finite scan.
    """
    if settings.resegment_range is None:
        return roi, None
    selected = values[roi]
    if not np.all(np.isfinite(selected)):
        raise ValueError("resegmentation requires finite selected intensities")
    thresholds = np.asarray(settings.resegment_range, dtype=np.float64)
    compared = selected
    centered: NDArray[np.float64] | None = None
    spread = 0.0
    if settings.resegment_mode == "sigma" or settings.resegment_outlier_sigma is not None:
        # Keep sigma boundaries in centered coordinates rather than rounding a mean
        # back onto the original intensity scale before inclusive comparisons. The ROI
        # values are rescaled by an exact power of two before subtraction, so opposite
        # values near 1e308 cannot overflow, and the offsets again so their squares
        # neither overflow at 1e200 nor flush at 1e-200 (inf or 0 deviation keeps
        # every or almost no voxel). Inclusive comparisons are unchanged by those
        # rescalings, and all-equal ROI values still give exactly zero spread. Only ROI
        # voxels are centered: the scale is the ROI's, and outside values are discarded.
        scaled = _unit_scaled(selected)
        offsets = _unit_scaled(scaled - float(scaled[0]))
        centered = offsets - float(np.mean(offsets))
        spread = float(np.std(offsets))
    if settings.resegment_mode == "relative":
        thresholds = thresholds * np.max(selected)
    elif settings.resegment_mode == "sigma":
        assert centered is not None
        compared = centered
        thresholds = spread * thresholds
    thresholds = np.sort(thresholds)
    low = float(thresholds[0])
    high = float(thresholds[1]) if len(thresholds) == 2 else None
    retained_samples = compared >= low
    if high is not None:
        retained_samples &= compared <= high
    if settings.resegment_outlier_sigma is not None:
        assert centered is not None
        retained_samples &= np.abs(centered) <= settings.resegment_outlier_sigma * spread
    if not np.any(retained_samples):
        raise ValueError("resegmentation must retain at least one selected voxel")
    retained = np.zeros_like(roi)
    retained[roi] = retained_samples
    return retained, selected[retained_samples] if keep_samples else None


def _check_gray_levels(levels: int, settings: Settings) -> None:
    """Validate an actual lattice against index capacity and an explicit caller cap."""
    if not 1 <= levels <= np.iinfo(np.int32).max:
        raise ValueError("gray-level range exceeds int32 indexing capacity")
    if settings.max_gray_levels is not None and levels > settings.max_gray_levels:
        raise ValueError("gray-level range exceeds the explicit max_gray_levels limit")


def _lattice_bins(
    selected: NDArray[np.float64], origin: Fraction, width: Fraction,
    first: int, levels: int, *, closed_last: bool = False,
) -> NDArray[np.int32]:
    """Classify exact binary64 samples with bounded host lattice storage.

    A bounded threshold table keeps ordinary image preprocessing vectorized. Larger
    lattices classify only distinct sampled values by integer-ratio floor division.
    The threshold table is merely a representation: rounded coincident edges and an
    edge outside binary64 cannot change the exact mathematical classification.
    """
    if levels <= 65536:
        exact = [origin + index * width for index in range(first, first + levels + 1)]
        try:
            edges = np.array([float(edge) for edge in exact], dtype=np.float64)
        except OverflowError:
            # A finite maximum can occupy a bin whose upper edge exceeds binary64.
            # Direct integer-ratio classification does not need that edge.
            pass
        else:
            below = np.array([
                Fraction(float(edge)) < target
                for edge, target in zip(edges, exact, strict=True)
            ], dtype=np.bool_)
            with np.errstate(over="ignore"):
                edges = np.where(below, np.nextafter(edges, np.inf), edges)
            # Repeated upward-rounded thresholds correctly skip intervals containing
            # no representable source value; searchsorted preserves original indices.
            bins = np.searchsorted(edges, selected, side="right")
            if closed_last:
                bins = np.minimum(bins, levels)
            return bins.astype(np.int32)

    values, inverse = np.unique(selected, return_inverse=True)
    anchor_n, anchor_d = origin.numerator, origin.denominator
    width_n, width_d = width.numerator, width.denominator

    def classify(value: np.float64) -> int:
        numerator, denominator = float(value).as_integer_ratio()
        index = ((numerator * anchor_d - anchor_n * denominator) * width_d) // (
            denominator * anchor_d * width_n
        ) - first + 1
        # Fixed-count binning includes the maximum in its final interval.
        return min(index, levels) if closed_last else index

    bins = np.fromiter((classify(value) for value in values), dtype=np.int32, count=values.size)
    return bins[inverse]


def _exact_bins(selected: NDArray[np.float64], settings: Settings) -> NDArray[np.int32]:
    """Build exact indices while preserving the requested origin and interval width."""
    minimum, maximum = Fraction(float(selected.min())), Fraction(float(selected.max()))
    if settings.bin_count is not None:
        # Evaluate floor(B*(x-min)/(max-min)) over the supplied binary64 values.
        # A nearest-rounded 1/3 boundary would incorrectly promote float64(1/3).
        count = settings.bin_count
        _check_gray_levels(count, settings)
        return _lattice_bins(
            selected, minimum, (maximum - minimum) / count, 0, count, closed_last=True,
        )
    width = Fraction(float(settings.bin_width))
    # IBSI fixed bin size takes X_gl,min from the lower bound of a user-set
    # resegmentation range, so gray-level indices are comparable across samples; a
    # data-driven ROI minimum shifts every index-weighted texture feature per sample.
    # Relative and sigma ranges are data-driven and keep the sampled minimum.
    resegment_anchor = (
        settings.bin_anchor is None and settings.bin_origin == "resegment-or-minimum"
        and settings.resegment_mode == "absolute" and bool(settings.resegment_range)
    )
    if resegment_anchor:
        assert settings.resegment_range is not None
        anchor = Fraction(float(min(settings.resegment_range)))
    elif settings.bin_anchor is None and settings.bin_origin == "resegment-or-minimum":
        anchor = minimum
    else:
        anchor = Fraction(0 if settings.bin_anchor is None else float(settings.bin_anchor))
    if settings.bin_anchor is not None and minimum < anchor:
        raise ValueError(
            f"bin_anchor {settings.bin_anchor} is above the selected minimum {float(minimum)}"
        )
    if resegment_anchor and minimum < anchor:
        # Only unmasked voxel kernels sample below the range. Extending the lattice down
        # to those samples would renumber every ROI gray level by the background minimum,
        # silently abandoning the fixed anchor; refuse them as an explicit anchor does.
        raise ValueError(
            f"sampled minimum {float(minimum)} lies below the resegmentation lower bound "
            f"{float(anchor)} that anchors IBSI fixed-bin-size discretization; use "
            "masked_kernel=True or a resegment_range that covers the sampled image"
        )
    first = (
        floor(minimum / width)
        if settings.bin_anchor is None and settings.bin_origin == "zero-lattice"
        else 0
    )
    last = floor((maximum - anchor) / width)
    levels = last - first + 1
    _check_gray_levels(levels, settings)
    return _lattice_bins(selected, anchor, width, first, levels)


def _integer_fixed_width_bins(
    selected: NDArray[np.float64], settings: Settings,
) -> tuple[NDArray[np.int32], tuple[int, int]] | None:
    """Exact integer division for bounded integral sources and an integral lattice.

    Every admitted edge is an integer of magnitude below 2**33, hence is exactly
    representable in binary64. Dividing the nonnegative int32 offset by the integer
    width therefore selects the same interval as the rational-edge comparisons.
    Other sources/lattices and all questionable error cases keep ``_exact_bins``.
    """
    if settings.bin_count is not None:
        return None
    width = float(settings.bin_width)
    limit = int(np.iinfo(np.int32).max)
    if not 1 <= width <= limit or not width.is_integer():
        return None
    minimum, maximum = float(selected.min()), float(selected.max())
    if max(abs(minimum), abs(maximum)) > 2**24:
        return None
    # The conversion is bounded before it runs. Comparing against the exact source
    # refuses fractional values, including adjacent binary64 boundary neighbors.
    integers = selected.astype(np.int32)
    if not np.all(selected == integers):
        return None
    lower, upper, step = int(minimum), int(maximum), int(width)
    resegment_anchor = (
        settings.bin_anchor is None and settings.bin_origin == "resegment-or-minimum"
        and settings.resegment_mode == "absolute" and bool(settings.resegment_range)
    )
    if resegment_anchor:
        assert settings.resegment_range is not None
        anchor = float(min(settings.resegment_range))
    elif settings.bin_anchor is None and settings.bin_origin == "resegment-or-minimum":
        anchor = minimum
    else:
        anchor = 0.0 if settings.bin_anchor is None else float(settings.bin_anchor)
    if not -limit - 1 <= anchor <= limit or not anchor.is_integer():
        return None
    if (settings.bin_anchor is not None or resegment_anchor) and minimum < anchor:
        # The unrounded path raises the anchor-specific error.
        return None
    origin = int(anchor)
    first = lower // step if (
        settings.bin_anchor is None and settings.bin_origin == "zero-lattice"
    ) else 0
    last = (upper - origin) // step
    levels = last - first + 1
    if levels > limit or (
        settings.max_gray_levels is not None and levels > settings.max_gray_levels
    ):
        return None
    edge = origin + first * step
    if not -limit - 1 <= edge <= limit or not 0 <= lower - edge <= upper - edge <= limit:
        return None
    np.subtract(integers, np.int32(edge), out=integers)
    np.floor_divide(integers, np.int32(step), out=integers)
    np.add(integers, np.int32(1), out=integers)
    return integers, (lower, upper)


def prepare_sampled_input(
    image: HostArray, mask: HostArray, settings: Settings, *, voxel: bool,
) -> tuple[PreparedInput, NDArray[np.float64]]:
    """Validate inputs, then normalize and resegment, keeping the gathered samples.

    Args:
        image: Real-valued (int, uint or float) array, 2-D or 3-D, NumPy axis order.
        mask: Bool or integer array of the same shape.
        settings: Extraction settings; ``settings.label`` selects the ROI.
        voxel: True for voxel mode, which needs at least two ROI voxels and, with
            ``masked_kernel=False``, samples the whole image.

    Returns:
        The prepared masks and image, and the float64 samples used for binning.

    Raises:
        ValueError: If shapes or dtypes are unsupported, the settings are invalid, the
            label is absent, resegmentation or normalization fails, or sampled intensities
            are non-finite.
    """
    image, mask = np.asarray(image), np.asarray(mask)
    if image.ndim not in (2, 3) or image.shape != mask.shape or any(s == 0 for s in image.shape):
        raise ValueError("image and mask must share a nonempty 2D or 3D array shape")
    if image.dtype.kind not in "iuf" or mask.dtype.kind not in "biu":
        raise ValueError("image must be real numeric and mask must contain integer labels")
    settings.validate(image.ndim)
    values = normalize_values(image, settings)
    roi = mask == settings.label
    if not np.any(roi):
        raise ValueError("selected label is absent from mask")
    original_roi = roi
    roi, retained_samples = _resegment_samples(
        values, roi, settings, keep_samples=not (voxel and not settings.masked_kernel),
    )
    shape_roi = roi if settings.resegment_shape else original_roi
    if voxel and np.count_nonzero(roi) < 2:
        raise ValueError("voxel extraction requires at least two selected voxels")
    sampling = np.ones_like(roi) if voxel and not settings.masked_kernel else roi
    selected = values[sampling] if retained_samples is None else retained_samples
    if retained_samples is None and not np.all(np.isfinite(selected)):
        raise ValueError("sampled image intensities must be finite")
    return PreparedInput(values, roi, shape_roi, sampling), selected


def prepare_derived_input(
    image: HostArray, original: PreparedInput,
) -> tuple[PreparedInput, NDArray[np.float64]]:
    """Validate derived pixels while reusing the original, already selected masks.

    The pipeline disables normalization and resegmentation for derived images. Its
    original preparation therefore supplies the intensity/sampling/morphology masks,
    but each derived image still needs its own exact conversion and sampled-value
    finite check before its independent discretization.
    """
    image = np.asarray(image)
    if image.shape != original.image.shape:
        raise ValueError("image and mask must share a nonempty 2D or 3D array shape")
    if image.dtype.kind not in "iuf":
        raise ValueError("image must be real numeric and mask must contain integer labels")
    values = exact_float64(image)
    selected = values[original.sampling_mask]
    if not np.all(np.isfinite(selected)):
        raise ValueError("sampled image intensities must be finite")
    return (
        PreparedInput(values, original.roi, original.shape_roi, original.sampling_mask),
        selected,
    )


def prepare(image: HostArray, mask: HostArray, settings: Settings, *, voxel: bool) -> Prepared:
    """Prepare, validate and discretize the sampled voxels into gray-level bins.

    Binning uses ``settings.bin_count`` if set, otherwise ``settings.bin_width``. With a
    fixed width, an explicit ``settings.bin_anchor`` fixes the first edge and gray level 1
    at the anchor; ``None`` follows ``settings.bin_origin``: the absolute resegmentation
    lower bound (else the sampled minimum) for ``"resegment-or-minimum"``, or a zero-origin
    lattice, renumbered from the minimum bin, for ``"zero-lattice"``.
    Nonconstant fixed-count bins use exact source-value thresholds and include the maximum
    in the final bin. A constant ROI uses bin 1 for ``constant_roi_bin="first"`` and the
    midpoint bin ``bin_count // 2 + 1`` for ``"middle"``. When only shape
    features are selected, binning is skipped (``bins=None``, one gray level).
    The same applies when selected intensity features read only source intensities
    (first-order features except Entropy and Uniformity): unused bin settings cannot
    restrict their input domain.

    Raises:
        ValueError: As ``prepare_sampled_input``, or if the gray-level count exceeds
            ``settings.max_gray_levels`` or the sampled values lie below ``bin_anchor`` or
            below the absolute resegmentation bound that anchors IBSI fixed-width bins
            (possible only for unmasked voxel kernels).
    """
    data, selected = prepare_sampled_input(image, mask, settings, voxel=voxel)
    return discretize_prepared(data, selected, settings)


def discretize_prepared(
    data: PreparedInput, selected: NDArray[np.float64], settings: Settings,
) -> Prepared:
    """Discretize validated samples without repeating source or mask preparation.

    Only selections that read bins construct a lattice. Raw intensity statistics,
    shape, local intensity and IVH leave ``bins=None`` with the unused-lattice
    sentinel ``gray_levels=1``; IVH constructs its separate lattice later.
    """
    assign = any(
        not name.startswith(("shape", "localintensity_", "ivh_", "firstorder_"))
        or name in {"firstorder_Entropy", "firstorder_Uniformity"}
        for name in settings.features
    )
    if not assign:
        return Prepared(data.image, data.roi, data.shape_roi, data.sampling_mask, None, 1)
    bins = np.zeros(data.image.shape, dtype=np.int32)
    if settings.bin_count is not None and selected.min() == selected.max():
        level = 1 if settings.constant_roi_bin == "first" else settings.bin_count // 2 + 1
        _check_gray_levels(level, settings)
        bins[data.sampling_mask] = level
        return Prepared(data.image, data.roi, data.shape_roi, data.sampling_mask, bins, level)
    integer = _integer_fixed_width_bins(selected, settings)
    if integer is not None:
        # Exact integer classification also supplies the source bounds that admit
        # later integer first-order paths without rescanning the samples.
        digitized, integer_bounds = integer
    else:
        digitized = _exact_bins(selected, settings)
        integer_bounds = None
    ng = int(digitized.max())
    # Sampled levels are positive int32 values and every other bin is zero, so
    # their maximum is the full array's maximum without another full-grid scan.
    _check_gray_levels(ng, settings)
    bins[data.sampling_mask] = digitized
    return Prepared(
        data.image, data.roi, data.shape_roi, data.sampling_mask, bins, ng, integer_bounds,
    )


def directions(shape: tuple[int, ...], settings: Settings) -> tuple[tuple[int, ...], ...]:
    """Return the voxel offsets that define neighbor pairs, in array-axis order.

    An offset is kept if its Chebyshev (maximum-norm) length is in ``settings.distances``,
    its first nonzero component is positive (one of each opposite pair), it fits inside
    ``shape``, and, with ``force2d_axis`` set, it has no component along that axis. Offsets
    are listed from +radius down to -radius per axis, the order of PyRadiomics' angle list.
    """
    radius = max(settings.distances)
    offsets = product(range(radius, -radius - 1, -1), repeat=len(shape))
    return tuple(
        offset for offset in offsets
        if max(map(abs, offset)) in settings.distances
        and next((v for v in offset if v), 0) > 0
        and all(abs(v) < s for v, s in zip(offset, shape, strict=True))
        and (settings.force2d_axis is None or offset[settings.force2d_axis] == 0)
    )
