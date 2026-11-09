"""Derived feature images (PyRadiomics image types) computed on Metal.

Each transform yields ``FeatureImage`` objects whose names become the feature-key prefix,
for example ``wavelet-HHL`` or ``log-sigma-2-0-mm-3D``. Existing names follow
PyRadiomics; the IBSI 2 filters use names that include their scientific parameters and
padding convention. The IBSI 2 filter kernels and response maps live in
:mod:`mlxradiomics.filters`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np

from mlxradiomics.config.filtering import (
    FilterPadding,
    FourierGrid,
    LogKernel,
    OrientationPooling,
    RieszBase,
)
from mlxradiomics.config.transforms import (
    ImageType,
    TransformSettings,
    gabor_scale_pairs,
    validate_filter_shape,
)
from mlxradiomics.config.wavelets import wavelet_filters
from mlxradiomics.filters import (
    FFT_AVAILABLE,
    FFT_DISABLED_REASON,
    gabor_responses,
    laws_energy,
    laws_response,
    log_response,
    mean_response,
    require_isotropic_spacing,
    riesz_log_responses,
    riesz_simoncelli_responses,
    separable_wavelet_responses,
    simoncelli_response,
)
from mlxradiomics.filters.lbp2d import lbp2d
from mlxradiomics.filters.riesz_steering import align_riesz, riesz_multi_indices
from mlxradiomics.numerics.runtime import evaluate, mx


@dataclass(frozen=True)
class FeatureImage:
    """A derived image: ``name`` is the feature-key prefix, ``image`` a float32 MLX array."""

    name: str
    image: mx.array


def _periodic_filter(
    image: mx.array, coefficients: tuple[float, ...], axis: int, dilation: int,
) -> mx.array:
    """Convolve periodically along ``axis`` with taps ``dilation`` voxels apart.

    This is the stationary (a trous) wavelet convolution, with the same phase as
    PyWavelets ``swtn`` with ``norm=False``. Each tap is evaluated before the next
    is gathered, so only one shifted copy is resident instead of one per tap.
    """
    positions = mx.arange(image.shape[axis], dtype=mx.int32)
    result = mx.zeros_like(image)
    for k, coefficient in enumerate(coefficients):
        indexes = (positions + (len(coefficients) // 2 - k) * dilation) % image.shape[axis]
        result = result + mx.take(image, indexes, axis=axis) * coefficient
        evaluate(result)
    return result


def _wavelet_codes(ndim: int, axis2d: int | None) -> tuple[str, ...]:
    """Return wavelet axis codes in output order (approximation first)."""
    codes = [""]
    for axis in range(ndim - 1, -1, -1):
        if axis != axis2d:
            codes = [name + suffix for name in codes for suffix in "LH"]
    return tuple(codes)


def _wavelet_prefix(output_level: int) -> str:
    """Return the image-name prefix for wavelet ``output_level``."""
    return "wavelet" if output_level == 1 else f"wavelet{output_level}"


def _log_name(sigma: float, options: TransformSettings) -> str:
    """Return the derived image name of a LoG response at ``sigma`` mm."""
    if options.log.kernel is LogKernel.RECURSIVE:
        return f"log-sigma-{str(sigma).replace('.', '-')}-mm-3D"
    return "-".join((
        "log-sampled", f"s{_tag(sigma)}mm", f"c{_tag(options.log.cutoff)}",
        _padding_tag(options.log.padding, options.log.constant),
        "2D" if options.log.planar else "3D",
    ))


def _tag(value: float) -> str:
    """Return a lossless identifier tag for one binary64 parameter value.

    ``repr`` is the shortest decimal that round-trips to the same float, so distinct
    scientific parameters keep distinct names, including adjacent sigmas and
    orientations. Whole floats drop the trailing ``.0`` to preserve readable names
    such as ``s1mm``. A leading sign becomes ``m`` and the decimal point ``p``; ``repr`` never
    emits ``m`` or ``p`` and always signs an exponent, so the substitution stays
    injective over representable parameter values.
    """
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _padding_tag(padding: FilterPadding, constant: float) -> str:
    """Return the name tag for a padding mode, including the constant value."""
    return (padding.value if padding is not FilterPadding.CONSTANT
            else f"constant{_tag(constant)}")


def _mean_name(options: TransformSettings) -> str:
    """Return the derived image name of the mean filter response."""
    name = f"mean-M{options.mean.support}-{_padding_tag(options.mean.padding,
                                                        options.mean.constant)}"
    return f"{name}-2D" if options.mean.planar else name


def _laws_name(options: TransformSettings) -> str:
    """Return the derived image name of the Laws response or energy map."""
    parts = [
        "laws", options.laws.kernel,
        _padding_tag(options.laws.padding, options.laws.constant),
    ]
    if options.laws.pooling is not OrientationPooling.NONE:
        parts.extend(("pool", options.laws.pooling.value))
    if options.laws.energy_distance is not None:
        parts.extend(("energy", f"d{options.laws.energy_distance}"))
    return "-".join(parts)


def _gabor_plane_tag(options: TransformSettings, ndim: int) -> str:
    """Return the Gabor plane tag: empty in 2-D, else ``planes`` or ``axis<k>``."""
    if ndim == 2:
        return ""
    return "planes" if options.gabor.slice_axis is None else (
        f"axis{options.gabor.slice_axis}"
    )


def _gabor_base_name(
    options: TransformSettings, sigma: float, wavelength: float,
    thetas: tuple[float, ...],
) -> str:
    """Return the Gabor image name without its plane tag."""
    parts = [
        "gabor",
        f"s{_tag(sigma)}mm",
        f"l{_tag(wavelength)}mm",
        f"g{_tag(options.gabor.gamma)}",
    ]
    if options.gabor.pooling is OrientationPooling.NONE:
        # One lossless angle tag per unpooled orientation map.
        (theta,) = thetas
        parts.append(f"t{_tag(theta)}")
    else:
        # A request has one theta set, so pooled maps cannot collide on orientations;
        # the exact angles are recorded in the transform metadata. Listing every
        # lossless angle would exceed the 255-byte file-name limit of exported maps
        # (16 IBSI orientations need about 430 characters), so only the count appears.
        parts.append(f"o{len(thetas)}")
    parts.extend((
        options.gabor.response.value,
        _padding_tag(options.gabor.padding, options.gabor.constant),
    ))
    if options.gabor.pooling is not OrientationPooling.NONE:
        parts.extend(("pool", options.gabor.pooling.value))
    return "-".join(parts)


def _gabor_names(options: TransformSettings, ndim: int) -> tuple[str, ...]:
    """Return every Gabor derived image name, one per map in output order."""
    plane = _gabor_plane_tag(options, ndim)
    names: list[str] = []
    for sigma, wavelength in gabor_scale_pairs(options):
        if options.gabor.pooling is OrientationPooling.NONE:
            names.extend(
                "-".join(filter(None, (_gabor_base_name(
                    options, sigma, wavelength, (theta,),
                ), plane)))
                for theta in options.gabor.theta
            )
        else:
            # Pooling collapses the orientations into one map named by their count.
            names.append(
                "-".join(filter(None, (_gabor_base_name(
                    options, sigma, wavelength, options.gabor.theta,
                ), plane)))
            )
    return tuple(names)


def _riesz_order_tag(order: tuple[int, ...]) -> str:
    """Return the name tag of a Riesz order, preserving IBSI ``(k1, ..., kD)`` order."""
    return "l" + "-".join(str(power) for power in order)


def _riesz_name(
    options: TransformSettings, order: tuple[int, ...], scale: float,
) -> str:
    """Return the derived image name of one Riesz component.

    ``scale`` is the LoG sigma in mm for a LoG base and the integer Simoncelli level
    for a Simoncelli base.
    """
    parts = ["riesz", options.riesz.base.value]
    if options.riesz.base is RieszBase.LOG:
        parts.extend((f"s{_tag(scale)}mm", f"c{_tag(options.riesz.log_cutoff)}"))
    else:
        parts.append(f"L{int(scale)}")
    parts.append(_riesz_order_tag(order))
    if options.riesz.base is RieszBase.LOG:
        parts.append(_padding_tag(options.riesz.padding, options.riesz.constant))
    if options.riesz.grid is not FourierGrid.DFT:
        parts.append(options.riesz.grid.value)
    if options.riesz.tensor_sigma is not None:
        parts.extend(("aligned", f"t{_tag(options.riesz.tensor_sigma)}mm"))
    return "-".join(parts)


def _riesz_computed_orders(
    options: TransformSettings, ndim: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the Riesz orders to filter: the requested ones, or the steering banks.

    Steering mixes every component of one total order, so it needs the complete
    order-``L`` bank of each requested ``L`` plus the first-order bank that builds
    the structure tensor. Orders are IBSI ``(k1, ..., kD)`` tuples throughout.
    """
    if not options.riesz.steering:
        return options.riesz.orders
    orders = list(riesz_multi_indices(ndim, 1))
    for length in sorted({sum(order) for order in options.riesz.orders}):
        orders.extend(riesz_multi_indices(ndim, length))
    return tuple(dict.fromkeys(orders))


def _riesz_requested(
    options: TransformSettings,
    computed: tuple[tuple[int, ...], ...],
    responses: list[mx.array],
    spacing: tuple[float, ...],
) -> list[mx.array]:
    """Steer the computed banks if requested and return maps in ``riesz_orders`` order."""
    if options.riesz.tensor_sigma is None:
        return responses
    maps = dict(zip(computed, responses, strict=True))
    ndim = len(spacing)
    first_order = [maps[order] for order in riesz_multi_indices(ndim, 1)]
    aligned: dict[tuple[int, ...], mx.array] = {}
    for length in sorted({sum(order) for order in options.riesz.orders}):
        bank = riesz_multi_indices(ndim, length)
        aligned.update(zip(bank, align_riesz(
            [maps[order] for order in bank], first_order, length, spacing,
            options.riesz.tensor_sigma, options.riesz.padding,
        ), strict=True))
    return [aligned[order] for order in options.riesz.orders]


def _separable_wavelet_name(options: TransformSettings, band: str) -> str:
    """Return the derived image name of one separable wavelet band."""
    parts = [
        "separablewavelet", options.separable_wavelet.name, f"L{options.separable_wavelet.level}",
        band, _padding_tag(options.separable_wavelet.padding,
                           options.separable_wavelet.constant),
    ]
    if options.separable_wavelet.pooling is not OrientationPooling.NONE:
        parts.extend(("pool", options.separable_wavelet.pooling.value))
    if options.separable_wavelet.planar:
        parts.append("2D")
    return "-".join(parts)


def _simoncelli_name(options: TransformSettings, level: int) -> str:
    """Return the derived image name of one Simoncelli level."""
    name = f"simoncelli-L{level}"
    if options.simoncelli.grid is not FourierGrid.DFT:
        name = f"{name}-{options.simoncelli.grid.value}"
    return f"{name}-2D" if options.simoncelli.planar else name


def _require_unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Reject repeated derived-image names before any image or sink is produced."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        if name in seen:
            repeated.add(name)
        seen.add(name)
    if repeated:
        raise ValueError(
            "derived image names must be unique; repeated: "
            + ", ".join(sorted(repeated))
        )
    return names


_DISABLED_FFT_IMAGE_TYPES = frozenset((ImageType.SIMONCELLI, ImageType.RIESZ))


def _require_available_image_types(kinds: tuple[ImageType, ...]) -> None:
    """Refuse image types whose FFT execution is disabled on the installed MLX.

    Runs before name planning, image conversion or any transform, so a request
    that mixes ``ORIGINAL`` with a disabled type cannot yield its supported
    images first and then fail.
    """
    if not FFT_AVAILABLE and any(kind in _DISABLED_FFT_IMAGE_TYPES for kind in kinds):
        raise ValueError(FFT_DISABLED_REASON)


def feature_image_names(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    options: TransformSettings,
    *,
    force2d_axis: int | None = None,
) -> tuple[str, ...]:
    """Return the derived-image prefixes ``feature_images`` will yield, without computing.

    The order and spelling match ``FeatureImage.name`` exactly. This lets callers size
    streaming storage before extraction starts.

    Raises:
        ValueError: If a requested FFT-based image type is unavailable on the installed
            MLX release (FFT filters need 0.32.5 or later), the shape or spacing is
            inconsistent, the 2-D axis is outside the image, LBP3D is requested for
            non-3-D data, a filter's axis count does not match the image dimensionality,
            or two selected configurations generate the same derived-image name.
    """
    _require_available_image_types(options.image_types)
    ndim = len(shape)
    if ndim not in (2, 3) or len(spacing) != ndim:
        raise ValueError("feature image names require aligned 2D/3D spacing")
    if force2d_axis is not None and force2d_axis not in range(ndim):
        raise ValueError("force2d_axis must index a NumPy image axis")
    names: list[str] = []
    for kind in options.image_types:
        if kind == ImageType.LBP3D:
            if ndim != 3:
                raise ValueError("LBP3D requires a 3D image")
            names.extend(f"lbp-3D-m{level}" for level in range(1, options.lbp3d.levels + 1))
            names.append("lbp-3D-k")
        elif kind == ImageType.WAVELET:
            codes = _wavelet_codes(ndim, force2d_axis)
            for level in range(1, options.wavelet.level + 1):
                prefix = _wavelet_prefix(level)
                names.extend(f"{prefix}-{code}" for code in codes[1:])
            # The transform yields details at every level and only the final approximation.
            names.append(f"{_wavelet_prefix(options.wavelet.level)}-{codes[0]}")
        elif kind == ImageType.LOG:
            if options.log.planar and ndim != 3:
                raise ValueError("planar LoG filtering requires a 3-D image")
            # Check the filter's grid preconditions here so an unsupported LoG fails
            # before the Original image or any earlier derived image is extracted.
            if options.log.kernel is LogKernel.SAMPLED:
                require_isotropic_spacing(spacing[1:] if options.log.planar else spacing)
            else:
                from mlxradiomics.filters.recursive_gaussian import validate_log_domain

                for sigma in options.log.sigma:
                    validate_log_domain(shape, spacing, sigma)
            names.extend(_log_name(sigma, options) for sigma in options.log.sigma)
        elif kind == ImageType.MEAN:
            if options.mean.planar and ndim != 3:
                raise ValueError("planar mean filtering requires a 3-D image")
            names.append(_mean_name(options))
        elif kind == ImageType.LAWS:
            validate_filter_shape(ndim, options, kind)
            names.append(_laws_name(options))
        elif kind == ImageType.GABOR:
            validate_filter_shape(ndim, options, kind)
            names.extend(_gabor_names(options, ndim))
        elif kind == ImageType.SIMONCELLI:
            if options.simoncelli.planar and ndim != 3:
                raise ValueError("planar Simoncelli filtering requires a 3-D image")
            names.extend(_simoncelli_name(options, level)
                         for level in options.simoncelli.levels)
        elif kind == ImageType.SEPARABLE_WAVELET:
            validate_filter_shape(ndim, options, kind)
            names.extend(_separable_wavelet_name(options, band)
                         for band in options.separable_wavelet.bands)
        elif kind == ImageType.RIESZ:
            validate_filter_shape(ndim, options, kind)
            if options.riesz.base is RieszBase.LOG:
                require_isotropic_spacing(spacing)
                names.extend(
                    _riesz_name(options, order, sigma)
                    for sigma in options.riesz.log_sigma
                    for order in options.riesz.orders
                )
            else:
                names.extend(
                    _riesz_name(options, order, level)
                    for level in options.riesz.simoncelli_levels
                    for order in options.riesz.orders
                )
        else:
            names.append(kind.value)
    return _require_unique_names(tuple(names))


def _wavelets(
    image: mx.array, options: TransformSettings, axis2d: int | None,
    window: tuple[slice, ...] | None = None,
) -> Iterator[FeatureImage]:
    """Yield stationary-wavelet detail images and the final approximation.

    Each name is ``wavelet-<codes>`` for level 1 or ``wavelet<level>-<codes>`` for deeper
    levels, with one code per filtered axis: ``L`` (low-pass) or ``H`` (high-pass), listed
    from the last array axis to the first. With ``axis2d`` set, that axis is not filtered
    and has no code. The last image is the approximation (all ``L``) of the final level.

    Level l uses filter dilation 2**(l-1), the standard a-trous transform with
    unnormalized wavelet filters. Filtered axes are periodically extended to a multiple
    of 2**(start_level + levels), then cropped to the original grid. ``start_level``
    discards that many initial decompositions but retains their approximation as input
    to the returned levels; it does not restart filtering from the original image.

    With ``window`` (bounded ``start:stop`` slices), images are returned cropped to it.
    Each output voxel is a fixed-order tap sum over voxels at most
    ``len(filter) // 2 * (2**levels - 1)`` away along each filtered axis, so a filtered
    axis whose window plus that reach lies inside the image is filtered over that span
    alone; the periodic extension then touches only voxels outside the window, and
    window values are bit-identical to whole-image filtering. Other filtered axes keep
    their full length and wrap, and the unfiltered ``axis2d`` is cut to the window.
    """
    wavelet = wavelet_filters(options.wavelet.name)
    axes = tuple(i for i in range(image.ndim - 1, -1, -1) if i != axis2d)
    total_levels = options.wavelet.start_level + options.wavelet.level
    multiple = 1 << total_levels
    crop = tuple(slice(0, size) for size in image.shape)
    if window is not None:
        reach = len(wavelet.low) // 2 * (multiple - 1)
        source: list[slice] = []
        output: list[slice] = []
        for axis, (part, size) in enumerate(zip(window, image.shape, strict=True)):
            start, stop = cast(int, part.start), cast(int, part.stop)
            if axis not in axes:
                source.append(part)
                output.append(slice(0, stop - start))
            elif start >= reach and stop + reach <= size:
                source.append(slice(start - reach, stop + reach))
                output.append(slice(reach, reach + stop - start))
            else:
                source.append(slice(0, size))
                output.append(part)
        image = image[tuple(source)]
        crop = tuple(output)
    for axis in axes:
        length = image.shape[axis]
        padded_length = ((length + multiple - 1) // multiple) * multiple
        if padded_length != length:
            image = mx.take(image, mx.arange(padded_length, dtype=mx.int32) % length, axis=axis)
    approximation = image
    for level in range(total_levels):
        decompositions = {"": approximation}
        for axis in axes:
            decompositions = {
                name + suffix: _periodic_filter(values, coefficients, axis, 1 << level)
                for name, values in decompositions.items()
                for suffix, coefficients in (("L", wavelet.low), ("H", wavelet.high))
            }
        approximation = decompositions.pop("L" * len(axes))
        evaluate(approximation, *decompositions.values())
        if level < options.wavelet.start_level:
            continue
        output_level = level - options.wavelet.start_level + 1
        prefix = _wavelet_prefix(output_level)
        for name, values in decompositions.items():
            yield FeatureImage(f"{prefix}-{name}", values[crop])
    yield FeatureImage(f"{_wavelet_prefix(options.wavelet.level)}-{'L' * len(axes)}",
                       approximation[crop])


def _gradient(image: mx.array, spacing: tuple[float, ...], use_spacing: bool) -> mx.array:
    """Gradient magnitude from central differences ``(x[i+1] - x[i-1]) / 2`` per axis.

    Edge voxels are replicated at the boundary, so edge differences span one step but are
    still halved. A difference of finite opposite-sign values near the float32 limit is
    formed from halved operands instead, so it cannot overflow before the division. The
    magnitude divides every component by the largest one before squaring and restores
    it after the square root: squaring ``1e-30`` underflowed and squaring ``1e20``
    overflowed although both magnitudes are normal float32 values.
    """
    components: list[mx.array] = []
    for axis, size in enumerate(image.shape):
        indexes = mx.arange(size, dtype=mx.int32)
        left = mx.take(image, mx.maximum(indexes - 1, 0), axis=axis)
        right = mx.take(image, mx.minimum(indexes + 1, size - 1), axis=axis)
        step = spacing[axis] if use_spacing else 1.0
        difference = right - left
        halved = mx.where(
            mx.isinf(difference) & mx.isfinite(right) & mx.isfinite(left),
            right * 0.5 - left * 0.5,
            difference * 0.5,
        )
        components.append(halved / step)
    largest = mx.abs(components[0])
    for component in components[1:]:
        largest = mx.maximum(largest, mx.abs(component))
    divisor = mx.where((largest > 0) & mx.isfinite(largest), largest, 1)
    squared = mx.zeros_like(image)
    for component in components:
        squared = squared + (component / divisor) ** 2
    return mx.where(mx.isinf(largest), largest, largest * mx.sqrt(squared))


def feature_images(
    image: mx.array,
    spacing: tuple[float, ...],
    options: TransformSettings,
    *,
    force2d_axis: int | None = None,
    mask: mx.array | None = None,
    source_dtype: np.dtype[np.generic] | None = None,
    source_residual: mx.array | None = None,
    source_tail: mx.array | None = None,
    window: tuple[slice, ...] | None = None,
    _finite_input: bool = False,
) -> Iterator[FeatureImage]:
    """Yield the derived images selected in ``options``, in ``options.image_types`` order.

    The IBSI 2 filters are fully specified by ``options`` and the physical ``spacing``:
    mean supports and Laws energy distances are in voxels; Gabor and Riesz LoG scales are
    in mm and are converted through the supplied spacing. Laws, Gabor and Riesz LoG
    response maps record their padding method in the derived-image name. Simoncelli is a
    full-image Fourier band-pass and therefore always uses periodic boundaries; a
    Simoncelli-based Riesz filter is likewise periodic.

    Args:
        image: 2-D or 3-D array of finite intensities (cast to float32). Axes follow NumPy
            order.
        spacing: Voxel spacing in mm per array axis; used by gradient, LoG and Gabor. The
            length is checked here; finite positive values are a caller precondition.
        options: Image types and their parameters.
        force2d_axis: Axis excluded from wavelet filtering, and the slice axis for LBP2D of a
            3-D image (the first axis when None). Direct callers must supply a valid axis;
            ``extract`` validates it through ``Settings``.
        mask: Boolean ROI mask; required for LBP3D, which is computed only inside it.
        source_dtype: dtype of the original image (default ``image.dtype``). LBP3D uses
            it to round interpolated samples to the nearest integer and clip them to the
            dtype range, mimicking integer assignment in PyRadiomics. LBP2D keeps its
            fractional variance output for integer sources.
        source_tail: Optional third float32 source word for exact LBP2D categorical
            comparisons; accepted source is image plus residual plus tail.
        source_residual: Float32 ``float64 source - float32 image``, used by LBP3D to
            distinguish constant source fields and by LBP2D to compare
            samples at source precision.
        window: Optional bounded ``start:stop`` slices of ``image``; every yielded image
            is cropped to it. Wavelets then filter only the window plus their tap reach
            where that stays inside the image, with bit-identical window values; other
            transforms still run on the whole image before cropping.
        _finite_input: Internal proof from the extraction pipeline's full float32 host
            validation; avoids repeating that validation and readback on the GPU.

    Yields:
        ``FeatureImage`` objects. Square, square-root, logarithm and exponential images use
        the maximum absolute intensity as a scale, as PyRadiomics does. LoG needs 3-D data.

    Raises:
        ValueError: If a requested FFT-based image type is unavailable on the installed
            MLX release (FFT filters need 0.32.5 or later), shapes or spacing are
            inconsistent, intensities are not finite, a transform is undefined for the
            input (all-zero image for square, logarithm and exponential; missing mask
            for LBP3D), or a transform's own constraints fail.
    """
    for feature in _feature_images(
        image, spacing, options, force2d_axis=force2d_axis, mask=mask,
        source_dtype=source_dtype, source_residual=source_residual,
        source_tail=source_tail, window=window, _finite_input=_finite_input,
    ):
        # Wavelets arrive cropped; full-grid images are cut to the window here. A cropped
        # wavelet shares the full shape only when the window spans the whole image, where
        # these slices are the identity.
        if window is not None and feature.image.shape == image.shape:
            feature = FeatureImage(feature.name, feature.image[window])
        yield feature


def _feature_images(
    image: mx.array,
    spacing: tuple[float, ...],
    options: TransformSettings,
    *,
    force2d_axis: int | None = None,
    mask: mx.array | None = None,
    source_dtype: np.dtype[np.generic] | None = None,
    source_residual: mx.array | None = None,
    source_tail: mx.array | None = None,
    window: tuple[slice, ...] | None = None,
    _finite_input: bool = False,
) -> Iterator[FeatureImage]:
    """Generate ``feature_images`` outputs; only wavelets are cropped to ``window``."""
    if image.ndim not in (2, 3) or len(spacing) != image.ndim:
        raise ValueError("feature images require aligned 2D/3D spacing")
    # Validate the complete name set before the first transform runs or a streaming
    # consumer opens a sink: a repeated prefix would overwrite one requested output.
    feature_image_names(image.shape, spacing, options, force2d_axis=force2d_axis)
    if not _finite_input and not bool(mx.all(mx.isfinite(image)).item()):
        raise ValueError("feature image intensities must be finite")
    source_dtype = source_dtype or np.dtype(str(image.dtype).split(".")[-1])
    image = image.astype(mx.float32)
    if source_residual is not None:
        source_residual = source_residual.astype(mx.float32)
    if source_tail is not None:
        source_tail = source_tail.astype(mx.float32)
    maximum = mx.max(mx.abs(image))
    zero_scale: bool | None = None
    for kind in options.image_types:
        if kind == ImageType.LBP2D:
            values = lbp2d(
                image,
                options.lbp2d.samples,
                options.lbp2d.radius,
                options.lbp2d.method,
                force2d_axis or 0,
                source_residual,
                source_tail,
            )
            yield FeatureImage(kind.value, values)
            continue
        if kind == ImageType.LBP3D:
            from mlxradiomics.filters.lbp3d import lbp3d_images

            if mask is None:
                raise ValueError("LBP3D requires an explicit label mask")
            yield from (
                FeatureImage(name, values)
                for name, values in lbp3d_images(
                    image, mask, options, source_dtype, source_residual,
                )
            )
            continue
        if kind == ImageType.WAVELET:
            yield from _wavelets(image, options, force2d_axis, window)
            continue
        if kind == ImageType.LOG:
            if options.log.kernel is LogKernel.SAMPLED:
                for sigma in options.log.sigma:
                    values = log_response(
                        image, spacing, sigma, options.log.cutoff, options.log.planar,
                        options.log.padding, options.log.constant,
                    )
                    evaluate(values)
                    yield FeatureImage(_log_name(sigma, options), values)
                continue
            from mlxradiomics.filters.recursive_gaussian import laplacian_gaussian

            for sigma in options.log.sigma:
                yield FeatureImage(
                    _log_name(sigma, options), laplacian_gaussian(image, spacing, sigma),
                )
            continue
        if kind == ImageType.MEAN:
            values = mean_response(
                image, options.mean.support, options.mean.padding, options.mean.constant,
                options.mean.planar,
            )
            evaluate(values)
            yield FeatureImage(_mean_name(options), values)
            continue
        if kind == ImageType.LAWS:
            values = laws_response(
                image, options.laws.kernel, options.laws.padding, options.laws.constant,
                options.laws.pooling,
            )
            if options.laws.energy_distance is not None:
                values = laws_energy(
                    values, options.laws.kernel, options.laws.energy_distance,
                    options.laws.padding, options.laws.constant,
                )
            evaluate(values)
            yield FeatureImage(_laws_name(options), values)
            continue
        if kind == ImageType.GABOR:
            for sigma, wavelength in gabor_scale_pairs(options):
                responses = gabor_responses(
                    image, spacing, sigma, wavelength, options.gabor.gamma,
                    options.gabor.theta, options.gabor.response,
                    options.gabor.slice_axis, options.gabor.padding,
                    options.gabor.constant, options.gabor.pooling,
                )
                plane = _gabor_plane_tag(options, image.ndim)
                for index, values in enumerate(responses):
                    thetas = (
                        options.gabor.theta
                        if options.gabor.pooling is not OrientationPooling.NONE
                        else (options.gabor.theta[index],)
                    )
                    name = "-".join(filter(None, (
                        _gabor_base_name(options, sigma, wavelength, thetas), plane,
                    )))
                    evaluate(values)
                    yield FeatureImage(name, values)
            continue
        if kind == ImageType.SEPARABLE_WAVELET:
            wavelet = wavelet_filters(options.separable_wavelet.name)
            responses = separable_wavelet_responses(
                image, wavelet.low, wavelet.high, options.separable_wavelet.bands,
                options.separable_wavelet.level, options.separable_wavelet.padding,
                options.separable_wavelet.constant, options.separable_wavelet.pooling,
                options.separable_wavelet.planar,
            )
            for band, values in zip(options.separable_wavelet.bands, responses, strict=True):
                yield FeatureImage(_separable_wavelet_name(options, band), values)
            continue
        if kind == ImageType.SIMONCELLI:
            for level in options.simoncelli.levels:
                values = simoncelli_response(
                    image, level, options.simoncelli.grid, options.simoncelli.planar,
                )
                evaluate(values)
                yield FeatureImage(_simoncelli_name(options, level), values)
            continue
        if kind == ImageType.RIESZ:
            computed = _riesz_computed_orders(options, image.ndim)
            if options.riesz.base is RieszBase.LOG:
                for sigma in options.riesz.log_sigma:
                    responses = _riesz_requested(options, computed, riesz_log_responses(
                        image, spacing, sigma, options.riesz.log_cutoff,
                        computed, options.riesz.grid,
                        options.riesz.padding, options.riesz.constant,
                    ), spacing)
                    for order, values in zip(
                        options.riesz.orders, responses, strict=True,
                    ):
                        evaluate(values)
                        yield FeatureImage(_riesz_name(options, order, sigma), values)
            else:
                for level in options.riesz.simoncelli_levels:
                    responses = _riesz_requested(options, computed, riesz_simoncelli_responses(
                        image, level, computed, options.riesz.grid,
                    ), spacing)
                    for order, values in zip(
                        options.riesz.orders, responses, strict=True,
                    ):
                        evaluate(values)
                        yield FeatureImage(_riesz_name(options, order, level), values)
            continue
        if kind in (ImageType.SQUARE, ImageType.LOGARITHM, ImageType.EXPONENTIAL):
            if zero_scale is None:
                zero_scale = float(cast(float, maximum.item())) == 0
            if zero_scale:
                raise ValueError(f"the {kind} transform is undefined for all-zero images")
        values = {
            ImageType.ORIGINAL: lambda: image,
            ImageType.SQUARE: lambda: (image / mx.sqrt(maximum)) ** 2,
            # Arrange each definition so representable outputs do not require products
            # outside float32 range. Power also preserves the exponential endpoint M^1.
            ImageType.SQUARE_ROOT: lambda: (
                mx.sign(image) * mx.sqrt(mx.abs(image)) * mx.sqrt(maximum)
            ),
            ImageType.LOGARITHM: lambda: (
                mx.sign(image) * mx.log1p(mx.abs(image)) * (maximum / mx.log1p(maximum))
            ),
            ImageType.EXPONENTIAL: lambda: mx.power(maximum, image / maximum),
            ImageType.GRADIENT: lambda: _gradient(image, spacing, options.gradient.use_spacing),
        }[kind]()
        evaluate(values)
        yield FeatureImage(kind.value, values)
