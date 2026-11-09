"""Derived-image input preparation, source precision and undefined LBP values."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.contracts.results import Result
from mlxradiomics.contracts.types import Features, HostArray, VoxelSink
from mlxradiomics.contracts.validity import accumulate_validity, empty_validity
from mlxradiomics.imaging.preprocessing import PreparedInput
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import array_from_bool, mx
from mlxradiomics.workflows.engine import extract_with_shape_mask, input_window


@dataclass(frozen=True)
class DerivedInputs:
    """Derived-image settings and the original masks on the required sampling window."""

    settings: Settings
    mask: HostArray
    window: tuple[slice, ...] | None
    crop: tuple[slice, ...]
    masks: PreparedInput


def prepare_derived_inputs(
    data: PreparedInput,
    mask: HostArray,
    settings: Settings,
    transforms: TransformSettings,
    spacing: tuple[float, ...],
    voxel: bool,
) -> DerivedInputs:
    """Preserve intensity and morphology support while preparing the derived-image window."""
    intensity_names = tuple(name for name in settings.features if not name.startswith("shape"))
    derived_settings = replace(
        settings,
        features=intensity_names,
        normalize=False,
        remove_outliers=None,
        resegment_range=None,
        resegment_outlier_sigma=None,
    )
    derived_types = tuple(kind for kind in transforms.image_types if kind is not ImageType.ORIGINAL)
    # Keep source integers: narrowing or inferring a signed dtype can lose large labels.
    # This mask fixes the derived image's intensity ROI. GLDZM morphology is passed
    # separately below so it keeps the shape mask rather than this reduced support.
    # Without resegmentation every selected label is already correct; other labels
    # are ignored by the prepared masks.
    derived_mask = (
        np.where(data.roi, mask, 0)
        if derived_types and settings.resegment_range is not None
        else mask
    )
    # ROI extraction of a derived image reads only the window that ``extract`` would
    # crop it to (masks padded by the neighbor and peak reach). Derived settings skip
    # normalization and resegmentation, so binning and checks see only sampled values;
    # cropping on the GPU first skips full-volume host copies and preparation and
    # leaves the cropped extraction inputs identical. Wavelets filter only that window
    # plus their tap reach (see ``feature_images``).
    window = (
        input_window(data.image, derived_mask, data.shape_roi, derived_settings, spacing)
        if derived_types and not voxel
        else None
    )
    crop = window or tuple(slice(None) for _ in range(data.image.ndim))
    # The reused original masks, cut to the same window as the derived pixels.
    window_masks = replace(
        data,
        image=data.image[crop],
        roi=data.roi[crop],
        shape_roi=data.shape_roi[crop],
        sampling_mask=data.sampling_mask[crop],
    )
    return DerivedInputs(derived_settings, derived_mask, window, crop, window_masks)


def feature_arrays(
    image: HostArray,
    data: PreparedInput,
    settings: Settings,
    transforms: TransformSettings,
    spacing: tuple[float, ...],
    window: tuple[slice, ...] | None,
) -> Iterator[tuple[str, HostArray | None]]:
    """Build source-precision residents and iterate evaluated transforms on the caller's stream."""
    derived_types = tuple(kind for kind in transforms.image_types if kind is not ImageType.ORIGINAL)
    source32: NDArray[np.float32] | None = None
    if derived_types:
        # Filters run on the float32 image, so a float64 value beyond float32 range
        # is non-finite there too. Check before the LBP source-precision split, whose
        # residual arithmetic would otherwise warn or misreport the cause.
        with np.errstate(over="ignore"):
            source32 = data.image.astype(np.float32)
        finite = bool(np.all(np.isfinite(source32)))
        if not finite:
            raise ValueError("feature image intensities must be finite")
    # LBP2D comparisons and LBP3D constant-field detection need source residuals.
    source_residual = None
    source_tail = None
    if source32 is not None and any(
        t in transforms.image_types for t in (ImageType.LBP2D, ImageType.LBP3D)
    ):
        exact_source = not settings.normalize and (
            (image.dtype.kind == "f" and image.dtype.itemsize <= 4)
            or (image.dtype.kind in "iu" and image.dtype.itemsize <= 2)
        )
        # These source dtypes are exact in float32 before normalization.
        # Their omitted residual/tail are the LBP primitives' implicit zeros.
        if not exact_source:
            low, tail = filter_source_words(
                data.image,
                source32,
                tail=ImageType.LBP2D in transforms.image_types,
            )
            source_residual = mx.array(low, dtype=mx.float32)
            if tail is not None:
                source_tail = mx.array(tail, dtype=mx.float32)
            del low, tail
    # ``feature_images`` rejects a non-finite image because its filters need the
    # whole domain. The Original image is the input itself, so dispatch it straight
    # to ``extract``: that accepts non-finite values outside the sampled ROI, as a
    # direct call does, and avoids a full float32 copy of the source.
    image_items: Iterator[tuple[str, HostArray | None]]
    if derived_types:
        assert source32 is not None
        resident_image = mx.array(source32, dtype=mx.float32)
        del source32
        image_items = (
            (
                feature_image.name,
                None
                if feature_image.name == ImageType.ORIGINAL.value
                else cast(HostArray, np.asarray(feature_image.image)),
            )
            for feature_image in feature_images(
                resident_image,
                spacing,
                transforms,
                force2d_axis=settings.force2d_axis,
                mask=(
                    array_from_bool(data.roi) if ImageType.LBP3D in transforms.image_types else None
                ),
                source_dtype=np.dtype(np.float64) if settings.normalize else image.dtype,
                source_residual=source_residual,
                source_tail=source_tail,
                window=window,
                _finite_input=True,
            )
        )
    else:
        image_items = iter((("original", None),))
    return image_items


def extract_lbp_kurtosis(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings,
    spacing: tuple[float, ...],
    voxel: bool,
    histogram_method: str,
    voxel_sink: VoxelSink | None,
    shape_mask: HostArray | None = None,
    _prepared_masks: PreparedInput | None = None,
) -> Result:
    """Propagate undefined sampled kurtosis to each affected feature neighborhood.

    ROI features are undefined if any selected sample is undefined. Voxel features are
    undefined only where the masked kernel contains such a sample. A finite selected
    value temporarily fills undefined positions so discretization keeps the finite
    image's boundaries; every result using a fill is replaced by NaN before delivery.
    Harmonic images and the strict finite-input contract of ``extract`` are unaffected.
    ``shape_mask`` is forwarded to ``extract_with_shape_mask`` so GLDZM keeps the
    original morphology.
    """
    roi = mask == settings.label if _prepared_masks is None else _prepared_masks.roi
    invalid = roi & ~np.isfinite(image)
    if not np.any(invalid):
        return extract_with_shape_mask(
            image,
            mask,
            settings=settings,
            spacing=spacing,
            voxel=voxel,
            histogram_method=histogram_method,
            voxel_sink=voxel_sink,
            shape_mask=shape_mask,
            _prepared_masks=_prepared_masks,
        )
    finite = np.asarray(image[roi & ~invalid], dtype=np.float64)
    fill = float(finite.min()) if finite.size else (settings.bin_anchor or 0.0)
    safe = np.where(invalid, fill, image)
    outputs: Features = {}
    validity = {
        f"original_{name}": empty_validity(image.size - int(roi.sum()) if voxel else 0)
        for name in settings.features
    }
    if voxel:
        from scipy.ndimage import maximum_filter

        widths = tuple(
            1 if axis == settings.force2d_axis else 2 * settings.kernel_radius + 1
            for axis in range(image.ndim)
        )
        affected = maximum_filter(invalid, size=widths, mode="constant", cval=0)
        retained = (
            {name: np.zeros(image.shape, dtype=np.float64) for name in validity}
            if voxel_sink is None
            else {}
        )

        def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
            selected = affected[tuple(centers.T)]
            corrected = {
                name: np.where(selected, np.float32(np.nan), values)
                for name, values in batch.items()
            }
            for name, values in corrected.items():
                accumulate_validity(validity[name], values)
                if voxel_sink is None:
                    retained[name][tuple(centers.T)] = values
            if voxel_sink is not None:
                voxel_sink(centers, corrected)

        result = extract_with_shape_mask(
            safe,
            mask,
            settings=settings,
            spacing=spacing,
            voxel=True,
            histogram_method=histogram_method,
            voxel_sink=consume,
            shape_mask=shape_mask,
            _prepared_masks=_prepared_masks,
        )
        outputs.update(retained)
    else:
        result = extract_with_shape_mask(
            safe,
            mask,
            settings=settings,
            spacing=spacing,
            histogram_method=histogram_method,
            shape_mask=shape_mask,
            _prepared_masks=_prepared_masks,
        )
        outputs.update({name: float("nan") for name in validity})
        for counts in validity.values():
            counts["nan"] = 1
    result.metadata["feature_validity"] = validity
    if not finite.size:
        result.metadata["gray_levels"] = 0
    return Result(outputs, result.metadata)


def filter_source_words(
    image: NDArray[np.float64],
    high: NDArray[np.float32],
    *,
    tail: bool,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Split LBP source words, reusing the filter input and subtraction scratch.

    The subtraction order is ``((image - high) - low) - tail``, reusing one scratch
    array. The caller has already proved the float32 input finite.
    """
    remainder = image - high
    low = remainder.astype(np.float32)
    if not tail:
        return low, None
    np.subtract(remainder, low, out=remainder)
    third = remainder.astype(np.float32)
    np.subtract(remainder, third, out=remainder)
    if np.any(remainder != 0):
        raise ValueError(
            "LBP2D source sign is unresolved: source precision cannot be "
            "represented by three float32 words"
        )
    return low, third
