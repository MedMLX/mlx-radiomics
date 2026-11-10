"""Image-file loading and spatial preprocessing (mask correction, resampling) with SimpleITK.

The grid and padding equations follow PyRadiomics' ``resampleImage`` by default. Everything
here runs on the host; feature arithmetic still runs on Metal. Spacing arguments use NumPy
array-axis order, while origin and direction keep ITK's physical x/y/z order.

An explicit opt-in ``IbsiInterpolation`` instead builds the IBSI interpolation grid: a
ceil-sized, centered grid that covers the source extent, float64 image interpolation
(linear by default or cubic B-spline), nearest-integer intensity rounding, and a linearly
interpolated float32 binary ROI mask binarized at a partial-volume threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import IbsiInterpolation, ImageSettings, Interpolator
from mlxradiomics.contracts.types import (
    ImageGeometryPayload,
)
from mlxradiomics.imaging.io import Volume, read_image, read_image_dimension, read_mask
from mlxradiomics.imaging.preprocessing import exact_float64, normalize_values, round_intensities

if TYPE_CHECKING:
    import SimpleITK as sitk


@dataclass(frozen=True)
class ImagePair:
    """Aligned volumes, with the source image grid retained by the file loader."""

    image: Volume
    mask: Volume
    source_geometry: ImageGeometryPayload | None = None

    def __post_init__(self) -> None:
        """Reject an image and mask that differ in shape or physical geometry."""
        if self.image.array.shape != self.mask.array.shape or (
            self.image.spacing != self.mask.spacing or self.image.origin != self.mask.origin
            or self.image.direction != self.mask.direction
        ):
            raise ValueError("aligned image and mask must share geometry")


@dataclass
class LoadedCase:
    """Keep one case's validated source pixels for independent label runs.

    Loading is lazy so dimension-dependent parameter validation still precedes
    full image I/O. Source files form one case snapshot; failed reads are also
    reused. No label mask, crop, normalization or spatial result is shared.
    Release this object before moving to the next study case.
    """

    image_path: Path
    mask_path: Path
    _dimension: int | None = field(default=None, init=False)
    _volumes: tuple[Volume, Volume] | None = field(default=None, init=False)
    _read_error: str | None = field(default=None, init=False)

    def dimension(self) -> int:
        """Read the dimension once, without loading or validating pixels."""
        if self._read_error is not None:
            raise ValueError(self._read_error)
        if self._dimension is None:
            try:
                self._dimension = read_image_dimension(self.image_path)
            except (ValueError, RuntimeError, OSError, EOFError) as error:
                self._read_error = str(error)
                raise
        return self._dimension

    def volumes(self) -> tuple[Volume, Volume]:
        """Load source image/mask pixels once, retaining their native types."""
        if self._read_error is not None:
            raise ValueError(self._read_error)
        if self._volumes is None:
            try:
                self._volumes = read_image(self.image_path), read_mask(self.mask_path)
            except (ValueError, RuntimeError, OSError, EOFError) as error:
                # Store text rather than a traceback retaining reader buffers.
                self._read_error = str(error)
                raise
        return self._volumes


def _itk(volume: Volume) -> sitk.Image:
    """Convert a ``Volume`` to a SimpleITK image.

    ITK stores spacing x-first, the reverse of the NumPy array axes; origin and
    direction are already in ITK order.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing(volume.spacing[::-1])
    image.SetOrigin(volume.origin)
    image.SetDirection(volume.direction)
    return image


def _volume(image: sitk.Image) -> Volume:
    """Convert a SimpleITK image to a ``Volume``, reversing spacing into array-axis order."""
    import SimpleITK as sitk

    return Volume(sitk.GetArrayFromImage(image), image.GetSpacing()[::-1],
                  image.GetOrigin(), image.GetDirection())


def _same_geometry(image: sitk.Image, mask: sitk.Image) -> bool:
    """Return whether image and mask share a grid within ITK's tolerances.

    Sizes must be equal; spacing and origin must agree within 1e-6 of the smallest
    image spacing, as in ITK's geometry check, so the test stays in voxel units at
    any physical scale; direction cosines must agree within 1e-6.
    """
    coordinate = 1e-6 * min(image.GetSpacing())
    return image.GetSize() == mask.GetSize() and all(
        np.allclose(a, b, rtol=0, atol=atol) for a, b, atol in (
            (image.GetSpacing(), mask.GetSpacing(), coordinate),
            (image.GetOrigin(), mask.GetOrigin(), coordinate),
            (image.GetDirection(), mask.GetDirection(), 1e-6),
        )
    )


def _roi_bounds(
    image: sitk.Image, mask: sitk.Image, label: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the label's bounding box as ``(lower, upper)`` voxel indexes, upper exclusive.

    Both are in ITK x/y/z order. The box corners, mapped through physical space, must lie
    inside the image extent (a tolerance of 0.001 voxel beyond its outer edges).

    Raises:
        ValueError: If the label is absent or the ROI extends outside the image.
    """
    import SimpleITK as sitk

    # The ITK mask stays alive throughout this read-only bounds check.
    locations = np.argwhere(sitk.GetArrayViewFromImage(mask) == label)
    if not len(locations):
        raise ValueError("selected label is absent from mask")
    lower, upper = locations.min(axis=0)[::-1], locations.max(axis=0)[::-1] + 1
    corners = product(*zip(lower - 0.5, upper - 0.5, strict=True))
    physical = [mask.TransformContinuousIndexToPhysicalPoint(tuple(map(float, point)))
                for point in corners]
    indexes = np.asarray([image.TransformPhysicalPointToContinuousIndex(p) for p in physical])
    if np.any(indexes < -0.501) or np.any(indexes > np.asarray(image.GetSize()) - 0.499):
        raise ValueError("mask ROI lies outside the physical image bounds")
    return lower.astype(np.int64), upper.astype(np.int64)


def _require_interpolable_labels(mask: sitk.Image) -> None:
    """Reject integer labels outside the exact float64 range used by ITK interpolation."""
    import SimpleITK as sitk

    # The image stays alive while this borrowed view is inspected; no mask copy is needed.
    view = sitk.GetArrayViewFromImage(mask)
    if view.dtype.itemsize <= 4:
        return
    # read_mask established integer pixels; SimpleITK's remaining integer types are 64-bit.
    values = cast(NDArray[np.int64] | NDArray[np.uint64], view)
    if int(values.min()) < -(2**53) or int(values.max()) > 2**53:
        raise ValueError(
            "spatial interpolation requires labels in exact integer range [-2**53, 2**53]"
        )


def _correct_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Correct the mask by nearest neighbor, preserving exact integers when already aligned."""
    import SimpleITK as sitk

    if _same_geometry(image, mask):
        mask.CopyInformation(image)
        return mask
    _require_interpolable_labels(mask)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(mask)


def _resample(
    image: sitk.Image, mask: sitk.Image, settings: ImageSettings, label: int,
) -> tuple[sitk.Image, sitk.Image]:
    """Resample image and mask to ``settings.resampled_spacing`` and crop to the ROI.

    The grid is aligned to the mask origin and cropped to the ROI bounding box plus
    ``pad_distance`` (PyRadiomics ``resampleImage``). If the spacing already
    matches the image, a misaligned mask is corrected and both are cropped; an
    aligned mask needs no interpolation. That branch uses the image's direction;
    the general resampling branch uses the mask's direction.
    """
    import SimpleITK as sitk

    requested = settings.resampled_spacing
    if requested is None or len(requested) != image.GetDimension():
        raise ValueError("resampled_spacing must match the image dimension")
    lower, upper = _roi_bounds(image, mask, label)
    source_spacing = np.asarray(mask.GetSpacing())
    spacing = np.asarray(requested[::-1])
    spacing = np.where((spacing == 0) | (upper - lower == 1), source_spacing, spacing)
    if np.allclose(image.GetSpacing(), spacing):
        mask = _correct_mask(image, mask)
        lower, upper = _roi_bounds(image, mask, label)
        lower = np.maximum(lower - settings.pad_distance, 0)
        upper = np.minimum(upper + settings.pad_distance, image.GetSize())
        size, index = (upper - lower).tolist(), lower.tolist()
        return sitk.RegionOfInterest(image, size, index), sitk.RegionOfInterest(mask, size, index)
    _require_interpolable_labels(mask)
    ratio = source_spacing / spacing
    new_lower = np.maximum(np.floor((lower - 0.5) * ratio - settings.pad_distance), 0)
    new_upper = np.minimum(np.ceil((upper - 0.5) * ratio + settings.pad_distance),
                           np.ceil(np.asarray(mask.GetSize()) * ratio) - 1)
    origin_index = 0.5 * (spacing - source_spacing) / source_spacing + new_lower / ratio
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((new_upper - new_lower + 1).astype(int).tolist())
    resampler.SetOutputSpacing(spacing.tolist())
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputOrigin(mask.TransformContinuousIndexToPhysicalPoint(origin_index.tolist()))
    interpolators = {Interpolator.NEAREST: sitk.sitkNearestNeighbor,
                     Interpolator.LINEAR: sitk.sitkLinear,
                     Interpolator.BSPLINE: sitk.sitkBSpline}
    resampler.SetInterpolator(interpolators[settings.interpolator])
    # Interpolate intensities in float64 regardless of the source storage type. Keeping
    # an integer pixel type would apply ITK's range-clamped cast after every interpolation
    # and make the same numeric image depend on how it was encoded. Masks keep their label
    # dtype and nearest-neighbor path.
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    result_image = resampler.Execute(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(mask.GetPixelID())
    return result_image, resampler.Execute(mask)


def _ibsi_grid(
    image: sitk.Image, spacing: tuple[float, ...],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the centered IBSI grid as ITK-order size, spacing and world origin.

    ``spacing`` follows NumPy array-axis order; a zero entry keeps the source spacing. The
    per-axis size is ``ceil(n * s / s_b)`` on the exact supplied binary64 spacing ratios.
    The origin is shifted so the grids share a physical center, with the index-space
    offset rotated by the image direction:

    ``origin_b = origin_a + D @ ((s_a (n_a - 1) - s_b (n_b - 1)) / 2)``.
    """
    dimension = image.GetDimension()
    size = np.asarray(image.GetSize(), dtype=np.float64)
    source = np.asarray(image.GetSpacing(), dtype=np.float64)
    target = np.asarray(spacing[::-1], dtype=np.float64)
    kept = (target == 0) | (target == source)
    target = np.where(kept, source, target)
    # Round only the final rational extent. Floating multiplication/division can move
    # an exact integer above its boundary; an epsilon would erase real fractional cells.
    shape = np.asarray([
        ceil(int(n) * Fraction(float(s)) / Fraction(float(t)))
        for n, s, t in zip(size, source, target, strict=True)
    ], dtype=np.int64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(dimension, dimension)
    offset = (source * (size - 1) - target * (shape - 1)) / 2
    origin = np.asarray(image.GetOrigin(), dtype=np.float64) + direction @ offset
    return shape, target, origin


def _inside_source_centers(
    source: sitk.Image, shape: NDArray[np.int64], spacing: NDArray[np.float64],
    origin: NDArray[np.float64], direction: tuple[float, ...],
) -> NDArray[np.bool_]:
    """Flag output voxels whose centers lie within the source voxel-center hull per axis.

    The output grid is given in ITK order (``shape``, ``spacing``, ``origin`` and flattened
    ``direction``); the result is a NumPy-order boolean array. A center is inside when its
    continuous source index is in ``[0, n - 1]`` on every axis, up to a tolerance of
    ``1e-6`` voxel that absorbs binary64 coordinate roundoff.
    """
    dimension = len(shape)
    target_direction = np.asarray(direction, dtype=np.float64).reshape(dimension, dimension)
    source_direction = np.asarray(source.GetDirection(), dtype=np.float64).reshape(
        dimension, dimension,
    )
    source_spacing = np.asarray(source.GetSpacing(), dtype=np.float64)
    # Continuous source index = linear @ output_index + offset, exact for any pair of grids.
    linear = source_direction.T @ target_direction * spacing[None, :] / source_spacing[:, None]
    offset = (
        source_direction.T @ (origin - np.asarray(source.GetOrigin(), dtype=np.float64))
        / source_spacing
    )
    last = np.asarray(source.GetSize(), dtype=np.float64) - 1
    tolerance = 1e-6
    inside = np.ones(tuple(int(n) for n in shape[::-1]), dtype=bool)
    # Visit one slab of the slowest NumPy axis (last ITK axis) at a time to bound memory.
    grids = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape[:-1]), indexing="ij")
    for slab in range(int(shape[-1])):
        valid = np.ones(grids[0].shape, dtype=bool)
        for row in range(dimension):
            coordinate = offset[row] + linear[row, dimension - 1] * slab
            for column, grid in enumerate(grids):
                coordinate = coordinate + linear[row, column] * grid
            valid &= (coordinate >= -tolerance) & (coordinate <= last[row] + tolerance)
        # ``valid`` is ITK-ordered over the leading axes; NumPy order reverses them.
        inside[slab] = valid.T
    return inside


# ITK prefilters a cubic B-spline with a mirror boundary. Replicating the edge voxels first
# reproduces the "nearest" boundary of scipy.ndimage.map_coordinates (which pads 12 voxels
# itself); the mirror condition's influence decays by roughly 0.27 per voxel, so 32 voxels
# leave it below 1e-18 at the original border.
_SPLINE_EDGE_PAD = 32


def _edge_padded(image: sitk.Image) -> sitk.Image:
    """Return ``image`` with edge-replicated voxels added on every side, grid unchanged."""
    import SimpleITK as sitk

    pad = _SPLINE_EDGE_PAD
    dimension = image.GetDimension()
    values = np.pad(sitk.GetArrayFromImage(image), pad, mode="edge")
    padded = sitk.GetImageFromArray(values)
    padded.SetSpacing(image.GetSpacing())
    padded.SetDirection(image.GetDirection())
    shift = np.asarray(image.GetDirection(), dtype=np.float64).reshape(dimension, dimension) @ (
        pad * np.asarray(image.GetSpacing(), dtype=np.float64)
    )
    padded.SetOrigin((np.asarray(image.GetOrigin(), dtype=np.float64) - shift).tolist())
    return padded


def _resample_ibsi(
    image: sitk.Image, mask: sitk.Image, settings: IbsiInterpolation, label: int,
    mask_extrapolation: str = "zero", spline_boundary: str = "nearest",
) -> tuple[sitk.Image, sitk.Image]:
    """Interpolate image and mask onto the centered IBSI grid.

    Intensities are rounded to the nearest integer before interpolation, as the reference
    data instructions require of converted NIfTI arrays, and rounded again afterwards, so
    the interpolated result is integer-valued as well. The intensity image uses
    ``settings.image_interpolator`` in float64; ``BSPLINE`` is SimpleITK's cubic B-spline
    (order 3) with ITK's internal mirror boundary conditions. The requested label is
    selected before interpolation, so other labels never blend into its occupancy.
    Occupancy is interpolated linearly to float32, independently of the image
    interpolator, before thresholding to exactly ``label`` or 0. Output voxel centers
    beyond the outermost source voxel centers along any axis lie outside the sampled
    domain. ``mask_extrapolation="zero"`` leaves them out of the mask (MIRP registers
    masks with constant zero fill); ``"nearest"`` copies the nearest source occupancy,
    which grows an ROI at the image border. The intensity image always uses the nearest
    source intensity.
    """
    import SimpleITK as sitk

    if len(settings.spacing) != image.GetDimension():
        raise ValueError("IBSI spacing must match the image dimension")
    view = sitk.GetArrayViewFromImage(mask)
    _require_interpolable_labels(mask)
    shape, spacing, origin = _ibsi_grid(image, settings.spacing)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(shape.tolist())
    resampler.SetOutputSpacing(spacing.tolist())
    resampler.SetOutputOrigin(origin.tolist())
    resampler.SetOutputDirection(image.GetDirection())
    # Reuse the nearest boundary intensity for points outside the source-domain buffer
    # instead of a fill value. This is external extrapolation; a cubic interpolator's
    # internal boundary handling is ITK's mirror condition and is not changed here.
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    image_interpolators = {
        Interpolator.LINEAR: sitk.sitkLinear,
        Interpolator.BSPLINE: sitk.sitkBSpline,
    }
    resampler.SetInterpolator(image_interpolators[settings.image_interpolator])
    source = image
    if settings.round_intensities:
        rounded_source = sitk.GetImageFromArray(
            round_intensities(sitk.GetArrayFromImage(image)),
        )
        rounded_source.CopyInformation(image)
        source = rounded_source
    cast_source = sitk.Cast(source, sitk.sitkFloat64)
    if spline_boundary == "nearest" and settings.image_interpolator is Interpolator.BSPLINE:
        cast_source = _edge_padded(cast_source)
    resampled_image = resampler.Execute(cast_source)
    # Interpolate occupancy, not label magnitude: the partial-volume threshold is a
    # fraction of one voxel regardless of the selected positive integer label.
    occupancy = sitk.GetImageFromArray((view == label).astype(np.float32))
    occupancy.CopyInformation(mask)
    # IBSI recommends full (32-bit) precision for ROI interpolation. Store occupancy in
    # that standard representation before thresholding: double-precision coordinate
    # roundoff must not exclude a mathematically half-covered voxel as 0.4999999999999999.
    # Reset the reused resampler: the mask is always linearly interpolated, never with a
    # BSPLINE image choice.
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampled_mask = resampler.Execute(occupancy)
    values = sitk.GetArrayFromImage(resampled_image)
    if settings.round_intensities:
        values = round_intensities(values)
    result_image = sitk.GetImageFromArray(values)
    result_image.CopyInformation(resampled_image)
    interpolated_mask = cast(NDArray[np.float32], sitk.GetArrayFromImage(resampled_mask))
    binary = np.zeros(interpolated_mask.shape, dtype=view.dtype)
    # Compare in the occupancy's float32 representation. A float64 threshold would
    # exclude an exactly 0.7-covered voxel stored as float32 0.699999988, and NumPy would
    # pick that precision only for NumPy scalar thresholds, not for Python floats.
    binary[interpolated_mask >= np.float32(settings.mask_threshold)] = label
    if mask_extrapolation == "zero":
        binary[~_inside_source_centers(mask, shape, spacing, origin, image.GetDirection())] = 0
    result_mask = sitk.GetImageFromArray(binary)
    result_mask.CopyInformation(resampled_mask)
    return result_image, result_mask


def load_image_pair(
    image_path: Path, mask_path: Path, *, settings: Settings | None = None,
    image_settings: ImageSettings | None = None,
) -> ImagePair:
    """Read an image and mask and bring them onto one grid.

    Normalization (if ``settings.normalize``) is applied to the full image before any
    resampling. Then, if ``image_settings.ibsi_interpolation`` is set, both are interpolated
    onto the centered IBSI grid; otherwise, if ``image_settings.resampled_spacing`` is set,
    both are resampled and cropped around the ROI; otherwise a geometry mismatch is
    corrected only when ``image_settings.correct_mask`` is true.

    Args:
        image_path: NIfTI or NRRD image (2-D or 3-D).
        mask_path: NIfTI or NRRD integer label mask.
        settings: Supplies the label and normalization options.
        image_settings: Resampling, IBSI interpolation and mask-correction options.

    Returns:
        Image and mask as ``Volume`` objects with NumPy-order arrays. Interpolated
        intensities are float64; mask labels keep their source integer dtype.

    Raises:
        ValueError: If a file is missing, dimensions or geometry disagree
            without correction, the label is absent or lies outside the image, or
            preprocessing removes the label. Spatial interpolation rejects masks with
            labels outside ``[-2**53, 2**53]`` to prevent rounding distinct labels together;
            IBSI interpolation selects binary occupancy for the requested label and
            refuses normalization with ``round_intensities=True``. 64-bit integer
            intensities must be exactly representable in float64.
        RuntimeError: If SimpleITK cannot read a supplied file.
    """
    source, labels = read_image(image_path), read_mask(mask_path)
    return prepare_image_pair(source, labels, settings=settings, image_settings=image_settings)


def prepare_image_pair(
    source: Volume, labels: Volume, *, settings: Settings | None = None,
    image_settings: ImageSettings | None = None,
) -> ImagePair:
    """Prepare one label from validated source volumes without modifying them.

    Every call independently normalizes, corrects/resamples and crops using its
    label settings, exactly as ``load_image_pair`` does after reading the files.
    """
    import SimpleITK as sitk

    options, spatial = settings or Settings(), image_settings or ImageSettings()
    options.validate(source.array.ndim)
    # Only 64-bit integers can lose source identity when converted to float64.
    # Validate them before resampling/rounding; other dtypes need no throwaway
    # full-volume float64 conversion here.
    if source.array.dtype.kind in "iu" and source.array.dtype.itemsize == 8:
        exact_float64(source.array)
    if options.normalize and spatial.ibsi_interpolation is not None and (
        spatial.ibsi_interpolation.round_intensities
    ):
        # Normalization runs first; integer rounding would collapse z-scores to a
        # handful of levels and silently change every intensity feature.
        raise ValueError(
            "normalize=True cannot be combined with IBSI interpolation rounding; "
            "set IbsiInterpolation(round_intensities=False)"
        )
    image, mask = _itk(source), _itk(labels)
    if image.GetDimension() != mask.GetDimension():
        raise ValueError("image and mask dimensions must match")
    _roi_bounds(image, mask, options.label)
    if options.normalize:
        normalized = sitk.GetImageFromArray(normalize_values(source.array, options))
        normalized.CopyInformation(image)
        image = normalized
    if spatial.ibsi_interpolation is not None:
        image, mask = _resample_ibsi(
            image, mask, spatial.ibsi_interpolation, options.label, options.mask_extrapolation,
            options.spline_boundary,
        )
    elif spatial.resampled_spacing is not None:
        image, mask = _resample(image, mask, spatial, options.label)
    elif not _same_geometry(image, mask):
        if not spatial.correct_mask:
            raise ValueError("geometry mismatch; explicitly enable correct_mask or resampling")
        mask = _correct_mask(image, mask)
    else:
        mask.CopyInformation(image)
    if not np.any(sitk.GetArrayViewFromImage(mask) == options.label):
        raise ValueError("spatial preprocessing removed the selected label")
    return ImagePair(_volume(image), _volume(mask), {
        "shape": source.array.shape, "spacing": source.spacing,
        "origin": source.origin, "direction": source.direction,
    })
