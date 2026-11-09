"""Host-only physical-grid and interpolation settings."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class Interpolator(StrEnum):
    """Image interpolators; PyRadiomics-style resampling masks use nearest neighbor."""

    NEAREST = "nearest"
    LINEAR = "linear"
    BSPLINE = "bspline"


@dataclass(frozen=True)
class IbsiInterpolation:
    """Explicit IBSI interpolation settings for the file-based extraction and export workflows.

    Implements the interpolation grid of the IBSI reference-data configurations: the
    per-axis dimension is ``ceil(n * spacing / target)``, the grid is centered on the source
    grid so both share a physical center, and the full grid is retained (no ROI crop or
    pad). The intensity image is interpolated in float64, linearly by default; keeping
    slice centers unchanged makes configuration B effectively bilinear within each slice.
    ``BSPLINE`` selects SimpleITK's cubic B-spline (order 3), which uses ITK's internal
    mirror boundary conditions. Both image interpolators use the nearest source intensity
    rather than a fill value for transformed points outside the source-domain buffer. The
    ROI mask is always interpolated linearly as float32 occupancy and binarized at
    ``mask_threshold``, independently of the image interpolator, so its voxels are exactly
    ``label`` or 0. The selected label is binarized before interpolation; other labels
    cannot blend into its occupancy. Outside the source domain the mask is empty or copies
    the nearest occupancy according to ``Settings.mask_extrapolation``. This is separate
    from, and does not change, the
    PyRadiomics-style resampling defaults.

    Attributes:
        spacing: Target spacing in mm, one entry per array axis (2 or 3). A zero entry keeps
            the source spacing on that axis; configuration B keeps the 3 mm slice
            spacing and resamples the in-plane axes to 2 mm.
        round_intensities: Round source and interpolated intensities to the nearest integer,
            ties to even, before re-segmentation and discretization.
        mask_threshold: Partial-volume threshold in ``(0, 1]``; interpolated mask voxels at
            or above it become ``label``.
        image_interpolator: Interpolator for the intensity image: ``LINEAR`` (default) or
            ``BSPLINE`` (cubic B-spline, order 3). The ROI mask always uses linear
            interpolation.
    """

    spacing: tuple[float, ...]
    round_intensities: bool = True
    mask_threshold: float = 0.5
    image_interpolator: Interpolator = Interpolator.LINEAR

    def __post_init__(self) -> None:
        """Reject invalid spacing, threshold, rounding or image interpolator values."""
        if type(self.round_intensities) is not bool:
            raise ValueError("round_intensities must be boolean")
        if len(self.spacing) not in (2, 3) or any(
            not isfinite(value) or value < 0 for value in self.spacing
        ):
            raise ValueError("IBSI spacing requires 2 or 3 finite nonnegative values")
        if not isfinite(self.mask_threshold) or not 0 < self.mask_threshold <= 1:
            raise ValueError("mask_threshold must be finite and in (0, 1]")
        if type(self.image_interpolator) is not Interpolator:
            raise ValueError("image_interpolator must be an Interpolator value")
        if self.image_interpolator not in (Interpolator.LINEAR, Interpolator.BSPLINE):
            raise ValueError("image_interpolator must be linear or bspline")


@dataclass(frozen=True)
class ImageSettings:
    """Spatial preprocessing options for the file-based extraction and export workflows.

    Attributes:
        resampled_spacing: Target spacing in mm, one entry per array axis (2 or 3). A zero
            entry keeps the source spacing on that axis, and so does an axis on which the ROI
            spans a single voxel. None disables resampling.
        interpolator: Image interpolation (B-spline by default). Intensities are
            interpolated in float64 independently of their file storage type; masks keep
            their integer label type and use nearest-neighbor interpolation.
        correct_mask: Resample the mask onto the image grid by nearest neighbor when their
            geometries differ. Without it (and without resampling), a geometry mismatch raises.
            Interpolated masks require all labels in the exact integer range
            ``[-2**53, 2**53]``; same-grid loading and cropping preserve larger integers.
        pad_distance: Voxels kept around the ROI bounding box when resampling and cropping.
        ibsi_interpolation: Explicit IBSI centered-grid interpolation. Mutually exclusive
            with ``resampled_spacing`` and ``correct_mask``; ``pad_distance`` does not apply
            because IBSI keeps the full source extent.
    """

    resampled_spacing: tuple[float, ...] | None = None
    interpolator: Interpolator = Interpolator.BSPLINE
    correct_mask: bool = False
    pad_distance: int = 5
    ibsi_interpolation: IbsiInterpolation | None = None

    def __post_init__(self) -> None:
        """Reject an untyped interpolator, non-boolean flags or invalid spacing and padding."""
        if type(self.interpolator) is not Interpolator:
            raise ValueError("interpolator must be an Interpolator value")
        if type(self.correct_mask) is not bool:
            raise ValueError("correct_mask must be boolean")
        if type(self.pad_distance) is not int or self.pad_distance < 0:
            raise ValueError("pad_distance must be a nonnegative integer")
        if self.resampled_spacing is not None and (
            len(self.resampled_spacing) not in (2, 3)
            or any(not isfinite(v) or v < 0 for v in self.resampled_spacing)
        ):
            raise ValueError("resampled_spacing requires 2 or 3 finite nonnegative values")
        if self.ibsi_interpolation is not None:
            if type(self.ibsi_interpolation) is not IbsiInterpolation:
                raise ValueError("ibsi_interpolation must be an IbsiInterpolation value")
            if self.resampled_spacing is not None:
                raise ValueError("ibsi_interpolation and resampled_spacing are mutually exclusive")
            if self.correct_mask:
                raise ValueError(
                    "ibsi_interpolation interpolates the mask; correct_mask must be False"
                )
