"""Host arrays, streaming consumers and typed execution/report payloads."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Literal, NotRequired, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.contracts.json import JsonValue

type HostArray = NDArray[np.generic]
type FloatArray = NDArray[np.float64]
type Features = dict[str, float | FloatArray]


class VoxelSink(Protocol):
    """Consume evaluated float32 feature batches at NumPy-order ROI coordinates.

    Arrays are borrowed for the call; retain copies if they are needed afterwards.
    """

    def __call__(
        self,
        centers: NDArray[np.int32],
        features: Mapping[str, NDArray[np.float32]],
        /,
    ) -> None:
        """Receive ``(C, D)`` int32 centers and one ``(C,)`` float32 array per feature."""
        ...


class VoxelSinkFactory(Protocol):
    """Create one sink per derived image, entering before and exiting after its batches.

    The context manager must flush or release that image's storage on exit so a caller
    can export many derived images with bounded resident buffers and scratch space.
    """

    def __call__(self, image_name: str, /) -> AbstractContextManager[VoxelSink]:
        """Return the context-managed sink for derived image ``image_name``."""
        ...


class SettingsPayload(TypedDict):
    """Plain-dict copy of ``Settings`` (``dataclasses.asdict``), as stored in metadata."""

    gldm_alpha: int
    bin_width: float
    bin_count: int | None
    bin_anchor: float | None
    normalize: bool
    normalize_scale: float
    remove_outliers: float | None
    resegment_range: tuple[float, ...] | None
    resegment_mode: Literal["absolute", "relative", "sigma"]
    resegment_outlier_sigma: float | None
    resegment_shape: bool
    label: int
    distances: tuple[int, ...]
    symmetrical_glcm: bool
    force2d_axis: int | None
    masked_kernel: bool
    kernel_radius: int
    voxel_array_shift: float
    scratch_mib: int
    max_gray_levels: int | None
    features: tuple[str, ...]
    covariance_divisor: str
    ellipsoid_area: str
    kurtosis: str
    zero_variance_moments: str
    ngtdm_singular_cases: str
    neighborhood_support: str
    bin_origin: str
    constant_roi_bin: str
    mask_extrapolation: str
    spline_boundary: str
    glcm_aggregation: str
    glrlm_aggregation: str
    glszm_aggregation: str
    gldzm_aggregation: str
    ngtdm_aggregation: str
    gldm_aggregation: str
    slice_axis: int
    neighborhood_radius: int
    ivh_bin_width: float | None
    ivh_bin_count: int | None
    ivh_intensity_range: tuple[float, float] | None


class ImageGeometryPayload(TypedDict):
    """Physical placement of the arrays that were processed.

    ``shape`` and ``spacing`` follow NumPy array-axis order. ``origin`` and ``direction``
    keep the ITK physical x/y/z order; ``direction`` is the flattened row-major matrix.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


class FeatureValidity(TypedDict):
    """Counts over computed feature samples; excluded map background is counted separately.

    ``valid_zero`` is a subset of ``finite``, including either sign of zero.
    ``nan`` marks undefined values; infinities retain their signs.
    """

    finite: int
    valid_zero: int
    nan: int
    positive_infinity: int
    negative_infinity: int
    excluded: int


class RuntimeMetadata(TypedDict):
    """Non-secret runtime provenance, without local paths or device identifiers."""

    versions: dict[str, str]
    python: str
    platform: str
    device_name: str
    mlx_enable_tf32: str | None


class TransformPayload(TypedDict):
    """Effective ``TransformSettings`` serialized without importing filter code."""

    scratch_mib: int
    image_types: tuple[str, ...]
    wavelet: str
    wavelet_level: int
    wavelet_start_level: int
    sigma: tuple[float, ...]
    log_kernel: str
    log_cutoff: float
    log_padding: str
    log_constant: float
    log_planar: bool
    lbp2d_radius: float
    lbp2d_samples: int
    lbp2d_method: str
    lbp3d_levels: int
    lbp3d_radius: float
    lbp3d_subdivision: int
    gradient_use_spacing: bool
    mean_support: int
    mean_padding: str
    mean_constant: float
    mean_planar: bool
    laws_kernel: str
    laws_padding: str
    laws_constant: float
    laws_pooling: str
    laws_energy_distance: int | None
    gabor_sigma: tuple[float, ...]
    gabor_wavelength: tuple[float, ...]
    gabor_gamma: float
    gabor_theta: tuple[float, ...]
    gabor_response: str
    gabor_pooling: str
    gabor_slice_axis: int | None
    gabor_padding: str
    gabor_constant: float
    simoncelli_levels: tuple[int, ...]
    simoncelli_padding: str
    simoncelli_grid: str
    simoncelli_planar: bool
    riesz_base: str
    riesz_orders: tuple[tuple[int, ...], ...]
    riesz_log_sigma: tuple[float, ...]
    riesz_log_cutoff: float
    riesz_simoncelli_levels: tuple[int, ...]
    riesz_padding: str
    riesz_constant: float
    riesz_grid: str
    riesz_steering: bool
    riesz_tensor_sigma: float | None
    separable_wavelet: str
    separable_wavelet_bands: tuple[str, ...]
    separable_wavelet_level: int
    separable_wavelet_padding: str
    separable_wavelet_constant: float
    separable_wavelet_pooling: str
    separable_wavelet_planar: bool


class IbsiPayload(TypedDict):
    """Effective IBSI interpolation settings, in NumPy spacing order.

    ``image_interpolator`` names the intensity-image interpolator; the ROI mask is always
    interpolated linearly and is therefore not represented separately.
    """

    spacing: tuple[float, ...]
    round_intensities: bool
    mask_threshold: float
    image_interpolator: str


class ImageSettingsPayload(TypedDict):
    """Effective host grid-processing settings, in NumPy spacing order."""

    resampled_spacing: tuple[float, ...] | None
    interpolator: str
    correct_mask: bool
    pad_distance: int
    ibsi_interpolation: IbsiPayload | None


class ExecutionMetadata(TypedDict):
    """Provenance of one extraction, returned as ``Result.metadata``.

    ``spacing`` is in the unit of the supplied spacing (mm for image files), one entry per
    array axis. ``gray_levels`` is the highest gray-level bin index (1 when only shape
    features were requested). ``batch_size`` and ``scratch_estimate_bytes`` are set in
    voxel mode, ``gray_levels_by_image`` when derived images are extracted, and
    ``image_geometry`` for file extraction. ``source_geometry`` retains the file's
    grid before spatial processing. Feature validity counts classify computed samples only;
    ``excluded_voxels`` records the array volume outside the processed intensity ROI.
    """

    settings: SettingsPayload
    device: str
    precision: str
    voxel: bool
    spacing: tuple[float, ...]
    histogram_method: str
    preprocessing: str
    gray_levels: int
    runtime: RuntimeMetadata
    array_shape: tuple[int, ...]
    array_dtype: str
    roi_voxels: int
    excluded_voxels: int
    feature_validity: dict[str, FeatureValidity]
    aggregation: NotRequired[dict[str, str]]
    ivh_lattice: NotRequired[dict[str, float | int]]
    gray_levels_by_image: NotRequired[dict[str, int]]
    batch_size: NotRequired[int]
    scratch_estimate_bytes: NotRequired[int]
    first_order_batch_size: NotRequired[int]
    first_order_scratch_estimate_bytes: NotRequired[int]
    first_order_cache_bytes: NotRequired[int]
    image_geometry: NotRequired[ImageGeometryPayload]
    source_geometry: NotRequired[ImageGeometryPayload]
    image_settings: NotRequired[ImageSettingsPayload]
    transforms: NotRequired[TransformPayload]


class VoxelMemory(TypedDict):
    """Delivery batch size and peak scratch estimate, including cached feature vectors.

    Optional first-order fields describe a larger computation group. Its results are
    sliced into complete delivery batches; it does not change the sink's feature keys.
    """

    batch_size: int
    scratch_estimate_bytes: int
    first_order_batch_size: NotRequired[int]
    first_order_scratch_estimate_bytes: NotRequired[int]
    first_order_cache_bytes: NotRequired[int]


class RunDetails(TypedDict):
    """Run metadata; ``shape`` and ``spacing`` follow the processed image array axis order."""

    voxel: bool
    label: int
    shape: list[int]
    spacing: list[float]
    feature_count: int
    execution: dict[str, JsonValue]
