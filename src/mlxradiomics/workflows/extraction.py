"""Feature extraction on derived images (wavelet, LoG, LBP, ...), via ``engine.extract``."""


import os
import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.contracts.metadata import file_metadata, transform_payload
from mlxradiomics.contracts.results import Result
from mlxradiomics.contracts.types import (
    ExecutionMetadata,
    Features,
    FeatureValidity,
    FloatArray,
    HostArray,
    SettingsPayload,
    VoxelSink,
    VoxelSinkFactory,
)
from mlxradiomics.imaging.preprocessing import prepare_sampled_input
from mlxradiomics.imaging.spatial import load_image_pair
from mlxradiomics.imaging.transforms import feature_image_names
from mlxradiomics.numerics.runtime import (
    mx,
    releases_cache,
    require_retained_capacity,
)
from mlxradiomics.workflows.derived import (
    extract_lbp_kurtosis,
    feature_arrays,
    prepare_derived_inputs,
)
from mlxradiomics.workflows.engine import PREPROCESSING_DESCRIPTION, extract_with_shape_mask
from mlxradiomics.workflows.engine import extract as extract_original


def _forward_prefixed(
    centers: NDArray[np.int32],
    batch: Mapping[str, NDArray[np.float32]],
    *,
    prefix: str,
    target: VoxelSink,
) -> None:
    """Prefix one image's feature columns and forward its evaluated batch."""
    target(centers, {
        name.replace("original_", prefix + "_", 1): values
        for name, values in batch.items()
    })


def _validate_mode(
    mode: Literal["roi", "voxel"], sink: VoxelSink | None,
    sink_factory: VoxelSinkFactory | None,
) -> bool:
    """Validate ``mode`` with its sink arguments and return whether it is voxel mode."""
    if mode not in ("roi", "voxel"):
        raise ValueError("mode must be 'roi' or 'voxel'")
    if sink is not None and sink_factory is not None:
        raise ValueError("pass only one of sink or sink_factory")
    if mode == "roi" and (sink is not None or sink_factory is not None):
        raise ValueError("sink and sink_factory require mode='voxel'")
    if sink is not None and not callable(sink):
        raise ValueError("sink must be callable")
    if sink_factory is not None and not callable(sink_factory):
        raise ValueError("sink_factory must be callable")
    return mode == "voxel"


@overload
def extract(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    transforms: TransformSettings | None = None,
    spacing: tuple[float, ...] | None = None,
    mode: Literal["roi"] = "roi",
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result[float]:
    ...


@overload
def extract(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    transforms: TransformSettings | None = None,
    spacing: tuple[float, ...] | None = None,
    mode: Literal["voxel"],
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result[FloatArray]:
    ...


@overload
def extract(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    transforms: TransformSettings | None = None,
    spacing: tuple[float, ...] | None = None,
    mode: Literal["roi", "voxel"],
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result:
    ...


@releases_cache
def extract(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    transforms: TransformSettings | None = None,
    spacing: tuple[float, ...] | None = None,
    mode: Literal["roi", "voxel"] = "roi",
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result:
    """Extract ROI or voxel features for the requested image transforms on Metal.

    Image and mask must share a nonempty 2-D or 3-D NumPy array shape. Spacing
    follows that array-axis order (mm) and defaults to 1.0 per axis. Shape features
    use the original inputs once; intensity feature keys carry each image
    transform's name. Normalization and resegmentation, including a combined
    absolute-range and outlier-sigma intersection, are applied once before the derived
    images are computed and are not repeated on them. Each derived image is discretized
    separately with ``settings.bin_width`` or ``bin_count``; the gray-level count of each
    is recorded in ``metadata["gray_levels_by_image"]``. A derived image keeps the
    original shape/morphology mask selected by ``resegment_shape``: the intensity ROI and
    the GLDZM zone-boundary mask are carried separately, so filtering cannot replace the
    unfiltered morphology with the resegmented intensity support.

    Undefined LBP3D kurtosis samples propagate NaN to the affected ROI or voxel
    neighborhoods, with validity counts, while the harmonic-band features remain usable.

    For original-image features, only sampled values (the
    selected ROI, or the whole image with unmasked voxel kernels) must be finite, so a
    non-finite background is accepted when ``Original`` is the only requested image type.
    Every derived filter instead requires the entire image to be finite because its
    transform is defined on the whole array.

    ``mode="roi"`` returns one Python float per feature. ``mode="voxel"`` returns
    float64 maps on the input grid, with zero outside the ROI. ``sink`` receives
    evaluated float32 batches as ``(centers, features)``; centers have shape
    ``(batch, ndim)`` in NumPy axis order and each feature has shape ``(batch,)``.
    ``sink_factory(image_name)`` instead supplies a context-managed sink per image.
    Sinks require voxel mode, are mutually exclusive and leave ``features`` empty.
    Arrays passed to sinks are borrowed; retain copies if needed after the call.

    Args:
        image: Aligned 2D or 3D integer or floating-point NumPy array.
        mask: Boolean or integer label array with the same shape.
        settings: Feature selection and preprocessing; defaults to ``Settings()``.
        transforms: Derived images to include; defaults to the original image.
        spacing: Positive millimeters per NumPy axis. Omission warns and uses unit spacing.
        mode: ``"roi"`` for scalars or ``"voxel"`` for neighborhood maps.
        sink: Optional consumer of evaluated voxel batches.
        sink_factory: Optional per-image context manager for a voxel consumer.

    Returns:
        ``Result[float]`` in ROI mode or ``Result[NDArray[np.float64]]`` in voxel mode.

    Raises:
        ValueError: If the inputs, mode or settings are invalid, LBP3D is
            combined with unmasked voxel mode, a requested derived filter encounters
            non-finite image values, no requested image can produce the selected
            features, or a sink is supplied in ROI mode.
        MemoryError: If all retained float64 maps together exceed one quarter of
            unified memory. A sink or sink factory streams without this payload.
    """
    if spacing is None:
        warnings.warn(
            "Array extraction assumes 1 mm spacing on every axis. Pass spacing explicitly "
            "in NumPy axis order, or use extract_files to read physical geometry.",
            UserWarning, stacklevel=2,
        )
    voxel = _validate_mode(mode, sink, sink_factory)
    voxel_sink, voxel_sink_factory = sink, sink_factory
    histogram_method = "mlx"
    streaming = sink is not None or sink_factory is not None
    settings, transforms = settings or Settings(), transforms or TransformSettings()
    shape_names = tuple(n for n in settings.features if n.startswith("shape"))
    intensity_names = tuple(n for n in settings.features if not n.startswith("shape"))
    if (
        not voxel and transforms.image_types == (ImageType.ORIGINAL,)
        and not (shape_names and intensity_names)
    ):
        # The original-only ROI route needs no filtered-image preparation. The
        # engine already validates, normalizes and resegments the complete source.
        result = extract_original(
            image, mask, settings=settings, spacing=spacing,
            histogram_method=histogram_method,
        )
        metadata = result.metadata
        metadata["gray_levels_by_image"] = (
            {"original": metadata["gray_levels"]} if intensity_names else {}
        )
        metadata["transforms"] = transform_payload(transforms)
        return Result(result.features, metadata)
    prepared_input = prepare_sampled_input(image, mask, settings, voxel=voxel)
    data = prepared_input[0]
    spacing = spacing or (1.0,) * data.image.ndim
    if len(spacing) != data.image.ndim or any(not np.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("spacing must contain one finite positive value per array axis")
    if ImageType.LBP3D in transforms.image_types and voxel and not settings.masked_kernel:
        raise ValueError(
            "pinned LBP3D leaves outside-ROI values undefined; unmasked voxels unsupported"
        )
    if voxel and not streaming and intensity_names:
        image_names = feature_image_names(
            data.image.shape, spacing, transforms, force2d_axis=settings.force2d_axis,
        )
        require_retained_capacity(data.image.size, len(intensity_names) * len(image_names))
    output: Features = {}
    gray_levels_by_image: dict[str, int] = {}
    validity: dict[str, FeatureValidity] = {}
    metadata: ExecutionMetadata | None = None
    if shape_names:
        shape = extract_with_shape_mask(
            image,
            mask,
            settings=replace(settings, features=shape_names),
            spacing=spacing,
            voxel=voxel,
            histogram_method=histogram_method,
            _prepared_input=prepared_input,
        )
        output.update(shape.features)
        metadata = shape.metadata
        validity.update(metadata["feature_validity"])
    if ImageType.ORIGINAL not in transforms.image_types:
        prepared_input = None
    if intensity_names:
        inputs = prepare_derived_inputs(data, mask, settings, transforms, spacing, voxel)
        derived_settings, derived_mask = inputs.settings, inputs.mask
        crop, window_masks = inputs.crop, inputs.masks
        with mx.stream(mx.gpu):
            image_items = feature_arrays(image, data, settings, transforms, spacing, inputs.window)
            for name, derived in image_items:
                with ExitStack() as image_resources:
                    image_sink = (
                        image_resources.enter_context(voxel_sink_factory(name))
                        if voxel_sink_factory is not None else None
                    )
                    consume = image_sink or voxel_sink
                    stream_sink: VoxelSink | None = None
                    if consume is not None:
                        stream_sink = partial(
                            _forward_prefixed, prefix=name, target=consume,
                        )
                    if derived is None:
                        result = extract_with_shape_mask(
                            image,
                            mask,
                            settings=replace(settings, features=intensity_names),
                            spacing=spacing,
                            voxel=voxel,
                            histogram_method=histogram_method,
                            voxel_sink=stream_sink,
                            _prepared_input=prepared_input,
                        )
                        prepared_input = None
                    elif name == "lbp-3D-k":
                        result = extract_lbp_kurtosis(
                            derived,
                            derived_mask[crop],
                            settings=derived_settings,
                            spacing=spacing,
                            voxel=voxel,
                            histogram_method=histogram_method,
                            voxel_sink=stream_sink,
                            shape_mask=data.shape_roi[crop],
                            _prepared_masks=window_masks,
                        )
                    else:
                        result = extract_with_shape_mask(
                            derived,
                            derived_mask[crop],
                            settings=derived_settings,
                            spacing=spacing,
                            voxel=voxel,
                            histogram_method=histogram_method,
                            voxel_sink=stream_sink,
                            # The filtered/derived image keeps the shape ROI that the
                            # original settings derived; it is not reconstructed from
                            # the intensity ROI or the filtered intensities.
                            shape_mask=data.shape_roi[crop],
                            _prepared_masks=window_masks,
                        )
                metadata = result.metadata
                gray_levels_by_image[name] = metadata["gray_levels"]
                validity.update({
                    feature_name.replace("original_", name + "_", 1): counts
                    for feature_name, counts in metadata["feature_validity"].items()
                })
                output.update(
                    {
                        feature_name.replace("original_", name + "_", 1): values
                        for feature_name, values in result.features.items()
                    }
                )
    if metadata is None:
        raise ValueError("no requested feature image can produce selected features")
    metadata["settings"] = cast(SettingsPayload, asdict(settings))
    metadata["gray_levels_by_image"] = gray_levels_by_image
    metadata["feature_validity"] = validity
    metadata["transforms"] = transform_payload(transforms)
    if ImageType.LBP3D in transforms.image_types:
        metadata["runtime"]["versions"].update(
            {name: version(name) for name in ("scipy", "trimesh")}
        )
    metadata["array_dtype"] = str(image.dtype)
    # A cropped derived-image result reports its window; the call's array is the input.
    metadata["array_shape"] = tuple(int(size) for size in data.image.shape)
    metadata["roi_voxels"] = int(np.count_nonzero(data.roi))
    metadata["excluded_voxels"] = data.image.size - metadata["roi_voxels"]
    metadata["preprocessing"] = PREPROCESSING_DESCRIPTION
    return Result(output, metadata)


@overload
def extract_files(
    image_path: str | os.PathLike[str], mask_path: str | os.PathLike[str], *,
    settings: Settings | None = None,
    image_settings: ImageSettings | None = None, transforms: TransformSettings | None = None,
    mode: Literal["roi"] = "roi",
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result[float]:
    ...


@overload
def extract_files(
    image_path: str | os.PathLike[str], mask_path: str | os.PathLike[str], *,
    settings: Settings | None = None,
    image_settings: ImageSettings | None = None, transforms: TransformSettings | None = None,
    mode: Literal["voxel"],
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result[FloatArray]:
    ...


@overload
def extract_files(
    image_path: str | os.PathLike[str], mask_path: str | os.PathLike[str], *,
    settings: Settings | None = None,
    image_settings: ImageSettings | None = None, transforms: TransformSettings | None = None,
    mode: Literal["roi", "voxel"],
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result:
    ...


def extract_files(
    image_path: str | os.PathLike[str], mask_path: str | os.PathLike[str], *,
    settings: Settings | None = None,
    image_settings: ImageSettings | None = None, transforms: TransformSettings | None = None,
    mode: Literal["roi", "voxel"] = "roi",
    sink: VoxelSink | None = None,
    sink_factory: VoxelSinkFactory | None = None,
) -> Result:
    """Read an image and mask file, preprocess on the host, and extract features on Metal.

    Combines ``load_image_pair`` with ``extract``. Normalization and outlier removal
    happen while loading, so they are switched off in the extraction call. The result's
    ``metadata["image_geometry"]`` describes the grid actually used, after any crop or
    resampling; ``source_geometry`` retains the input grid and ``image_settings``
    records the effective spatial options.

    Args:
        image_path: NIfTI or NRRD image.
        mask_path: NIfTI or NRRD integer label mask.
        settings: Feature selection and preprocessing options.
        image_settings: Resampling, IBSI interpolation and mask-correction options.
        transforms: Derived images to compute; defaults to the original image only.
        mode: ``"roi"`` for scalar features or ``"voxel"`` for maps.
        sink: Optional evaluated-batch consumer in voxel mode, as in
            ``extract``. Coordinates follow the processed grid after resampling
            or cropping; no dense feature maps are retained when a sink is supplied.
        sink_factory: Optional context-managed voxel consumer per derived image.

    Returns:
        A ``Result`` whose spacing came from the processed image.

    Raises:
        ValueError: If the files, geometry or label fail :func:`load_image_pair`, or
            the settings, mode or sink are invalid as in :func:`extract`.
        RuntimeError: If SimpleITK cannot read a supplied file.
    """
    _validate_mode(mode, sink, sink_factory)
    options = settings or Settings()
    pair = load_image_pair(
        Path(image_path).expanduser(), Path(mask_path).expanduser(),
        settings=options, image_settings=image_settings,
    )
    extraction = replace(options, normalize=False, remove_outliers=None)
    result = extract(
        pair.image.array, pair.mask.array, settings=extraction, transforms=transforms,
        spacing=pair.image.spacing, mode=mode, sink=sink, sink_factory=sink_factory,
    )
    metadata = file_metadata(
        result.metadata, pair.image, settings=options, source_geometry=pair.source_geometry,
        image_settings=image_settings or ImageSettings(),
    )
    return Result(result.features, metadata)
