"""Original-image admission, preparation and dispatch to ROI or voxel extraction."""

from collections.abc import Mapping
from dataclasses import asdict, replace
from importlib.metadata import version
from math import floor, isfinite, isnan, prod
from os import environ
from platform import platform, python_version
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics import __version__
from mlxradiomics.config.settings import Settings
from mlxradiomics.contracts.results import Result
from mlxradiomics.contracts.types import (
    ExecutionMetadata,
    HostArray,
    MapDtype,
    SettingsPayload,
    VoxelArray,
    VoxelSink,
)
from mlxradiomics.contracts.validity import empty_validity
from mlxradiomics.features.aggregation import resolved_aggregations
from mlxradiomics.features.ivh import prepare_ivh
from mlxradiomics.features.morphology import require_pairwise_admission
from mlxradiomics.imaging.preprocessing import (
    PreparedInput,
    discretize_prepared,
    prepare_derived_input,
    prepare_sampled_input,
    validate_exact_float64,
)
from mlxradiomics.numerics.runtime import (
    mx,
    releases_cache,
    require_retained_capacity,
    validate_map_dtype,
)
from mlxradiomics.voxel.extraction import extract_voxels
from mlxradiomics.workflows.roi import extract_roi, occupied_window

PREPROCESSING_DESCRIPTION = (
    "host normalization, resegmentation and discretization; native Metal image filters/features"
)


@releases_cache
def extract(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    spacing: tuple[float, ...] | None = None,
    voxel: bool = False,
    histogram_method: str = "mlx",
    voxel_sink: VoxelSink | None = None,
    map_dtype: MapDtype = "float64",
) -> Result:
    """Extract original-image features from aligned arrays (NumPy axis order).

    Internal engine behind :func:`mlxradiomics.workflows.extraction.extract`, which
    adds derived images, sinks per image and the public contract. Host preprocessing
    bins in float64; feature arithmetic uses float32 on Metal. This function does not
    resample or filter images. GLCM features average only the directions with a pair
    at that sample. MCC averages only directions with at least two occupied row and
    column gray levels in their local matrices; it is NaN when none qualify, including
    empty and flat regions. Map backgrounds are 0.

    Args:
        image: 2-D or 3-D integer or float array, in the image's own intensity units (HU
            for CT). Axes follow NumPy order, so ``spacing`` has one entry per array axis.
        mask: Bool or integer array of the same shape. Voxels equal to ``settings.label``
            form the ROI.
        settings: Feature selection and preprocessing; defaults to ``Settings()`` (all 93
            non-shape features, bin width 25, label 1).
        spacing: Voxel spacing per array axis, finite and positive, in mm for physical
            results. Defaults to 1.0 per axis. It scales shape features and TotalEnergy
            (the spacing product is voxel volume in 3-D, pixel area in 2-D); texture
            features do not use it.
        voxel: False for one value per feature over the ROI. True for a map per feature,
            computed in a ``(2 * kernel_radius + 1)``-wide neighborhood around each ROI
            voxel (shape features are unavailable).
        histogram_method: ``"mlx"`` or ``"metal"``, the GLCM counting route; see
            ``histogram.histogram``.
        voxel_sink: Optional batch consumer, valid only in voxel mode. Receives
            ``(centers, features)`` after evaluation: int32 NumPy-order coordinates of
            shape ``(batch, ndim)`` and float32 arrays of shape ``(batch,)`` keyed
            ``original_<family>_<Name>``. No full maps are allocated and the returned
            ``features`` mapping is empty. Sink exceptions propagate.
        map_dtype: Retained-map storage; defaults to float64. Float32 retains the same
            evaluated float32 batch values using half the map payload.

    Returns:
        A ``Result`` with keys ``original_<family>_<Name>``, or an empty feature mapping
        when ``voxel_sink`` consumes the batches (see ``Result``).

    Raises:
        ValueError: If the method, settings, image, mask or spacing are invalid or
            unsupported: the label is absent, the gray-level count exceeds
            ``max_gray_levels``, voxel mode has fewer than two ROI voxels or a neighborhood
            larger than ``scratch_mib``, shape features are requested in voxel mode or
            ``shape`` on a non-3-D mask, ``shape2D`` lacks a single-slice ROI, or the image
            has 2**30 or more voxels, ``voxel_sink`` is supplied in ROI mode, or
            shape Moran's I or Geary's C is requested for more than
            ``morphology.MAX_PAIRWISE_VOXELS`` ROI voxels.
        RuntimeError: If the Metal histogram route is requested without Metal, or an
            iterative GPU solver (GLSZM zone labeling, Jacobi eigenvalues) does not
            converge.
        MemoryError: If retained maps would exceed one quarter of unified
            memory. Supply ``voxel_sink`` to stream evaluated batches instead.
    """
    return extract_with_shape_mask(
        image,
        mask,
        settings=settings,
        spacing=spacing,
        voxel=voxel,
        histogram_method=histogram_method,
        voxel_sink=voxel_sink,
        map_dtype=map_dtype,
    )


def input_window(
    image: NDArray[np.generic],
    mask: HostArray,
    shape_mask: HostArray | None,
    settings: Settings,
    spacing: tuple[float, ...] | None,
) -> tuple[slice, ...] | None:
    """ROI-mode crop of the inputs that leaves every selected feature unchanged.

    The window holds the label and shape masks padded by the largest neighbor reach and,
    when a local-intensity peak is selected, by the physical sphere's reach on each axis,
    clipped to the image. Texture windows and GLDZM border shells therefore keep their
    content, and a sphere is truncated only where the image itself ends. Direct ROI
    extraction validates settings and whole-source 64-bit integer admission, then
    prepares and discretizes only this window. Normalization retains the complete image
    because its statistics cover every voxel.
    """
    labels = np.asarray(mask)
    if (
        settings.normalize or image.ndim not in (2, 3) or labels.shape != image.shape
        or labels.dtype.kind not in "biu"
    ):
        return None
    occupied = labels == settings.label
    if shape_mask is not None:
        shape = np.asarray(shape_mask, dtype=np.bool_)
        if shape.shape != image.shape:
            return None
        occupied = occupied | shape
    if not np.any(occupied):
        return None
    reach = [max(*settings.distances, settings.neighborhood_radius)] * image.ndim
    if any(name.startswith("localintensity_") for name in settings.features):
        steps = spacing or (1.0,) * image.ndim
        if len(steps) != image.ndim or any(not np.isfinite(v) or v <= 0 for v in steps):
            return None
        radius = (3000 / (4 * np.pi)) ** (1 / 3)
        reach = [max(value, floor(radius / step)) for value, step in zip(reach, steps, strict=True)]
    return occupied_window(occupied, reach)


def extract_with_shape_mask(
    image: HostArray,
    mask: HostArray,
    *,
    settings: Settings | None = None,
    spacing: tuple[float, ...] | None = None,
    voxel: bool = False,
    histogram_method: str = "mlx",
    voxel_sink: VoxelSink | None = None,
    map_dtype: MapDtype = "float64",
    shape_mask: HostArray | None = None,
    _prepared_input: tuple[PreparedInput, NDArray[np.float64]] | None = None,
    _prepared_masks: PreparedInput | None = None,
) -> Result:
    """Extract with an optional explicit morphology mask for derived images.

    ``workflows.extraction.extract`` supplies ``shape_mask`` so that the original shape
    ROI survives when resegmentation does not apply to shape features. It overrides only
    ``Prepared.shape_roi``; intensity sampling and resegmentation keep using ``mask``.
    Callers within ``workflows`` pass ``_prepared_input`` for an already validated source,
    or ``_prepared_masks`` for a derived image whose pixels still need validation.
    Discretization remains specific to this image and selected feature subset.
    Direct ROI extraction prepares and discretizes only the window every selected
    feature reads (``input_window``); the complete source is still checked for exact
    64-bit integer conversion, and normalization retains the complete image.
    This is the implementation of ``extract``.
    """
    if histogram_method not in {"mlx", "metal"}:
        raise ValueError("histogram_method must be mlx or metal")
    if voxel_sink is not None and not voxel:
        raise ValueError("voxel_sink requires voxel=True")
    validate_map_dtype(map_dtype)
    settings = settings or Settings()
    if voxel and any(name.startswith("shape") for name in settings.features):
        raise ValueError("shape features are unavailable in voxel mode")
    if voxel and any(name.startswith("localintensity_") for name in settings.features):
        raise ValueError("physical local-intensity peak features are unavailable in voxel mode")
    if voxel and voxel_sink is None:
        require_retained_capacity(image.size, len(settings.features), map_dtype)
    source_shape = tuple(int(size) for size in np.shape(image))
    # Direct ROI extraction prepares only the window every selected feature reads.
    # Private prepared inputs arrive in their caller's coordinates and are not cropped
    # again here; the shape/local-intensity crop below still applies to them.
    preparation_window: tuple[slice, ...] | None = None
    if _prepared_input is not None:
        prepared, sampled = _prepared_input
    elif _prepared_masks is not None:
        prepared, sampled = prepare_derived_input(image, _prepared_masks)
    else:
        source_image, source_mask = np.asarray(image), np.asarray(mask)
        preparation_image, preparation_mask = source_image, source_mask
        if (
            not voxel and source_image.ndim in (2, 3)
            and source_mask.shape == source_image.shape
            and all(size > 0 for size in source_shape) and prod(source_shape) < 2**30
            and source_image.dtype.kind in "iuf" and source_mask.dtype.kind in "biu"
        ):
            # Reject invalid settings before planning, exactly as whole-image
            # preparation would. Normalization keeps the complete image, and
            # input_window refuses to plan a window for it.
            settings.validate(source_image.ndim)
            if shape_mask is not None:
                shape_mask = np.asarray(shape_mask, dtype=np.bool_)
            window = input_window(source_image, source_mask, shape_mask, settings, spacing)
            if window is not None:
                # The complete source reached exact_float64 before windowing, and
                # values outside the ROI still decided admission. Keep that whole-source
                # check without converting the whole image.
                validate_exact_float64(source_image)
                preparation_image = source_image[window]
                preparation_mask = source_mask[window]
                preparation_window = window
        prepared, sampled = prepare_sampled_input(
            preparation_image, preparation_mask, settings, voxel=voxel,
        )
    data = discretize_prepared(prepared, sampled, settings)
    if shape_mask is not None:
        shape_mask = np.asarray(shape_mask, dtype=np.bool_)
        # Direct preparation cropped the source, but shape_mask still covers the source
        # array the caller passed. Other routes prepare the caller's image in full.
        expected_shape = source_shape if preparation_window is not None else data.image.shape
        if shape_mask.shape != expected_shape:
            raise ValueError("shape_mask must match the image shape")
        if preparation_window is not None:
            shape_mask = shape_mask[preparation_window]
        data = replace(data, shape_roi=shape_mask)
    spacing = spacing or (1.0,) * data.image.ndim
    if len(spacing) != data.image.ndim or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("spacing must contain one finite positive value per array axis")
    if prod(source_shape) >= 2**30:
        raise ValueError("image exceeds supported integer indexing/count range")
    if (
        not voxel and preparation_window is None
        and any(name.startswith(("shape", "localintensity_")) for name in settings.features)
    ):
        # Prepared inputs may still cover the caller's full grid; crop them after the
        # settings, source and shape errors above. Direct preparation already used this
        # same window, so cropping it again would only repeat the mask reductions.
        window = input_window(data.image, mask, shape_mask, settings, spacing)
        if window is not None:
            data = replace(
                data, image=data.image[window], roi=data.roi[window],
                shape_roi=data.shape_roi[window], sampling_mask=data.sampling_mask[window],
                bins=data.bins[window] if data.bins is not None else None,
            )
    dtype = mx.float32
    roi_voxels = int(np.count_nonzero(data.roi))
    if not voxel:
        # Refuse quadratic pair work before any other feature is computed.
        require_pairwise_admission(
            [name.removeprefix("shape_") for name in settings.features], roi_voxels,
        )
    excluded_voxels = prod(source_shape) - roi_voxels
    validity = {f"original_{name}": empty_validity(excluded_voxels if voxel else 0)
                for name in settings.features}
    ivh = (prepare_ivh(data.image, data.sampling_mask, settings, sampled=sampled)
           if any(name.startswith("ivh_") for name in settings.features) else None)
    metadata: ExecutionMetadata = {
        "settings": cast(SettingsPayload, asdict(settings)),
        "device": "gpu",
        "precision": "float32",
        "voxel": voxel,
        "spacing": spacing,
        "histogram_method": histogram_method,
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "gray_levels": data.gray_levels,
        "runtime": {
            "versions": {"mlxradiomics": __version__, "numpy": np.__version__,
                         "mlx": version("mlx"), "medmlx-core": version("medmlx-core")},
            "python": python_version(), "platform": platform(),
            "device_name": str(mx.device_info(mx.gpu)["device_name"]),
            "mlx_enable_tf32": environ.get("MLX_ENABLE_TF32"),
        },
        "array_shape": source_shape,
        "array_dtype": str(image.dtype),
        "roi_voxels": roi_voxels,
        "excluded_voxels": excluded_voxels,
        "feature_validity": validity,
        "aggregation": resolved_aggregations(settings, data.image.ndim),
    }
    features: Mapping[str, float | VoxelArray]
    if voxel:
        metadata["map_dtype"] = map_dtype
    if ivh is not None:
        metadata["ivh_lattice"] = {"origin": ivh.origin, "step": ivh.step, "levels": ivh.levels}
    with mx.stream(mx.gpu):
        if voxel:
            # ROI statistics reuse these host samples. Voxel neighborhoods gather
            # their own values, so release the full sampled vector before batching.
            del sampled
            features, memory = extract_voxels(
                data, settings, dtype, prod(spacing), histogram_method, voxel_sink, validity, ivh,
                map_dtype,
            )
            metadata.update(
                batch_size=memory["batch_size"],
                scratch_estimate_bytes=memory["scratch_estimate_bytes"],
            )
            if "first_order_batch_size" in memory:
                metadata["first_order_batch_size"] = memory["first_order_batch_size"]
            if "first_order_scratch_estimate_bytes" in memory:
                metadata["first_order_scratch_estimate_bytes"] = (
                    memory["first_order_scratch_estimate_bytes"]
                )
            if "first_order_cache_bytes" in memory:
                metadata["first_order_cache_bytes"] = memory["first_order_cache_bytes"]
        else:
            features = extract_roi(
                data, settings, dtype, prod(spacing), histogram_method, spacing, ivh, sampled,
            )
            for name, value in features.items():
                # extract_roi returns Python floats; classify them without allocating scalar
                # NumPy arrays or dispatching five ufunc reductions per feature.
                scalar = value
                counts = validity[name]
                if isfinite(scalar):
                    counts["finite"] += 1
                    counts["valid_zero"] += int(scalar == 0)
                elif isnan(scalar):
                    counts["nan"] += 1
                elif scalar > 0:
                    counts["positive_infinity"] += 1
                else:
                    counts["negative_infinity"] += 1
    return Result(features, metadata)
