"""Original-image ROI feature orchestration."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.aggregation import aggregated_texture_statistics
from mlxradiomics.features.cooccurrence import aggregate_roi_glcm
from mlxradiomics.features.first_order import first_order, first_order_histogram, first_order_inputs
from mlxradiomics.features.first_order_source import prepare_first_order_source
from mlxradiomics.features.ivh import IVHLattice, ivh_statistics
from mlxradiomics.features.selection import explicit_texture
from mlxradiomics.features.texture import texture_statistics
from mlxradiomics.imaging.preprocessing import (
    Prepared,
)
from mlxradiomics.numerics.runtime import (
    array_from_bool,
    evaluate,
    mx,
    trim_cache,
)
from mlxradiomics.workflows.roi_memory import check_roi_count_buffers


def _roi_window(data: Prepared, settings: Settings) -> tuple[slice, ...]:
    """Return the ROI bounding box padded by the largest neighbor reach, clipped to the image.

    Mask-based families see only voxels inside the sampling or shape mask, so a window
    containing both masks gives the same matrices as the whole grid. The pad keeps
    ``directions`` unchanged: an offset fits the window exactly when it fits the image,
    because each window axis is either the whole image axis or longer than the reach.
    It also leaves a non-ROI shell for GLDZM border distances.
    """
    occupied = data.sampling_mask | data.shape_roi
    reach = max(*settings.distances, settings.neighborhood_radius)
    return occupied_window(occupied, [reach] * occupied.ndim)


def occupied_window(
    occupied: NDArray[np.bool_],
    reach: list[int],
) -> tuple[slice, ...]:
    """Bound a nonempty mask, narrowing each later reduction to prior exact bounds."""
    shape = occupied.shape
    window: list[slice] = []
    for axis, size in enumerate(shape):
        present = np.flatnonzero(
            np.any(
                occupied,
                axis=tuple(i for i in range(occupied.ndim) if i != axis),
            )
        )
        low, stop = int(present[0]), int(present[-1]) + 1
        window.append(slice(max(0, low - reach[axis]), min(size, stop + reach[axis])))
        # Later axes retain their source coordinates. Narrow only axes already bounded,
        # without the halo, since it contains no additional occupied voxels.
        exact = [slice(None)] * occupied.ndim
        exact[axis] = slice(low, stop)
        occupied = occupied[tuple(exact)]
    return tuple(window)


def _roi_shapes(
    data: Prepared,
    settings: Settings,
    spacing: tuple[float, ...],
) -> dict[str, mx.array]:
    """Evaluate shape selections without constructing intensity-grid residents."""
    values: dict[str, mx.array] = {}
    if any(name.startswith("shape") for name in settings.features):
        from mlxradiomics.features.geometry.mesh import MeshGeometryContext
        from mlxradiomics.features.shape import shape_statistics

        if any(name.startswith("shape_") for name in settings.features):
            from mlxradiomics.features.morphology import INTENSITY_MORPHOLOGY, morphology_statistics

            if data.roi.ndim != 3:
                raise ValueError("shape requires a three-dimensional mask")
            shape_names = tuple(
                name.removeprefix("shape_")
                for name in settings.features
                if name.startswith("shape_")
            )
            geometric = tuple(name for name in shape_names if name not in INTENSITY_MORPHOLOGY)
            intensity_shape = tuple(name for name in shape_names if name in INTENSITY_MORPHOLOGY)
            mesh_context = (
                MeshGeometryContext(data.shape_roi, spacing, settings.scratch_mib)
                if geometric and "IntegratedIntensity" in intensity_shape
                else None
            )
            if geometric:
                values.update(
                    shape_statistics(
                        data.shape_roi,
                        spacing,
                        settings.scratch_mib,
                        geometric,
                        covariance_divisor=settings.covariance_divisor,
                        ellipsoid_area=settings.ellipsoid_area,
                        _mesh_context=mesh_context,
                    )
                )
            if intensity_shape:
                values.update(
                    morphology_statistics(
                        data.image,
                        data.roi,
                        data.shape_roi,
                        spacing,
                        settings.scratch_mib,
                        intensity_shape,
                        _mesh_context=mesh_context,
                    )
                )
        if any(name.startswith("shape2D_") for name in settings.features):
            roi, plane_spacing = data.shape_roi, spacing
            if roi.ndim == 3:
                axis = settings.force2d_axis
                if axis is None:
                    raise ValueError("shape2D requires 2D data or force2d_axis")
                occupied = np.flatnonzero(np.any(roi, axis=tuple(i for i in range(3) if i != axis)))
                if len(occupied) != 1:
                    raise ValueError("shape2D ROI must occupy exactly one slice")
                roi = np.take(roi, int(occupied[0]), axis=axis)
                plane_spacing = tuple(s for i, s in enumerate(spacing) if i != axis)
            shape2d_names = tuple(
                name.removeprefix("shape2D_")
                for name in settings.features
                if name.startswith("shape2D_")
            )
            values.update(
                shape_statistics(
                    roi,
                    plane_spacing,
                    settings.scratch_mib,
                    shape2d_names,
                    covariance_divisor=settings.covariance_divisor,
                    ellipsoid_area=settings.ellipsoid_area,
                )
            )
    return values


def _roi_local_intensity(
    data: Prepared,
    settings: Settings,
    spacing: tuple[float, ...],
) -> dict[str, mx.array]:
    """Evaluate the physical sphere directly from its source and center mask."""
    names = tuple(
        name.removeprefix("localintensity_")
        for name in settings.features
        if name.startswith("localintensity_")
    )
    if not names:
        return {}
    from mlxradiomics.features.local_intensity import local_intensity_statistics

    return local_intensity_statistics(
        data.image,
        data.roi,
        spacing,
        settings.scratch_mib,
        names,
    )


def _roi_intensity(
    data: Prepared,
    settings: Settings,
    dtype: mx.Dtype,
    volume: float,
    method: str,
    spacing: tuple[float, ...],
    ivh: IVHLattice | None,
    sampled: NDArray[np.float64],
    window: tuple[slice, ...] | None,
) -> dict[str, mx.array]:
    """Compute the selected features once over the whole ROI, keyed ``original_<name>``.

    First-order statistics reuse the validated compact samples in source traversal order;
    excluded positions never enter their reductions or percentile sort. Texture families retain
    the padded ROI window needed for spatial relationships. Local intensity, whose
    physical sphere reads non-ROI voxels, and host-side shape families retain the
    input window with the complete morphology support and physical sphere halo.
    """
    values = _roi_local_intensity(data, settings, spacing)
    first_names = tuple(
        name.removeprefix("firstorder_")
        for name in settings.features
        if name.startswith("firstorder_")
    )
    if first_names:
        first_mask = mx.ones((1, sampled.size), dtype=mx.bool_)
        first_bins = (
            mx.array(data.bins[data.sampling_mask][None, :])
            if data.bins is not None
            and any(name in {"Entropy", "Uniformity"} for name in first_names)
            else None
        )
        if set(first_names) <= {"Entropy", "Uniformity"}:
            assert first_bins is not None
            first = first_order_histogram(
                first_bins,
                first_mask,
                data.gray_levels,
                dtype=dtype,
                selected=first_names,
            )
        else:
            source = prepare_first_order_source(
                sampled,
                np.broadcast_to(np.bool_(True), sampled.shape),
                sampled=sampled,
                integer_bounds=data.integer_source_bounds,
            )
            inputs = first_order_inputs(first_names, source, shift=settings.voxel_array_shift)
            first = first_order(
                mx.array(source.high, dtype=dtype),
                first_mask,
                first_bins,
                data.gray_levels,
                shift=settings.voxel_array_shift,
                voxel_volume=volume,
                # Source parts retain ordering and residual-scale intensity differences.
                residual=mx.array(source.low, dtype=dtype) if inputs.residual else None,
                source_bits=(
                    mx.array(source.source_bits, dtype=mx.uint64) if inputs.source_bits else None
                ),
                representation_error=(
                    mx.array(source.representation_error, dtype=dtype)
                    if inputs.representation_error
                    else None
                ),
                selected=first_names,
                kurtosis=settings.kurtosis,
                zero_variance_moments=settings.zero_variance_moments,
                source=source,
            )
            del source
        if any(not name.startswith("firstorder_") for name in settings.features):
            # Complete scalar outputs before texture allocations so deferred graphs
            # do not retain the first-order source arrays through later memory peaks.
            evaluate(*first.values())
            trim_cache()
        values.update({f"firstorder_{k}": v for k, v in first.items()})
    if all(
        name.startswith(("firstorder_", "shape", "localintensity_")) for name in settings.features
    ):
        return values
    assert window is not None
    image = data.image[window]
    mask = array_from_bool(data.sampling_mask[window])
    if ivh is not None:
        values.update(
            {
                f"ivh_{name}": value
                for name, value in ivh_statistics(
                    mx.reshape(mx.array(ivh.bins[window]), (1, -1)),
                    mx.reshape(mask, (1, -1)),
                    ivh.levels,
                    ivh.origin,
                    ivh.step,
                    selected=tuple(
                        name.removeprefix("ivh_")
                        for name in settings.features
                        if name.startswith("ivh_")
                    ),
                ).items()
            }
        )
    if data.bins is None:
        return values
    bins = mx.array(data.bins[window])
    if any(name.startswith("intensityhistogram_") for name in settings.features):
        from mlxradiomics.features.intensity_histogram import histogram_statistics

        values.update(
            {
                f"intensityhistogram_{name}": value
                for name, value in histogram_statistics(
                    mx.reshape(bins, (1, -1)),
                    mx.reshape(mask, (1, -1)),
                    data.gray_levels,
                    selected=tuple(
                        name.removeprefix("intensityhistogram_")
                        for name in settings.features
                        if name.startswith("intensityhistogram_")
                    ),
                ).items()
            }
        )
    if explicit_texture(settings):
        values.update(
            aggregated_texture_statistics(
                mx.reshape(bins, (1, -1)),
                mx.reshape(mask, (1, -1)),
                image.shape,
                data.gray_levels,
                settings,
                morphology=mx.reshape(array_from_bool(data.shape_roi[window]), (1, -1)),
                histogram_method=method,
            )
        )
    elif any(name.startswith("glcm_") for name in settings.features):
        values.update(aggregate_roi_glcm(bins, mask, data.gray_levels, settings, dtype, method))
    if not explicit_texture(settings):
        values.update(
            texture_statistics(
                mx.reshape(bins, (1, -1)),
                mx.reshape(mask, (1, -1)),
                image.shape,
                data.gray_levels,
                settings,
            )
        )
    return values


def extract_roi(
    data: Prepared,
    settings: Settings,
    dtype: mx.Dtype,
    volume: float,
    method: str,
    spacing: tuple[float, ...],
    ivh: IVHLattice | None,
    sampled: NDArray[np.float64],
) -> dict[str, float]:
    """Complete selected ROI arithmetic and transfer only requested scalars."""
    spatial = any(
        not name.startswith(("firstorder_", "shape", "localintensity_"))
        for name in settings.features
    )
    window = _roi_window(data, settings) if spatial else None
    shape = data.image[window].shape if window is not None else data.image.shape
    check_roi_count_buffers(shape, data.gray_levels, settings)
    values = _roi_intensity(data, settings, dtype, volume, method, spacing, ivh, sampled, window)
    values.update(_roi_shapes(data, settings, spacing))
    selected = {name: values[name] for name in settings.features}
    evaluate(*selected.values())
    return {f"original_{k}": float(cast(float, v.item())) for k, v in selected.items()}
