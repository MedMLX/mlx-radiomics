"""Voxel scheduling and evaluated feature batches."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.contracts.types import (
    FeatureValidity,
    FloatArray,
    VoxelMemory,
    VoxelSink,
)
from mlxradiomics.contracts.validity import accumulate_validity
from mlxradiomics.features.aggregation import aggregated_texture_statistics
from mlxradiomics.features.cooccurrence import aggregate_glcm, pair_slices
from mlxradiomics.features.first_order import first_order, first_order_histogram, first_order_inputs
from mlxradiomics.features.first_order_source import FirstOrderAdmission, prepare_first_order_source
from mlxradiomics.features.intensity_histogram import histogram_statistics
from mlxradiomics.features.ivh import IVHLattice, ivh_statistics
from mlxradiomics.features.selection import explicit_texture
from mlxradiomics.features.sparse_glcm import use_sparse_glcm
from mlxradiomics.features.texture import texture_statistics
from mlxradiomics.imaging.preprocessing import (
    Prepared,
    directions,
)
from mlxradiomics.numerics.runtime import (
    array_from_bool,
    evaluate,
    mx,
)
from mlxradiomics.voxel.gather import NeighborhoodIndex, NeighborhoodPlan
from mlxradiomics.voxel.glcm import eligible as fused_glcm_eligible
from mlxradiomics.voxel.glcm import local_glcm
from mlxradiomics.voxel.texture import eligible as fused_texture_eligible
from mlxradiomics.voxel.texture import local_texture


class _FirstOrderResidents(NamedTuple):
    """Optional source-precision buffers retained across voxel batches."""

    residual: mx.array | None
    source_bits: mx.array | None
    representation_error: mx.array | None


class _VoxelResidents(NamedTuple):
    """Flattened device inputs shared by all voxel batches."""

    image: mx.array | None
    sampling_mask: mx.array
    bins: mx.array


class _TextureOffsets(NamedTuple):
    """Run directions and zone/dependence neighborhoods for fused texture kernels."""

    runs: tuple[tuple[int, ...], ...]
    zones: tuple[tuple[int, ...], ...]
    dependence: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _VoxelPlan:
    """Feature paths, neighborhood topology and scratch-bounded delivery schedule."""

    centers: NDArray[np.int32]
    first_names: tuple[str, ...]
    glcm_names: tuple[str, ...]
    has_source_first: bool
    has_intensity_histogram: bool
    has_texture: bool
    explicit_texture: bool
    kernel_axes: tuple[range, ...]
    local_shape: tuple[int, ...]
    pair_offsets: tuple[tuple[int, ...], ...]
    texture_offsets: _TextureOffsets
    fused_glcm: bool
    fused_texture: bool
    cache_indexes: bool
    memory: VoxelMemory


def _local_pair_indexes(
    shape: tuple[int, ...],
    offsets: tuple[tuple[int, ...], ...],
) -> Iterator[tuple[mx.array, mx.array]]:
    """Yield flat voxel indexes ``(left, right)`` of all pairs inside one local neighborhood."""
    indexes = np.arange(prod(shape), dtype=np.int32).reshape(shape)
    for delta in offsets:
        left, right = pair_slices(shape, delta)
        yield mx.array(indexes[left].reshape(-1)), mx.array(indexes[right].reshape(-1))


def _local_pairs(
    bins: mx.array,
    mask: mx.array,
    indexes: Iterable[tuple[mx.array, mx.array]],
) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
    """Yield left levels, right levels and joint validity for each index pair."""
    for left, right in indexes:
        yield bins[:, left], bins[:, right], mask[:, left] & mask[:, right]


def _voxel_first(
    values: mx.array,
    bins: mx.array,
    valid: mx.array,
    data: Prepared,
    settings: Settings,
    volume: float,
    source_resident: _FirstOrderResidents,
    index: NeighborhoodIndex,
    admission: FirstOrderAdmission,
) -> dict[str, mx.array]:
    """Compute the requested first-order features for one voxel batch, keyed ``firstorder_*``."""
    first = first_order(
        values,
        valid,
        bins,
        data.gray_levels,
        shift=settings.voxel_array_shift,
        voxel_volume=volume,
        residual=(
            None if source_resident.residual is None else index.gather(source_resident.residual)
        ),
        source_bits=(
            None if source_resident.source_bits is None
            else index.gather(source_resident.source_bits)
        ),
        representation_error=(
            None if source_resident.representation_error is None
            else index.gather(source_resident.representation_error)
        ),
        selected=tuple(
            name.removeprefix("firstorder_")
            for name in settings.features
            if name.startswith("firstorder_")
        ),
        kurtosis=settings.kurtosis,
        zero_variance_moments=settings.zero_variance_moments,
        source=admission,
    )
    return {f"firstorder_{k}": v for k, v in first.items()}


def _voxel_schedule(
    *,
    rows: int,
    per_center: int,
    first_per_center: int,
    first_features: int,
    budget: int,
    retained_bytes: int,
) -> VoxelMemory:
    """Group first-order work within the scratch budget.

    First-order groups span whole delivery batches, so a sink still receives every
    requested feature for each center exactly once. Their working set is capped at
    8 MiB (including retained topology). Completed float32 feature vectors are cached
    until delivery; the texture batch budget reserves those vectors explicitly.
    Small selections and budgets that cannot benefit keep the shared schedule.
    """
    batch = max(1, (budget - retained_bytes) // per_center)
    memory: VoxelMemory = {
        "batch_size": batch,
        "scratch_estimate_bytes": batch * per_center + retained_bytes,
    }
    if not first_features or rows <= batch:
        return memory
    first_budget = min(8 * 1024**2, budget) - retained_bytes
    # Reserve vectors from the existing batch's spare budget. Shrinking an expensive
    # texture batch to make room can add more texture work than the grouping saves.
    cache_groups = (budget - retained_bytes - batch * per_center) // (batch * first_features * 4)
    groups = min(
        first_budget // first_per_center // batch,
        (rows + batch - 1) // batch,
        cache_groups,
    )
    if groups < 2:
        return memory
    # A group caches one float32 value per first-order feature and center.
    first_batch = groups * batch
    cache_bytes = first_batch * first_features * 4
    first_scratch = first_batch * first_per_center + retained_bytes
    return {
        "batch_size": batch,
        "scratch_estimate_bytes": max(
            batch * per_center + retained_bytes + cache_bytes,
            first_scratch,
        ),
        "first_order_batch_size": first_batch,
        "first_order_scratch_estimate_bytes": first_scratch,
        "first_order_cache_bytes": cache_bytes,
    }


def _voxel_first_group(
    resident: _VoxelResidents,
    source_resident: _FirstOrderResidents | None,
    centers: NDArray[np.int32],
    offsets: mx.array,
    data: Prepared,
    settings: Settings,
    volume: float,
    plan: NeighborhoodPlan,
    admission: FirstOrderAdmission | None,
) -> dict[str, NDArray[np.float32]]:
    """Evaluate one bounded first-order group; retain only its result vectors."""
    index = plan.index(mx.array(centers, dtype=mx.int32), offsets)
    bins, valid = index.gather(resident.bins), index.valid(resident.sampling_mask)
    if source_resident is None:
        first = first_order_histogram(
            bins,
            valid,
            data.gray_levels,
            dtype=mx.float32,
            selected=tuple(
                name.removeprefix("firstorder_")
                for name in settings.features
                if name.startswith("firstorder_")
            ),
        )
        selected = {f"firstorder_{name}": value for name, value in first.items()}
    else:
        assert resident.image is not None and admission is not None
        selected = _voxel_first(
            index.gather(resident.image),
            bins,
            valid,
            data,
            settings,
            volume,
            source_resident,
            index,
            admission,
        )
    evaluate(*selected.values())
    return {name: np.asarray(value, dtype=np.float32) for name, value in selected.items()}


def _sparse_glcm_bytes(
    ng: int,
    names: tuple[str, ...],
    kernel_size: int,
    pairs: int,
) -> int:
    """Bound occupied-cell work and the selected features' full-lattice arrays.

    Coordinate sorting, exact counts and cell-wise feature expressions scale with
    observed pairs. Only marginal consumers retain arrays spanning ``Ng``; lazy
    unselected MI/entropy/ClusterShade expressions are never evaluated. ClusterShade
    reserves its sum distribution and integer-limb intermediates separately. MCC
    instead needs an occupied-state eigensystem, bounded by local voxel population.
    """
    selected = set(names)
    per_level = 0
    if selected & {"Imc1", "Imc2"}:
        per_level += 256  # Row/column counts, entropy terms and reduction workspace.
    if selected & {"Correlation"}:
        per_level += 32  # Integer row/column populations classify singleton marginals.
    if selected & {"DifferenceEntropy", "DifferenceVariance"}:
        per_level += 64
    if "SumEntropy" in selected:
        per_level += 128  # The sum axis has 2*Ng-1 entries.
    if "ClusterShade" in selected:
        per_level += 1024  # Sum-axis limbs, products and carries before reduction.
    cells = max(pairs, 1)
    working = cells * 256 + ng * per_level
    if "MCC" in selected:
        working += min(kernel_size, ng) ** 2 * 128
    return working


def _plan_voxels(data: Prepared, settings: Settings) -> _VoxelPlan:
    """Choose feature paths and batch sizes before allocating device residents."""
    shape = data.image.shape
    first_names = tuple(
        name.removeprefix("firstorder_")
        for name in settings.features
        if name.startswith("firstorder_")
    )
    first_features = len(first_names)
    has_source_first = any(name not in {"Entropy", "Uniformity"} for name in first_names)
    has_intensity_histogram = any(
        name.startswith("intensityhistogram_") for name in settings.features
    )
    has_texture = any(
        n.startswith(("gldm_", "ngtdm_", "glrlm_", "glszm_", "gldzm_")) for n in settings.features
    )
    glcm_names = tuple(
        name.removeprefix("glcm_") for name in settings.features if name.startswith("glcm_")
    )
    has_glcm = bool(glcm_names)
    explicit = explicit_texture(settings)
    r = settings.kernel_radius
    axes = tuple(
        range(1) if settings.force2d_axis == i else range(-r, r + 1) for i in range(len(shape))
    )
    kernel_size = prod(map(len, axes))
    local_shape = tuple(map(len, axes))
    all_pair_offsets = directions(local_shape, settings) if has_glcm else ()
    pair_offsets = all_pair_offsets if not explicit else ()
    pair_counts = tuple(
        prod(s - abs(v) for s, v in zip(local_shape, delta, strict=True)) for delta in pair_offsets
    )
    # Native GLCM maps of small neighborhoods use one fused pass per center and
    # direction, without dense Ng x Ng matrices (see local_glcm).
    fused_glcm = bool(pair_offsets) and fused_glcm_eligible(
        kernel_size,
        pair_counts,
        data.gray_levels,
        settings.symmetrical_glcm,
    )
    # Conservative working-set estimate, excluding resident input and final output arrays.
    if has_glcm and not fused_glcm:
        if use_sparse_glcm(data.gray_levels):
            # Explicit aggregation can retain and merge directions/slices. Their
            # combined pair population bounds sparse storage even when their gray
            # ranges contain large gaps. Native averaging releases each direction.
            populations = tuple(
                prod(s - abs(v) for s, v in zip(local_shape, delta, strict=True))
                for delta in all_pair_offsets
            )
            population = sum(populations) if explicit else max(populations, default=0)
            population *= 2 if settings.symmetrical_glcm else 1
            histogram_bytes = _sparse_glcm_bytes(
                data.gray_levels, glcm_names, kernel_size, population,
            )
        else:
            histogram_bytes = data.gray_levels**2 * 128
    elif has_glcm:
        histogram_bytes = max(data.gray_levels * 16, len(pair_offsets) * 4 * 32)
    else:
        histogram_bytes = data.gray_levels * 16
    if (
        first_names or has_intensity_histogram
        or any(n.startswith("ivh_") for n in settings.features)
    ):
        histogram_bytes = max(histogram_bytes, data.gray_levels * 16)
    # Native GLRLM/GLSZM/GLDM/NGTDM maps likewise evaluate occupied cells per center.
    fused_texture = (
        has_texture
        and not explicit
        and fused_texture_eligible(kernel_size, data.gray_levels)
    )
    texture_offsets = _TextureOffsets(runs=(), zones=(), dependence=())
    if fused_texture:
        unit = replace(settings, distances=(1,))
        zones = directions(local_shape, unit)
        dependence = directions(local_shape, settings)
        texture_offsets = _TextureOffsets(
            runs=directions(tuple(max(s, 2) for s in local_shape), unit),
            zones=zones + tuple(tuple(-v for v in d) for d in zones),
            dependence=dependence + tuple(tuple(-v for v in d) for d in dependence),
        )
        histogram_bytes = max(histogram_bytes, 17 * 4 * (len(texture_offsets.runs) + 2))
    elif has_texture:
        histogram_bytes = max(histogram_bytes, data.gray_levels * kernel_size * 64)
    per_center = kernel_size * (64 + 16 * len(shape)) + histogram_bytes + 4096
    budget = settings.scratch_mib * 1024**2
    if per_center > budget:
        raise ValueError("scratch_mib cannot hold one local histogram; reduce bins/kernel radius")
    index_bytes = 0 if fused_glcm else 8 * sum(pair_counts)
    # Reserve retained topology in the scratch budget; large neighborhoods stream one
    # direction's indexes at a time, already covered by the per-center working set.
    cache_indexes = index_bytes <= budget - per_center
    retained_bytes = index_bytes if cache_indexes else 0
    centers = np.argwhere(data.roi).astype(np.int32)
    memory = _voxel_schedule(
        rows=len(centers),
        per_center=per_center,
        first_per_center=kernel_size * (64 + 16 * len(shape)) + data.gray_levels * 16 + 4096,
        first_features=first_features if has_glcm else 0,
        budget=budget,
        retained_bytes=retained_bytes,
    )
    return _VoxelPlan(
        centers=centers,
        first_names=first_names,
        glcm_names=glcm_names,
        has_source_first=has_source_first,
        has_intensity_histogram=has_intensity_histogram,
        has_texture=has_texture,
        explicit_texture=explicit,
        kernel_axes=axes,
        local_shape=local_shape,
        pair_offsets=pair_offsets,
        texture_offsets=texture_offsets,
        fused_glcm=fused_glcm,
        fused_texture=fused_texture,
        cache_indexes=cache_indexes,
        memory=memory,
    )


def extract_voxels(
    data: Prepared,
    settings: Settings,
    dtype: mx.Dtype,
    volume: float,
    method: str,
    sink: VoxelSink | None,
    validity: dict[str, FeatureValidity],
    ivh: IVHLattice | None,
) -> tuple[dict[str, FloatArray], VoxelMemory]:
    """Compute feature maps by batches of ROI voxels, each the center of a local kernel.

    Return float64 maps (0 outside the ROI), or send evaluated float32 batches to ``sink``
    and return no maps. Both routes use the same feature arithmetic and batch schedule.
    Mixed first-order/GLCM selections may compute first-order features in larger
    groups while delivering complete feature batches. Their cached result vectors
    are included in the scratch estimate chosen from ``settings.scratch_mib``.

    Raises:
        ValueError: If one center's working set exceeds the scratch budget.
    """
    shape = data.image.shape
    schedule = _plan_voxels(data, settings)
    centers = schedule.centers
    memory = schedule.memory
    batch_size = memory["batch_size"]
    first_batch_size = memory.get("first_order_batch_size", 0)
    # Every selected map reads only these centers plus the kernel's exact reach.
    # Validation/binning have already seen the complete source. Keep their global
    # lattice and the original output coordinates while bounding resident storage.
    reach = np.array(
        [0 if settings.force2d_axis == i else settings.kernel_radius for i in range(len(shape))],
        dtype=np.int64,
    )
    lower = np.maximum(centers.min(axis=0).astype(np.int64) - reach, 0)
    upper = np.minimum(centers.max(axis=0).astype(np.int64) + reach + 1, shape)
    window = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    resident_centers = centers - lower.astype(np.int32) if np.any(lower) else centers
    data = replace(
        data,
        image=data.image[window],
        roi=data.roi[window],
        bins=data.bins[window],
        sampling_mask=data.sampling_mask[window],
        shape_roi=data.shape_roi[window],
    )
    resident_shape = data.image.shape
    local = np.array(list(product(*schedule.kernel_axes)), dtype=np.int32)
    offsets = mx.array(local, dtype=mx.int32)
    neighborhood = NeighborhoodPlan(resident_shape)
    pair_indexes = (
        tuple(_local_pair_indexes(schedule.local_shape, schedule.pair_offsets))
        if schedule.cache_indexes and not schedule.fused_glcm
        else None
    )
    resident = _VoxelResidents(
        mx.reshape(mx.array(data.image, dtype=dtype), (-1,)) if schedule.has_source_first else None,
        mx.reshape(array_from_bool(data.sampling_mask), (-1,)),
        mx.reshape(mx.array(data.bins), (-1,)),
    )
    # Source-precision buffers serve only selected first-order operations and guards.
    # Proven zero words stay implicit; unused bits/errors never cross the device boundary.
    source_resident: _FirstOrderResidents | None = None
    source_admission: FirstOrderAdmission | None = None
    ivh_resident = mx.reshape(mx.array(ivh.bins[window]), (-1,)) if ivh is not None else None
    morphological_resident = (
        mx.reshape(array_from_bool(data.shape_roi), (-1,))
        if any(name.startswith("gldzm_") for name in settings.features)
        else None
    )
    if schedule.has_source_first:
        samples = data.image[data.sampling_mask]
        source = prepare_first_order_source(
            samples.reshape(1, -1),
            np.ones((1, samples.size), dtype=np.bool_),
            sampled=samples,
            integer_bounds=data.integer_source_bounds,
        )
        source_admission = source.neighborhood_admission()
        inputs = first_order_inputs(
            schedule.first_names, source_admission, shift=settings.voxel_array_shift,
        )
        low_resident = error_resident = None
        if inputs.residual:
            low = np.zeros(resident_shape, dtype=np.float32)
            low[data.sampling_mask] = source.low.reshape(-1)
            low_resident = mx.reshape(mx.array(low, dtype=dtype), (-1,))
            del low
        if inputs.representation_error:
            error = np.zeros(resident_shape, dtype=np.float32)
            error[data.sampling_mask] = source.representation_error.reshape(-1)
            error_resident = mx.reshape(mx.array(error, dtype=dtype), (-1,))
            del error
        source_resident = _FirstOrderResidents(
            low_resident,
            (
                mx.reshape(mx.array(data.image.view(np.uint64)), (-1,))
                if inputs.source_bits
                else None
            ),
            error_resident,
        )
        del source, samples
        evaluate(*(array for array in source_resident if array is not None))
    evaluate(
        *(array for array in resident if array is not None),
        *(index for pair in (pair_indexes or ()) for index in pair),
    )
    outputs = (
        {f"original_{name}": np.zeros(shape, dtype=np.float64) for name in settings.features}
        if sink is None
        else {}
    )
    first_cache: dict[str, NDArray[np.float32]] = {}
    for start in range(0, len(centers), batch_size):
        if first_batch_size and start % first_batch_size == 0:
            # Release the previous group before allocating the next group's scratch.
            first_cache.clear()
            first_cache = _voxel_first_group(
                resident,
                source_resident,
                resident_centers[start : start + first_batch_size],
                offsets,
                data,
                settings,
                volume,
                neighborhood,
                source_admission,
            )
        chunk = centers[start : start + batch_size]
        chunk_device = mx.array(resident_centers[start : start + batch_size], dtype=mx.int32)
        needs_gather = (
            (bool(schedule.first_names) and not first_batch_size)
            or schedule.has_intensity_histogram
            or ivh is not None
            or (schedule.explicit_texture and (bool(schedule.glcm_names) or schedule.has_texture))
            or (bool(schedule.glcm_names) and not schedule.fused_glcm)
            or (schedule.has_texture and not schedule.fused_texture)
        )
        index = neighborhood.index(chunk_device, offsets) if needs_gather else None
        bins = index.gather(resident.bins) if index is not None else None
        valid = index.valid(resident.sampling_mask) if index is not None else None
        batch: dict[str, mx.array] = {}
        if ivh is not None and ivh_resident is not None:
            assert index is not None and valid is not None
            ivh_bins = index.gather(ivh_resident)
            batch.update(
                {
                    f"ivh_{name}": value
                    for name, value in ivh_statistics(
                        ivh_bins,
                        valid,
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
        if bool(schedule.first_names) and not first_batch_size:
            assert index is not None and bins is not None and valid is not None
            if source_resident is None:
                first = first_order_histogram(
                    bins,
                    valid,
                    data.gray_levels,
                    dtype=dtype,
                    selected=schedule.first_names,
                )
                batch.update({f"firstorder_{name}": value for name, value in first.items()})
            else:
                assert resident.image is not None and source_admission is not None
                batch.update(
                    _voxel_first(
                        index.gather(resident.image),
                        bins,
                        valid,
                        data,
                        settings,
                        volume,
                        source_resident,
                        index,
                        source_admission,
                    )
                )
        if schedule.has_intensity_histogram:
            assert bins is not None and valid is not None
            batch.update(
                {
                    f"intensityhistogram_{name}": value
                    for name, value in histogram_statistics(
                        bins,
                        valid,
                        data.gray_levels,
                        selected=tuple(
                            name.removeprefix("intensityhistogram_")
                            for name in settings.features
                            if name.startswith("intensityhistogram_")
                        ),
                    ).items()
                }
            )
        if schedule.explicit_texture and (bool(schedule.glcm_names) or schedule.has_texture):
            assert index is not None and bins is not None and valid is not None
            morphology = valid
            if morphological_resident is not None and settings.masked_kernel:
                morphology = index.valid(morphological_resident)
            batch.update(
                aggregated_texture_statistics(
                    bins,
                    valid,
                    schedule.local_shape,
                    data.gray_levels,
                    settings,
                    morphology=morphology,
                    histogram_method=method,
                    fixed_columns=True,
                )
            )
        elif schedule.fused_glcm:
            batch.update(
                local_glcm(
                    resident.bins,
                    resident.sampling_mask,
                    resident_shape,
                    chunk_device,
                    schedule.local_shape,
                    schedule.pair_offsets,
                    data.gray_levels,
                    settings.symmetrical_glcm,
                    schedule.glcm_names,
                )
            )
        elif bool(schedule.glcm_names):
            assert bins is not None and valid is not None
            batch.update(
                aggregate_glcm(
                    _local_pairs(
                        bins,
                        valid,
                        pair_indexes
                        if pair_indexes is not None
                        else _local_pair_indexes(schedule.local_shape, schedule.pair_offsets),
                    ),
                    data.gray_levels,
                    settings,
                    dtype,
                    len(chunk),
                    method,
                )
            )
        if schedule.fused_texture:
            batch.update(
                local_texture(
                    resident.bins,
                    resident.sampling_mask,
                    resident_shape,
                    chunk_device,
                    schedule.local_shape,
                    schedule.texture_offsets.runs,
                    schedule.texture_offsets.zones,
                    schedule.texture_offsets.dependence,
                    settings.gldm_alpha,
                    settings.features,
                )
            )
        elif schedule.has_texture and not schedule.explicit_texture:
            assert bins is not None and valid is not None
            batch.update(
                texture_statistics(
                    bins,
                    valid,
                    schedule.local_shape,
                    data.gray_levels,
                    settings,
                    fixed_columns=True,
                )
            )
        selected = {name: batch[name] for name in settings.features if name not in first_cache}
        evaluate(*selected.values())
        first_start = start % first_batch_size if first_batch_size else 0
        host = {
            f"original_{name}": first_cache[name][first_start : first_start + len(chunk)]
            if name in first_cache
            else np.asarray(selected[name], dtype=np.float32)
            for name in settings.features
        }
        for name, values in host.items():
            accumulate_validity(validity[name], values)
        if sink is None:
            for name, values in host.items():
                outputs[name][tuple(chunk.T)] = values
        else:
            # File export already stores float32. Spooling the evaluated values directly
            # avoids retaining a float64 full-volume array for every selected feature.
            sink(chunk, host)
        if first_batch_size:
            del bins, valid, index, selected, batch, host
    return outputs, memory
