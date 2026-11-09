"""IBSI texture aggregation: build, merge, evaluate, then average where specified.

Directional ``2d_average`` weights every nonempty slice/direction matrix equally;
a slice with more valid directions therefore contributes more matrices. Families
with one matrix per slice weight nonempty slices equally. Merging combines integer
matrices before nonlinear feature evaluation and retains percentage denominators.
"""

from collections.abc import Iterator
from dataclasses import replace
from math import prod

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.distributions import ngtdm_statistics, size_statistics
from mlxradiomics.features.glcm import glcm_statistics
from mlxradiomics.features.matrices import (
    NeighborhoodPreparation,
    TextureMatrix,
    glcm_matrices,
    merge_matrices,
    neighborhood_matrix,
    prepare_neighborhood,
    prepare_zone_labels,
    run_matrices,
    zone_matrix,
)
from mlxradiomics.features.sparse_glcm import sparse_glcm_statistics
from mlxradiomics.features.sparse_size import sparse_size_statistics
from mlxradiomics.numerics.reduction import CompensatedTotal
from mlxradiomics.numerics.runtime import evaluate, mx


def resolved_aggregations(settings: Settings, ndim: int) -> dict[str, str]:
    """Resolve ``native`` aggregation per family and dimensionality into explicit modes."""
    choices = {
        "glcm": settings.glcm_aggregation,
        "glrlm": settings.glrlm_aggregation,
        "glszm": settings.glszm_aggregation,
        "gldzm": settings.gldzm_aggregation,
        "ngtdm": settings.ngtdm_aggregation,
        "gldm": settings.gldm_aggregation,
    }
    result: dict[str, str] = {}
    for family, mode in choices.items():
        if not any(name.startswith(f"{family}_") for name in settings.features):
            continue
        if mode == "native":
            if family in {"glcm", "glrlm"}:
                mode = (
                    "2d_average"
                    if ndim == 2
                    else "2.5d_direction_merge"
                    if settings.force2d_axis is not None
                    else "3d_average"
                )
            else:
                mode = (
                    "2d_average"
                    if ndim == 2
                    else "2.5d_merge"
                    if settings.force2d_axis is not None
                    else "3d"
                )
        result[family] = mode
    return result


def _domains(
    bins: mx.array,
    mask: mx.array,
    morphology: mx.array,
    shape: tuple[int, ...],
    settings: Settings,
    mode: str,
) -> Iterator[tuple[mx.array, mx.array, mx.array, tuple[int, ...], Settings]]:
    """Yield the spatial domains a texture family is computed over.

    3-D modes and 2-D inputs yield the whole array once. Slice-wise modes yield one
    flattened plane per index along the slice axis, with ``force2d_axis`` cleared
    so each plane is treated as a native 2-D image.
    """
    if len(shape) == 2 or mode.startswith("3d"):
        yield bins, mask, morphology, shape, settings
        return
    axis = settings.slice_axis if settings.force2d_axis is None else settings.force2d_axis
    # Identical morphology and sampling masks share the same slice view. Families
    # that do not consume morphology also use this pair-only domain preparation.
    same_mask = morphology is mask
    arrays = (bins, mask) if same_mask else (bins, mask, morphology)
    spatial = tuple(mx.reshape(array, (bins.shape[0], *shape)) for array in arrays)
    plane_shape = tuple(size for i, size in enumerate(shape) if i != axis)
    planar = replace(settings, force2d_axis=None)
    for index in range(shape[axis]):
        plane = tuple(
            mx.reshape(mx.take(array, index, axis=axis + 1), (bins.shape[0], -1))
            for array in spatial
        )
        yield plane[0], plane[1], plane[1] if same_mask else plane[2], plane_shape, planar


def _matrices(
    family: str,
    bins: mx.array,
    mask: mx.array,
    morphology: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    method: str,
    fixed_columns: bool,
    neighborhood: NeighborhoodPreparation | None = None,
    labels: mx.array | None = None,
) -> Iterator[TextureMatrix]:
    """Yield the texture matrices of ``family`` for one spatial domain."""
    if family == "glcm":
        yield from glcm_matrices(bins, mask, shape, ng, settings, method)
    elif family == "glrlm":
        yield from run_matrices(bins, mask, shape, ng, settings)
    elif family in {"gldm", "ngtdm"}:
        yield neighborhood_matrix(
            bins, mask, shape, ng, settings, family,
            prepared=neighborhood,
        )
    else:
        yield zone_matrix(
            bins, mask, shape, ng, settings, morphology if family == "gldzm" else None,
            fixed_columns=fixed_columns,
            labels=labels,
        )


def _groups(
    family: str,
    bins: mx.array,
    mask: mx.array,
    morphology: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    mode: str,
    method: str,
    fixed_columns: bool,
    neighborhood: NeighborhoodPreparation | None = None,
    labels: mx.array | None = None,
) -> Iterator[TextureMatrix]:
    """Yield the matrices of ``family`` grouped as ``mode`` aggregates them.

    Merging modes sum matrices across directions and/or slices before features are
    computed; averaging modes yield each direction or slice separately.
    """
    merged: TextureMatrix | None = None
    by_direction: list[TextureMatrix] = []
    for b, m, morph, domain_shape, effective in _domains(
        bins,
        mask,
        morphology if family == "gldzm" else mask,
        shape,
        settings,
        mode,
    ):
        slice_merged: TextureMatrix | None = None
        for direction, matrix in enumerate(
            _matrices(
                family,
                b,
                m,
                morph,
                domain_shape,
                ng,
                effective,
                method,
                fixed_columns,
                neighborhood,
                labels,
            )
        ):
            if mode in {"2d_average", "3d_average", "3d"}:
                yield matrix
            elif mode == "2d_slice_merge":
                slice_merged = (
                    matrix if slice_merged is None else merge_matrices([slice_merged, matrix])
                )
            elif mode == "2.5d_direction_merge":
                if direction == len(by_direction):
                    by_direction.append(matrix)
                else:
                    by_direction[direction] = merge_matrices([by_direction[direction], matrix])
            else:
                merged = matrix if merged is None else merge_matrices([merged, matrix])
        if slice_merged is not None:
            yield slice_merged
    if merged is not None:
        yield merged
    yield from by_direction


def _glcm_count_check_needed(shape: tuple[int, ...], settings: Settings, mode: str) -> bool:
    """Whether geometry alone fails to prove the merged GLCM population below 2**31.

    Each direction contributes at most one pair per voxel, doubled by symmetry. Averaged
    matrices and direction-wise slice merges contain at most the full image population.
    Other modes can merge directions: the complete radius cube, including unused shells
    and out-of-grid offsets, conservatively bounds their number without reading counts.
    Requests outside this proof check each matrix's count on the device.
    """
    directions_bound = 1
    if mode not in {"2d_average", "3d_average", "2.5d_direction_merge"}:
        radius = max(settings.distances)
        directions_bound = ((2 * radius + 1) ** len(shape) - 1) // 2
    symmetry = 2 if settings.symmetrical_glcm else 1
    return prod(shape) * directions_bound * symmetry >= 2**31


def aggregated_texture_statistics(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    *,
    morphology: mx.array | None = None,
    histogram_method: str = "mlx",
    fixed_columns: bool = False,
) -> dict[str, mx.array]:
    """Evaluate selected texture families under their effective aggregation settings.

    GLCM totals must remain below 2**31 even after merging. Thus integer contingency
    products remain below 2**62 in the dependence formulas. Large merged requests fail
    explicitly rather than overflow the proof used by the native information routines.
    ``fixed_columns`` gives zone matrices a batch-independent size axis (voxel mode).
    """
    output: dict[str, mx.array] = {}
    morphological = mask if morphology is None else morphology
    aggregations = resolved_aggregations(settings, len(shape))

    def shared_domain(first: str, second: str) -> bool:
        mode = aggregations.get(first)
        return (
            mode is not None and mode == aggregations.get(second)
            and (len(shape) == 2 or mode.startswith("3d"))
        )

    share_neighborhood = shared_domain("gldm", "ngtdm")
    share_zones = shared_domain("glszm", "gldzm")
    shared_neighborhood: NeighborhoodPreparation | None = None
    shared_labels: mx.array | None = None
    neighborhood_users = zone_users = 0
    for family, mode in aggregations.items():
        neighborhood = None
        labels = None
        # A single-domain preparation is consumed by both matching families and
        # then released. Multi-slice aggregations keep their independent walks;
        # retaining every slice's arrays would grow storage with the slice count.
        if share_neighborhood and family in {"gldm", "ngtdm"}:
            if shared_neighborhood is None:
                shared_neighborhood = prepare_neighborhood(bins, mask, shape, settings)
            neighborhood = shared_neighborhood
        if share_zones and family in {"glszm", "gldzm"}:
            if shared_labels is None:
                shared_labels = prepare_zone_labels(bins, mask, shape, settings)
            labels = shared_labels
        names = tuple(
            name.removeprefix(f"{family}_")
            for name in settings.features
            if name.startswith(f"{family}_")
        )
        # Signed features (ClusterShade, Correlation) cancel across matrices; a plain
        # running sum would make the average depend on slice and direction order.
        totals = {name: CompensatedTotal(bins.shape[0]) for name in names}
        ordinary = tuple(name for name in names if name != "MCC")
        denominators = (
            {"ordinary": mx.zeros((bins.shape[0],), dtype=mx.int32)} if ordinary else {}
        )
        if "MCC" in names:
            denominators["MCC"] = mx.zeros((bins.shape[0],), dtype=mx.int32)
        check_glcm_count = family == "glcm" and _glcm_count_check_needed(shape, settings, mode)
        for matrix in _groups(
            family, bins, mask, morphological, shape, ng, settings, mode, histogram_method,
            fixed_columns, neighborhood, labels,
        ):
            if family == "glcm":
                if check_glcm_count:
                    count = mx.sum(
                        matrix.counts.astype(mx.int64), axis=tuple(range(1, matrix.counts.ndim)),
                    )
                    if bool(mx.any(count >= 2**31).item()):
                        raise ValueError("merged GLCM count must be below 2**31")
                if matrix.cell_keys is None:
                    features, active = glcm_statistics(matrix.counts, mx.float32, names)
                else:
                    features, active = sparse_glcm_statistics(
                        matrix.cell_keys, matrix.counts, ng, names, settings.scratch_mib,
                    )
            elif family == "ngtdm":
                assert matrix.differences is not None
                features = ngtdm_statistics(
                    matrix.counts, matrix.differences, names, total_count=matrix.voxels,
                )
                active = matrix.voxels > 0
                if settings.ngtdm_singular_cases == "ibsi-caps":
                    # IBSI QCDE/NQ30/1X9X specify these finite degenerate conventions.
                    if "Coarseness" in names:
                        features["Coarseness"] = mx.minimum(features["Coarseness"], 1e6)
                    if "Busyness" in names:
                        present = mx.sum((matrix.counts > 0).astype(mx.int32), axis=1)
                        features["Busyness"] = mx.where(present == 1, 0, features["Busyness"])
                    if "Strength" in names:
                        features["Strength"] = mx.where(
                            mx.sum(matrix.differences, axis=1) == 0, 0, features["Strength"],
                        )
            else:
                count = mx.sum(
                    matrix.counts.astype(mx.int64), axis=tuple(range(1, matrix.counts.ndim)),
                )
                if matrix.cell_keys is None:
                    features = size_statistics(
                        matrix.counts, family, names, voxel_count=matrix.voxels, total_count=count,
                    )
                else:
                    assert matrix.column_count is not None
                    features = sparse_size_statistics(
                        matrix.cell_keys, matrix.counts, matrix.column_count, family, names,
                        voxel_count=matrix.voxels, total_count=count,
                    )
                active = count > 0
            if ordinary:
                denominators["ordinary"] = denominators["ordinary"] + active.astype(mx.int32)
            mcc_defined = active
            if "MCC" in names:
                mcc_defined = active & ~mx.isnan(features["MCC"])
                denominators["MCC"] = denominators["MCC"] + mcc_defined.astype(mx.int32)
            for name in names:
                # Ordinary NaN values still count an active matrix. Only
                # MCC excludes matrices whose second eigenvalue is undefined.
                defined = mcc_defined if name == "MCC" else active
                totals[name].add(mx.where(defined, features[name], 0))
            evaluate(*(a for t in totals.values() for a in t.arrays()), *denominators.values())
        for name, total in totals.items():
            denominator = denominators["MCC" if name == "MCC" else "ordinary"]
            output[f"{family}_{name}"] = mx.where(
                denominator > 0, total.value() / denominator, float("nan"),
            )
        if neighborhood is not None:
            neighborhood_users += 1
            if neighborhood_users == 2:
                shared_neighborhood = None
        if labels is not None:
            zone_users += 1
            if zone_users == 2:
                shared_labels = None
    return output
