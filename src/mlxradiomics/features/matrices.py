"""Integer texture matrices, separate from merging and feature evaluation.

Each batch row has its own spatial domain. GLDZM uses the morphological mask for
Manhattan boundary distance (boundary voxels have distance one), and the intensity
mask for 8/26-connected equal-level zones. Both calculations run on Metal.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import prod
from typing import cast

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.sparse_glcm import use_sparse_glcm
from mlxradiomics.features.sparse_size import use_sparse_size
from mlxradiomics.features.texture import (
    count_matrix,
    neighborhood_arrays,
    ngtdm_level_reductions,
    run_kernel,
    zone_labels,
)
from mlxradiomics.imaging.preprocessing import directions
from mlxradiomics.numerics.histogram import histogram
from mlxradiomics.numerics.protocols import IndexedAddition, IndexedMinimum
from mlxradiomics.numerics.runtime import evaluate, mx
from mlxradiomics.numerics.sparse_counts import coordinate_counts, merge_counts


@dataclass(frozen=True)
class TextureMatrix:
    """Counts and associated sample count; NGTDM also carries difference sums."""

    counts: mx.array
    voxels: mx.array
    differences: mx.array | None = None
    cell_keys: mx.array | None = None
    column_count: int | None = None


@dataclass(frozen=True)
class NeighborhoodPreparation:
    """Per-voxel neighborhood outputs and the dependence matrix's column count."""

    arrays: dict[str, mx.array]
    dependence_columns: int


def glcm_matrices(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    method: str,
) -> Iterator[TextureMatrix]:
    """Yield one optionally symmetric GLCM for each admitted direction."""
    spatial_bins = mx.reshape(bins, (bins.shape[0], *shape))
    spatial_mask = mx.reshape(mask, (bins.shape[0], *shape))
    voxels = mx.sum(mask.astype(mx.int64), axis=1)
    for delta in directions(shape, settings):
        left = (
            slice(None),
            *(slice(max(0, -v), min(s, s - v)) for s, v in zip(shape, delta, strict=True)),
        )
        right = (
            slice(None),
            *(slice(max(0, v), min(s, s + v)) for s, v in zip(shape, delta, strict=True)),
        )
        valid = mx.reshape(spatial_mask[left] & spatial_mask[right], (bins.shape[0], -1))
        if use_sparse_glcm(ng):
            keys, counts = coordinate_counts(
                mx.reshape(spatial_bins[left], valid.shape),
                mx.reshape(spatial_bins[right], valid.shape), valid, ng,
                symmetric=settings.symmetrical_glcm,
            )
            yield TextureMatrix(counts, voxels, cell_keys=keys, column_count=ng)
            continue
        counts = histogram(
            mx.reshape(spatial_bins[left], valid.shape),
            mx.reshape(spatial_bins[right], valid.shape),
            valid,
            ng,
            method=method,
        )
        if settings.symmetrical_glcm:
            counts = counts + mx.swapaxes(counts, 1, 2)
        yield TextureMatrix(counts, voxels)


def run_matrices(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
) -> Iterator[TextureMatrix]:
    """Yield unit-direction run matrices, including valid singleton-only directions."""
    offsets = directions(tuple(max(s, 2) for s in shape), replace(settings, distances=(1,)))
    voxels = mx.sum(mask.astype(mx.int64), axis=1)
    for offset in offsets:
        lengths, starts = run_kernel(shape, offset)(
            inputs=[bins, mask],
            grid=(bins.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[bins.shape] * 2,
            output_dtypes=[mx.int32, mx.bool_],
        )
        yield TextureMatrix(count_matrix(bins, lengths, starts, ng, max(shape)), voxels)


def prepare_neighborhood(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    settings: Settings,
    families: tuple[str, ...] = ("gldm", "ngtdm"),
) -> NeighborhoodPreparation:
    """Prepare the neighborhood arrays of ``families`` for one spatial domain on Metal."""
    if settings.neighborhood_support == "closed-radius":
        settings = replace(settings, distances=tuple(range(1, settings.neighborhood_radius + 1)))
    offsets = directions(shape, settings)
    offsets += tuple(tuple(-v for v in d) for d in offsets)
    arrays = neighborhood_arrays(
        bins, mask, shape, offsets, settings.gldm_alpha, families,
    )
    return NeighborhoodPreparation(arrays, len(offsets) + 1)


def neighborhood_matrix(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    family: str,
    *,
    prepared: NeighborhoodPreparation | None = None,
) -> TextureMatrix:
    """Build GLDM or NGTDM with explicit shell or IBSI closed-radius support.

    ``prepared`` may reuse outputs from the same bins, mask, shape and effective
    neighborhood settings; it must contain the requested family's outputs.
    """
    source = (
        prepare_neighborhood(bins, mask, shape, settings, (family,))
        if prepared is None else prepared
    )
    arrays = source.arrays
    if family == "gldm":
        return TextureMatrix(
            count_matrix(bins, arrays["dependences"], mask, ng, source.dependence_columns),
            mx.sum(mask.astype(mx.int64), axis=1),
        )
    occurrences, differences = ngtdm_level_reductions(
        bins, arrays["eligible"], arrays["difference"], ng,
    )
    return TextureMatrix(
        occurrences,
        mx.sum(occurrences, axis=1),
        differences,
    )


def boundary_distances(mask: mx.array, shape: tuple[int, ...], axis: int | None) -> mx.array:
    """Exact integer Manhattan distance to background and active image faces.

    ``mask`` contains flattened spatial domains, one per batch row. ``axis`` is
    excluded for force-2D distances; slices never exchange distances along it.
    For each active axis, prefix/suffix minima of ``d[i] - i`` / ``d[i] + i``
    give ``min_j(d[j] + abs(i-j))``. Native Metal scans avoid convergence checks
    and remain parallel on long, thin domains. Boundary voxels have distance one.
    """
    axes = tuple(i for i in range(len(shape)) if i != axis)
    # The nearest pair of active image faces bounds every final distance. A
    # size-one or size-two active axis makes every selected voxel a boundary.
    bound = (min(shape[i] for i in axes) + 1) // 2
    if bound == 1 or mask.size == 0:
        return mask.astype(mx.int32)
    distance = mx.reshape(
        mx.where(mask, bound, 0).astype(mx.int32), (mask.shape[0], *shape),
    )
    for dimension in axes:
        size = shape[dimension]
        position = mx.reshape(
            mx.arange(size, dtype=mx.int32),
            (1,) * (dimension + 1) + (size,) + (1,) * (len(shape) - dimension - 1),
        )
        left = mx.cummin(distance - position, axis=dimension + 1) + position
        right = mx.cummin(distance + position, axis=dimension + 1, reverse=True) - position
        distance = mx.minimum(mx.minimum(left, right), mx.minimum(position + 1, size - position))
    return mx.reshape(distance, mask.shape)


def prepare_zone_labels(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    settings: Settings,
) -> mx.array:
    """Prepare the equal-level partition independently of zone-size/distance counts."""
    return zone_labels(bins, mask, shape, directions(shape, replace(settings, distances=(1,))))


def zone_matrix(
    bins: mx.array,
    mask: mx.array,
    shape: tuple[int, ...],
    ng: int,
    settings: Settings,
    morphology: mx.array | None = None,
    *,
    fixed_columns: bool = False,
    labels: mx.array | None = None,
) -> TextureMatrix:
    """Build a GLSZM, or GLDZM when a morphological mask is supplied.

    Distance-zone percentage uses the intensity voxel count carried alongside the
    matrix. Distance columns cannot reconstruct voxel count from their indices.
    With ``fixed_columns`` the size/distance axis spans ``prod(shape)``, a bound on
    every zone size and boundary distance, so voxel results do not depend on batching.
    ``labels`` may reuse the partition for the same bins, intensity mask, spatial
    shape and connectivity. Morphological containment and distances are still checked
    independently for GLDZM.
    """
    width = prod(shape)
    ids = mx.broadcast_to(mx.arange(width, dtype=mx.int32)[None, :], bins.shape)
    if labels is None:
        labels = prepare_zone_labels(bins, mask, shape, settings)
    rows = mx.arange(bins.shape[0], dtype=mx.int32)[:, None]
    indices = mx.reshape(rows * width + labels, (-1,))
    if morphology is None:
        storage = mx.zeros((bins.size,), dtype=mx.int32)
        sizes = cast(IndexedAddition, storage.at[indices]).add(
            mx.reshape(mask.astype(mx.int32), (-1,))
        )
    else:
        if morphology is not mask and bool(mx.any(mask & ~morphology).item()):
            raise ValueError("GLDZM intensity mask must lie inside its morphological mask")
        distance = boundary_distances(morphology, shape, settings.force2d_axis)
        storage = mx.full((bins.size,), width + 1, dtype=mx.int32)
        sizes = cast(IndexedMinimum, storage.at[indices]).minimum(
            mx.reshape(mx.where(mask, distance, width + 1), (-1,))
        )
    sizes = mx.reshape(sizes, bins.shape)
    starts = mask & (ids == labels)
    # GLSZM size storage is nonzero only at masked roots. GLDZM also has sentinel
    # entries, so only that family needs a masked maximum. The shared Metal histogram
    # skips invalid starts before reading sizes; its invalid size entries need no fill.
    maximum_values = sizes if morphology is None else mx.where(starts, sizes, 0)
    maximum = width if fixed_columns else max(1, cast(int, mx.max(maximum_values).item()))
    if not fixed_columns and use_sparse_size(ng, maximum):
        keys, counts = coordinate_counts(bins, sizes, starts, maximum)
        return TextureMatrix(
            counts, mx.sum(mask.astype(mx.int64), axis=1),
            cell_keys=keys, column_count=maximum,
        )
    counts = count_matrix(bins, sizes, starts, ng, maximum)
    return TextureMatrix(counts, mx.sum(mask.astype(mx.int64), axis=1))


def merge_matrices(matrices: list[TextureMatrix]) -> TextureMatrix:
    """Merge integer counts and their denominators, padding absent size columns."""
    if not matrices:
        raise ValueError("cannot merge an empty matrix collection")
    if any(matrix.cell_keys is not None for matrix in matrices):
        # All matrices of a family share one storage policy for their lattice.
        # Dense size matrices can occur on slices with shorter observed zones;
        # encode those counts at their actual indices before combining them.
        columns = max(
            matrix.column_count if matrix.column_count is not None else matrix.counts.shape[2]
            for matrix in matrices
        )
        keys: list[mx.array] = []
        counts: list[mx.array] = []
        for matrix in matrices:
            if matrix.cell_keys is None:
                ng, ns = matrix.counts.shape[1:]
                encoded = mx.arange(ng, dtype=mx.int64)[:, None] * columns
                encoded = encoded + mx.arange(ns, dtype=mx.int64)[None, :]
                keys.append(mx.broadcast_to(
                    mx.reshape(encoded, (1, -1)), (matrix.counts.shape[0], ng * ns),
                ))
                counts.append(mx.reshape(matrix.counts, (matrix.counts.shape[0], -1)))
            else:
                assert matrix.column_count is not None
                keys.append(
                    (matrix.cell_keys // matrix.column_count) * columns
                    + matrix.cell_keys % matrix.column_count
                )
                counts.append(matrix.counts)
        merged_keys, merged_counts = merge_counts(keys, counts)
        voxels = matrices[0].voxels.astype(mx.int64)
        for matrix in matrices[1:]:
            voxels = voxels + matrix.voxels
        evaluate(merged_keys, merged_counts, voxels)
        return TextureMatrix(merged_counts, voxels, cell_keys=merged_keys, column_count=columns)
    shape = tuple(
        max(matrix.counts.shape[axis] for matrix in matrices)
        for axis in range(matrices[0].counts.ndim)
    )
    def padded_counts(matrix: TextureMatrix) -> mx.array:
        counts = matrix.counts.astype(mx.int64)
        if counts.shape == shape:
            return counts
        padding = [(0, end - start) for start, end in zip(counts.shape, shape, strict=True)]
        return mx.pad(counts, padding)

    # Integer counts need no zero initializer. Differences are added left to right in
    # float32, and each subsequent merge is evaluated to bound lazy storage.
    total = padded_counts(matrices[0])
    voxels = matrices[0].voxels.astype(mx.int64)
    differences = matrices[0].differences
    for matrix in matrices[1:]:
        total = total + padded_counts(matrix)
        voxels = voxels + matrix.voxels
        if matrix.differences is not None:
            differences = (
                matrix.differences if differences is None else differences + matrix.differences
            )
        evaluate(total, voxels, *((differences,) if differences is not None else ()))
    if len(matrices) == 1:
        evaluate(total, voxels, *((differences,) if differences is not None else ()))
    return TextureMatrix(total, voxels, differences)
