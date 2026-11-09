"""Co-occurrence counting and directional aggregation."""

from collections.abc import Iterable, Iterator

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.glcm import glcm_statistics
from mlxradiomics.features.sparse_glcm import sparse_glcm_statistics, use_sparse_glcm
from mlxradiomics.imaging.preprocessing import (
    directions,
)
from mlxradiomics.numerics.eigen import maximal_correlation
from mlxradiomics.numerics.histogram import histogram, roi_histograms
from mlxradiomics.numerics.reduction import CompensatedTotal, directional_mean
from mlxradiomics.numerics.runtime import (
    evaluate,
    mx,
)
from mlxradiomics.numerics.sparse_counts import coordinate_counts


def _mcc_group(
    counts: list[mx.array],
    active: list[mx.array],
    batch: int,
) -> tuple[mx.array, mx.array]:
    """Return the summed MCC and defined-direction count for one bounded group.

    ``maximal_correlation`` returns NaN for a direction whose local matrix has no
    second eigenvalue, so only defined directions contribute to the average.
    """
    if not counts:
        return (
            mx.zeros((batch,), dtype=mx.float32),
            mx.zeros((batch,), dtype=mx.int32),
        )
    block = counts[0] if len(counts) == 1 else mx.concatenate(counts, axis=0)
    values = mx.reshape(maximal_correlation(block), (len(counts), batch))
    mask = mx.stack(active, axis=0) & ~mx.isnan(values)
    total = mx.sum(mx.where(mask, values, 0), axis=0)
    defined = mx.sum(mask.astype(mx.int32), axis=0)
    return total, defined


def aggregate_glcm(
    pairs: Iterable[tuple[mx.array, mx.array, mx.array]],
    ng: int,
    settings: Settings,
    dtype: mx.Dtype,
    batch: int,
    method: str,
) -> dict[str, mx.array]:
    """Average the selected GLCM features over directions, for ``batch`` rows.

    Each direction's counts come from ``pairs`` (left bins, right bins, valid mask), are
    optionally symmetrized, and give one value per row. Directions in which a row has no
    valid pair are skipped for that row. MCC is also skipped where its local matrix has
    fewer than two occupied row or column gray levels, because its second eigenvalue is
    then absent. A feature is NaN when no direction defines it for that row.
    """
    names = tuple(
        name.removeprefix("glcm_") for name in settings.features if name.startswith("glcm_")
    )
    ordinary = tuple(name for name in names if name != "MCC")
    # Compensated: signed directional values must not cancel order-dependently.
    totals = {name: CompensatedTotal(batch, dtype) for name in ordinary}
    angles = mx.zeros((batch,), dtype=mx.int32)
    mcc_total = mx.zeros((batch,), dtype=mx.float32)
    mcc_angles = mx.zeros((batch,), dtype=mx.int32)
    pending_counts: list[mx.array] = []
    pending_active: list[mx.array] = []
    per_direction = max(1, batch * ng * ng * 4)
    group = max(1, (64 * 1024**2) // per_direction)
    for left, right, valid in pairs:
        if use_sparse_glcm(ng):
            keys, sparse = coordinate_counts(
                left, right, valid, ng, symmetric=settings.symmetrical_glcm,
            )
            values, active = sparse_glcm_statistics(
                keys, sparse, ng, names, settings.scratch_mib,
            )
            for name in ordinary:
                totals[name].add(mx.where(active, values[name], 0))
            angles = angles + active.astype(mx.int32)
            if "MCC" in names:
                defined = active & ~mx.isnan(values["MCC"])
                mcc_total = mcc_total + mx.where(defined, values["MCC"], 0)
                mcc_angles = mcc_angles + defined.astype(mx.int32)
            evaluate(angles, *(a for t in totals.values() for a in t.arrays()),
                     mcc_total, mcc_angles)
            continue
        counts = histogram(left, right, valid, ng, method=method)
        if settings.symmetrical_glcm:
            counts = counts + mx.swapaxes(counts, 1, 2)
        if ordinary:
            # Nonnegative counts let ``glcm_statistics`` classify empty directions.
            values, active = glcm_statistics(counts, dtype, ordinary)
            for name in ordinary:
                totals[name].add(mx.where(active, values[name], 0))
        else:
            active = mx.sum(counts.astype(dtype), axis=(1, 2)) > 0
        angles = angles + active.astype(mx.int32)
        evaluate(
            angles,
            *(a for t in totals.values() for a in t.arrays()),
            *((counts,) if "MCC" in names else ()),
        )
        if "MCC" in names:
            pending_counts.append(counts)
            pending_active.append(active)
            if len(pending_counts) == group:
                total, defined = _mcc_group(pending_counts, pending_active, batch)
                mcc_total = mcc_total + total
                mcc_angles = mcc_angles + defined
                pending_counts, pending_active = [], []
    if pending_counts:
        total, defined = _mcc_group(pending_counts, pending_active, batch)
        mcc_total = mcc_total + total
        mcc_angles = mcc_angles + defined
    result = {
        f"glcm_{k}": mx.where(angles > 0, v.value() / angles, float("nan"))
        for k, v in totals.items()
    }
    if "MCC" in names:
        result["glcm_MCC"] = mx.where(mcc_angles > 0, mcc_total / mcc_angles, float("nan"))
    return result


def aggregate_roi_glcm(
    bins: mx.array,
    mask: mx.array,
    ng: int,
    settings: Settings,
    dtype: mx.Dtype,
    method: str,
) -> dict[str, mx.array]:
    """Batch ordinary ROI directions from exact resident-grid counts.

    Count storage and the statistics' conservative working-set estimate must fit
    the scratch budget; larger selections use the directional pair route. Directional
    feature sums and the MCC grouping are taken in direction order.
    """
    offsets = directions(tuple(bins.shape), settings)
    count_bytes = len(offsets) * ng * ng * 4
    per_direction = ng * ng * 128
    budget = settings.scratch_mib * 1024**2
    if (
        use_sparse_glcm(ng)
        or not offsets
        or len(offsets) > 13
        or count_bytes > 64 * 1024**2
        or 2 * count_bytes + per_direction > budget
    ):
        return aggregate_glcm(roi_pairs(bins, mask, settings), ng, settings, dtype, 1, method)
    group = min(len(offsets), max(1, (budget - 2 * count_bytes) // per_direction))
    counts = roi_histograms(bins, mask, offsets, ng)
    if settings.symmetrical_glcm:
        counts = counts + mx.swapaxes(counts, 1, 2)
    names = tuple(
        name.removeprefix("glcm_") for name in settings.features if name.startswith("glcm_")
    )
    ordinary = tuple(name for name in names if name != "MCC")
    blocks: list[mx.array] = []
    active_rows: list[mx.array] = []
    for start in range(0, len(offsets), group):
        stop = min(start + group, len(offsets))
        block = counts[start:stop]
        if ordinary:
            values, active = glcm_statistics(block, dtype, ordinary)
            feature_block = mx.stack([values[name] for name in ordinary])
            evaluate(feature_block, active)
            blocks.append(feature_block)
        else:
            active = mx.sum(block.astype(dtype), axis=(1, 2)) > 0
        active_rows.extend(active[index : index + 1] for index in range(stop - start))
    output: dict[str, mx.array] = {}
    if ordinary:
        means = directional_mean(
            mx.concatenate(blocks, axis=1),
            mx.concatenate(active_rows),
            compensated=True,
        )
        output.update(
            {f"glcm_{name}": means[index : index + 1] for index, name in enumerate(ordinary)}
        )
    if "MCC" in names:
        total, defined = _mcc_group(
            [counts[index : index + 1] for index in range(len(offsets))],
            active_rows,
            1,
        )
        output["glcm_MCC"] = mx.where(defined > 0, total / defined, float("nan"))
    return output


def pair_slices(
    shape: tuple[int, ...],
    offset: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return slices selecting voxel ``p`` and voxel ``p + offset``.

    Only positions whose partner lies inside an array of ``shape`` are selected.
    """
    return (
        tuple(slice(max(0, -v), min(s, s - v)) for s, v in zip(shape, offset, strict=True)),
        tuple(slice(max(0, v), min(s, s + v)) for s, v in zip(shape, offset, strict=True)),
    )


def roi_pairs(
    bins: mx.array,
    mask: mx.array,
    settings: Settings,
) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
    """Yield ``(left, right, valid)`` voxel-pair arrays of shape ``(1, pairs)`` per direction.

    ``left`` and ``right`` hold the gray-level bins of voxels ``p`` and ``p + offset``; a
    pair is valid only if both voxels are in ``mask``. Directions follow
    ``preprocessing.directions``.
    """
    for offset in directions(bins.shape, settings):
        left, right = pair_slices(bins.shape, offset)
        yield (
            mx.reshape(bins[left], (1, -1)),
            mx.reshape(bins[right], (1, -1)),
            mx.reshape(mask[left] & mask[right], (1, -1)),
        )
