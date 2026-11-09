"""Run, zone and dependence features from occupied integer count cells on Metal.

The coordinates retain original gray levels and sizes. Empty lattice cells do
not enter the storage or reductions, including for gray/size nonuniformity.
"""

from collections.abc import Collection

from mlxradiomics.numerics.count_statistics import count_entropy
from mlxradiomics.numerics.reduction import sum_rows
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.numerics.sparse_counts import coalesce


def use_sparse_size(ng: int, ns: int) -> bool:
    """Select sparse ROI storage above a 64 MiB estimated dense working lattice.

    The representation depends only on the gray and size axes, so a memory budget
    cannot change a feature's reduction order. Gray and size indices are preserved.
    Voxel neighborhoods retain their fixed dense reduction tree and are separately
    bounded by the voxel scheduler's working-memory estimate.
    """
    return ng * ns * 128 > 64 * 1024**2


def sparse_size_statistics(
    keys: mx.array,
    counts: mx.array,
    column_count: int,
    family: str,
    selected: Collection[str] | None = None,
    *,
    voxel_count: mx.array | int | None = None,
    total_count: mx.array | None = None,
) -> dict[str, mx.array]:
    """Evaluate size-matrix features from padded ``(batch, occupied cells)`` arrays.

    Each key is ``(gray_level - 1) * column_count + size - 1``. Counts are
    nonnegative integers; padded entries have zero count. GLDZM requires an
    explicit voxel count because its distance index cannot reconstruct volume.
    All count and feature arithmetic runs on Metal. Only the number of occupied
    cells and marginal entries crosses to the host to allocate compact reductions.
    """
    spec = {
        "gldm": ("Dependence", "SmallDependence", "LargeDependence", "LowGrayLevel",
                 "HighGrayLevel", "DependenceEntropy", "DependenceVariance", None),
        "glrlm": ("RunLength", "ShortRun", "LongRun", "LowGrayLevelRun",
                  "HighGrayLevelRun", "RunEntropy", "RunVariance", "RunPercentage"),
        "glszm": ("SizeZone", "SmallArea", "LargeArea", "LowGrayLevelZone",
                  "HighGrayLevelZone", "ZoneEntropy", "ZoneVariance", "ZonePercentage"),
        "gldzm": ("ZoneDistance", "SmallDistance", "LargeDistance", "LowGrayLevelZone",
                  "HighGrayLevelZone", "ZoneDistanceEntropy", "ZoneDistanceVariance",
                  "ZonePercentage"),
    }
    if family not in spec:
        raise ValueError(f"unknown size-statistics family: {family}")
    size, small, large, low, high, entropy, variance, percentage = spec[family]
    requested = None if selected is None else frozenset(selected)
    if requested is not None and not requested:
        return {}
    if family == "gldzm" and voxel_count is None:
        raise ValueError("gldzm requires an explicit voxel_count")
    totals = (mx.sum(counts.astype(mx.int64), axis=1) if total_count is None
              else total_count.astype(mx.int64))
    voxels = (
        None if voxel_count is None else
        mx.full((counts.shape[0],), voxel_count, dtype=mx.int64)
        if isinstance(voxel_count, int) else voxel_count.astype(mx.int64)
    )

    def wanted(name: str) -> bool:
        return requested is None or name in requested

    def reduce(values: mx.array) -> mx.array:
        return sum_rows(values[None, :])[0]

    def marginal(indices: mx.array, weights: mx.array) -> tuple[mx.array, mx.array]:
        return coalesce(mx.where(weights > 0, indices, -1), weights)

    def integer_variance(
        indices: mx.array, weights: mx.array, safe_total: mx.array,
    ) -> mx.array:
        # Subtract original integer indices before converting to float32. This
        # preserves small variation between adjacent million-voxel zone sizes.
        reference = indices[mx.argmax(weights)]
        offsets = indices - reference
        mean = mx.sum(weights.astype(mx.int64) * offsets).astype(mx.float32) / safe_total
        centered = offsets.astype(mx.float32) - mean
        return reduce(weights.astype(mx.float32) * centered**2) / safe_total

    rows: list[dict[str, mx.array]] = []
    for row in range(counts.shape[0]):
        # Fix the occupied-cell order and remove batch padding before any
        # floating-point reduction, independently of other rows' cell counts.
        row_keys, row_counts = coalesce(mx.where(counts[row] > 0, keys[row], -1), counts[row])
        exact = row_counts.astype(mx.int64)
        total = totals[row]
        total_f = total.astype(mx.float32)
        safe_total = mx.maximum(total, 1).astype(mx.float32)
        i, j = row_keys // column_count + 1, row_keys % column_count + 1
        p = exact.astype(mx.float32) / safe_total
        i_squared, j_squared = i.astype(mx.float32)**2, j.astype(mx.float32)**2
        output: dict[str, mx.array] = {}

        if wanted(small + "Emphasis"):
            output[small + "Emphasis"] = reduce(p / j_squared)
        if wanted(large + "Emphasis"):
            output[large + "Emphasis"] = reduce(p * j_squared)
        for indices, group, variance_name in (
            (i, "GrayLevel", "GrayLevelVariance"), (j, size, variance),
        ):
            if any(wanted(name) for name in (
                group + "NonUniformity", group + "NonUniformityNormalized", variance_name,
            )):
                levels, weights = marginal(indices, exact)
                if wanted(group + "NonUniformity") or wanted(group + "NonUniformityNormalized"):
                    squared = reduce(weights.astype(mx.float32)**2)
                    if wanted(group + "NonUniformity"):
                        output[group + "NonUniformity"] = squared / total_f
                    if wanted(group + "NonUniformityNormalized"):
                        output[group + "NonUniformityNormalized"] = squared / total_f**2
                if wanted(variance_name):
                    output[variance_name] = integer_variance(levels, weights, safe_total)
        if wanted(entropy):
            output[entropy] = count_entropy(exact[None, :], total[None], mx.float32)[0]
        if wanted(low + "Emphasis"):
            output[low + "Emphasis"] = reduce(p / i_squared)
        if wanted(high + "Emphasis"):
            output[high + "Emphasis"] = reduce(p * i_squared)
        if wanted(small + "LowGrayLevelEmphasis"):
            output[small + "LowGrayLevelEmphasis"] = reduce(p / (i_squared * j_squared))
        if wanted(small + "HighGrayLevelEmphasis"):
            output[small + "HighGrayLevelEmphasis"] = reduce(p * i_squared / j_squared)
        if wanted(large + "LowGrayLevelEmphasis"):
            output[large + "LowGrayLevelEmphasis"] = reduce(p * j_squared / i_squared)
        if wanted(large + "HighGrayLevelEmphasis"):
            output[large + "HighGrayLevelEmphasis"] = reduce(p * i_squared * j_squared)
        if family == "gldm" and wanted("DependenceCountEnergy"):
            output["DependenceCountEnergy"] = reduce(p**2)
        if family == "gldm" and wanted("DependencePercentage"):
            denominator = total if voxels is None else voxels[row]
            output["DependencePercentage"] = total_f / denominator.astype(mx.float32)
        if percentage is not None and wanted(percentage):
            if family == "gldzm":
                assert voxels is not None
                denominator = voxels[row]
            else:
                denominator = mx.sum(exact * j)
            output[percentage] = total_f / denominator.astype(mx.float32)
        if requested is not None and (unknown := requested.difference(output)):
            raise ValueError(f"unknown {family} statistics: {', '.join(sorted(unknown))}")
        rows.append(output)
    return {name: mx.stack([row[name] for row in rows]) for name in rows[0]}
