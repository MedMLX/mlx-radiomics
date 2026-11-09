"""Features computed from count matrices: GLDM, GLRLM, GLSZM, GLDZM and NGTDM.

Inputs have a leading batch axis (one row per ROI or per voxel center). GLDM, GLRLM,
GLSZM and GLDZM matrices hold counts; NGTDM also includes sums of neighborhood
differences, which need not be integers. Names and formulas follow the PyRadiomics
feature classes except where the mathematical definition is corrected here: NGTDM
normalizes over centers with a nonempty neighborhood and uses IEEE singular-value
semantics instead of finite sentinels, and variance features center integer indices
before converting them to float32.
"""

from collections.abc import Collection
from math import isqrt
from typing import cast

from mlxradiomics.config.settings import NGTDM
from mlxradiomics.numerics.count_statistics import count_entropy
from mlxradiomics.numerics.reduction import CompensatedTotal, sum_rows
from mlxradiomics.numerics.runtime import evaluate, mx


def size_statistics(
    counts: mx.array,
    family: str,
    selected: Collection[str] | None = None,
    *,
    voxel_count: mx.array | int | None = None,
    total_count: mx.array | None = None,
) -> dict[str, mx.array]:
    """Compute GLDM, GLRLM, GLSZM or GLDZM features from count matrices.

    Args:
        counts: Array ``(batch, Ng, Ns)``. Axis 1 is the gray level ``i = 1..Ng`` and axis
            2 the dependence size, run length, zone size or zone distance ``j = 1..Ns``.
            Gray levels keep their original indices, so absent levels stay in the matrix
            with zero counts.
        family: ``"gldm"``, ``"glrlm"``, ``"glszm"`` or ``"gldzm"``. Selects the feature
            names, for example ``SmallDependenceEmphasis``, ``ShortRunEmphasis``,
            ``SmallAreaEmphasis`` or ``SmallDistanceEmphasis``.
        selected: Statistic names without the family prefix, or ``None`` to compute every
            statistic. Only requested features are constructed.
        voxel_count: Number of voxels represented by each row, as an integer or
            ``(batch,)`` integer array. Required for ``"gldzm"`` because a distance index
            cannot reconstruct voxel count from ``distance * zone count``. For ``"gldm"``
            it supplies the explicit denominator of ``DependencePercentage``; when
            omitted, the GLDM construction guarantees ``N_s == N_v`` and the ratio is one.
        total_count: Optional exact integer sum of each count matrix, reusable by aggregation.

    Returns:
        Feature name without the family prefix mapped to a ``(batch,)`` float32 array, in
        the same order as the all-statistics result. ``RunPercentage``, ``ZonePercentage``
        and ``DependencePercentage`` are included only for the families that define them.
        Empty count matrices have undefined normalized features; callers must exclude
        inactive directions before aggregating them.

    Notes:
        Gray-level and size/distance variances select an occupied integer reference
        index, subtract it in integer arithmetic, and only then convert the offsets to
        float32. This retains a small variance when all occupied indices share a large
        common offset (for example zone sizes 524288 and 524289). Entropies evaluate
        integer counts directly through
        :func:`mlxradiomics.numerics.count_statistics.count_entropy`.

    Raises:
        ValueError: If ``family`` is unknown, ``selected`` names a statistic that
            ``family`` does not produce, or ``"gldzm"`` is requested without
            ``voxel_count``.
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
    counts_int = counts.astype(mx.int64)
    # Integer totals are exact and independent of reduction order. ``sum_rows`` supports
    # float32/int32 only, so integer totals use the regular reduction.
    total = (mx.sum(counts_int, axis=(1, 2)) if total_count is None
             else total_count.astype(mx.int64))
    total_f = total.astype(mx.float32)
    safe_total_f = mx.maximum(total, 1).astype(mx.float32)
    counts_f = counts_int.astype(mx.float32)
    # ``j`` is the length-Ns index vector shared by emphasis, variance, cross terms and
    # percentage.
    j = mx.arange(1, counts.shape[2] + 1, dtype=mx.float32)[None, None, :]

    if family == "gldzm" and voxel_count is None:
        raise ValueError("gldzm requires an explicit voxel_count")
    if voxel_count is None:
        voxels = None
    else:
        voxels = (
            mx.array(voxel_count, dtype=mx.int64)
            if isinstance(voxel_count, int)
            else voxel_count.astype(mx.int64)
        )

    def wanted(name: str) -> bool:
        return requested is None or name in requested

    marginals: dict[int, mx.array] = {}

    def marginal(axis: int) -> mx.array:
        # Nonuniformity and variance use the same exact integer marginal. Build
        # it only for a requested consumer, retaining its original reduction axis.
        if axis not in marginals:
            marginals[axis] = mx.sum(counts_int, axis=axis)
        return marginals[axis]

    def integer_variance(marginal: mx.array) -> mx.array:
        """Weighted variance of index offsets, centered in integer arithmetic.

        The reference is the most frequent occupied index, so the bulk of the counts
        receive small offsets.  Index shifts cancel because only differences matter.
        """
        reference = mx.argmax(marginal, axis=1).astype(mx.int32)
        offsets = (
            mx.arange(marginal.shape[1], dtype=mx.int32)[None, :] - reference[:, None]
        )
        mean_offset = (
            mx.sum(marginal * offsets.astype(mx.int64), axis=1).astype(mx.float32)
            / safe_total_f
        )
        centered = offsets.astype(mx.float32) - mean_offset[:, None]
        return (
            sum_rows(marginal.astype(mx.float32) * centered**2) / safe_total_f
        )

    # Only requested statistics are constructed; building the others would cost host
    # time for graphs that are discarded. ``selected=None`` requests every statistic.
    need_p = True if requested is None else any(name != percentage for name in requested)
    features: dict[str, mx.array] = {}
    if need_p:
        p = counts_f / safe_total_f[:, None, None]
        i = mx.arange(1, counts.shape[1] + 1, dtype=mx.float32)[None, :, None]
        i_squared, j_squared = i**2, j**2
        if wanted(small + "Emphasis"):
            features[small + "Emphasis"] = sum_rows(p / j_squared)
        if wanted(large + "Emphasis"):
            features[large + "Emphasis"] = sum_rows(p * j_squared)
        if wanted("GrayLevelNonUniformity") or wanted("GrayLevelNonUniformityNormalized"):
            pg = marginal(2).astype(mx.float32)
            squared = sum_rows(pg**2)
            if wanted("GrayLevelNonUniformity"):
                features["GrayLevelNonUniformity"] = squared / total_f
            if wanted("GrayLevelNonUniformityNormalized"):
                features["GrayLevelNonUniformityNormalized"] = squared / total_f**2
        if wanted(size + "NonUniformity") or wanted(size + "NonUniformityNormalized"):
            ps = marginal(1).astype(mx.float32)
            squared = sum_rows(ps**2)
            if wanted(size + "NonUniformity"):
                features[size + "NonUniformity"] = squared / total_f
            if wanted(size + "NonUniformityNormalized"):
                features[size + "NonUniformityNormalized"] = squared / total_f**2
        if wanted("GrayLevelVariance"):
            features["GrayLevelVariance"] = integer_variance(marginal(2))
        if wanted(variance):
            features[variance] = integer_variance(marginal(1))
        if wanted(entropy):
            features[entropy] = count_entropy(
                mx.reshape(counts_int, (counts.shape[0], -1)), total, mx.float32
            )
        if wanted(low + "Emphasis"):
            features[low + "Emphasis"] = sum_rows(p / i_squared)
        if wanted(high + "Emphasis"):
            features[high + "Emphasis"] = sum_rows(p * i_squared)
        if wanted(small + "LowGrayLevelEmphasis"):
            features[small + "LowGrayLevelEmphasis"] = sum_rows(p / (i_squared * j_squared))
        if wanted(small + "HighGrayLevelEmphasis"):
            features[small + "HighGrayLevelEmphasis"] = sum_rows(p * i_squared / j_squared)
        if wanted(large + "LowGrayLevelEmphasis"):
            features[large + "LowGrayLevelEmphasis"] = sum_rows(p * j_squared / i_squared)
        if wanted(large + "HighGrayLevelEmphasis"):
            features[large + "HighGrayLevelEmphasis"] = sum_rows(p * i_squared * j_squared)
        if family == "gldm" and wanted("DependenceCountEnergy"):
            features["DependenceCountEnergy"] = sum_rows(p**2)
    if family == "gldm" and wanted("DependencePercentage"):
        denominator = total if voxels is None else voxels
        features["DependencePercentage"] = total_f / denominator.astype(mx.float32)
    if percentage is not None and wanted(percentage):
        if family == "gldzm":
            assert voxels is not None
            features[percentage] = total_f / voxels.astype(mx.float32)
        else:
            features[percentage] = total_f / sum_rows(counts_f * j)
    if requested is not None:
        unknown = requested.difference(features)
        if unknown:
            raise ValueError(f"unknown {family} statistics: {', '.join(sorted(unknown))}")
    return features


def _large_ngtdm_pair_sums(
    n: mx.array, s: mx.array, p: mx.array, selected: Collection[str],
) -> dict[str, mx.array]:
    """Reduce occupied-level pairs in bounded native Metal tiles.

    Zero-count columns contribute no pairs. GPU sorting/gathering removes their
    storage while keeping the original integer gray-level indices. One host scalar
    read sizes the compact arrays; all numerical feature work remains on Metal.
    Each pair tensor has at most 262,144 elements (or one per batch row if larger).
    Evaluating compensated totals after each tile releases the pair intermediates.
    The inputs and selection buffers are O(batch * Ng), not O(Ng²).
    """
    present = n > 0
    occupied = sum_rows(present.astype(mx.int32))
    width = max(1, cast(int, mx.max(occupied).item()))
    original = mx.arange(n.shape[1], dtype=mx.int32)[None, :]
    order = mx.argsort(mx.where(present, original, n.shape[1]), axis=1)[:, :width]
    n = mx.take_along_axis(n, order, axis=1)
    s = mx.take_along_axis(s, order, axis=1)
    p = mx.take_along_axis(p, order, axis=1)
    levels = order.astype(mx.int32) + 1
    evaluate(n, s, p, levels)

    totals = {name: CompensatedTotal(n.shape[0]) for name in selected if name != "Coarseness"}
    tile = max(1, isqrt(262144 // n.shape[0]))
    for first in range(0, width, tile):
        ni = n[:, first:first + tile, None]
        pi = p[:, first:first + tile, None]
        si = s[:, first:first + tile, None]
        i = levels[:, first:first + tile, None]
        for second in range(0, width, tile):
            nj = n[:, None, second:second + tile]
            pj = p[:, None, second:second + tile]
            sj = s[:, None, second:second + tile]
            j = levels[:, None, second:second + tile]
            pairs = (ni > 0) & (nj > 0)
            difference = (i - j).astype(mx.float32)
            squared_difference = difference**2
            if "Contrast" in totals:
                totals["Contrast"].add(sum_rows(pi * pj * squared_difference))
            if "Busyness" in totals:
                # Integer products preserve near-equal i*n_i even above 2**24.
                products_i = ni.astype(mx.int64) * i.astype(mx.int64)
                products_j = nj.astype(mx.int64) * j.astype(mx.int64)
                differences = mx.abs(products_i - products_j)
                totals["Busyness"].add(sum_rows(
                    mx.where(pairs, differences, 0).astype(mx.float32),
                ))
            if "Complexity" in totals:
                denominator = mx.where(pairs, pi + pj, 1)
                totals["Complexity"].add(sum_rows(mx.where(
                    pairs, mx.abs(difference) * (pi * si + pj * sj) / denominator, 0,
                )))
            if "Strength" in totals:
                totals["Strength"].add(sum_rows(mx.where(
                    pairs, (pi + pj) * squared_difference, 0,
                )))
            evaluate(*(value for total in totals.values() for value in total.arrays()))
    return {name: total.value() for name, total in totals.items()}


def ngtdm_statistics(
    n: mx.array, s: mx.array, selected: Collection[str] | None = None, *,
    total_count: mx.array | None = None,
) -> dict[str, mx.array]:
    """Compute selected NGTDM features from per-gray-level neighborhood statistics.

    ``n`` is an integer ``(batch, Ng)`` array of eligible-center occurrences, and ``s``
    contains corresponding float32 absolute neighborhood-difference sums. Optional
    ``total_count`` reuses their exact row population. A caller-selected subset avoids
    constructing unrelated pair graphs. Lattices above 1,024 levels use bounded
    occupied-level tiles with their original indices; smaller lattices use dense
    pair reductions.

    No eligible centers gives NaN. A constant eligible region has infinite Coarseness,
    zero Contrast/Complexity, and undefined Busyness/Strength; positive-over-zero ratios
    remain infinite. Profile-specific finite degenerate conventions belong to the caller.
    """
    requested = frozenset(NGTDM if selected is None else selected)
    unknown = requested.difference(NGTDM)
    if unknown:
        raise ValueError(f"unknown NGTDM statistics: {', '.join(sorted(unknown))}")
    if not requested:
        return {}
    total = (sum_rows(n.astype(mx.int32)) if total_count is None
             else total_count.astype(mx.int32))
    p = n / mx.maximum(total[:, None], 1)
    defined = total > 0
    output: dict[str, mx.array] = {}
    # These small reductions are shared; discarded lazy values launch no GPU work.
    ps = sum_rows(p * s)
    sum_s = sum_rows(s)
    if "Coarseness" in requested:
        output["Coarseness"] = mx.where(
            defined, mx.where(ps > 0, 1 / ps, float("inf")), float("nan"),
        )
    if requested - {"Coarseness"}:
        bounded_pairs = (
            _large_ngtdm_pair_sums(n, s, p, requested) if n.shape[1] > 1024 else None
        )
        levels = mx.arange(1, n.shape[1] + 1, dtype=mx.float32)[None, :]
        present = n > 0
        pi, pj = p[:, :, None], p[:, None, :]
        i, j = levels[:, :, None], levels[:, None, :]
        pairs = present[:, :, None] & present[:, None, :]
        level_difference = i - j
        squared_difference = level_difference**2
        if "Contrast" in requested:
            ng = sum_rows(present.astype(mx.int32))
            pair_sum = (bounded_pairs["Contrast"] if bounded_pairs is not None
                        else sum_rows(pi * pj * squared_difference))
            contrast = pair_sum * sum_s / mx.maximum(total, 1)
            output["Contrast"] = mx.where(
                defined, mx.where(ng > 1, contrast / mx.maximum(ng * (ng - 1), 1), 0),
                float("nan"),
            )
        if "Busyness" in requested:
            # Keep level products integral until subtraction, including counts >2**24.
            if bounded_pairs is not None:
                busyness = bounded_pairs["Busyness"]
            else:
                products = n.astype(mx.int64) * mx.arange(1, n.shape[1] + 1, dtype=mx.int64)
                differences = mx.abs(products[:, :, None] - products[:, None, :])
                busyness = sum_rows(mx.where(pairs, differences, 0).astype(mx.float32))
            output["Busyness"] = mx.where(
                defined,
                mx.where(busyness > 0, sum_rows(n.astype(mx.float32) * s) / busyness,
                         mx.where(ps > 0, float("inf"), float("nan"))),
                float("nan"),
            )
        if "Complexity" in requested:
            # Excluded pairs retain a harmless denominator, avoiding an unused 0/0.
            if bounded_pairs is not None:
                complexity = bounded_pairs["Complexity"]
            else:
                denominator = mx.where(pairs, pi + pj, 1)
                si, sj = s[:, :, None], s[:, None, :]
                complexity = sum_rows(mx.where(
                    pairs, mx.abs(level_difference) * (pi * si + pj * sj) / denominator, 0,
                ))
            output["Complexity"] = mx.where(
                defined, complexity / mx.maximum(total, 1), float("nan"),
            )
        if "Strength" in requested:
            strength = (bounded_pairs["Strength"] if bounded_pairs is not None
                        else sum_rows(mx.where(pairs, (pi + pj) * squared_difference, 0)))
            output["Strength"] = mx.where(
                defined,
                mx.where(sum_s > 0, strength / sum_s,
                         mx.where(strength > 0, float("inf"), float("nan"))),
                float("nan"),
            )
    return output
