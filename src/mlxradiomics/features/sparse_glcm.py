"""Co-occurrence features from occupied cells, retaining original gray indices.

Native Metal sorting and integer scatter addition build coordinate-form counts.
Storage scales with observed voxel pairs and one-dimensional gray marginals,
rather than the square of the discretization range. Shape reads transfer only
integer sizes; all counting and feature arithmetic remain on the GPU.
"""

from typing import cast

from mlxradiomics.features.glcm import EPS, cluster_shade, one_minus_exp
from mlxradiomics.numerics.count_statistics import count_entropy, log1p
from mlxradiomics.numerics.eigen import maximal_correlation
from mlxradiomics.numerics.protocols import IndexedAddition
from mlxradiomics.numerics.reduction import compensated_sum_rows, sum_rows
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.numerics.sparse_counts import coalesce


def use_sparse_glcm(ng: int) -> bool:
    """Choose storage from the lattice, independently of batching and scratch settings.

    At most 512 levels keep the dense intermediates near 32 MiB per matrix.
    Larger lattices use occupied cells; this changes storage, not admissibility.
    """
    return ng > 512


def _marginal(indices: mx.array, counts: mx.array, size: int) -> mx.array:
    storage = mx.zeros((size,), dtype=mx.int32)
    return cast(IndexedAddition, storage.at[indices.astype(mx.int32)]).add(counts)


def _row_statistics(
    keys: mx.array,
    counts: mx.array,
    ng: int,
    features: tuple[str, ...],
    scratch_mib: int,
) -> dict[str, mx.array]:
    exact = counts.astype(mx.int64)
    total = mx.sum(exact)
    safe_total = mx.maximum(total, 1)
    n = safe_total.astype(mx.float32)
    i, j = keys // ng + 1, keys % ng + 1
    rows = _marginal(i - 1, counts, ng)
    columns = _marginal(j - 1, counts, ng)
    differences = _marginal(mx.abs(i - j), counts, ng)
    sums = _marginal(i + j - 2, counts, 2 * ng - 1)
    p = counts.astype(mx.float32) / n
    sx, sy = mx.sum(exact * i), mx.sum(exact * j)
    dx, dy = (
        (i * safe_total - sx).astype(mx.float32) / n,
        (j * safe_total - sy).astype(mx.float32) / n,
    )
    cluster = ((i + j) * safe_total - sx - sy).astype(mx.float32) / n
    delta_integer = mx.abs(i - j)
    delta = delta_integer.astype(mx.float32)

    def reduce(values: mx.array) -> mx.array:
        return sum_rows(values[None, :])[0]

    def entropy(values: mx.array) -> mx.array:
        return count_entropy(values[None, :], total[None], mx.float32)[0]

    vx, vy = reduce(p * dx**2), reduce(p * dy**2)
    hx, hy = entropy(rows), entropy(columns)
    row_cells, column_cells = rows[(i - 1).astype(mx.int32)], columns[(j - 1).astype(mx.int32)]
    denominator_int = row_cells.astype(mx.int64) * column_cells.astype(mx.int64)
    denominator = mx.maximum(denominator_int, 1).astype(mx.float32)
    residual = exact * total - denominator_int
    ratio = residual.astype(mx.float32) / denominator
    quotient = counts.astype(mx.float32) * n / denominator
    logs = mx.where(quotient > 0.5, log1p(ratio), mx.log(mx.maximum(quotient, 1e-38)))
    direct = compensated_sum_rows(mx.where(counts > 0, counts * logs, 0)[None, :])[0] / n
    # Near independence, all cells in the product support must be occupied;
    # missing cells have ratio=-1 and therefore cannot use the Taylor series.
    full_support = mx.sum((counts > 0).astype(mx.int64)) == (
        mx.sum((rows > 0).astype(mx.int64)) * mx.sum((columns > 0).astype(mx.int64))
    )
    use_series = full_support & (mx.max(mx.where(counts > 0, mx.abs(ratio), 0)) <= 0.5)
    small_ratio = mx.where(use_series, ratio, 0)
    polynomial = mx.zeros_like(small_ratio)
    for order in range(24, 1, -1):
        polynomial = small_ratio * polynomial + (1.0 / order if order % 2 else -1.0 / order)
    linear = reduce(mx.where(counts > 0, residual.astype(mx.float32) ** 2 / denominator, 0))
    linear = linear / (n * n)
    correction = reduce(counts * small_ratio**2 * polynomial) / n
    information = mx.where(use_series, linear + correction, direct) / 0.6931471805599453
    mean_difference = reduce(p * delta)
    centered_difference = (
        (delta_integer * safe_total - mx.sum(exact * delta_integer)).astype(mx.float32) / n
    )
    all_values = {
        "Autocorrelation": reduce(p * i.astype(mx.float32) * j.astype(mx.float32)),
        "ClusterProminence": reduce(p * cluster**4),
        "ClusterShade": cluster_shade(sums.astype(mx.int64)[None, :], total[None], mx.float32)[0],
        "ClusterTendency": reduce(p * cluster**2),
        "Contrast": reduce(p * delta**2),
        "Correlation": mx.where(
            (mx.sum((rows > 0).astype(mx.int32)) == 1)
            | (mx.sum((columns > 0).astype(mx.int32)) == 1),
            1,
            reduce(p * dx * dy) / (mx.sqrt(vx * vy) + EPS),
        ),
        "DifferenceAverage": mean_difference,
        "DifferenceEntropy": entropy(differences),
        "DifferenceVariance": reduce(p * centered_difference**2),
        "Dissimilarity": mean_difference,
        "Id": reduce(p / (1 + delta)),
        "Idm": reduce(p / (1 + delta**2)),
        "Idmn": reduce(p / (1 + delta**2 / float(ng) ** 2)),
        "Idn": reduce(p / (1 + delta / float(ng))),
        "Imc1": mx.where(
            mx.maximum(hx, hy) > 0, -information / mx.maximum(mx.maximum(hx, hy), 1e-38), 0
        ),
        "Imc2": mx.sqrt(mx.maximum(one_minus_exp(2 * information), 0)),
        "InverseVariance": reduce(mx.where(delta > 0, p / mx.maximum(delta**2, 1), 0)),
        "JointAverage": sx.astype(mx.float32) / n,
        "JointEnergy": reduce(p**2),
        "JointEntropy": entropy(counts),
        "MaximumProbability": mx.max(p),
        "SumAverage": reduce(p * (i + j).astype(mx.float32)),
        "SumEntropy": entropy(sums),
        "SumSquares": vx,
    }
    all_values["SumVariance"] = all_values["ClusterTendency"]
    if "MCC" in features:
        occupied, _ = coalesce(
            mx.concatenate([mx.where(counts > 0, i, -1), mx.where(counts > 0, j, -1)])
        )
        size = occupied.size
        if size * size * 128 > scratch_mib * 1024**2:
            raise ValueError("MCC occupied-state eigensystem exceeds scratch_mib")
        r = mx.where(counts > 0, mx.searchsorted(occupied, i), 0).astype(mx.int32)
        c = mx.where(counts > 0, mx.searchsorted(occupied, j), 0).astype(mx.int32)
        compact = mx.reshape(_marginal(r * size + c, counts, size * size), (1, size, size))
        all_values["MCC"] = maximal_correlation(compact)[0]
    return {name: all_values[name] for name in features}


def sparse_glcm_statistics(
    keys: mx.array,
    counts: mx.array,
    ng: int,
    features: tuple[str, ...],
    scratch_mib: int,
) -> tuple[dict[str, mx.array], mx.array]:
    """Evaluate occupied-cell rows with full-lattice normalization."""
    rows = [
        _row_statistics(k, c, ng, features, scratch_mib) for k, c in zip(keys, counts, strict=True)
    ]
    return (
        {name: mx.stack([row[name] for row in rows]) for name in features},
        mx.sum(counts.astype(mx.int64), axis=1) > 0,
    )
