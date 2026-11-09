"""GLCM feature arithmetic from exact integer co-occurrence counts."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.count_statistics import count_entropy, log1p
from mlxradiomics.numerics.eigen import maximal_correlation
from mlxradiomics.numerics.reduction import sum_rows as _sum_rows
from mlxradiomics.numerics.runtime import mx

# Same epsilon as PyRadiomics (numpy.spacing(1)); used only as the correlation
# denominator floor. Entropies and mutual information use explicit zero support.
EPS = 2.220446049250313e-16
_LN2 = 0.6931471805599453


def _mutual_information_bits(
    counts: mx.array,
    total: mx.array,
    rows: mx.array,
    columns: mx.array,
    dtype: mx.Dtype,
) -> mx.array:
    """Mutual information in bits from exact int64 GLCM counts.

    Evaluates ``sum c*log(c*N/(row*col))/N``. The log ratio comes from the exact int64
    residual ``c*N - row*col``; for rows whose largest ratio is small, the leading
    residual sum is evaluated as the nonnegative Pearson term plus
    ``sum c*(log1p(r)-r)``, avoiding cancellation between positive and negative cells.
    Other rows evaluate the log ratio directly. With ``N < 2**31`` every int64 product
    stays below ``2**62``.
    """
    safe_total = mx.maximum(total, 1)
    total_f = safe_total.astype(dtype)
    counts_f = counts.astype(dtype)
    rows_f = mx.where(rows > 0, rows, 1).astype(dtype)
    columns_f = mx.where(columns > 0, columns, 1).astype(dtype)
    valid = (rows[:, :, None] > 0) & (columns[:, None, :] > 0)
    denominator = rows_f[:, :, None] * columns_f[:, None, :]
    residual = counts * total[:, None, None] - rows[:, :, None] * columns[:, None, :]
    residual_f = residual.astype(dtype)
    ratio = mx.where(valid, residual_f / denominator, 0)
    support = counts > 0
    # Pearson linear term X^2/N = sum r^2/(row*col)/N^2 is nonnegative, so its
    # small value is computed without subtracting comparable positive cells.
    linear = _sum_rows(
        mx.where(
            valid,
            (residual_f / rows_f[:, :, None]) * (residual_f / columns_f[:, None, :]),
            0,
        )
    ) / (total_f * total_f)
    # For |ratio| <= 0.5, log1p(ratio) - ratio is well conditioned as a power series.
    series_mask = mx.max(mx.where(valid, mx.abs(ratio), 0), axis=(1, 2)) <= 0.5
    series_ratio = mx.where(series_mask[:, None, None], ratio, 0)
    polynomial = mx.zeros_like(series_ratio)
    for k in range(24, 1, -1):
        coefficient = 1.0 / k if k % 2 else -1.0 / k
        polynomial = series_ratio * polynomial + coefficient
    correction = series_ratio * series_ratio * polynomial
    series = linear + _sum_rows(mx.where(support, counts_f * correction, 0)) / total_f
    # Direct evaluation is used when some ratio is too large for the series; log(q)
    # retains a rare negative-dependence cell that rounds to ratio == -1 in float32.
    quotient = counts_f * total[:, None, None] / denominator
    log_ratio = mx.where(
        quotient > 0.5,
        log1p(ratio),
        mx.log(mx.maximum(quotient, 1e-38)),
    )
    direct = _sum_rows(mx.where(support, counts_f * log_ratio, 0)) / total_f
    return mx.where(series_mask, series, direct) / _LN2


def _glcm_statistics(counts: mx.array, dtype: mx.Dtype) -> tuple[dict[str, mx.array], mx.array]:
    """Compute one direction's GLCM features and the row activity flag.

    Returns the feature mapping and a boolean ``active`` that is False for rows with
    no counted pair. MCC is excluded because it needs an eigensolver (see
    ``glcm_statistics``).
    """
    count = _sum_rows(counts.astype(dtype))
    p = counts.astype(dtype) / mx.maximum(count[:, None, None], 1)
    ng = counts.shape[1]
    levels = mx.arange(1, ng + 1, dtype=dtype)
    i, j = levels[None, :, None], levels[None, None, :]
    # Marginal means are exact rationals Sx/N of int64 counts. Centered coordinates are
    # formed from the exact int64 numerators i*N - Sx before one float32 rounding, so a
    # symmetric distribution yields exactly negated offsets and a gray-level translation
    # leaves them unchanged. A rounded float32 mean would contaminate
    # every offset and turn exact odd-moment cancellation into a nonzero ClusterShade.
    # Absolute indices remain in place for the features whose definitions use them.
    exact = counts.astype(mx.int64)
    total = mx.maximum(mx.sum(exact, axis=(1, 2)), 1)
    total_f = total.astype(dtype)
    int_levels = mx.arange(1, ng + 1, dtype=mx.int64)
    sx = mx.sum(mx.sum(exact, axis=2) * int_levels[None, :], axis=1)
    sy = mx.sum(mx.sum(exact, axis=1) * int_levels[None, :], axis=1)
    scaled_levels = int_levels[None, :] * total[:, None]
    dx = ((scaled_levels - sx[:, None]).astype(dtype) / total_f[:, None])[:, :, None]
    dy = ((scaled_levels - sy[:, None]).astype(dtype) / total_f[:, None])[:, None, :]
    cluster_numerator = (
        scaled_levels[:, :, None] + scaled_levels[:, None, :] - (sx + sy)[:, None, None]
    )
    cluster = cluster_numerator.astype(dtype) / total_f[:, None, None]
    ux = (sx.astype(dtype) / total_f)[:, None, None]
    vx = _sum_rows(p * dx**2)
    vy = _sum_rows(p * dy**2)
    scale = mx.sqrt(vx * vy)
    corr = _sum_rows(p * dx * dy) / (scale + EPS)
    delta = mx.abs(i - j)
    extra = _glcm_distributions(counts, p, levels, ux, dx, cluster, delta, dtype)
    difference_average = _sum_rows(p * delta)
    values = {
        **extra,
        "Autocorrelation": _sum_rows(p * i * j),
        "Contrast": _sum_rows(p * delta**2),
        # A singleton marginal is exactly constant, even if its rounded mean is not.
        "Correlation": mx.where(
            (mx.sum(mx.any(counts > 0, axis=2), axis=1) == 1)
            | (mx.sum(mx.any(counts > 0, axis=1), axis=1) == 1), 1, corr,
        ),
        "DifferenceAverage": difference_average,
        "Dissimilarity": difference_average,
        "JointEnergy": _sum_rows(p**2),
        "Id": _sum_rows(p / (1 + delta)),
        "Idm": _sum_rows(p / (1 + delta**2)),
    }
    # IBSI defines these names as the same quantities, not as alternate formulas.
    values["SumVariance"] = values["ClusterTendency"]
    return values, count > 0


type _Segment = tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]


@lru_cache(maxsize=64)
def _glcm_segments(ng: int) -> tuple[_Segment, ...]:
    """Return (order, starts, ends) grouping flattened GLCM cells by |i-j| and by i+j.

    Every difference 0..ng-1 and every sum 2..2*ng occurs at least once, so no
    group is empty.
    """
    index = np.arange(ng, dtype=np.intp)
    segments: list[_Segment] = []
    for key in (np.abs(index[:, None] - index[None, :]).ravel(),
                (index[:, None] + index[None, :]).ravel()):
        order = np.argsort(key, kind="stable")
        ends = np.searchsorted(
            key[order], np.arange(int(key.max()) + 1, dtype=np.intp), side="right",
        )
        starts = np.concatenate(([0], ends[:-1]))
        segments.append((order, starts, ends))
    return tuple(segments)


def _glcm_sum_difference_counts(exact: mx.array, ng: int) -> tuple[mx.array, mx.array]:
    """Sum int64 GLCM counts over each |i-j| and i+j group in one pass per group set.

    Cells are permuted so each group is contiguous; an exclusive int64 prefix sum then
    gives every group total as a difference of two prefix values. Integer addition is
    exact, so the result is identical to summing each group separately, without the
    3*ng full-matrix masked reductions.
    """
    flat = mx.reshape(exact, (exact.shape[0], ng * ng))
    zero = mx.zeros((flat.shape[0], 1), dtype=flat.dtype)
    totals: list[mx.array] = []
    for order, starts, ends in _glcm_segments(ng):
        prefix = mx.concatenate(
            [zero, mx.cumsum(mx.take(flat, mx.array(order), axis=1), axis=1)], axis=1,
        )
        totals.append(
            mx.take(prefix, mx.array(ends), axis=1) - mx.take(prefix, mx.array(starts), axis=1)
        )
    return totals[0], totals[1]


# Exact ClusterShade numerators use base-2**24 int64 limbs: limb products stay below
# 2**48, so short convolutions and gray-level sums never overflow int64, and every
# normalized limb converts to float32 exactly.
_LIMB_BITS = 24
_LIMB_MASK = (1 << _LIMB_BITS) - 1
_LIMB_COUNT = 9  # N < 2**31 and signed-int32 gray indices give |Q| < 2**190.
_SMALL_SHADE_LEVELS = 2**14


def _limbs(value: mx.array, count: int) -> list[mx.array]:
    """Split int64 ``value`` into ``count`` base-``2**24`` limbs, least significant first.

    Floor-mod/floor-div splitting keeps the lower limbs in ``[0, 2**24)``; the top
    limb keeps the sign.
    """
    limbs: list[mx.array] = []
    for _ in range(count - 1):
        limbs.append(mx.bitwise_and(value, _LIMB_MASK))
        value = mx.right_shift(value, _LIMB_BITS)
    return [*limbs, value]


def _normalized(limbs: list[mx.array], count: int) -> list[mx.array]:
    """Pad ``limbs`` to ``count`` entries and propagate carries into canonical limb form."""
    zero = mx.zeros_like(limbs[0])
    work = [*limbs, *([zero] * (count - len(limbs)))]
    for index in range(count - 1):
        work[index + 1] = work[index + 1] + mx.right_shift(work[index], _LIMB_BITS)
        work[index] = mx.bitwise_and(work[index], _LIMB_MASK)
    return work


def _product(left: list[mx.array], right: list[mx.array]) -> list[mx.array]:
    """Multiply two limb vectors without carry propagation (schoolbook convolution)."""
    terms = [mx.zeros_like(left[0]) for _ in range(len(left) + len(right) - 1)]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            terms[i + j] = terms[i + j] + a * b
    return terms


def cluster_shade(sum_counts: mx.array, total: mx.array, dtype: mx.Dtype) -> mx.array:
    """ClusterShade ``sum_k s_k (k - mu)**3 / N`` from exact int64 sum-distribution counts.

    ``s_k`` counts pairs with ``i + j = k`` and ``mu = S / N`` with ``S = sum_k k s_k``.
    Writing ``mu = m + f / N`` (``m = S // N``, ``0 <= f < N``) and ``a_k = k - m``, the
    moment is ``Q / N**3`` with the integer
    ``Q = N**2 A3 - 3 f N A2 + 2 f**3``, ``Ar = sum_k s_k a_k**r`` (``A1 = f``).
    ``Q`` is formed exactly in int64 limbs and rounded only by the final division, so
    cancelling moments are exactly zero and other values keep float32 relative accuracy.
    Rounding probabilities and cubic products first (then summing, compensated or not)
    already loses the cancellation: counts ``[2, 12, 2, 8]`` at offsets
    ``[-300, -100, 100, 200]`` have moment 0 but rounded terms summing to 0.0859375.

    Requires ``N < 2**31`` and signed-int32 gray indices. Larger lattices split
    powers into limbs before count weighting; no potentially overflowing square
    or cube is formed in a single int64 word.
    """
    width = sum_counts.shape[1]
    n = mx.maximum(total, 1)
    k = mx.arange(2, width + 2, dtype=mx.int64)[None, :]
    floor_mean = mx.floor_divide(mx.sum(sum_counts * k, axis=1), n)
    a = k - floor_mean[:, None]
    f = mx.sum(sum_counts * a, axis=1)
    if width <= 2 * _SMALL_SHADE_LEVELS - 1:
        weighted_square = sum_counts * a * a
        a2_limbs = _limbs(mx.sum(weighted_square, axis=1), 3)
        a3 = [mx.sum(limb * a, axis=1) for limb in _limbs(weighted_square, 3)]
        a3 = _normalized(a3, 5)
    else:
        # Each normalized power limb has magnitude below 2**24 (the signed
        # top limb is bounded likewise). Weighting and summing stays below
        # N * 2**24 < 2**55, independently of the number of occupied states.
        offsets = _limbs(a, 2)
        squares = _normalized(_product(offsets, offsets), 3)
        cubes = _normalized(_product(squares, offsets), 4)
        a2_limbs = _normalized(
            [mx.sum(sum_counts * limb, axis=1) for limb in squares], 5,
        )
        a3 = _normalized(
            [mx.sum(sum_counts * limb, axis=1) for limb in cubes], 6,
        )
    first = _product(_limbs(n * n, 3), a3)
    second = _product(_limbs(f * n, 3), a2_limbs)
    third = _product(_limbs(f * f, 3), _limbs(f, 2))
    zero = mx.zeros_like(n)
    q = [
        (first[i] if i < len(first) else zero)
        - 3 * (second[i] if i < len(second) else zero)
        + 2 * (third[i] if i < len(third) else zero)
        for i in range(max(len(first), len(second), len(third)))
    ]
    q = _normalized(q, _LIMB_COUNT)
    negative = q[-1] < 0
    magnitude = _normalized([mx.where(negative, -limb, limb) for limb in q], _LIMB_COUNT)
    # Horner from the top limb, dividing each limb by N**3 first: every partial value is
    # at most the final moment, so nothing overflows or flushes to a subnormal.
    n_f = n.astype(dtype)
    cube = n_f * n_f * n_f
    value = mx.zeros(n.shape, dtype=dtype)
    for limb in reversed(magnitude):
        value = value * float(1 << _LIMB_BITS) + limb.astype(dtype) / cube
    return mx.where(negative, -value, value)


def one_minus_exp(y: mx.array) -> mx.array:
    """Return ``1 - exp(-y)`` for ``y >= 0`` to a few float32 ulps.

    MLX's GPU ``expm1`` is about ``3.5e-6`` relative off for arguments in ``[0.1, 1]``
    (MLX 0.32.4) but accurate below that. ``u = expm1(-y/16)`` stays in the accurate
    range for the GLCM ``y = 2 MI <= 28``, and ``exp(-2t) - 1 = u (u + 2)`` with
    ``u = exp(-t) - 1`` doubles back without cancellation, since ``u + 2`` lies in
    ``(1, 2]``. Over ``y`` in ``[0, 30]`` the worst error is ``2.9e-7`` relative.
    """
    u = mx.expm1(y * -0.0625)
    for _ in range(4):
        u = u * (u + 2)
    return -u


def _glcm_distributions(
    counts: mx.array,
    p: mx.array,
    levels: mx.array,
    ux: mx.array,
    dx: mx.array,
    cluster: mx.array,
    delta: mx.array,
    dtype: mx.Dtype,
) -> dict[str, mx.array]:
    """GLCM features from integer-derived centered offsets.

    ``dx`` is ``i - mu_x`` and ``cluster`` is ``i + j - mu_x - mu_y``, each rounded once
    from exact int64 numerators (see ``_glcm_statistics``). The odd ClusterShade moment
    is evaluated exactly from integer sum-distribution counts (see ``cluster_shade``);
    including lattices with large gaps between occupied gray levels.
    """
    ng = p.shape[1]
    i, j = levels[None, :, None], levels[None, None, :]
    exact = counts.astype(mx.int64)
    total = mx.sum(exact, axis=(1, 2))
    total_f = mx.maximum(total, 1).astype(dtype)
    # Sum and difference distributions stay as integer counts until entropy is
    # evaluated; converting to probabilities first rounds the dominant state to one and
    # deletes its small positive entropy contribution.
    difference_counts, sum_counts = _glcm_sum_difference_counts(exact, ng)
    differences = difference_counts.astype(dtype) / total_f[:, None]
    rows = mx.sum(exact, axis=2)
    columns = mx.sum(exact, axis=1)
    joint_entropy = count_entropy(mx.reshape(exact, (exact.shape[0], -1)), total, dtype)
    hx = count_entropy(rows, total, dtype)
    hy = count_entropy(columns, total, dtype)
    information = _mutual_information_bits(exact, total, rows, columns, dtype)
    denominator = mx.maximum(hx, hy)
    mean_difference = _sum_rows(p * delta)
    k = mx.arange(ng, dtype=p.dtype)[None, :]
    shade = {"ClusterShade": cluster_shade(sum_counts, total, dtype)}
    return {
        **shade,
        "JointAverage": ux[:, 0, 0],
        "ClusterProminence": _sum_rows(p * cluster**4),
        "ClusterTendency": _sum_rows(p * cluster**2),
        "DifferenceEntropy": count_entropy(difference_counts, total, dtype),
        "DifferenceVariance": _sum_rows(differences * (k - mean_difference[:, None])**2),
        # Imc1 = -MI/max(H(px), H(py)); MI is zero for independence and equals HXY
        # for a diagonal GLCM, so a two-state diagonal is exactly -1.
        "Imc1": mx.where(
            denominator > 0,
            -information / mx.where(denominator > 0, denominator, 1),
            0,
        ),
        "Imc2": mx.sqrt(mx.maximum(one_minus_exp(2 * information), 0)),
        "Idmn": _sum_rows(p / (1 + delta**2 / ng**2)),
        "Idn": _sum_rows(p / (1 + delta / ng)),
        "InverseVariance": _sum_rows(mx.where(delta > 0, p / mx.maximum(delta**2, 1), 0)),
        "JointEntropy": joint_entropy,
        "MaximumProbability": mx.max(p, axis=(1, 2)),
        "SumAverage": _sum_rows(p * (i + j)),
        "SumEntropy": count_entropy(sum_counts, total, dtype),
        "SumSquares": _sum_rows(p * dx**2),
    }


@lru_cache(maxsize=32)
def _compiled_glcm(
    dtype: mx.Dtype, features: tuple[str, ...],
) -> Callable[[mx.array], tuple[dict[str, mx.array], mx.array]]:
    """Return a compiled GLCM feature function for ``dtype`` and ``features``.

    The requested outputs are selected before tracing so unrequested feature
    arithmetic is pruned from the compiled graph.
    """
    @mx.compile
    def calculate(counts: mx.array) -> tuple[dict[str, mx.array], mx.array]:
        values, active = _glcm_statistics(counts, dtype)
        return {name: values[name] for name in features}, active

    return calculate


def glcm_statistics(
    counts: mx.array, dtype: mx.Dtype, features: tuple[str, ...],
) -> tuple[dict[str, mx.array], mx.array]:
    """Compute the requested GLCM features for one direction's counts.

    With fewer than 2**30 voxels, symmetrization gives a total ``N < 2**31``; the int64
    products used by the information features and MCC then stay below ``2**62``.

    Args:
        counts: GLCM counts ``(batch, Ng, Ng)``, already symmetrized if requested.
        dtype: Arithmetic dtype (float32).
        features: Feature names without the ``glcm_`` prefix.

    Returns:
        ``(values, active)``: a name-to-``(batch,)`` mapping, and a boolean ``(batch,)``
        that is False where the direction had no counted pair. The caller averages values
        over active directions, as PyRadiomics does.
    """
    ordinary = tuple(name for name in features if name != "MCC")
    values, active = _compiled_glcm(dtype, ordinary)(counts)
    if "MCC" in features:
        values["MCC"] = maximal_correlation(counts)
    return values, active

