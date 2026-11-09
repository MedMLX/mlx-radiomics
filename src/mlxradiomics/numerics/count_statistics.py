"""Entropy of integer count distributions, shared by every feature family.

The source probabilities of a dominant state can round to one in float32, which
silently deletes that state's small positive entropy contribution.  Keeping counts
in int64 until the exact deficit is formed avoids that rounding; normalisation
happens only inside the logarithm.
"""

from mlxradiomics.numerics.reduction import sum_rows
from mlxradiomics.numerics.runtime import mx

_LN2 = 0.6931471805599453


def log1p(x: mx.array) -> mx.array:
    """Return ``log(1 + x)`` accurately for small ``x``, also inside ``mx.compile``.

    MLX's Metal ``log1p`` uses the compensation ``x*log(1+x)/((1+x)-1)``. In fused
    kernels the Metal compiler may simplify ``(1+x)-1`` to ``x``, which leaves the naive
    ``log(1+x)`` and relative errors of tens of percent near ``|x| = 1e-7``. For
    ``|x| <= 0.5`` this evaluates
    ``2*atanh(x/(2+x))`` as an odd series with ``|t| <= 1/3``, which has no
    cancellation for a compiler to remove; the truncation error is below ``1e-10``
    relative. Larger arguments use ``mx.log1p``, where ``1 + x`` has no cancellation.
    """
    small = mx.abs(x) <= 0.5
    near = mx.where(small, x, 0)
    t = near / (2 + near)
    t2 = t * t
    polynomial = mx.full(t.shape, 1.0 / 19, dtype=t.dtype)
    for k in range(8, -1, -1):
        polynomial = polynomial * t2 + 1.0 / (2 * k + 1)
    return mx.where(small, 2 * t * polynomial, mx.log1p(mx.where(small, 1, x)))


def count_entropy(counts: mx.array, total: mx.array, dtype: mx.Dtype) -> mx.array:
    """Return Shannon entropy in bits of integer counts normalised by row totals.

    Args:
        counts: Nonnegative integer array ``(batch, states)``.  Additional trailing
            axes are not interpreted; flatten them before calling.
        total: Integer array ``(batch,)`` holding each row's sum.  It may be an
            independent exact total (for example a marginal count) as long as the
            row's probabilities are ``counts / total``.
        dtype: Floating-point arithmetic dtype (float32 in this package).

    Returns:
        ``(batch,)`` entropy in bits.  Zero counts contribute the limit zero, and an
        empty row (``total == 0``) contributes zero rather than producing NaN.

    Notes:
        Counts and totals must be nonnegative and ``counts <= total``.  The exact
        integer deficit ``counts - total`` is formed before conversion.  When a
        state holds more than half the mass, ``log1p`` of that deficit evaluates
        the logarithm near one without materialising a probability that rounds to
        one; other occupied states use direct ``log2(count / total)``.
    """
    counts = counts.astype(mx.int64)
    total = total.astype(mx.int64)
    safe_total = mx.maximum(total, 1)
    total_f = safe_total.astype(dtype)
    counts_f = counts.astype(dtype)
    ratio = (counts - total[:, None]).astype(dtype) / total_f[:, None]
    log_probability = mx.where(
        counts * 2 > total[:, None],
        log1p(ratio) / _LN2,
        mx.log2(mx.maximum(counts_f, 1) / total_f[:, None]),
    )
    log_probability = mx.where(counts > 0, log_probability, 0)
    return -sum_rows(counts_f * log_probability) / total_f
