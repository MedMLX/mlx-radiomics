"""Symmetric eigenspectra by cyclic Jacobi rotations entirely on Metal."""

from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx


def _lanes(size: int) -> int:
    """Threads cooperating on one matrix: the next power of two of ``size``, capped at 128.

    The lanes split each rotation's row and column updates.
    """
    lanes = 1
    while lanes < min(size, 128):
        lanes *= 2
    return lanes


@lru_cache(maxsize=32)
def _jacobi_kernel(size: int, lanes: int) -> MetalKernel:
    """Compile the cyclic Jacobi eigenvalue kernel for ``size`` x ``size`` matrices."""
    # One threadgroup per matrix. Rotations keep the serial cyclic order; the lanes split
    # only each rotation's row/column update, whose ``k`` entries are disjoint, so every
    # element sees the same float32 operations as a one-thread sweep. All lanes read the
    # same pivot values and therefore take the same skip and convergence branches, which
    # keeps the barriers uniform. The first barrier stops lane 0 overwriting the pivots
    # before another lane has read them; the second publishes the rotation.
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_jacobi",
            input_names=["matrix"],
            output_names=["work", "eigenvalues", "converged"],
            source=f"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        uint base = row * {size * size};
        for (uint k = lane; k < {size * size}; k += {lanes}) work[base + k] = matrix[base + k];
        threadgroup_barrier(mem_flags::mem_device);
        bool done = false;
        for (int sweep = 0; sweep < 100; sweep++) {{
            done = true;
            for (int p = 0; p < {size}; p++) {{
                for (int q = p + 1; q < {size}; q++) {{
                    float apq = work[base + p * {size} + q];
                    float app = work[base + p * {size} + p];
                    float aqq = work[base + q * {size} + q];
                    // A large unrelated diagonal must not hide a small correlated block.
                    float threshold = (1e-7f * sqrt(abs(app))) * sqrt(abs(aqq));
                    if (abs(apq) <= threshold) continue;
                    done = false;
                    float delta = 0.5f * aqq - 0.5f * app;
                    float scale = max(abs(delta), abs(apq));
                    float d = delta / scale, off = apq / scale;
                    float t = copysign(1.0f, delta) * off /
                              (abs(d) + sqrt(d * d + off * off));
                    float c = rsqrt(1 + t * t), s = t * c;
                    threadgroup_barrier(mem_flags::mem_device);
                    for (uint k = lane; k < {size}; k += {lanes}) {{
                        if (k == uint(p) || k == uint(q)) continue;
                        float akp = work[base + k * {size} + p];
                        float akq = work[base + k * {size} + q];
                        work[base + k * {size} + p] =
                            work[base + p * {size} + k] = c * akp - s * akq;
                        work[base + k * {size} + q] =
                            work[base + q * {size} + k] = s * akp + c * akq;
                    }}
                    if (lane == 0) {{
                        work[base + p * {size} + p] = app - t * apq;
                        work[base + q * {size} + q] = aqq + t * apq;
                        work[base + p * {size} + q] = work[base + q * {size} + p] = 0;
                    }}
                    threadgroup_barrier(mem_flags::mem_device);
                }}
            }}
            if (done) break;
        }}
        for (uint p = lane; p < {size}; p += {lanes})
            eigenvalues[row * {size} + p] = work[base + p * {size} + p];
        if (lane == 0) converged[row] = done;
        """,
        ),
    )


def symmetric_eigenvalues(matrix: mx.array) -> mx.array:
    """Return the eigenvalues of a batch of symmetric matrices in ascending order.

    The kernel runs one threadgroup per matrix in float32 and does not check symmetry.
    A sweep converges when every off-diagonal magnitude is at most ``1e-7`` times
    the geometric mean of its two diagonal magnitudes. This pair-local criterion
    resolves small correlated blocks independently of other axes' scales. Rotation
    coefficients use scaled operands to avoid squaring an overflowing Jacobi ratio.

    Args:
        matrix: Float32 array ``(batch, n, n)``.

    Returns:
        Float32 array ``(batch, n)``.

    Raises:
        ValueError: If ``matrix`` is not a batch of square matrices.
        RuntimeError: If any matrix has not converged after 100 sweeps.
    """
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("Jacobi expects batched square symmetric matrices")
    size = matrix.shape[1]
    if matrix.shape[0] == 0:
        # An empty grid has nothing to compute, and Metal faults on an unallocated input.
        return mx.zeros((0, size), dtype=mx.float32)
    if matrix.dtype != mx.float32:
        raise ValueError("Jacobi expects float32 matrices")
    if size == 1:
        # A singleton spectrum has no off-diagonal entries or rotations, and its
        # sole value is already sorted. Retain the original float32 words directly.
        return matrix[:, 0, :]
    lanes = _lanes(size)
    _, values, converged = _jacobi_kernel(size, lanes)(
        inputs=[matrix],
        grid=(matrix.shape[0] * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=[matrix.shape, (matrix.shape[0], size), (matrix.shape[0],)],
        output_dtypes=[mx.float32, mx.float32, mx.bool_],
    )
    evaluate(values, converged)
    if not bool(mx.all(converged).item()):
        raise RuntimeError("Metal Jacobi eigenvalue calculation did not converge")
    return mx.sort(values, axis=1)


def maximal_correlation(counts: mx.array) -> mx.array:
    """Compute the GLCM Maximal Correlation Coefficient for one direction per batch row.

    For a contingency table with joint probability ``p`` and marginals ``px``, ``py``,
    MCC is the second singular value of ``p / sqrt(px py)``. That matrix has a fixed unit
    singular component along ``sqrt(px)`` and ``sqrt(py)``; the centered standardized
    residual ``(c N - row*col) / (N sqrt(row*col))`` removes it exactly before the
    Gram matrix is formed, so weak dependence is not rounded away by the dominant unit
    component. The numerator is evaluated in int64 before conversion to float32; with a
    supported total ``N < 2**31`` its products stay below ``2**62``.

    A second singular value exists only when the occupied row and column marginal states
    span at least two levels, so an empty matrix, a flat matrix, or a matrix with a
    single occupied row or column state is undefined rather than assigned a numerical
    convention.

    Args:
        counts: GLCM counts ``(batch, Ng, Ng)`` for a single direction. ``Ng`` is the
            global gray-level count; validity uses only the occupied entries of each
            matrix.

    Returns:
        Float32 array ``(batch,)``. Entries whose matrix has fewer than two occupied row
        or column states are NaN. Other entries are the largest singular value of the
        centered standardized residual, which is a genuine zero for exact independence.
    """
    if counts.shape[1] == 1:
        return mx.full((counts.shape[0],), float("nan"), dtype=mx.float32)
    occupied = counts > 0
    defined = (mx.sum(mx.any(occupied, axis=2).astype(mx.int32), axis=1) >= 2) & (
        mx.sum(mx.any(occupied, axis=1).astype(mx.int32), axis=1) >= 2
    )
    exact = counts.astype(mx.int64)
    total = mx.sum(exact, axis=(1, 2))
    rows = mx.sum(exact, axis=2)
    columns = mx.sum(exact, axis=1)
    total_f = total.astype(mx.float32)
    rows_f = rows.astype(mx.float32)
    columns_f = columns.astype(mx.float32)
    valid = (rows[:, :, None] > 0) & (columns[:, None, :] > 0)
    numerator = exact * total[:, None, None] - rows[:, :, None] * columns[:, None, :]
    denominator = total_f[:, None, None] * mx.sqrt(
        mx.where(valid, rows_f[:, :, None] * columns_f[:, None, :], 1)
    )
    normalized = mx.where(
        valid, numerator.astype(mx.float32) / mx.where(valid, denominator, 1), 0
    )
    gram = normalized @ mx.swapaxes(normalized, 1, 2)
    mcc = mx.sqrt(mx.maximum(symmetric_eigenvalues(gram)[:, -1], 0))
    return mx.where(defined, mcc, float("nan"))
