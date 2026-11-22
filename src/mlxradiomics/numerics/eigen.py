"""Symmetric eigenspectra by cyclic Jacobi rotations entirely on Metal."""

from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

# A 64-square float32 matrix uses 16 KiB, half the Apple silicon threadgroup budget.
_LOCAL_MATRIX_SIZE = 64


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
    elements = size * size
    local = size <= _LOCAL_MATRIX_SIZE
    storage = (
        f"threadgroup float work[{elements}];" if local
        else "device float* work = workspace + base;"
    )
    memory = "mem_threadgroup" if local else "mem_device"
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_jacobi",
            input_names=["matrix"],
            output_names=["eigenvalues", "converged"] if local else [
                "workspace", "eigenvalues", "converged"
            ],
            source=f"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        uint base = row * {elements};
        {storage}
        for (uint k = lane; k < {elements}; k += {lanes}) work[k] = matrix[base + k];
        threadgroup_barrier(mem_flags::{memory});
        bool done = false;
        for (int sweep = 0; sweep < 100; sweep++) {{
            done = true;
            for (int p = 0; p < {size}; p++) {{
                for (int q = p + 1; q < {size}; q++) {{
                    float apq = work[p * {size} + q];
                    float app = work[p * {size} + p];
                    float aqq = work[q * {size} + q];
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
                    threadgroup_barrier(mem_flags::{memory});
                    for (uint k = lane; k < {size}; k += {lanes}) {{
                        if (k == uint(p) || k == uint(q)) continue;
                        float akp = work[k * {size} + p];
                        float akq = work[k * {size} + q];
                        work[k * {size} + p] = work[p * {size} + k] = c * akp - s * akq;
                        work[k * {size} + q] = work[q * {size} + k] = s * akp + c * akq;
                    }}
                    if (lane == 0) {{
                        work[p * {size} + p] = app - t * apq;
                        work[q * {size} + q] = aqq + t * apq;
                        work[p * {size} + q] = work[q * {size} + p] = 0;
                    }}
                    threadgroup_barrier(mem_flags::{memory});
                }}
            }}
            if (done) break;
        }}
        for (uint p = lane; p < {size}; p += {lanes})
            eigenvalues[row * {size} + p] = work[p * {size} + p];
        if (lane == 0) converged[row] = done;
        """,
        ),
    )


# The padded matrix and rotation data fit in 32 KiB of threadgroup storage.
_PARALLEL_MAX_SIZE = 88
_PARALLEL_LANES = 256


@lru_cache(maxsize=32)
def _parallel_jacobi_kernel(size: int) -> MetalKernel:
    """Rotate disjoint pairs of axes together in threadgroup memory."""
    width = size + size % 2
    pairs = width // 2
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_jacobi_parallel",
            input_names=["matrix"],
            output_names=["eigenvalues", "converged"],
            source=f"""
        constexpr uint n = {size};
        constexpr uint width = {width};
        // An odd stride separates column accesses across memory banks.
        constexpr uint stride = width + 1;
        constexpr uint anchor = width - 1;
        constexpr uint pairs = {pairs};
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        threadgroup float work[width * stride];
        threadgroup uint2 axes[pairs];
        threadgroup float2 rotation[pairs];
        threadgroup bool rotated[pairs];
        threadgroup bool done;

        // An extra zero axis lets an odd-sized matrix use complete pairings.
        for (uint k = lane; k < width * width; k += {_PARALLEL_LANES}) {{
            uint i = k / width, j = k % width;
            work[i * stride + j] = (i < n && j < n) ? matrix[(row * n + i) * n + j] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sweep = 0; sweep < 100; ++sweep) {{
            bool changed = false;
            // Rotate all disjoint pairs in one round. Holding one axis
            // fixed and rotating the others visits every pair per sweep.
            for (uint round = 0; round < anchor; ++round) {{
                if (lane < pairs) {{
                    uint p = lane == 0 ? anchor : (round + lane) % anchor;
                    uint q = (round + anchor - lane) % anchor;
                    axes[lane] = uint2(p, q);
                    float apq = work[p * stride + q];
                    float app = work[p * stride + p];
                    float aqq = work[q * stride + q];
                    float threshold = (1e-7f * sqrt(abs(app))) * sqrt(abs(aqq));
                    float2 cs = float2(1.0f, 0.0f);
                    if (!(abs(apq) <= threshold)) {{
                        changed = true;
                        float delta = 0.5f * aqq - 0.5f * app;
                        float scale = max(abs(delta), abs(apq));
                        float d = delta / scale, off = apq / scale;
                        float t = copysign(1.0f, delta) * off /
                                  (abs(d) + sqrt(d * d + off * off));
                        float c = rsqrt(1 + t * t);
                        cs = float2(c, t * c);
                        work[p * stride + p] = app - t * apq;
                        work[q * stride + q] = aqq + t * apq;
                        work[p * stride + q] = work[q * stride + p] = 0;
                    }}
                    rotation[lane] = cs;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Each off-diagonal 2x2 block has one writer, so these
                // two-sided updates can run in place.
                for (uint block = lane; block < pairs * pairs; block += {_PARALLEL_LANES}) {{
                    uint i = block / pairs, j = block % pairs;
                    if (i >= j) continue;
                    float2 a = rotation[i], b = rotation[j];
                    if (a.y == 0 && b.y == 0) continue;
                    uint2 left = axes[i], right = axes[j];
                    uint p = left.x, q = left.y, r = right.x, s = right.y;
                    float pr = work[p * stride + r], ps = work[p * stride + s];
                    float qr = work[q * stride + r], qs = work[q * stride + s];
                    float x = a.x * pr - a.y * qr, y = a.x * ps - a.y * qs;
                    float z = a.x * qr + a.y * pr, w = a.x * qs + a.y * ps;
                    work[p * stride + r] = work[r * stride + p] = b.x * x - b.y * y;
                    work[p * stride + s] = work[s * stride + p] = b.x * y + b.y * x;
                    work[q * stride + r] = work[r * stride + q] = b.x * z - b.y * w;
                    work[q * stride + s] = work[s * stride + q] = b.x * w + b.y * z;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (lane < pairs) rotated[lane] = changed;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {{
                bool any_rotated = false;
                for (uint k = 0; k < pairs; ++k) any_rotated |= rotated[k];
                done = !any_rotated;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (done) break;
        }}
        if (lane < n) eigenvalues[row * n + lane] = work[lane * stride + lane];
        if (lane == 0) converged[row] = done;
            """,
        ),
    )


def symmetric_eigenvalues(matrix: mx.array) -> mx.array:
    """Return the eigenvalues of a batch of symmetric matrices in ascending order.

    The kernel runs one threadgroup per matrix in float32 and does not check symmetry.
    Small matrices use cyclic rotations in threadgroup memory. Medium matrices
    rotate disjoint axis pairs in parallel there. Larger matrices use a device buffer.
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
    parallel = _LOCAL_MATRIX_SIZE < size <= _PARALLEL_MAX_SIZE
    lanes = _PARALLEL_LANES if parallel else _lanes(size)
    shapes: list[tuple[int, ...]] = [(matrix.shape[0], size), (matrix.shape[0],)]
    dtypes = [mx.float32, mx.bool_]
    if not parallel and size > _LOCAL_MATRIX_SIZE:
        shapes.insert(0, matrix.shape)
        dtypes.insert(0, mx.float32)
    kernel = _parallel_jacobi_kernel(size) if parallel else _jacobi_kernel(size, lanes)
    *_, values, converged = kernel(
        inputs=[matrix],
        grid=(matrix.shape[0] * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
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
