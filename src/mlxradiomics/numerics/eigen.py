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


def symmetric_eigenvalues(matrix: mx.array) -> mx.array:
    """Return the eigenvalues of a batch of symmetric matrices in ascending order.

    The kernel runs one threadgroup per matrix in float32 and does not check symmetry.
    Small matrices fit in threadgroup memory; larger matrices use a device buffer.
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
    shapes: list[tuple[int, ...]] = [(matrix.shape[0], size), (matrix.shape[0],)]
    dtypes = [mx.float32, mx.bool_]
    if size > _LOCAL_MATRIX_SIZE:
        shapes.insert(0, matrix.shape)
        dtypes.insert(0, mx.float32)
    *_, values, converged = _jacobi_kernel(size, lanes)(
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


# Floats of threadgroup storage the largest-eigenvalue kernel may use (32 KiB).
_THREADGROUP_FLOATS = 8192


def _largest_lanes(size: int) -> int:
    """Threads per matrix: whole SIMD groups, the next power of two of ``size``, at most 256."""
    lanes = 32
    while lanes < min(size, 256):
        lanes *= 2
    return lanes


def _largest_local(size: int, lanes: int) -> bool:
    """Whether the matrix, two vectors and the lane scratch fit in threadgroup memory."""
    return size * size + 2 * size + lanes + lanes // 32 + 4 <= _THREADGROUP_FLOATS


@lru_cache(maxsize=32)
def _largest_kernel(size: int, lanes: int) -> MetalKernel:
    """Compile the largest-eigenvalue kernel for ``size`` x ``size`` symmetric matrices."""
    # One threadgroup per matrix. The matrix is first scaled by a power of two so its
    # largest magnitude lies in [0.5, 1), which is exact and keeps squares away from
    # overflow and underflow. Householder reflections then reduce it to a symmetric
    # tridiagonal matrix (d, e): reflection k zeroes column k below the subdiagonal and
    # updates the trailing block as A -= v w^T + w v^T, with p = beta A v and
    # w = p - (beta v.p / 2) v. Lanes split the reductions, the matrix-vector product
    # and the rank-two update; every lane derives the same scalars from the same
    # threadgroup partials in the same order, so branches and barriers stay uniform.
    # The largest eigenvalue lies between max(d) and the Gershgorin bound. Each round
    # evaluates the Sturm count at ``lanes`` interior points, one per lane, and keeps
    # the subinterval where the count first reaches ``size``; rounds stop when the
    # interval no longer shrinks in float32 or after 16 rounds.
    elements = size * size
    simd_groups = lanes // 32
    local = _largest_local(size, lanes)
    storage = (
        f"threadgroup float work[{elements}];" if local
        else "device float* work = workspace + base;"
    )
    memory = "mem_flags::mem_threadgroup" if local else (
        "mem_flags::mem_threadgroup | mem_flags::mem_device"
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_largest_eigenvalue",
            input_names=["matrix"],
            output_names=["largest"] if local else ["workspace", "largest"],
            source=f"""
        const int N = {size};
        const uint L = {lanes};
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_id = simdgroup_index_in_threadgroup;
        uint base = row * {elements};
        {storage}
        threadgroup float v[N];
        threadgroup float w[N];
        threadgroup float partial[{simd_groups}];
        threadgroup int reached[L];

        float magnitude = 0.0f;
        for (uint k = lane; k < {elements}; k += L) {{
            float value = matrix[base + k];
            work[k] = value;
            magnitude = max(magnitude, abs(value));
        }}
        magnitude = simd_max(magnitude);
        if (simd_lane == 0) partial[simd_id] = magnitude;
        threadgroup_barrier({memory});
        magnitude = partial[0];
        for (uint g = 1; g < {simd_groups}; g++) magnitude = max(magnitude, partial[g]);
        if (magnitude == 0.0f) {{
            // A zero matrix has only zero eigenvalues.
            if (lane == 0) largest[row] = 0.0f;
            return;
        }}
        int exponent;
        frexp(magnitude, exponent);
        float down = ldexp(1.0f, -exponent);
        threadgroup_barrier({memory});
        for (uint k = lane; k < {elements}; k += L) work[k] *= down;
        threadgroup_barrier({memory});

        for (int k = 0; k < N - 2; k++) {{
            int first = k + 1;
            int count = N - first;
            float sigma = 0.0f;
            for (int i = first + 1 + int(lane); i < N; i += int(L)) {{
                float x = work[i * N + k];
                sigma += x * x;
            }}
            sigma = simd_sum(sigma);
            if (simd_lane == 0) partial[simd_id] = sigma;
            threadgroup_barrier({memory});
            sigma = 0.0f;
            for (uint g = 0; g < {simd_groups}; g++) sigma += partial[g];
            if (sigma == 0.0f) {{
                // Column k is already tridiagonal; its subdiagonal stays in place.
                threadgroup_barrier({memory});
                continue;
            }}
            float x0 = work[first * N + k];
            float alpha = sqrt(x0 * x0 + sigma);
            if (x0 > 0.0f) alpha = -alpha;
            float v0 = x0 - alpha;
            float beta = 2.0f / (sigma + v0 * v0);
            for (int i = first + int(lane); i < N; i += int(L))
                v[i] = i == first ? v0 : work[i * N + k];
            threadgroup_barrier({memory});
            float dot = 0.0f;
            for (int i = first + int(lane); i < N; i += int(L)) {{
                float product = 0.0f;
                for (int j = first; j < N; j++) product += work[i * N + j] * v[j];
                product *= beta;
                w[i] = product;
                dot += v[i] * product;
            }}
            dot = simd_sum(dot);
            if (simd_lane == 0) partial[simd_id] = dot;
            threadgroup_barrier({memory});
            dot = 0.0f;
            for (uint g = 0; g < {simd_groups}; g++) dot += partial[g];
            float half_k = 0.5f * beta * dot;
            for (int i = first + int(lane); i < N; i += int(L)) w[i] -= half_k * v[i];
            threadgroup_barrier({memory});
            for (int index = int(lane); index < count * count; index += int(L)) {{
                int i = first + index / count;
                int j = first + index % count;
                work[i * N + j] -= v[i] * w[j] + w[i] * v[j];
            }}
            if (lane == 0) work[first * N + k] = alpha;
            threadgroup_barrier({memory});
        }}

        // Diagonal in v, |subdiagonal| in w. The largest eigenvalue is at least max(d)
        // and at most the Gershgorin bound, widened by a few ulps for its rounding.
        for (int i = int(lane); i < N; i += int(L)) {{
            v[i] = work[i * N + i];
            w[i] = i + 1 < N ? abs(work[(i + 1) * N + i]) : 0.0f;
        }}
        threadgroup_barrier({memory});
        float lo = v[0], hi = -INFINITY, previous = 0.0f, largest_e = 0.0f;
        for (int i = 0; i < N; i++) {{
            lo = max(lo, v[i]);
            hi = max(hi, v[i] + previous + w[i]);
            previous = w[i];
            largest_e = max(largest_e, w[i]);
        }}
        float pivmin = FLT_MIN * max(1.0f, largest_e * largest_e);
        hi += 4.0f * FLT_EPSILON * max(abs(hi), abs(lo)) + pivmin;
        for (int round = 0; round < 16 && lo < hi; round++) {{
            float x = lo + (hi - lo) * (float(lane + 1) / float(L + 1));
            float q = v[0] - x;
            if (abs(q) <= pivmin) q = -pivmin;
            int below = q < 0.0f ? 1 : 0;
            for (int i = 1; i < N; i++) {{
                q = (v[i] - x) - (w[i - 1] * w[i - 1]) / q;
                if (abs(q) <= pivmin) q = -pivmin;
                below += q < 0.0f ? 1 : 0;
            }}
            reached[lane] = below == N ? 1 : 0;
            threadgroup_barrier({memory});
            float next_lo = lo, next_hi = hi;
            for (uint l = 0; l < L; l++) {{
                float point = lo + (hi - lo) * (float(l + 1) / float(L + 1));
                if (reached[l]) {{ next_hi = point; break; }}
                next_lo = point;
            }}
            threadgroup_barrier({memory});
            bool stalled = next_lo == lo && next_hi == hi;
            lo = next_lo;
            hi = next_hi;
            if (stalled) break;
        }}
        if (lane == 0) largest[row] = ldexp(0.5f * (lo + hi), exponent);
        """,
        ),
    )


def largest_symmetric_eigenvalue(matrix: mx.array) -> mx.array:
    """Return the largest eigenvalue of each matrix in a batch of symmetric matrices.

    One threadgroup per matrix reduces it to tridiagonal form with Householder
    reflections in float32 and brackets the largest eigenvalue by parallel Sturm-count
    bisection. The bracket stops shrinking at float32 resolution, so the result is the
    midpoint of an interval a few ulps wide around the tridiagonal matrix's largest
    eigenvalue. No iteration can fail to converge, so the host never waits on a
    convergence flag. Symmetry is not checked; the full matrix is read.

    Args:
        matrix: Float32 array ``(batch, n, n)``.

    Returns:
        Float32 array ``(batch,)``.

    Raises:
        ValueError: If ``matrix`` is not a batch of float32 square matrices.
    """
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("expected batched square symmetric matrices")
    size = matrix.shape[1]
    if matrix.shape[0] == 0:
        return mx.zeros((0,), dtype=mx.float32)
    if matrix.dtype != mx.float32:
        raise ValueError("expected float32 matrices")
    if size == 1:
        return matrix[:, 0, 0]
    lanes = _largest_lanes(size)
    shapes: list[tuple[int, ...]] = [(matrix.shape[0],)]
    dtypes = [mx.float32]
    if not _largest_local(size, lanes):
        shapes.insert(0, matrix.shape)
        dtypes.insert(0, mx.float32)
    *_, largest = _largest_kernel(size, lanes)(
        inputs=[matrix],
        grid=(matrix.shape[0] * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return largest


def maximal_correlation(counts: mx.array, *, group: int | None = None) -> mx.array:
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

    MLX selects its matrix-multiplication algorithm from the batch size, so one Gram
    product over a larger batch can round differently from separate products. A caller
    that stacks several directions passes ``group``, the row count of each original
    batch; each group then forms its Gram matrices in its own product, as a separate
    call would, while the eigenvalue kernel processes every row in one dispatch.

    Args:
        counts: GLCM counts ``(batch, Ng, Ng)`` for a single direction. ``Ng`` is the
            global gray-level count; validity uses only the occupied entries of each
            matrix.
        group: Rows per stacked direction; ``batch`` must be a multiple of it. None
            forms all Gram matrices in one product.

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
    if group is None or group >= counts.shape[0]:
        gram = normalized @ mx.swapaxes(normalized, 1, 2)
    else:
        if group <= 0 or counts.shape[0] % group:
            raise ValueError("MCC group must divide the batch")
        parts = [
            normalized[start:start + group]
            for start in range(0, counts.shape[0], group)
        ]
        gram = mx.concatenate([part @ mx.swapaxes(part, 1, 2) for part in parts])
    mcc = mx.sqrt(mx.maximum(largest_symmetric_eigenvalue(gram), 0))
    return mx.where(defined, mcc, float("nan"))
