"""Minimum-volume and covariance ellipsoids, areas and densities."""

from functools import lru_cache
from math import frexp, ldexp, pi
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import DS_HEADER, scale_pow2
from mlxradiomics.features.geometry.hull import scipy_hull
from mlxradiomics.features.geometry.mesh import MeshGeometry
from mlxradiomics.features.geometry.moments import spectrum_kernel
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

_MVEE_TOLERANCE = 1e-3


_MVEE_MAX_ITERATIONS = 100_000


# Khachiyan steps per ascent launch; the bound keeps each launch short for large hulls
# and the 100,000-step cap exact.
_MVEE_STEPS_PER_LAUNCH = 1024


# Weighted center and second moments of a hull point cloud. Both the Khachiyan step and
# the final covariance use this reduction: a bare float32 covariance rounds the small
# eigenvalue of a thin shape away, and the cofactor inverse then accepts percent-level
# axis errors. Each product and sum carries a float-float low word, so the centered
# covariance keeps about 48 significant bits on Metal. The caller declares the
# threadgroup accumulators (_MVEE_ACCUMULATORS) at kernel scope and binds the read-only
# weight words to ``w_hi``/``w_lo``, so the ascent kernel can repeat this body in a loop.
_MVEE_ACCUMULATORS = r"""
        threadgroup float acc_hi[10][256];
        threadgroup float acc_lo[10][256];
"""


_MVEE_ACCUMULATE = r"""
        float2 wsum = float2(0.0f, 0.0f);
        float2 s0 = float2(0.0f, 0.0f), s1 = float2(0.0f, 0.0f);
        float2 s2 = float2(0.0f, 0.0f);
        float2 m00 = float2(0.0f, 0.0f), m01 = float2(0.0f, 0.0f);
        float2 m02 = float2(0.0f, 0.0f), m11 = float2(0.0f, 0.0f);
        float2 m12 = float2(0.0f, 0.0f), m22 = float2(0.0f, 0.0f);
        for (uint i = tid; i < n; i += 256) {
            float2 w = float2(w_hi[i], w_lo[i]);
            float2 p0 = float2(points_hi[i * 3], points_lo[i * 3]);
            float2 p1 = float2(points_hi[i * 3 + 1], points_lo[i * 3 + 1]);
            float2 p2 = float2(points_hi[i * 3 + 2], points_lo[i * 3 + 2]);
            wsum = ds_add(wsum, w);
            s0 = ds_add(s0, ds_mul(w, p0));
            s1 = ds_add(s1, ds_mul(w, p1));
            s2 = ds_add(s2, ds_mul(w, p2));
            m00 = ds_add(m00, ds_mul(ds_mul(w, p0), p0));
            m01 = ds_add(m01, ds_mul(ds_mul(w, p0), p1));
            m02 = ds_add(m02, ds_mul(ds_mul(w, p0), p2));
            m11 = ds_add(m11, ds_mul(ds_mul(w, p1), p1));
            m12 = ds_add(m12, ds_mul(ds_mul(w, p1), p2));
            m22 = ds_add(m22, ds_mul(ds_mul(w, p2), p2));
        }
        float2 accumulator[10];
        accumulator[0] = wsum;
        accumulator[1] = s0;
        accumulator[2] = s1;
        accumulator[3] = s2;
        accumulator[4] = m00;
        accumulator[5] = m01;
        accumulator[6] = m02;
        accumulator[7] = m11;
        accumulator[8] = m12;
        accumulator[9] = m22;
        for (uint k = 0; k < 10; k++) {
            acc_hi[k][tid] = accumulator[k].x;
            acc_lo[k][tid] = accumulator[k].y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                for (uint k = 0; k < 10; k++) {
                    float2 value = ds_add(
                        float2(acc_hi[k][tid], acc_lo[k][tid]),
                        float2(acc_hi[k][tid + stride], acc_lo[k][tid + stride]));
                    acc_hi[k][tid] = value.x;
                    acc_lo[k][tid] = value.y;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
"""


@lru_cache(maxsize=1)
def _mvee_ascent_kernel() -> MetalKernel:
    """Metal kernel for up to ``limit[0]`` compensated Khachiyan barycentric-ascent steps.

    Inputs are the centered hull point words, the current weight words and the step
    limit. Each step forms the weighted covariance and its cofactor inverse in
    double-single arithmetic, evaluates every point's lifted value and updates the
    weights. One threadgroup runs the steps in sequence, so the loop needs no host
    synchronization; it stops after the first step whose update norm is at most the
    tolerance, or after a step that loses positive definiteness or produces a non-finite
    weight. Every thread reads and writes only its own strided weight indices, and all
    cross-thread values pass through threadgroup memory behind barriers, so the returned
    weights are those of the first converged step.

    ``state[0]`` is 0 when the limit was reached without stopping, 1 when converged,
    2 when the covariance lost positive definiteness (weights then hold the previous
    step) and 3 when a weight became non-finite. ``state[1]`` counts the steps run.
    """
    tolerance = repr(float(_MVEE_TOLERANCE)) + "f"
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_mvee_ascent",
            header=DS_HEADER,
            input_names=["points_hi", "points_lo", "weights_hi", "weights_lo", "limit"],
            output_names=["new_weights_hi", "new_weights_lo", "state"],
            source=r"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = points_hi_shape[0];
        uint steps = uint(limit[0]);
        """
            + _MVEE_ACCUMULATORS
            + r"""
        threadgroup float center_hi[3];
        threadgroup float center_lo[3];
        threadgroup float inverse_hi[6];
        threadgroup float inverse_lo[6];
        threadgroup float max_hi[256];
        threadgroup float max_lo[256];
        threadgroup uint max_index[256];
        threadgroup float change_hi[256];
        threadgroup float change_lo[256];
        threadgroup int status;
        threadgroup bool finite_flags[256];
        threadgroup int stop;
        for (uint i = tid; i < n; i += 256) {
            new_weights_hi[i] = weights_hi[i];
            new_weights_lo[i] = weights_lo[i];
        }
        if (tid == 0) {
            state[0] = 0;
            state[1] = 0;
        }
        const device float* w_hi = new_weights_hi;
        const device float* w_lo = new_weights_lo;
        for (uint iteration = 0; iteration < steps; iteration++) {
        """
            + _MVEE_ACCUMULATE
            + r"""
            if (tid == 0) {
                float2 total = float2(acc_hi[0][0], acc_lo[0][0]);
                float2 c0 = ds_ratio(float2(acc_hi[1][0], acc_lo[1][0]), total);
                float2 c1 = ds_ratio(float2(acc_hi[2][0], acc_lo[2][0]), total);
                float2 c2 = ds_ratio(float2(acc_hi[3][0], acc_lo[3][0]), total);
                center_hi[0] = c0.x; center_lo[0] = c0.y;
                center_hi[1] = c1.x; center_lo[1] = c1.y;
                center_hi[2] = c2.x; center_lo[2] = c2.y;
                float2 a = ds_sub(ds_ratio(float2(acc_hi[4][0], acc_lo[4][0]), total),
                                  ds_mul(c0, c0));
                float2 b = ds_sub(ds_ratio(float2(acc_hi[5][0], acc_lo[5][0]), total),
                                  ds_mul(c0, c1));
                float2 c = ds_sub(ds_ratio(float2(acc_hi[6][0], acc_lo[6][0]), total),
                                  ds_mul(c0, c2));
                float2 d = ds_sub(ds_ratio(float2(acc_hi[7][0], acc_lo[7][0]), total),
                                  ds_mul(c1, c1));
                float2 e = ds_sub(ds_ratio(float2(acc_hi[8][0], acc_lo[8][0]), total),
                                  ds_mul(c1, c2));
                float2 f = ds_sub(ds_ratio(float2(acc_hi[9][0], acc_lo[9][0]), total),
                                  ds_mul(c2, c2));
                float2 co00 = ds_sub(ds_mul(d, f), ds_mul(e, e));
                float2 co01 = ds_sub(ds_mul(c, e), ds_mul(b, f));
                float2 co02 = ds_sub(ds_mul(b, e), ds_mul(c, d));
                float2 co11 = ds_sub(ds_mul(a, f), ds_mul(c, c));
                float2 co12 = ds_sub(ds_mul(b, c), ds_mul(a, e));
                float2 co22 = ds_sub(ds_mul(a, d), ds_mul(b, b));
                float2 determinant = ds_add(
                    ds_mul(a, co00), ds_add(ds_mul(b, co01), ds_mul(c, co02)));
                bool positive = ds_gt(determinant, float2(0.0f, 0.0f));
                bool finite = ds_finite(determinant) && ds_finite(co00) && ds_finite(co01)
                    && ds_finite(co02) && ds_finite(co11) && ds_finite(co12)
                    && ds_finite(co22);
                if (positive && finite) {
                    float2 i00 = ds_ratio(co00, determinant);
                    float2 i01 = ds_ratio(co01, determinant);
                    float2 i02 = ds_ratio(co02, determinant);
                    float2 i11 = ds_ratio(co11, determinant);
                    float2 i12 = ds_ratio(co12, determinant);
                    float2 i22 = ds_ratio(co22, determinant);
                    inverse_hi[0] = i00.x; inverse_lo[0] = i00.y;
                    inverse_hi[1] = i01.x; inverse_lo[1] = i01.y;
                    inverse_hi[2] = i02.x; inverse_lo[2] = i02.y;
                    inverse_hi[3] = i11.x; inverse_lo[3] = i11.y;
                    inverse_hi[4] = i12.x; inverse_lo[4] = i12.y;
                    inverse_hi[5] = i22.x; inverse_lo[5] = i22.y;
                    status = 1;
                } else {
                    status = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (status == 0) {
                if (tid == 0) {
                    state[0] = 2;
                    state[1] = int(iteration + 1);
                }
                break;
            }
            float2 c0 = float2(center_hi[0], center_lo[0]);
            float2 c1 = float2(center_hi[1], center_lo[1]);
            float2 c2 = float2(center_hi[2], center_lo[2]);
            float2 i00 = float2(inverse_hi[0], inverse_lo[0]);
            float2 i01 = float2(inverse_hi[1], inverse_lo[1]);
            float2 i02 = float2(inverse_hi[2], inverse_lo[2]);
            float2 i11 = float2(inverse_hi[3], inverse_lo[3]);
            float2 i12 = float2(inverse_hi[4], inverse_lo[4]);
            float2 i22 = float2(inverse_hi[5], inverse_lo[5]);
            float2 local_max = float2(0.0f, 0.0f);
            uint local_index = 0;
            for (uint i = tid; i < n; i += 256) {
                float2 x0 = ds_sub(float2(points_hi[i * 3], points_lo[i * 3]), c0);
                float2 x1 = ds_sub(float2(points_hi[i * 3 + 1], points_lo[i * 3 + 1]), c1);
                float2 x2 = ds_sub(float2(points_hi[i * 3 + 2], points_lo[i * 3 + 2]), c2);
                float2 y0 = ds_add(ds_add(ds_mul(i00, x0), ds_mul(i01, x1)), ds_mul(i02, x2));
                float2 y1 = ds_add(ds_add(ds_mul(i01, x0), ds_mul(i11, x1)), ds_mul(i12, x2));
                float2 y2 = ds_add(ds_add(ds_mul(i02, x0), ds_mul(i12, x1)), ds_mul(i22, x2));
                float2 value = ds_add(ds_add(ds_mul(x0, y0), ds_mul(x1, y1)),
                                      ds_mul(x2, y2));
                if (ds_gt(value, local_max)) {
                    local_max = value;
                    local_index = i;
                }
            }
            max_hi[tid] = local_max.x;
            max_lo[tid] = local_max.y;
            max_index[tid] = local_index;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float2 other = float2(max_hi[tid + stride], max_lo[tid + stride]);
                    float2 current = float2(max_hi[tid], max_lo[tid]);
                    if (ds_gt(other, current)) {
                        max_hi[tid] = other.x;
                        max_lo[tid] = other.y;
                        max_index[tid] = max_index[tid + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float2 maximum = float2(max_hi[0], max_lo[0]);
            uint index = max_index[0];
            // Khachiyan's lifted value is M = 1 + values, so the step
            // (M - d - 1) / ((d + 1) (M - 1)) becomes (values - d) / ((d + 1) values).
            bool improve = ds_gt(maximum, float2(3.0f, 0.0f));
            float2 step = improve
                ? ds_ratio(ds_sub(maximum, float2(3.0f, 0.0f)),
                           ds_mul(float2(4.0f, 0.0f), maximum))
                : float2(0.0f, 0.0f);
            float2 change = float2(0.0f, 0.0f);
            bool finite_weights = true;
            for (uint i = tid; i < n; i += 256) {
                float2 old = float2(new_weights_hi[i], new_weights_lo[i]);
                float2 indicator = (i == index) ? float2(1.0f, 0.0f) : float2(0.0f, 0.0f);
                float2 updated = ds_add(ds_mul(old, ds_sub(float2(1.0f, 0.0f), step)),
                                        ds_mul(step, indicator));
                new_weights_hi[i] = updated.x;
                new_weights_lo[i] = updated.y;
                finite_weights = finite_weights && ds_finite(updated);
                float2 delta = ds_sub(updated, old);
                change = ds_add(change, ds_mul(delta, delta));
            }
            change_hi[tid] = change.x;
            change_lo[tid] = change.y;
            finite_flags[tid] = finite_weights;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float2 value = ds_add(float2(change_hi[tid], change_lo[tid]),
                                          float2(change_hi[tid + stride],
                                                 change_lo[tid + stride]));
                    change_hi[tid] = value.x;
                    change_lo[tid] = value.y;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                bool all_finite = true;
                for (uint k = 0; k < 256; k++) {
                    all_finite = all_finite && finite_flags[k];
                }
                // A non-finite weight takes precedence over convergence.
                int code = !all_finite ? 3
                    : (ds_le(ds_sqrt(float2(change_hi[0], change_lo[0])),
                             float2("""
            + tolerance
            + r""", 0.0f)) ? 1 : 0);
                stop = code;
                state[0] = code;
                state[1] = int(iteration + 1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (stop != 0) {
                break;
            }
        }
        """,
        ),
    )


@lru_cache(maxsize=1)
def _mvee_covariance_kernel() -> MetalKernel:
    """Metal kernel returning the compensated weighted covariance of one MVEE iterates.

    This repeats the step kernel's double-single reduction for the converged weights, so
    the spectrum is formed from the actual final covariance rather than from a rounded
    float32 matrix.
    """
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_mvee_covariance",
            header=DS_HEADER,
            input_names=["points_hi", "points_lo", "weights_hi", "weights_lo"],
            output_names=["covariance_hi", "covariance_lo"],
            source=r"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = points_hi_shape[0];
        // MLX may bind small inputs in the constant address space.
        auto w_hi = weights_hi;
        auto w_lo = weights_lo;
        """
            + _MVEE_ACCUMULATORS
            + _MVEE_ACCUMULATE
            + r"""
        if (tid == 0) {
            float2 total = float2(acc_hi[0][0], acc_lo[0][0]);
            float2 c0 = ds_ratio(float2(acc_hi[1][0], acc_lo[1][0]), total);
            float2 c1 = ds_ratio(float2(acc_hi[2][0], acc_lo[2][0]), total);
            float2 c2 = ds_ratio(float2(acc_hi[3][0], acc_lo[3][0]), total);
            float2 cov00 = ds_sub(ds_ratio(float2(acc_hi[4][0], acc_lo[4][0]), total),
                                  ds_mul(c0, c0));
            float2 cov01 = ds_sub(ds_ratio(float2(acc_hi[5][0], acc_lo[5][0]), total),
                                  ds_mul(c0, c1));
            float2 cov02 = ds_sub(ds_ratio(float2(acc_hi[6][0], acc_lo[6][0]), total),
                                  ds_mul(c0, c2));
            float2 cov11 = ds_sub(ds_ratio(float2(acc_hi[7][0], acc_lo[7][0]), total),
                                  ds_mul(c1, c1));
            float2 cov12 = ds_sub(ds_ratio(float2(acc_hi[8][0], acc_lo[8][0]), total),
                                  ds_mul(c1, c2));
            float2 cov22 = ds_sub(ds_ratio(float2(acc_hi[9][0], acc_lo[9][0]), total),
                                  ds_mul(c2, c2));
            covariance_hi[0] = cov00.x; covariance_lo[0] = cov00.y;
            covariance_hi[1] = cov01.x; covariance_lo[1] = cov01.y;
            covariance_hi[2] = cov02.x; covariance_lo[2] = cov02.y;
            covariance_hi[3] = cov01.x; covariance_lo[3] = cov01.y;
            covariance_hi[4] = cov11.x; covariance_lo[4] = cov11.y;
            covariance_hi[5] = cov12.x; covariance_lo[5] = cov12.y;
            covariance_hi[6] = cov02.x; covariance_lo[6] = cov02.y;
            covariance_hi[7] = cov12.x; covariance_lo[7] = cov12.y;
            covariance_hi[8] = cov22.x; covariance_lo[8] = cov22.y;
        }
        """,
        ),
    )


def _centered_ds_points(host: NDArray[np.float64]) -> tuple[mx.array, mx.array]:
    """Return float32 high/low words of float64-centered hull coordinates.

    Centering is evaluated in float64, where the offset subtraction loses nothing that
    matters for a float32 input. Splitting the centered value into two float32 words
    keeps about 48 significant bits for the compensated covariance while avoiding the
    large-coordinate cancellation that uncentered moments would introduce.
    """
    values = np.asarray(host, dtype=np.float64)
    centered = values - values.mean(axis=0)
    high = centered.astype(np.float32)
    low = (centered - high.astype(np.float64)).astype(np.float32)
    return mx.array(high), mx.array(low)


def _whitened_ds_points(
    host: NDArray[np.float64],
) -> tuple[mx.array, mx.array, NDArray[np.float64]]:
    """Return high/low words of whitened hull points and their per-axis scales.

    Khachiyan's lifted values ``x^T C^-1 x`` and therefore its weights are invariant
    under any invertible affine map of the points, so the ascent may run on a
    well-conditioned copy. The float64-centered points are rotated onto their principal
    axes and each axis is divided by its spread. Only the conditioning of this map
    matters, not its accuracy: an imprecise rotation from ``eigh`` leaves the whitened
    cloud well conditioned. The returned scales map the whitened covariance ``W`` back
    to ``D W D``, which has the physical spectrum because the rotation is orthogonal.
    """
    values = np.asarray(host, dtype=np.float64)
    centered = values - values.mean(axis=0)
    _, rotation = np.linalg.eigh(centered.T @ centered)
    rotated = centered @ rotation
    scales = np.sqrt(np.mean(rotated * rotated, axis=0))
    if not bool(np.all(np.isfinite(scales))) or not bool(np.all(scales > 0)):
        raise RuntimeError("MVEE hull points do not span three dimensions")
    return (*_centered_ds_points(rotated / scales), scales)


def mvee_semi_axes(
    host: NDArray[np.float64],
) -> tuple[mx.array, mx.array, mx.array]:
    """Return the MVEE semi-axes of float64 hull points in their own coordinate unit.

    Khachiyan barycentric coordinate ascent runs on Metal. Its 4x4 homogeneous solve is
    reduced to the Schur complement ``C = P diag(u) P^T - c c^T``, whose 3x3 inverse has
    a closed form. Center, covariance, inverse, lifted values and the weight update all
    use double-single arithmetic. The ascent runs on whitened points
    (_whitened_ds_points): the cofactor inverse of a thin oblique hull's covariance
    loses its long-axis entries to cancellation (a 1e5:1e-2 hull changed its long axis
    by 10% against an exact Khachiyan), while affine invariance leaves the weights
    unchanged. The stopping tolerance is the IBSI default ``tau = 0.001`` and the
    maximum is 100,000 steps. Steps run in chunks of up to 1,024 per launch and stop at
    the first step with ``||du|| <= tau``. Semi-axes follow ``sqrt(3 * eig(C))``, where
    ``C = D W D`` is the whitened covariance ``W`` graded by the axis scales ``D`` and
    diagonalized by the shared compensated Jacobi spectrum, which keeps the small
    eigenvalues of such graded matrices to relative accuracy. The scales are first
    divided by a power of two near their maximum and that factor is restored exactly.

    Args:
        host: Float64 ``(n, 3)`` hull coordinates.

    Returns:
        Three float32 semi-axes in descending order.

    Raises:
        RuntimeError: If the points do not span three dimensions, the weighted
            covariance loses positive definiteness, an iteration becomes non-finite or
            does not converge within 100,000 steps, or the compensated spectrum cannot
            resolve the smallest axis above its roundoff bound.
    """
    if host.ndim != 2 or host.shape[1] != 3:
        raise ValueError("MVEE requires an (n, 3) point array")
    count = host.shape[0]
    if count < 4:
        raise RuntimeError("MVEE requires at least four hull vertices")
    points_hi, points_lo, scales = _whitened_ds_points(host)
    evaluate(points_hi, points_lo)
    weights_hi = mx.full((count,), 1.0 / count, dtype=mx.float32)
    weights_lo = mx.zeros((count,), dtype=mx.float32)
    remaining = _MVEE_MAX_ITERATIONS
    while True:
        # Each launch runs up to _MVEE_STEPS_PER_LAUNCH steps on the GPU and stops at the
        # first converged step, so the host waits once per chunk rather than once per
        # step.
        steps = min(_MVEE_STEPS_PER_LAUNCH, remaining)
        weights_hi, weights_lo, state = _mvee_ascent_kernel()(
            inputs=[
                points_hi,
                points_lo,
                weights_hi,
                weights_lo,
                mx.array([steps], dtype=mx.int32),
            ],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(count,), (count,), (2,)],
            output_dtypes=[mx.float32, mx.float32, mx.int32],
        )
        evaluate(weights_hi, weights_lo, state)
        code, executed = cast(list[int], state.tolist())
        if code == 2:
            raise RuntimeError("MVEE covariance is not positive definite")
        if code == 3:
            raise RuntimeError("MVEE iteration produced a non-finite weight")
        if code == 1:
            break
        remaining -= executed
        if remaining <= 0:
            raise RuntimeError(f"MVEE did not converge within {_MVEE_MAX_ITERATIONS} iterations")
    high, low = _mvee_covariance_kernel()(
        inputs=[points_hi, points_lo, weights_hi, weights_lo],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 3, 3), (1, 3, 3)],
        output_dtypes=[mx.float32, mx.float32],
    )
    whitened = (np.asarray(high, dtype=np.float64) + np.asarray(low, dtype=np.float64)).reshape(
        3, 3
    )
    _, exponent = frexp(float(np.max(scales)))
    grading = scales / ldexp(1.0, exponent)
    graded = whitened * np.outer(grading, grading)
    graded_hi = graded.astype(np.float32)
    graded_lo = (graded - graded_hi.astype(np.float64)).astype(np.float32)
    high = mx.array(graded_hi.reshape(1, 3, 3))
    low = mx.array(graded_lo.reshape(1, 3, 3))
    eigen = spectrum_kernel(3)(
        inputs=[high, low],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1, 3)],
        output_dtypes=[mx.float32],
    )[0][0]
    eigen = mx.sort(eigen)
    # Same compensated roundoff bound as shape_eigenvalues: a positive determinant does
    # not prove that the smallest weighted variance or its inverse survived cancellation.
    terms = max(1, (count + 255) // 256)
    tolerance = 8 * 3 * 3.552713678800501e-15 * terms * mx.max(mx.abs(high))
    checks = mx.stack(
        [
            mx.all(mx.isfinite(eigen)),
            mx.any(eigen < -tolerance),
            eigen[0] <= tolerance,
        ]
    )
    evaluate(checks)
    finite, negative, unresolved = cast(list[bool], checks.tolist())
    if not finite:
        raise RuntimeError("MVEE covariance spectrum is not finite")
    if negative:
        raise RuntimeError("MVEE covariance has an eigenvalue below its PSD roundoff bound")
    if unresolved:
        raise RuntimeError(
            "MVEE covariance is too ill-conditioned to resolve the smallest semi-axis"
        )
    semi = mx.sqrt(3 * eigen) * ldexp(1.0, exponent)
    evaluate(semi)
    if not np.all(np.isfinite(np.asarray(semi, dtype=np.float32))):
        raise RuntimeError("MVEE semi-axes are not finite")
    return semi[2], semi[1], semi[0]


def minimum_volume_ellipsoid_axes(
    points: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return the descending MVEE semi-axes of a float32 ``(n, 3)`` point cloud.

    The hull is built with SciPy and Khachiyan ascent runs on Metal, using the IBSI
    tolerance 0.001. This public wrapper isolates the geometry from shape-feature
    extraction so it can be checked against an independent Khachiyan implementation.

    Raises:
        ValueError: If ``points`` is not an ``(n, 3)`` array.
        RuntimeError: If the convex hull is degenerate, the compensated covariance loses
            positive definiteness, or its smallest axis cannot be resolved above the
            compensated roundoff bound.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("MVEE requires an (n, 3) point array")
    evaluate(points)
    host = np.asarray(points, dtype=np.float64)
    hull = scipy_hull(host)
    hull_points = host[hull.vertices]
    return mvee_semi_axes(hull_points)


def ellipsoid_area_normalized(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    """Evaluate the exact ellipsoid surface area for ``a >= b >= c``.

    This is the sum of the IBSI Legendre series with eccentricities referenced to the
    shortest axis, ``1-c²/a²`` and ``1-c²/b²`` (Tee, Surface Area of Ellipsoid
    Segment, equations 11–14). The IBSI manual's displayed eccentricities refer to
    the longest axis instead; that transcription does not recover spheroid areas.
    Spheres and spheroids use closed forms; triaxial axes use Carlson's ``R_G``, to
    which the corrected series converges.
    """
    evaluate(a, b, c)
    av, bv, cv = (float(cast(float, value.item())) for value in (a, b, c))
    if cv <= 0:
        return 2 * pi * a * b
    if av == cv:
        return 4 * pi * a * a
    if bv == cv:
        eccentricity = mx.sqrt(1 - (b / a) ** 2)
        return 2 * pi * b * (b + a * mx.arcsin(eccentricity) / eccentricity)
    if av == bv:
        ratio = c / a
        # q = e**2 = (1 - r)(1 + r); 1 - r is exact for r in [1/2, 1].
        squared = (1 - ratio) * (1 + ratio)
        if float(cast(float, squared.item())) <= 0.25:
            # Near the sphere log((1+e)/r)/e cancels: its float32 rounding error is
            # about 2**-24/e, which put a polar radius of 1 - 3*2**-24 above the unit
            # sphere's area. atanh(e)/e = sum q**k/(2k+1); with q <= 1/4 the terms
            # after k = 16 sum below 3e-12.
            atanh_ratio = mx.array(0.0)
            for k in range(16, -1, -1):
                atanh_ratio = atanh_ratio * squared + 1.0 / (2 * k + 1)
        else:
            eccentricity = mx.sqrt(squared)
            # log((1+e)/r) is atanh(e), retaining the small polar radius when
            # float32 would round e to one before arctanh.
            atanh_ratio = mx.log((1 + eccentricity) / ratio) / eccentricity
        return 2 * pi * a * a * (1 + ratio * ratio * atanh_ratio)
    # S = 4*pi*a*b*c*R_G(a**-2, b**-2, c**-2) (DLMF 19.33.2). Scaling by b**2 puts
    # the middle argument at one, where DLMF 19.21.10 has only non-negative terms:
    # S = 2*pi*(a*c*R_F(x, y, 1) + a*c*(1-x)*(y-1)/3*R_D(x, y, 1) + b**2).
    # The Legendre series in Tee's eccentricities decays only as (1-c**2/a**2)**n/n**2,
    # so moderately elongated axes such as 10:5:1 needed thousands of terms.
    inner, outer = b / a, b / c
    x, y = inner * inner, outer * outer
    one_minus_x = (1 - inner) * (1 + inner)
    y_minus_one = (outer - 1) * (outer + 1)
    rf, rd = _carlson_rf_rd(x, y, (av / cv) ** 2)
    return 2 * pi * (a * c * (rf + one_minus_x * y_minus_one / 3 * rd) + b * b)


def _carlson_rf_rd(x: mx.array, y: mx.array, spread: float) -> tuple[mx.array, mx.array]:
    """Return Carlson's ``R_F(x, y, 1)`` and ``R_D(x, y, 1)`` for ``x <= 1 <= y``.

    Duplication (Carlson 1995) keeps every argument at least ``x``, so the stop rule
    ``4**-m * Q < A_m`` holds once ``4**m`` exceeds ``19 * y / x``; ``spread`` is that
    host-known ratio. Both seventh-order closing series then truncate below 1e-7.
    """
    steps = max(1, int(np.ceil(np.log(19 * spread) / np.log(4))) + 1)
    z = mx.array(1.0)
    total, scale = mx.array(0.0), 1.0
    for _ in range(steps):
        root_x, root_y, root_z = mx.sqrt(x), mx.sqrt(y), mx.sqrt(z)
        lam = root_x * (root_y + root_z) + root_y * root_z
        total = total + scale / (root_z * (z + lam))
        scale /= 4
        x, y, z = (x + lam) / 4, (y + lam) / 4, (z + lam) / 4
    mean = (x + y + z) / 3
    dx, dy = 1 - x / mean, 1 - y / mean
    dz = -(dx + dy)
    e2, e3 = dx * dy - dz * dz, dx * dy * dz
    rf = (1 - e2 / 10 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / mx.sqrt(mean)
    mean = (x + y + 3 * z) / 5
    dx, dy = (mean - x) / mean, (mean - y) / mean
    dz = -(dx + dy) / 3
    xy, zz = dx * dy, dz * dz
    e2, e3 = xy - 6 * zz, (3 * xy - 8 * zz) * dz
    e4, e5 = 3 * (xy - zz) * zz, xy * zz * dz
    closing = (
        1 - 3 * e2 / 14 + e3 / 6 + 9 * e2 * e2 / 88 - 3 * e4 / 22 - 9 * e2 * e3 / 52 + 3 * e5 / 26
    )
    rd = 3 * total + scale * closing / (mean * mx.sqrt(mean))
    return rf, rd


IBSI_SERIES_ORDER = 20


def ellipsoid_area_ibsi_series(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    """Evaluate the ellipsoid area series as printed in the IBSI manual, ``a >= b >= c``.

    ``4*pi*a*b * sum_{nu=0}^{20} (alpha*beta)**nu / (1 - 4*nu**2) * P_nu(x)`` with
    ``alpha = sqrt(1 - b²/a²)``, ``beta = sqrt(1 - c²/a²)`` and
    ``x = (alpha² + beta²) / (2*alpha*beta)``, truncated at ``nu = 20`` as the manual
    states. Its longest-axis eccentricities are not the exact surface (see
    ``ellipsoid_area_normalized``), but they reproduce the published IBSI 1 ``RDD2``
    values, so ellipsoid_area="ibsi-series" uses this form for that feature only.

    ``Q_nu = (alpha*beta)**nu * P_nu(x)`` follows from Bonnet's recurrence as
    ``Q_{nu+1} = ((2nu+1) * s * Q_nu - nu * p² * Q_{nu-1}) / (nu+1)`` with
    ``s = (alpha² + beta²)/2`` and ``p = alpha*beta``, which never divides by ``p``;
    a sphere (``alpha = beta = 0``) gives ``4*pi*a²``.
    """
    evaluate(a, b, c)
    if float(cast(float, c.item())) <= 0:
        return 2 * pi * a * b
    inner, outer = b / a, c / a
    alpha2 = (1 - inner) * (1 + inner)
    beta2 = (1 - outer) * (1 + outer)
    half_sum, product2 = (alpha2 + beta2) / 2, alpha2 * beta2
    previous, current = mx.array(1.0), half_sum
    total = previous + current / (1 - 4)
    for nu in range(1, IBSI_SERIES_ORDER):
        following = ((2 * nu + 1) * half_sum * current - nu * product2 * previous) / (nu + 1)
        previous, current = current, following
        total = total + current / (1 - 4 * (nu + 1) ** 2)
    return 4 * pi * a * b * total


def enclosing_densities(
    geometry: MeshGeometry,
    volume: mx.array,
    area: mx.array,
    physical_scale: float,
) -> tuple[mx.array, mx.array]:
    """Return mesh volume and area divided by a normalized enclosing volume and area.

    ``volume`` and ``area`` are in units where the largest spacing is one. Their physical
    size is restored from the exact binary decomposition of that scale, so a
    representable density does not require an intermediate cubed or squared spacing.
    """
    scale, power = frexp(physical_scale)
    volume_density = scale_pow2(
        geometry.enclosed_norm / (volume * scale**3),
        geometry.enclosed_exponent - 3 * power,
    )
    area_density = scale_pow2(
        geometry.boundary_norm / (area * scale**2),
        geometry.boundary_exponent - 2 * power,
    )
    return (
        mx.where(volume > 0, volume_density, float("inf")),
        mx.where(area > 0, area_density, float("inf")),
    )


def aee_densities(
    geometry: MeshGeometry,
    eigen: mx.array,
    physical_scale: float,
    needs_area: bool,
    printed_series: bool = False,
) -> tuple[mx.array, mx.array]:
    """Return volume and area densities for the approximate enclosing ellipsoid.

    ``printed_series`` selects the IBSI manual's printed area series instead of the
    exact surface; only the area density changes.
    """
    semi = 2 * mx.sqrt(eigen)
    a, b, c = semi[-1], semi[-2], semi[0]
    ellipsoid_volume = (4 * pi / 3) * a * b * c
    area_of = ellipsoid_area_ibsi_series if printed_series else ellipsoid_area_normalized
    ellipsoid_area = area_of(a, b, c) if needs_area else mx.array(1.0)
    return enclosing_densities(geometry, ellipsoid_volume, ellipsoid_area, physical_scale)
