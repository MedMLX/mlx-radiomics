"""Compensated coordinate covariance, rank and shape eigenvalues."""

from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import DS_HEADER, cross_product
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx


@lru_cache(maxsize=2)
def _covariance_kernel(ndim: int) -> MetalKernel:
    """Metal kernel returning the compensated covariance of one centered point cloud.

    One threadgroup reduces the mean and then the covariance from the integer voxel
    coordinates and float32 spacing. Outputs are the high and low float32 words of the
    covariance in ``(1, ndim, ndim)`` arrays.
    """
    pairs = [(i, j) for i in range(ndim) for j in range(i, ndim)]
    decode = "\n".join(
        f"int c{d} = coordinates[i * {ndim} + {d}]; "
        f"float2 p{d} = float2(float(c{d}), float(c{d} - int(float(c{d}))));"
        for d in range(ndim)
    )
    sums = "\n".join(
        f"sums[{d}] = ds_add(sums[{d}], ds_mul(p{d}, float2(spacing[{d}], 0.0f)));"
        for d in range(ndim)
    )
    store_sums = "\n".join(
        f"sum_hi[{d}][tid] = sums[{d}].x; sum_lo[{d}][tid] = sums[{d}].y;" for d in range(ndim)
    )
    reduce_sums = "\n".join(
        f"{{ float2 value = ds_add(float2(sum_hi[{d}][tid], sum_lo[{d}][tid]), "
        f"float2(sum_hi[{d}][tid + stride], sum_lo[{d}][tid + stride])); "
        f"sum_hi[{d}][tid] = value.x; sum_lo[{d}][tid] = value.y; }}"
        for d in range(ndim)
    )
    read_mean = "\n".join(
        f"float2 mean{d} = float2(mean_hi[{d}], mean_lo[{d}]);" for d in range(ndim)
    )
    center = "\n".join(
        f"float2 x{d} = ds_sub(ds_mul(p{d}, float2(spacing[{d}], 0.0f)), mean{d});"
        for d in range(ndim)
    )
    accumulate = "\n".join(
        f"acc[{k}] = ds_add(acc[{k}], ds_mul(x{p}, x{q}));" for k, (p, q) in enumerate(pairs)
    )
    store_covariance = "\n".join(
        f"cov_hi[{k}][tid] = acc[{k}].x; cov_lo[{k}][tid] = acc[{k}].y;" for k in range(len(pairs))
    )
    reduce_covariance = "\n".join(
        f"{{ float2 value = ds_add(float2(cov_hi[{k}][tid], cov_lo[{k}][tid]), "
        f"float2(cov_hi[{k}][tid + stride], cov_lo[{k}][tid + stride])); "
        f"cov_hi[{k}][tid] = value.x; cov_lo[{k}][tid] = value.y; }}"
        for k in range(len(pairs))
    )
    write_covariance = "\n".join(
        f"{{ float2 value = ds_ratio(float2(cov_hi[{k}][0], cov_lo[{k}][0]), count); "
        f"covariance_hi[{p} * {ndim} + {q}] = value.x; "
        f"covariance_hi[{q} * {ndim} + {p}] = value.x; "
        f"covariance_lo[{p} * {ndim} + {q}] = value.y; "
        f"covariance_lo[{q} * {ndim} + {p}] = value.y; }}"
        for k, (p, q) in enumerate(pairs)
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=f"mlxradiomics_shape_covariance_{ndim}",
            header=DS_HEADER,
            input_names=["coordinates", "spacing"],
            output_names=["covariance_hi", "covariance_lo"],
            source=f"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = coordinates_shape[0];
        // Keep the integer row count exact beyond float32's 24-bit mantissa; the
        // covariance divisor must not round a large ROI's mean.
        float2 count = float2(float(n), float(int(n) - int(float(n))));
        threadgroup float sum_hi[{ndim}][256];
        threadgroup float sum_lo[{ndim}][256];
        threadgroup float cov_hi[{len(pairs)}][256];
        threadgroup float cov_lo[{len(pairs)}][256];
        threadgroup float mean_hi[{ndim}];
        threadgroup float mean_lo[{ndim}];
        float2 sums[{ndim}] = {{{", ".join("float2(0, 0)" for _ in range(ndim))}}};
        for (uint i = tid; i < n; i += 256) {{
            {decode}
            {sums}
        }}
        {store_sums}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                {reduce_sums}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid < {ndim}) {{
            float2 mean = ds_ratio(float2(sum_hi[tid][0], sum_lo[tid][0]), count);
            mean_hi[tid] = mean.x;
            mean_lo[tid] = mean.y;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {read_mean}
        float2 acc[{len(pairs)}] = {{{", ".join("float2(0, 0)" for _ in pairs)}}};
        for (uint i = tid; i < n; i += 256) {{
            {decode}
            {center}
            {accumulate}
        }}
        {store_covariance}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                {reduce_covariance}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            {write_covariance}
        }}
        """,
        ),
    )


@lru_cache(maxsize=2)
def spectrum_kernel(ndim: int) -> MetalKernel:
    """Metal kernel diagonalizing one 2x2 or 3x3 compensated covariance.

    Cyclic Jacobi rotations are applied to the high/low word pairs, so a small minor
    eigenvalue is formed from differences rather than from a rounded float32 matrix.
    Twenty-four sweeps are ample for these fixed-size matrices; the caller validates the
    resulting spectrum and rejects unresolved conditioning.
    """
    size = ndim * ndim
    sweeps = 24
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=f"mlxradiomics_shape_spectrum_{ndim}",
            header=DS_HEADER,
            input_names=["covariance_hi", "covariance_lo"],
            output_names=["eigenvalues"],
            source=f"""
        uint row = thread_position_in_grid.x;
        if (row >= covariance_hi_shape[0]) return;
        float2 matrix[{size}];
        for (uint k = 0; k < {size}; k++)
            matrix[k] = float2(covariance_hi[row * {size} + k],
                               covariance_lo[row * {size} + k]);
        for (int sweep = 0; sweep < {sweeps}; sweep++) {{
            for (int p = 0; p < {ndim}; p++) {{
                for (int q = p + 1; q < {ndim}; q++) {{
                    float2 apq = matrix[p * {ndim} + q];
                    float2 app = matrix[p * {ndim} + p];
                    float2 aqq = matrix[q * {ndim} + q];
                    // Form the half-difference from the high/low word pair. Subtracting
                    // float32 diagonals first would round 4194304.25 - 4194304 to zero and
                    // rotate a thin full-rank block as if it were exactly collinear.
                    float2 delta = ds_mul(float2(0.5f, 0.0f), ds_sub(aqq, app));
                    float scale = max(fabs(delta.x + delta.y), fabs(apq.x + apq.y));
                    if (scale == 0.0f) continue;
                    float2 d = ds_div(delta, scale), o = ds_div(apq, scale);
                    float2 denominator = ds_add(
                        ds_abs(d), ds_sqrt(ds_add(ds_mul(d, d), ds_mul(o, o))));
                    float2 t = ds_ratio(o, denominator);
                    if (delta.x < 0 || (delta.x == 0 && delta.y < 0)) t = ds_neg(t);
                    float2 c = ds_rsqrt(ds_add(float2(1, 0), ds_mul(t, t)));
                    float2 s = ds_mul(t, c);
                    for (int k = 0; k < {ndim}; k++) {{
                        if (k == p || k == q) continue;
                        float2 akp = matrix[k * {ndim} + p];
                        float2 akq = matrix[k * {ndim} + q];
                        matrix[k * {ndim} + p] = matrix[p * {ndim} + k] =
                            ds_sub(ds_mul(c, akp), ds_mul(s, akq));
                        matrix[k * {ndim} + q] = matrix[q * {ndim} + k] =
                            ds_add(ds_mul(s, akp), ds_mul(c, akq));
                    }}
                    float2 product = ds_mul(t, apq);
                    matrix[p * {ndim} + p] = ds_sub(matrix[p * {ndim} + p], product);
                    matrix[q * {ndim} + q] = ds_add(matrix[q * {ndim} + q], product);
                    matrix[p * {ndim} + q] = float2(0, 0);
                    matrix[q * {ndim} + p] = float2(0, 0);
                }}
            }}
        }}
        for (int p = 0; p < {ndim}; p++) {{
            float2 value = matrix[p * {ndim} + p];
            eigenvalues[row * {ndim} + p] = value.x + value.y;
        }}
        """,
        ),
    )


def coordinate_rank(coordinates: mx.array) -> mx.array:
    """Return the exact affine rank of integer voxel centers on Metal.

    Differences, cross products and scalar triple products fit int64 under the image's
    <2**30 voxel limit. Geometric rank therefore does not need a floating tolerance
    that could erase a thin but positive physical axis after anisotropic scaling.
    """
    vectors = coordinates.astype(mx.int64) - coordinates[0].astype(mx.int64)
    distinct = mx.any(vectors != 0, axis=1)
    first = vectors[mx.argmax(distinct.astype(mx.int32))]
    if coordinates.shape[1] == 2:
        area = vectors[:, 0] * first[1] - vectors[:, 1] * first[0]
        return mx.where(mx.any(area != 0), 2, mx.any(distinct).astype(mx.int32))
    normals = cross_product(vectors, first)
    independent = mx.any(normals != 0, axis=1)
    normal = normals[mx.argmax(independent.astype(mx.int32))]
    volume = mx.sum(vectors * normal, axis=1)
    return mx.where(
        mx.any(volume != 0), 3, mx.where(mx.any(independent), 2, mx.any(distinct).astype(mx.int32))
    )


def shape_eigenvalues(
    voxel_coordinates: NDArray[np.int64],
    spacing: tuple[float, ...],
    rank: mx.array,
    required_axes: frozenset[int],
) -> tuple[mx.array, float]:
    """Return the normalized ascending covariance spectrum and physical length scale.

    The covariance and its Jacobi spectrum use double-single arithmetic on Metal. This
    preserves a positive minor eigenvalue when a bare float32 covariance would round it
    to the major scale (for example two voxel pairs separated by 4096). The exact integer
    rank still fixes null axes to zero. A required positive eigenvalue below the
    compensated roundoff bound is rejected instead of returning a false zero axis. Only
    positive axes needed by the requested features must exceed that bound, so an
    unresolved minor axis does not invalidate a requested major length.

    Raises:
        RuntimeError: If the kernel returns a non-finite spectrum, has a negative value
            outside the roundoff bound, or cannot resolve a requested positive axis.
    """
    ndim = voxel_coordinates.shape[1]
    # Normalize before products so representable lengths do not require their squared
    # units to fit float32. Restore units after taking the square root.
    physical_scale = max(spacing)
    normalized_spacing = tuple(value / physical_scale for value in spacing)
    coordinates = mx.array(voxel_coordinates, dtype=mx.int32)
    high, low = _covariance_kernel(ndim)(
        inputs=[coordinates, mx.array(normalized_spacing, dtype=mx.float32)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, ndim, ndim), (1, ndim, ndim)],
        output_dtypes=[mx.float32, mx.float32],
    )
    eigen = spectrum_kernel(ndim)(
        inputs=[high, low],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1, ndim)],
        output_dtypes=[mx.float32],
    )[0][0]
    eigen = mx.sort(eigen)
    evaluate(eigen, rank)
    # Two float32 words retain about 48 significant bits; each thread accumulates its
    # share of the rows with a residual, so allow the per-thread term count in the
    # roundoff bound. This only identifies spectra cancellation has already consumed;
    # valid diagonal anisotropy well above single precision is not rejected.
    terms = max(1, (voxel_coordinates.shape[0] + 255) // 256)
    tolerance = 8 * ndim * 3.552713678800501e-15 * terms * mx.max(mx.abs(high))
    rank_value = int(cast(int, rank.item()))
    null_count = ndim - rank_value
    required_checks = [eigen[axis] <= tolerance for axis in required_axes if axis >= null_count]
    unresolved = (
        mx.any(mx.stack(required_checks)) if required_checks else mx.array(False, dtype=mx.bool_)
    )
    null_invalid = (
        mx.any(mx.abs(eigen[:null_count]) > tolerance)
        if null_count
        else mx.array(False, dtype=mx.bool_)
    )
    # All spectrum checks depend on the same tiny result; resolve their flags in one
    # host read.
    finite = mx.all(mx.isfinite(eigen))
    eigen = mx.where(mx.arange(ndim) < null_count, 0, eigen)
    checks = mx.stack([finite, unresolved, null_invalid, mx.any(eigen < -tolerance)])
    evaluate(checks)
    finite_value, unresolved_value, null_value, negative_value = cast(list[bool], checks.tolist())
    if not finite_value:
        raise RuntimeError("shape covariance spectrum is not finite")
    if unresolved_value:
        raise RuntimeError("shape covariance is too ill-conditioned to resolve a requested axis")
    if null_value:
        raise RuntimeError("shape covariance null axis exceeds the compensated roundoff bound")
    # Enforce the exact geometric nullspace even for large collinear/coplanar
    # populations. Full-rank positive eigenvalues are never thresholded away.
    if negative_value:
        raise RuntimeError("shape covariance has an eigenvalue below its PSD roundoff bound")
    return mx.maximum(eigen, 0), physical_scale
