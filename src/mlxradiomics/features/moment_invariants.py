"""Three-dimensional moment invariants of the ROI shape and intensity on Metal.

The family follows the ten moment invariants of PySERA 2.1.5
(``MomentInvariantsFeaturesExtractor``): five invariants of a density ``f`` over the
voxel centers, evaluated once for the shape (``f = 1`` on the morphological mask) and
once for the intensity (``f = (I - min) / (mean - min)`` on the intensity mask, so the
density is nonnegative with mean one). With the density's mass ``m000``, centroid and
central moments ``mu_pqr``, the scale-normalized moments are

    eta_pqr = mu_pqr / m000 ** ((p + q + r) / 3 + 1)

and the invariants are, for the second-order tensor ``A`` and third-order tensor ``T``,

    I1 = trace(A)                     F2 = sum(A_ij ** 2)
    I2 = second principal invariant   I3 = det(A)
    F3 = sum(T_ijk ** 2)

All five are unchanged by translation, rotation and isotropic scaling of the ROI; the
intensity set is also unchanged by an affine intensity transform with positive gain.

Moments use physical voxel-center coordinates and the voxel volume, so anisotropic
spacing is honored. PySERA uses voxel indices; the two agree on an isotropic grid.
Both densities have mass ``n * voxel_volume`` for ``n`` mask voxels, which makes every
``eta_pqr`` a density-weighted mean of products of the centered coordinates divided by
``(n * voxel_volume) ** (1 / 3)``. Weighted means do not depend on the scale of the
weights, so the intensity density enters as ``I - min`` and its mean is never formed.

Host NumPy work is limited to selecting source voxels, translating integer coordinates
and shifting and scaling source intensities. The centroid, the sixteen central moments
and the invariants are evaluated in double-single arithmetic in one Metal kernel, so the
cancellation in the centering, the determinant and the third-order sums is resolved
before the float32 results are written.
"""

from collections.abc import Collection
from functools import lru_cache
from math import exp, frexp, isfinite, log
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import DS_HEADER
from mlxradiomics.numerics.precision import prepare_source_pair
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

INVARIANTS = ("I1", "F2", "I2", "I3", "F3")
MOMENT_INVARIANTS = tuple(
    f"{density}{name}" for density in ("Intensity", "Shape") for name in INVARIANTS
)
_SUPPORTED = frozenset(MOMENT_INVARIANTS)
# One threadgroup reduces every moment. Sixteen double-single accumulators per thread
# fit Metal's threadgroup memory at this width.
_THREADS = 128
# Axis multisets of the central moments, in the order the kernel stores them:
# eta200, eta020, eta002, eta110, eta101, eta011, then eta300, eta030, eta003, eta210,
# eta201, eta120, eta102, eta021, eta012, eta111.
_SECOND = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_THIRD = (
    (0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
    (0, 1, 1), (0, 2, 2), (1, 1, 2), (1, 2, 2), (0, 1, 2),
)


@lru_cache(maxsize=1)
def _invariant_kernel() -> MetalKernel:
    """Metal kernel returning the five invariants of one weighted point cloud.

    Inputs are ``(n, 3)`` int32 voxel coordinates, the per-axis coordinate scale as high
    and low float32 words, and one nonnegative weight per row as high and low words.
    The first pass reduces the total weight and weighted coordinate sums; the second
    centers on the resulting centroid and reduces the sixteen central moments. Outputs
    are ``invariants`` in :data:`INVARIANTS` order and the total weight, both float32.
    A zero total weight leaves non-finite invariants for the caller to mark undefined.
    """
    moments = len(_SECOND) + len(_THIRD)
    accumulate = "\n".join(
        [f"acc[{k}] = ds_add(acc[{k}], ds_mul(wx[{a}], x[{b}]));"
         for k, (a, b) in enumerate(_SECOND)]
        + [f"acc[{k}] = ds_add(acc[{k}], ds_mul(ds_mul(wx[{a}], x[{b}]), x[{c}]));"
           for k, (a, b, c) in enumerate(_THIRD, start=len(_SECOND))]
    )
    reduce = f"""
        for (uint k = 0; k < COUNT; k++) {{
            part_hi[k][tid] = acc[k].x;
            part_lo[k][tid] = acc[k].y;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {_THREADS // 2}; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                for (uint k = 0; k < COUNT; k++) {{
                    float2 value = ds_add(
                        float2(part_hi[k][tid], part_lo[k][tid]),
                        float2(part_hi[k][tid + stride], part_lo[k][tid + stride]));
                    part_hi[k][tid] = value.x;
                    part_lo[k][tid] = value.y;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_moment_invariants",
            header=DS_HEADER,
            input_names=["coordinates", "scale_hi", "scale_lo", "weight_hi", "weight_lo"],
            output_names=["invariants", "total_weight"],
            source=f"""
        uint tid = thread_position_in_threadgroup.x;
        uint n = coordinates_shape[0];
        threadgroup float part_hi[{moments}][{_THREADS}];
        threadgroup float part_lo[{moments}][{_THREADS}];
        threadgroup float shared_hi[4];
        threadgroup float shared_lo[4];
        float2 scale[3];
        for (uint d = 0; d < 3; d++) scale[d] = float2(scale_hi[d], scale_lo[d]);
        float2 acc[{moments}];
        for (uint k = 0; k < {moments}; k++) acc[k] = float2(0, 0);
        for (uint i = tid; i < n; i += {_THREADS}) {{
            float2 w = float2(weight_hi[i], weight_lo[i]);
            acc[0] = ds_add(acc[0], w);
            for (uint d = 0; d < 3; d++) {{
                int c = coordinates[i * 3 + d];
                float2 p = float2(float(c), float(c - int(float(c))));
                acc[1 + d] = ds_add(acc[1 + d], ds_mul(w, ds_mul(p, scale[d])));
            }}
        }}
        {reduce.replace("COUNT", "4")}
        if (tid == 0) {{
            shared_hi[0] = part_hi[0][0];
            shared_lo[0] = part_lo[0][0];
        }} else if (tid < 4) {{
            float2 mean = ds_ratio(float2(part_hi[tid][0], part_lo[tid][0]),
                                   float2(part_hi[0][0], part_lo[0][0]));
            shared_hi[tid] = mean.x;
            shared_lo[tid] = mean.y;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 total = float2(shared_hi[0], shared_lo[0]);
        float2 center[3];
        for (uint d = 0; d < 3; d++) center[d] = float2(shared_hi[1 + d], shared_lo[1 + d]);
        for (uint k = 0; k < {moments}; k++) acc[k] = float2(0, 0);
        for (uint i = tid; i < n; i += {_THREADS}) {{
            float2 w = float2(weight_hi[i], weight_lo[i]);
            float2 x[3];
            float2 wx[3];
            for (uint d = 0; d < 3; d++) {{
                int c = coordinates[i * 3 + d];
                float2 p = float2(float(c), float(c - int(float(c))));
                x[d] = ds_sub(ds_mul(p, scale[d]), center[d]);
                wx[d] = ds_mul(w, x[d]);
            }}
            {accumulate}
        }}
        {reduce.replace("COUNT", str(moments))}
        if (tid == 0) {{
            float2 m[{moments}];
            for (uint k = 0; k < {moments}; k++)
                m[k] = ds_ratio(float2(part_hi[k][0], part_lo[k][0]), total);
            float2 off = ds_add(ds_add(ds_mul(m[3], m[3]), ds_mul(m[4], m[4])),
                                ds_mul(m[5], m[5]));
            float2 i1 = ds_add(ds_add(m[0], m[1]), m[2]);
            float2 f2 = ds_add(
                ds_add(ds_add(ds_mul(m[0], m[0]), ds_mul(m[1], m[1])), ds_mul(m[2], m[2])),
                ds_mul(float2(2, 0), off));
            float2 i2 = ds_sub(
                ds_add(ds_add(ds_mul(m[0], m[1]), ds_mul(m[0], m[2])), ds_mul(m[1], m[2])),
                off);
            float2 i3 = ds_add(
                ds_mul(ds_mul(m[0], m[1]), m[2]),
                ds_mul(float2(2, 0), ds_mul(ds_mul(m[3], m[4]), m[5])));
            i3 = ds_sub(i3, ds_mul(m[2], ds_mul(m[3], m[3])));
            i3 = ds_sub(i3, ds_mul(m[1], ds_mul(m[4], m[4])));
            i3 = ds_sub(i3, ds_mul(m[0], ds_mul(m[5], m[5])));
            float2 pure = float2(0, 0), mixed = float2(0, 0);
            for (uint k = 6; k < 9; k++) pure = ds_add(pure, ds_mul(m[k], m[k]));
            for (uint k = 9; k < 15; k++) mixed = ds_add(mixed, ds_mul(m[k], m[k]));
            float2 f3 = ds_add(
                ds_add(pure, ds_mul(float2(3, 0), mixed)),
                ds_mul(float2(6, 0), ds_mul(m[15], m[15])));
            invariants[0] = i1.x + i1.y;
            invariants[1] = f2.x + f2.y;
            invariants[2] = i2.x + i2.y;
            invariants[3] = i3.x + i3.y;
            invariants[4] = f3.x + f3.y;
            total_weight[0] = total.x + total.y;
        }}
        """,
        ),
    )


def _invariants(
    mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    weights: NDArray[np.float64] | None,
) -> mx.array:
    """Return the five invariants of one density as a float32 array of shape ``(5,)``.

    ``weights`` holds one nonnegative value per mask voxel in traversal order, or
    ``None`` for the unit density. All five are NaN when the weights sum to zero.

    Raises:
        RuntimeError: If the weights are not finite or a defined invariant leaves the
            float32 range.
    """
    coordinates = np.argwhere(mask)
    count = coordinates.shape[0]
    # Offsets from the bounding-box corner are exact small integers, and an axis the
    # ROI does not extend along contributes exact zeros to every moment.
    relative = (coordinates - coordinates.min(axis=0)).astype(np.int32)
    # eta_pqr is a weighted mean of coordinate products after dividing each coordinate
    # by cbrt(count * voxel_volume). The geometric-mean spacing keeps the cube root in
    # range for spacings whose product is not representable.
    geometric_mean = exp(sum(log(value) for value in spacing) / 3)
    scale = np.asarray(
        [value / geometric_mean / count ** (1 / 3) for value in spacing], dtype=np.float64,
    )
    scale_hi = scale.astype(np.float32)
    scale_lo = (scale - scale_hi.astype(np.float64)).astype(np.float32)
    if weights is None:
        weight_hi = np.ones(count, dtype=np.float32)
        weight_lo = np.zeros(count, dtype=np.float32)
    else:
        largest = float(weights.max())
        if not isfinite(largest):
            raise RuntimeError("moment invariant intensity range exceeds the float64 range")
        if largest == 0:
            return mx.full((len(INVARIANTS),), float("nan"), dtype=mx.float32)
        # The weights are nonnegative, so their sums do not cancel and the two float32
        # words leave a representation error far below float32 resolution of any moment.
        weight_hi, weight_lo, _ = prepare_source_pair(np.ldexp(weights, -frexp(largest)[1]))
    values, total = _invariant_kernel()(
        inputs=[
            mx.array(relative), mx.array(scale_hi), mx.array(scale_lo),
            mx.array(weight_hi), mx.array(weight_lo),
        ],
        grid=(_THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(len(INVARIANTS),), (1,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    defined = total[0] > 0
    resolved = mx.all(mx.isfinite(values)) | ~defined
    evaluate(values, defined, resolved)
    if not bool(resolved.item()):
        raise RuntimeError("moment invariants exceed the float32 range")
    return mx.where(defined, values, float("nan"))


def moment_invariant_statistics(
    image: NDArray[np.generic],
    intensity_mask: NDArray[np.bool_],
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    features: Collection[str],
) -> dict[str, mx.array]:
    """Compute selected moment invariants of the ROI shape and intensity.

    Args:
        image: Intensity image on the same grid as both masks.
        intensity_mask: Voxels whose intensities form the intensity density.
        morphology_mask: Segmented shape that forms the unit density.
        spacing: Voxel spacing per array axis in the requested length unit.
        features: Bare names from :data:`MOMENT_INVARIANTS`.

    Returns:
        ``momentinvariant_<Name>`` keys mapped to float32 arrays of shape ``(1,)``. The
        values are dimensionless. The intensity invariants are NaN for a constant
        intensity ROI, whose density ``(I - min) / (mean - min)`` is undefined.

    Raises:
        ValueError: If shapes, dimensions, spacing or feature names are invalid, a
            required mask is empty, or a selected intensity is not finite.
        RuntimeError: If the intensity range overflows float64 or a defined invariant
            leaves the float32 range.
    """
    requested = frozenset(features)
    unsupported = requested - _SUPPORTED
    if unsupported:
        raise ValueError(f"unsupported moment invariants: {', '.join(sorted(unsupported))}")
    if not requested:
        return {}
    image_array = np.asarray(image)
    if image_array.shape != intensity_mask.shape or image_array.shape != morphology_mask.shape:
        raise ValueError("image and both masks must have the same shape")
    if morphology_mask.ndim != 3:
        raise ValueError("moment invariants require three-dimensional masks")
    if len(spacing) != 3:
        raise ValueError("spacing must contain one value per mask axis")
    if any(not isfinite(float(value)) or value <= 0 for value in spacing):
        raise ValueError("spacing values must be finite and positive")
    values: dict[str, mx.array] = {}
    if any(name.startswith("Shape") for name in requested):
        if not np.any(morphology_mask):
            raise ValueError("morphology mask is empty")
        shape = _invariants(morphology_mask, spacing, None)
        values.update({f"Shape{name}": shape[k] for k, name in enumerate(INVARIANTS)})
    if any(name.startswith("Intensity") for name in requested):
        if not np.any(intensity_mask):
            raise ValueError("intensity mask is empty")
        samples = image_array[intensity_mask].astype(np.float64)
        if not np.all(np.isfinite(samples)):
            raise ValueError("moment invariants require finite source intensities")
        intensity = _invariants(intensity_mask, spacing, samples - samples.min())
        values.update({f"Intensity{name}": intensity[k] for k, name in enumerate(INVARIANTS)})
    return {
        f"momentinvariant_{name}": values[name][None]
        for name in MOMENT_INVARIANTS if name in requested
    }
