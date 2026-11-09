"""Cubic B-spline interpolation (prefilter and point sampling) on Metal, used by LBP3D."""

from functools import lru_cache
from math import prod
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx


@lru_cache(maxsize=64)
def _prefilter_kernel(length: int) -> MetalKernel:
    """Compile the mirror-boundary cubic B-spline prefilter for rows of ``length``."""
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_spline_prefilter",
            input_names=["image"],
            output_names=["coefficients"],
            source=f"""
        uint row = thread_position_in_grid.x;
        if (row >= image_shape[0]) return;
        uint base = row * {length};
        constexpr float z = -0.2679491924311227f;
        for (int k = 0; k < {length}; k++) coefficients[base + k] = image[base + k] * 6;
        float zn = pow(z, float({length - 1})), zi = z;
        float initial = coefficients[base] + zn * coefficients[base + {length - 1}];
        for (int k = 1; k < {length - 1}; k++) {{
            initial += zi * (coefficients[base + k] + zn * coefficients[base + {length - 1} - k]);
            zi *= z;
        }}
        coefficients[base] = initial / (1 - zn * zn);
        for (int k = 1; k < {length}; k++) coefficients[base + k] += z * coefficients[base + k - 1];
        coefficients[base + {length - 1}] =
            (z * coefficients[base + {length - 2}]
                + coefficients[base + {length - 1}]) * z / (z * z - 1);
        for (int k = {length - 2};
        k >= 0;
        k--) coefficients[base + k] = z * (coefficients[base + k + 1] - coefficients[base + k]);
        """,
        ),
    )


def spline_coefficients(image: mx.array) -> mx.array:
    """Return cubic B-spline coefficients of a float32 image for interpolation.

    Applies the recursive prefilter (pole ``sqrt(3) - 2``) along every axis longer than one
    voxel, with mirror boundary conditions.
    """
    for axis, length in enumerate(image.shape):
        if length == 1:
            continue
        arranged = mx.moveaxis(image, axis, -1)
        rows = mx.reshape(arranged, (-1, length))
        result = _prefilter_kernel(length)(
            inputs=[rows],
            grid=(rows.shape[0], 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[rows.shape],
            output_dtypes=[mx.float32],
        )[0]
        image = mx.moveaxis(mx.reshape(result, arranged.shape), -1, axis)
        evaluate(image)
    return image


@lru_cache(maxsize=32)
def _sample_kernel(shape: tuple[int, ...]) -> MetalKernel:
    """Compile the kernel sampling B-spline coefficients of ``shape`` at center offsets."""
    size = ",".join(str(v) for v in shape)
    strides = ",".join(str(prod(shape[i + 1 :])) for i in range(3))
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_cubic_samples",
            input_names=["coefficients", "centers", "offsets"],
            output_names=["samples"],
            header=f"constant int sizes[] = {{{size}}}; constant int strides[] = {{{strides}}};",
            source="""
        uint t = thread_position_in_grid.x;
        uint count = offsets_shape[0], row = t / count, point = t % count;
        if (row >= centers_shape[0]) return;
        int index[3][4]; float weight[3][4];
        for (int d = 0; d < 3; d++) {
            float coordinate = centers[row * 3 + d] + offsets[point * 3 + d];
            if (coordinate < 0 || coordinate > sizes[d] - 1) { samples[t] = 0; return; }
            int left = int(floor(coordinate)) - 1;
            float x = coordinate - floor(coordinate), y = 1 - x;
            weight[d][0] = y * y * y / 6;
            weight[d][1] = (x * x * (x - 2) * 3 + 4) / 6;
            weight[d][2] = (y * y * (y - 2) * 3 + 4) / 6;
            weight[d][3] = 1 - weight[d][0] - weight[d][1] - weight[d][2];
            for (int k = 0; k < 4; k++) {
                int period = 2 * (sizes[d] - 1);
                int q = period ? ((left + k) % period + period) % period : 0;
                index[d][k] = q < sizes[d] ? q : period - q;
            }
        }
        float value = 0;
        for (int z = 0; z < 4; z++) for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
            int flat = index[0][z] * strides[0] + index[1][y] * strides[1] + index[2][x];
            value += coefficients[flat] * weight[0][z] * weight[1][y] * weight[2][x];
        }
        samples[t] = value;
        """,
        ),
    )


def cubic_samples(coefficients: mx.array, centers: mx.array, offsets: mx.array) -> mx.array:
    """Sample the spline at ``centers[i] + offsets[j]`` (voxel index units, array-axis order).

    Args:
        coefficients: Output of ``spline_coefficients`` for a 3-D image.
        centers: Integer array ``(n_centers, 3)`` of voxel indexes.
        offsets: Float array ``(n_offsets, 3)`` of displacements in voxels.

    Returns:
        Float32 array ``(n_centers, n_offsets)``. A position outside ``[0, size - 1]`` on any
        axis gives 0; the 4-tap support near an edge reflects (mirror boundary).
    """
    return _sample_kernel(coefficients.shape)(
        inputs=[coefficients, centers, offsets],
        grid=(centers.shape[0] * offsets.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(centers.shape[0], offsets.shape[0])],
        output_dtypes=[mx.float32],
    )[0]
