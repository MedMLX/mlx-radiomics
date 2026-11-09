"""2-D local binary patterns (PyRadiomics ``lbp-2D``) on Metal."""

from functools import lru_cache
from math import cos, floor, pi, sin
from typing import cast

import numpy as np

from mlxradiomics.config.filtering import LBPMethod
from mlxradiomics.numerics.precision import PAIR_HEADER
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx


def _method_source(samples: int, method: LBPMethod) -> str:
    """Return Metal statements mapping the raw bit ``code`` to ``value`` for ``method``.

    ``DEFAULT`` needs none; ``value`` stays the raw code.
    """
    if method == LBPMethod.DEFAULT:
        return ""
    if method == LBPMethod.UNIFORM:
        return f"value = changes <= 2 ? ones : {samples + 1};"
    if method == LBPMethod.NONROTATION_UNIFORM:
        return f"""value = changes > 2 ? {samples * (samples - 1) + 2}
            : ones == 0 ? 0 : ones == {samples} ? {samples * (samples - 1) + 1}
            : 1 + (ones - 1) * {samples}
                + (first_one == 0 ? int(ones) - first_zero : {samples} - first_one);"""
    if method == LBPMethod.ROTATION_INVARIANT:
        return f"""for (uint k = 1; k < {samples}; k++) {{
            code = (code >> 1) | ((code & 1) << {samples - 1});
            value = min(value, float(code)); }}"""
    if method == LBPMethod.VARIANCE:
        # Two passes over pair-valued ring samples avoid subtracting large raw moments.
        return f"""Pair total = {{0, 0}};
            for (uint k = 0; k < {samples}; k++)
                total = add(total, subtract(ring[k], ring[0]));
            float inverse = 1.0f / {samples};
            Pair mean = multiply(total, Pair{{inverse, fma(-inverse, {samples}.0f, 1.0f)
                                                       / {samples}.0f}});
            float squares = 0;
            for (uint k = 0; k < {samples}; k++) {{
                Pair delta = subtract(subtract(ring[k], ring[0]), mean);
                float centered = delta.hi + delta.lo;
                squares += centered * centered;
            }}
            value = squares / {samples};
            if (value == 0) value = NAN;"""
    raise ValueError(f"unsupported LBP method: {method}")


def _float32_pair(value: float) -> tuple[float, float]:
    """Split ``value`` into a float32 high part and its float32 residual.

    The kernel reconstructs a declared coordinate or fraction from this pair.
    """
    high = np.float32(value)
    return float(high), float(np.float32(value - float(high)))


def _literal(value: float) -> str:
    """Format ``value`` as a Metal float literal from its shortest round-trip decimal."""
    return f"{repr(value)}f"


@lru_cache(maxsize=64)
def _lbp_kernel(samples: int, radius: float, method: LBPMethod) -> MetalKernel:
    """Compile the 2-D LBP kernel for ``samples`` ring points at ``radius``.

    Ring coordinates, their integer pixel offsets and pair-split fractional weights
    are emitted as constant tables in the kernel header.
    """
    # Sampling coordinates are declared to five decimals. Keep the
    # integer pixel offset separate from the fractional weight so the kernel
    # never loses the coordinate residual to a float32 subtraction.
    coordinates = [
        (
            round(-radius * sin(2 * pi * k / samples), 5),
            round(radius * cos(2 * pi * k / samples), 5),
        )
        for k in range(samples)
    ]
    row_floor = [int(floor(row)) for row, _ in coordinates]
    col_floor = [int(floor(col)) for _, col in coordinates]
    row_parts = [_float32_pair(row - row_floor[k]) for k, (row, _) in enumerate(coordinates)]
    col_parts = [_float32_pair(col - col_floor[k]) for k, (_, col) in enumerate(coordinates)]
    header = "constant int row_floor[] = {" + ",".join(map(str, row_floor)) + "};\n"
    header += "constant int col_floor[] = {" + ",".join(map(str, col_floor)) + "};\n"
    header += "constant float row_frac_hi[] = {" + ",".join(
        _literal(high) for high, _ in row_parts) + "};\n"
    header += "constant float row_frac_lo[] = {" + ",".join(
        _literal(low) for _, low in row_parts) + "};\n"
    header += "constant float col_frac_hi[] = {" + ",".join(
        _literal(high) for high, _ in col_parts) + "};\n"
    header += "constant float col_frac_lo[] = {" + ",".join(
        _literal(low) for _, low in col_parts) + "};\n"
    # Integer weights express the declared decimal coordinates exactly. Their
    # denominator is common to all four corners and does not affect the sign.
    weights: list[int] = []
    for k, (row, col) in enumerate(coordinates):
        ry = round((row - row_floor[k]) * 100000)
        rx = round((col - col_floor[k]) * 100000)
        weights.extend(((100000 - ry) * (100000 - rx),
                        (100000 - ry) * rx, ry * (100000 - rx), ry * rx))
    weight_parts = [_float32_pair(float(weight)) for weight in weights]
    for part, name in ((0, "weight_hi"), (1, "weight_lo")):
        header += "constant float " + name + "[] = {" + ",".join(
            _literal(pair[part]) for pair in weight_parts) + "};\n"
    header += """
    // Grow an exact, nonoverlapping expansion using error-free TwoSum.
    // At most 54 product components enter one interpolation comparison.
    void grow(thread float *expansion, thread int &count, float value) {
        int next = 0;
        for (int i = 0; i < count; ++i) {
            float sum = value + expansion[i];
            float virtual_b = sum - value;
            float error = (value - (sum - virtual_b)) + (expansion[i] - virtual_b);
            if (error != 0.0f) expansion[next++] = error;
            value = sum;
        }
        if (value != 0.0f || next == 0) expansion[next++] = value;
        count = next;
    }
    void exact_product(thread float *expansion, thread int &count, float a, float b) {
        float product = a * b;
        grow(expansion, count, fma(a, b, -product));
        grow(expansion, count, product);
    }
    """
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_lbp2d",
            header=PAIR_HEADER + header,
            input_names=["image", "residual", "tail"],
            output_names=["output"],
            compile_options={"math_mode": "safe"},
            source=f"""
        uint t = thread_position_in_grid.x;
        uint height = image_shape[1], width = image_shape[2];
        if (t >= image_shape[0] * height * width) return;
        uint base = (t / (height * width)) * height * width;
        uint row = (t / width) % height, col = t % width;
        bool bits[{samples}]; Pair ring[{samples}];
        uint code = 0, ones = 0; int first_one = -1, first_zero = -1;
        for (uint k = 0; k < {samples}; k++) {{
            int iy = row_floor[k], ix = col_floor[k];
            Pair wy = Pair{{row_frac_hi[k], row_frac_lo[k]}};
            Pair wx = Pair{{col_frac_hi[k], col_frac_lo[k]}};
            Pair corner[4];
            float expansion[64]; int count = 0;
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {{
                int yy = int(row) + iy + a, xx = int(col) + ix + b;
                float sample = 0, low = 0, tiny = 0;
                if (yy >= 0 && yy < int(height) && xx >= 0 && xx < int(width))
                    {{ sample = image[base + yy * width + xx];
                      low = residual[base + yy * width + xx];
                      tiny = tail[base + yy * width + xx]; }}
                corner[a * 2 + b] = subtract(Pair{{sample, low + tiny}},
                                             Pair{{image[t], residual[t] + tail[t]}});
                int wi = int(k) * 4 + a * 2 + b;
                exact_product(expansion, count, sample, weight_hi[wi]);
                exact_product(expansion, count, sample, weight_lo[wi]);
                exact_product(expansion, count, low, weight_hi[wi]);
                exact_product(expansion, count, low, weight_lo[wi]);
                exact_product(expansion, count, tiny, weight_hi[wi]);
                exact_product(expansion, count, tiny, weight_lo[wi]);
            }}
            // Nested lerps of corner differences: equal corners give exactly that
            // value, so a uniform ring keeps exactly zero variance. Four float-float
            // weight products need not sum to one and left a residual variance.
            Pair top = add(corner[0], multiply(wx, subtract(corner[1], corner[0])));
            Pair bottom = add(corner[2], multiply(wx, subtract(corner[3], corner[2])));
            ring[k] = add(top, multiply(wy, subtract(bottom, top)));
            // 10^10 = 100000^2 is exactly representable in float32.
            exact_product(expansion, count, -image[t], 10000000000.0f);
            exact_product(expansion, count, -residual[t], 10000000000.0f);
            exact_product(expansion, count, -tail[t], 10000000000.0f);
            bits[k] = expansion[count - 1] >= 0.0f;
            ones += bits[k]; code |= uint(bits[k]) << k;
            if (bits[k] && first_one < 0) first_one = k;
            if (!bits[k] && first_zero < 0) first_zero = k;
        }}
        uint changes = 0;
        for (uint k = 0; k < {samples - 1}; k++) changes += bits[k] != bits[k + 1];
        float value = code;
        {_method_source(samples, method)}
        output[t] = value;
        """,
        ),
    )


def lbp2d(image: mx.array, samples: int, radius: float, method: LBPMethod, axis: int,
          residual: mx.array | None = None, source_tail: mx.array | None = None) -> mx.array:
    """Compute a 2-D local binary pattern image, slice by slice.

    For each voxel, ``samples`` points on a circle of ``radius`` voxels (spacing is not used)
    are read by bilinear interpolation, with samples outside the slice taken as 0. A bit is
    set where the sample is at least the center value, and the bits are coded by ``method``.

    Sampling coordinates are declared to five decimal places. Categorical comparisons
    use exact integer bilinear weights with denominator ``100000**2`` and an error-free
    float32 expansion of the weighted source parts. Exact cancelling ties set the bit;
    small nonzero signs are never replaced by ties. Ring variance uses compensated
    float-float arithmetic rather than the categorical expansion.

    Every nonzero source part must lie in ``[2**-90, 2**90]`` in magnitude. Outside
    this range an explicit error prevents underflow or overflow from invalidating
    the expansion. A supplied residual describes the exact accepted source as
    ``image + residual + source_tail``; source conversion must reject any further loss.

    Args:
        image: Float32 array, 2-D, or 3-D processed as slices perpendicular to ``axis``.
        samples: Number of points on the circle.
        radius: Circle radius in voxels.
        method: Coding of the bit pattern.
        axis: Slice axis for a 3-D ``image``; ignored for 2-D.
        residual: Float32 ``float64 source - float32 image``. When given, sample-minus-center
            differences and weighted interpolation retain float-float pairs through the
            bit comparison. Ring variance uses two centered passes. Defaults to zeros.
        source_tail: Optional third float32 source word, used exactly in categorical
            comparisons. Variance merges it into the residual at float32 precision.

    Returns:
        Float32 array of the image's shape.
    """
    residual = mx.zeros_like(image) if residual is None else residual
    source_tail = mx.zeros_like(image) if source_tail is None else source_tail
    if residual.shape != image.shape or source_tail.shape != image.shape:
        raise ValueError("LBP2D source parts must match the image shape")
    for part in (image, residual, source_tail):
        magnitude = mx.abs(part)
        invalid = (~mx.isfinite(part)) | ((magnitude != 0) & (
            (magnitude < 2.0**-90) | (magnitude > 2.0**90)))
        if bool(mx.any(invalid).item()):
            raise ValueError("LBP2D source sign is unresolved outside the supported "
                             "nonzero source-part magnitude range [2**-90, 2**90]")
    slices = image[None] if image.ndim == 2 else mx.moveaxis(image, axis, 0)
    low = residual[None] if image.ndim == 2 else mx.moveaxis(residual, axis, 0)
    tail = source_tail[None] if image.ndim == 2 else mx.moveaxis(source_tail, axis, 0)
    output = _lbp_kernel(samples, radius, method)(
        inputs=[slices, low, tail],
        grid=(slices.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[slices.shape],
        output_dtypes=[mx.float32],
    )[0]
    return output[0] if image.ndim == 2 else mx.moveaxis(output, 0, axis)
