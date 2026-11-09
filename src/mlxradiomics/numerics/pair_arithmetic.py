"""Ordered float-float pointwise subtraction on the explicit MLX GPU stream.

First-order features subtract source pairs before rounding their low words away.
These kernels perform each float32 operation of that subtraction and its overflow
policy in one dispatch; the folded variant also applies the caller's row scaling,
word fold and mask selection. They do not replace or reorder any of the subsequent reduction trees.
"""

from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx

_HEADER = """
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    inline float2 subtract_pair(float high, float low,
                                float other_high, float other_low) {
        volatile float difference = high - other_high;
        volatile float around = difference - high;
        volatile float back = difference - around;
        volatile float left = high - back;
        volatile float right = (-other_high) - around;
        volatile float error = left + right;
        volatile float residual = low - other_low;
        error = error + residual;
        volatile float rounded = difference + error;
        volatile float adjustment = rounded - difference;
        return float2(rounded, error - adjustment);
    }
    inline float2 subtract_pair_or_overflow(float a, float b, float c, float d) {
        volatile float span = a - c;
        if (isinf(span) && isfinite(a) && isfinite(c)) {
            // Multiplication by one half is exact in this overflow branch:
            // both high operands must lie far above the normal-range floor.
            volatile float ah = a * 0.5f, bh = b * 0.5f;
            volatile float ch = c * 0.5f, dh = d * 0.5f;
            float2 reduced = subtract_pair(ah, bh, ch, dh);
            volatile float restored = reduced.x * 2.0f;
            volatile float restored_low = reduced.y * 2.0f;
            return float2(restored, isinf(restored) ? 0.0f : restored_low);
        }
        return subtract_pair(a, b, c, d);
    }
"""


@lru_cache(maxsize=1)
def _subtract_kernel() -> MetalKernel:
    """Compile the elementwise pair subtraction kernel."""
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_pair_subtract",
        input_names=["high", "low", "other_high", "other_low"],
        output_names=["difference_high", "difference_low"],
        header=_HEADER,
        source="""
            uint t = thread_position_in_grid.x;
            ulong size = 1;
            for (int axis = 0; axis < high_ndim; ++axis) size *= high_shape[axis];
            if (t >= size) return;
            float a = high[t], b = low[t], c = other_high[t], d = other_low[t];
            float2 result = subtract_pair_or_overflow(a, b, c, d);
            difference_high[t] = result.x;
            difference_low[t] = result.y;
        """,
    ))


def subtract_pairs(
    high: mx.array, low: mx.array, other_high: mx.array, other_low: mx.array,
) -> tuple[mx.array, mx.array]:
    """Subtract broadcast-compatible float32 pairs without folding their low words.

    Inputs follow NumPy broadcasting, including scalars and row/column centers.
    MLX's custom-kernel boundary makes the broadcast views row contiguous before
    linear indexing. Each element uses the two-sum subtraction, and finite overflowing
    high-part spans use a half-scale recovery path.
    """
    inputs = (high, low, other_high, other_low)
    if any(value.dtype != mx.float32 for value in inputs):
        raise ValueError("pair subtraction requires float32 inputs")
    high, low, other_high, other_low = mx.broadcast_arrays(*inputs)
    if high.size == 0:
        return high, low
    shape = high.shape
    # MLX binds rank-zero custom-kernel inputs by value. A one-element view keeps
    # the kernel's pointer indexing uniform; restore the broadcast shape afterwards.
    if not shape:
        high, low, other_high, other_low = (
            mx.reshape(value, (1,)) for value in (high, low, other_high, other_low)
        )
    result = _subtract_kernel()(
        inputs=[high, low, other_high, other_low], grid=(high.size, 1, 1),
        threadgroup=(min(256, high.size), 1, 1), output_shapes=[high.shape, high.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return mx.reshape(result[0], shape), mx.reshape(result[1], shape)


@lru_cache(maxsize=16)
def _folded_kernel(scaling: str, select: str, shared_other: bool) -> MetalKernel:
    """Compile the :func:`folded_pair_differences` variant for one scaling and selection."""
    inputs = ["high", "low", "other_high", "other_low"]
    other = "0" if shared_other else "row"
    operands = "a, b, c, d"
    scale = ""
    if scaling != "none":
        inputs.append("scale")
        operator = "*" if scaling == "multiply" else "/"
        scale = f"""
            float factor = scale[row];
            volatile float sa = a {operator} factor, sb = b {operator} factor;
            volatile float sc = c {operator} factor, sd = d {operator} factor;"""
        operands = "sa, sb, sc, sd"
    if select == "mask":
        inputs.append("mask")
        store = "folded[t] = mask[t] ? difference : 0.0f;"
    elif select == "finite":
        store = """volatile float span = a - c;
            folded[t] = isfinite(span) ? difference : span;"""
    else:
        store = "folded[t] = difference;"
    row = "" if shared_other and scaling == "none" else "\n            uint row = t / width;"
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_folded_pair_difference", input_names=inputs,
        output_names=["folded"], header=_HEADER,
        source=f"""
            uint t = thread_position_in_grid.x;
            uint width = high_shape[1];
            if (t >= high_shape[0] * width) return;{row}
            float a = high[t], b = low[t];
            float c = other_high[{other}], d = other_low[{other}];{scale}
            float2 result = subtract_pair_or_overflow({operands});
            volatile float difference = result.x + result.y;
            {store}
        """,
    ))


def folded_pair_differences(
    high: mx.array, low: mx.array, other_high: mx.array, other_low: mx.array, *,
    scale: mx.array | None = None, divide: bool = False, mask: mx.array | None = None,
    keep_overflow: bool = False,
) -> mx.array:
    """Fold row pairs minus one pair per row, as one dispatch without intermediates.

    ``high``/``low`` are ``(rows, n)`` float32. ``other_high``/``other_low`` hold one pair
    per row, ``(rows,)``, or one shared pair, ``(1,)``. With ``scale`` ``(rows,)``, all four
    operands are first multiplied (with ``divide``, divided) by the row's scale. Each
    element then takes :func:`subtract_pairs`'s two-sum and overflow path and adds the
    two result words. ``mask`` selects positive zero outside it; ``keep_overflow``
    instead returns the unscaled ``high - other_high`` wherever that is not finite. The
    result equals those MLX operations evaluated separately.
    """
    inputs = [high, low, other_high, other_low]
    scaling = "none"
    if scale is not None:
        inputs.append(scale)
        scaling = "divide" if divide else "multiply"
    select = "none"
    if mask is not None:
        inputs.append(mask)
        select = "mask"
    elif keep_overflow:
        select = "finite"
    return _folded_kernel(scaling, select, other_high.shape == (1,))(
        inputs=inputs, grid=(high.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[high.shape], output_dtypes=[mx.float32],
    )[0]
