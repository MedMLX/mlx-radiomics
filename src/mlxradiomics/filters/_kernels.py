"""Fused compensated spatial-filter primitives for float32 Metal arrays.

Each window is summed locally rather than subtracting two prefix sums. Prefix
subtraction changes cancellation and can erase weak windows far from a large
sample. The two-sum residual is retained between separable passes, just as in the
original array-expression implementation of the IBSI mean filter.
"""

from functools import lru_cache
from math import prod
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx

_THREADS = 256


@lru_cache(maxsize=128)
def _window_pair_kernel(ndim: int, axis: int, support: int) -> MetalKernel:
    """Compile one axis/support variant; image sizes remain runtime metadata."""
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_compensated_window_sum",
        input_names=["high", "low"],
        output_names=["sums", "residuals"],
        ensure_row_contiguous=True,
        header="""
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
        """,
        source=f"""
            size_t t = thread_position_in_grid.x;
            size_t stride = 1;
            for (int d = {axis + 1}; d < {ndim}; ++d) {{
                stride *= size_t(high_shape[d]);
            }}
            size_t width = size_t(high_shape[{axis}]) - {support - 1}u;
            size_t inner = width * stride;
            size_t count = inner;
            for (int d = 0; d < {axis}; ++d) {{
                count *= size_t(high_shape[d]);
            }}
            if (t >= count) return;
            // The output omits support-1 samples per outer block. Restore that
            // gap to address the first input sample of this forward window.
            size_t base = t + (t / inner) * {support - 1}u * stride;
            float total = high[base];
            float residual = low[base];
            for (int tap = 1; tap < {support}; ++tap) {{
                size_t index = base + size_t(tap) * stride;
                float term = high[index];
                float added = total + term;
                float back = added - total;
                float error = (total - (added - back)) + (term - back);
                residual = (residual + low[index]) + error;
                total = added;
            }}
            sums[t] = total;
            residuals[t] = residual;
        """,
    ))


def window_pair_sum(
    high: mx.array, low: mx.array, axis: int, support: int,
) -> tuple[mx.array, mx.array]:
    """Valid forward box sum on ``axis``, retaining the low part of every window.

    Inputs are float32 high/residual arrays with identical shapes. The result is
    shorter by ``support - 1`` on the selected axis. One thread owns one output,
    keeping compensation in registers instead of writing image-sized intermediate
    arrays for each tap. Operations preserve the original left-to-right two-sum
    schedule; the compiler cannot contract or reassociate it.
    """
    if high.shape != low.shape or high.dtype != mx.float32 or low.dtype != mx.float32:
        raise ValueError("window pairs need equal-shape float32 arrays")
    if not 0 <= axis < high.ndim:
        raise ValueError("window axis is outside the input dimensions")
    if support < 1 or support > high.shape[axis]:
        raise ValueError("window support must fit the input axis")
    if support == 1:
        return high, low
    shape = list(high.shape)
    shape[axis] -= support - 1
    output_shape = tuple(shape)
    count = prod(output_shape)
    if count == 0:
        return mx.zeros(output_shape, dtype=mx.float32), mx.zeros(output_shape, dtype=mx.float32)
    threads = ((count + _THREADS - 1) // _THREADS) * _THREADS
    sums, residuals = _window_pair_kernel(high.ndim, axis, support)(
        inputs=[high, low], grid=(threads, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[output_shape, output_shape], output_dtypes=[mx.float32, mx.float32],
    )
    return sums, residuals
