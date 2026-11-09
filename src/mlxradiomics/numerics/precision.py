"""Source (float64) ordering and float-float arithmetic for intensity features.

Host preprocessing retains float64 intensities, but native feature arithmetic runs on
float32 MLX streams. A float32 value plus its residual (``source - float32(source)``)
forms an approximately 48-bit float-float ("double-single") pair that retains source
variation where float32 alone would collapse distinct intensities, for example near
``1e8``. Pair operations
use compensated transforms (two-sum and fused multiply-add) in Metal; they do not
provide full float64 arithmetic or its exponent range.
"""

from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx

# Float-float ("double-single") arithmetic: a value is hi + lo, where lo holds the rounding
# error of hi, built from error-free sum and fused-multiply-add products.
PAIR_HEADER = """
struct Pair { float hi; float lo; };
Pair add(Pair a, Pair b) {
    float sum = a.hi + b.hi, v = sum - a.hi;
    float error = (a.hi - (sum - v)) + (b.hi - v) + a.lo + b.lo;
    float hi = sum + error;
    return {hi, error - (hi - sum)};
}
Pair negate(Pair a) { return {-a.hi, -a.lo}; }
Pair subtract(Pair a, Pair b) { return add(a, negate(b)); }
Pair multiply(Pair a, Pair b) {
    float value = a.hi * b.hi;
    float error = fma(a.hi, b.hi, -value) + a.hi * b.lo + a.lo * b.hi;
    float hi = value + error;
    return {hi, error - (hi - value)};
}
bool less(Pair a, Pair b) {
    // Equal high parts leave only the residual ordering.
    if (a.hi == b.hi) return a.lo < b.lo;
    // Opposite-sign high parts would overflow in subtract even though every finite
    // value is ordered by its sign. This keeps percentile bounds usable at +-3e38.
    bool a_negative = signbit(a.hi), b_negative = signbit(b.hi);
    if (a_negative != b_negative) return a_negative;
    // Same-sign subtraction cannot overflow; the compensated residual still decides
    // values whose high parts differ by one ulp.
    Pair d = subtract(a, b);
    return d.hi < 0 || (d.hi == 0 && d.lo < 0);
}
"""


def prepare_source_pair(
    source: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Prepare float32 source parts and their Metal-visible representation error.

    ``high`` is ``float32(source)`` and ``low`` is ``float32(source - high)``. Metal
    flushes float32 subnormals to zero, so either part below the smallest normal float32
    contributes zero here, as it does on the device. ``low`` has only 24 significant bits,
    so once ``source - high`` needs more (sources far larger than their mutual
    differences, for example ``1e12 + 1e-3 * k``) the pair no longer represents the
    source and the gap is returned. The float64 gap is rounded up to float32, so the
    result never understates it. Non-finite sources give NaN; callers read only sampled
    values.
    """
    tiny = float(np.finfo(np.float32).tiny)
    with np.errstate(invalid="ignore", over="ignore"):
        high = source.astype(np.float32)
        high64 = high.astype(np.float64)
        low64 = np.empty_like(high64, dtype=np.float64)
        np.subtract(source, high64, out=low64)
        low = low64.astype(np.float32)
        # Exactly represented normal values have neither a residual nor a Metal
        # flushing error. This includes ordinary float32 and integer CT sources.
        if not np.any(low64) and not np.any((high != 0) & (np.abs(high) < tiny)):
            return high, low, np.zeros_like(high, dtype=np.float32)
        # Reuse the two float64 scratch arrays for the Metal-visible parts and gap.
        # The returned raw parts stay untouched, including their subnormal values.
        high64[np.abs(high) < tiny] = np.float64(0)
        np.subtract(source, high64, out=high64)
        np.copyto(low64, low, casting="safe")
        low64[np.abs(low) < tiny] = np.float64(0)
        np.subtract(high64, low64, out=high64)
        np.abs(high64, out=high64)
        gap = high64
        error = gap.astype(np.float32)
        error = np.where(error < gap, np.nextafter(error, np.float32(np.inf)), error)
    return high, low, error


def pair_representation_error(source: NDArray[np.float64]) -> NDArray[np.float32]:
    """Bound the error in the source pair with Metal's subnormal flushing included.

    See :func:`prepare_source_pair`; callers needing the source parts can prepare
    all three together rather than repeating the float32 conversion and residual.
    """
    return prepare_source_pair(source)[2]


@lru_cache(maxsize=1)
def _robust_kernel() -> MetalKernel:
    """Compile the kernel marking valid samples between the 10th and 90th percentile pairs."""
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_robust_membership",
            header=PAIR_HEADER,
            input_names=["values", "residual", "valid", "highs", "lows"],
            output_names=["robust"],
            source="""
        uint t = thread_position_in_grid.x;
        uint width = values_shape[1];
        if (t >= values_shape[0] * width) return;
        uint base = (t / width) * 5;
        Pair lower = {highs[base], lows[base]};
        Pair upper = {highs[base + 4], lows[base + 4]};
        Pair value = {values[t], residual[t]};
        robust[t] = valid[t] && !less(value, lower) && !less(upper, value);
        """,
        ),
    )


@lru_cache(maxsize=2)
def _quantile_kernel(ordered_endpoints: bool = False) -> MetalKernel:
    """Compile the kernel interpolating the five percentile pairs per row.

    Exact rational interpolation positions avoid float32 rounding of ``(n - 1) * q``.
    """
    source = """
        uint t = thread_position_in_grid.x;
        uint rows = counts_shape[0];
        if (t >= rows * 5u) return;
        uint row = t / 5u, k = t % 5u;
        int n = counts[row] - 1;
        if (n < 0) { highs[t] = 0.0f; lows[t] = 0.0f; return; }
        int p = 1, den = 2;
        if (k == 0) { p = 1; den = 10; }
        else if (k == 1) { p = 1; den = 4; }
        else if (k == 3) { p = 3; den = 4; }
        else if (k == 4) { p = 9; den = 10; }
        int whole = n / den, partial = n % den;
        int lower = whole * p + (partial * p) / den;
        int remainder = (partial * p) % den;
        int upper = min(lower + 1, n);
        float fraction = float(remainder) / float(den);
        // Preserve the coefficient's rounding error before multiplying the endpoints.
        float fraction_low = fma(-fraction, float(den), float(remainder)) / float(den);
        uint width = values_shape[1], base = row * width;
        uint left = base + order[base + uint(lower)];
        uint right = base + order[base + uint(upper)];
        Pair a = {values[left], residual[left]}, b = {values[right], residual[right]};
        Pair weight = {fraction, fraction_low};
        // Opposite signs can overflow b-a even though every convex combination is
        // finite. Weighted endpoints have opposite signs and add without that span.
        Pair value = signbit(a.hi) != signbit(b.hi)
            ? add(multiply(a, subtract(Pair{1.0f, 0.0f}, weight)), multiply(b, weight))
            : add(a, multiply(subtract(b, a), weight));
        highs[t] = value.hi;
        lows[t] = value.lo;
        """
    if ordered_endpoints:
        # Only endpoint addressing changes. Rank fractions, compensated coefficient
        # division, sign handling and Pair interpolation keep their source order.
        source = source.replace(
            "uint left = base + order[base + uint(lower)];\n"
            "        uint right = base + order[base + uint(upper)];",
            "uint left = base + 2u * k;\n"
            "        uint right = left + 1u;",
        )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=("mlxradiomics_source_quantile_endpoints" if ordered_endpoints
                  else "mlxradiomics_source_quantiles"),
            header=PAIR_HEADER,
            input_names=(["values", "residual", "counts"] if ordered_endpoints
                         else ["values", "residual", "order", "counts"]),
            output_names=["highs", "lows"],
            source=source,
        ),
    )


@lru_cache(maxsize=1)
def _order_key_kernel() -> MetalKernel:
    """Compile the kernel mapping float64 source bits to unsigned sort keys.

    Invalid samples get the largest key, so they sort last.
    """
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_source_order_keys",
            input_names=["bits", "valid"],
            output_names=["keys"],
            source="""
        uint t = thread_position_in_grid.x;
        if (t >= bits_shape[0] * bits_shape[1]) return;
        ulong value = bits[t];
        ulong key = (value >> 63) != 0 ? ~value : value ^ (1ul << 63);
        keys[t] = valid[t] ? key : ~0ul;
        """,
        ),
    )


def source_order(bits: mx.array, valid: mx.array) -> mx.array:
    """Return, per row, indexes of the valid entries in source (float64) value order.

    IEEE-754 float64 bit patterns are mapped to unsigned keys that sort in numeric
    order (sign bit flipped for nonnegative values, all bits flipped for negative
    values), then sorted along the last axis. Invalid entries receive the largest key and
    sort after every valid entry.

    Args:
        bits: uint64 array ``(batch, n)`` holding the float64 source bit patterns.
        valid: Boolean array of the same shape; voxels belonging to each row's set.

    Returns:
        int32 array ``(batch, n)`` of column indexes into the source row.
    """
    # One integer pass forms the keys; MLX's argsort then orders them.
    keys = _order_key_kernel()(
        inputs=[bits, valid], grid=(bits.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[bits.shape], output_dtypes=[mx.uint64],
    )[0]
    return mx.argsort(keys, axis=1).astype(mx.int32)


def source_quantiles(
    values: mx.array,
    residual: mx.array,
    order: mx.array,
    count: mx.array,
) -> tuple[mx.array, mx.array]:
    """Linear-interpolated 10/25/50/75/90 percentiles at source precision.

    Interpolation follows the NumPy default definition: position ``(n - 1) * q`` with
    linear interpolation between the adjacent order statistics. The integer position and
    interpolation fraction are derived from integer arithmetic on the rational form
    of ``q``; compensated division retains the fractional coefficient's residual. Pair
    interpolation keeps residual-scale differences that a float32 value cannot represent.
    Opposite-sign endpoints use a convex weighted sum to avoid an overflowing span.

    Args:
        values: Float32 array ``(batch, n)`` of intensities.
        residual: Float32 array of the same shape, ``source - float32(source)``.
        order: int32 source-order indexes from :func:`source_order`.
        count: Number of valid voxels per row, shape ``(batch,)``.

    Returns:
        ``(high, low)`` float32 arrays shaped ``(batch, 5)``. For each row, the columns
        are the 10th, 25th, 50th, 75th and 90th percentiles as a float-float pair.
    """
    outputs = _quantile_kernel()(
        inputs=[values, residual, order, count],
        grid=(values.shape[0] * 5, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(values.shape[0], 5), (values.shape[0], 5)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs[0], outputs[1]


def source_quantile_endpoints(
    values: mx.array, residual: mx.array, count: mx.array,
) -> tuple[mx.array, mx.array]:
    """Interpolate ten preselected source endpoints with the source quantile kernel.

    Rows contain adjacent lower/upper endpoint pairs for the 10/25/50/75/90 percentiles.
    ``count`` remains the original population, so rational interpolation coefficients
    match :func:`source_quantiles` exactly. Only source endpoint addressing differs.
    """
    outputs = _quantile_kernel(ordered_endpoints=True)(
        inputs=[values, residual, count],
        grid=(values.shape[0] * 5, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(values.shape[0], 5), (values.shape[0], 5)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs[0], outputs[1]


def robust_membership(
    values: mx.array,
    residual: mx.array,
    valid: mx.array,
    quantiles_high: mx.array,
    quantiles_low: mx.array,
) -> mx.array:
    """Flag voxels within the 10th to 90th percentile range, comparing at source precision.

    PyRadiomics' RobustMeanAbsoluteDeviation keeps values between the 10th and 90th
    percentiles (NumPy linear interpolation, bounds inclusive) in float64. Float32 values
    can fall on the wrong side of a bound, so each comparison uses the float32 value plus
    its float32 residual in float-float arithmetic. Ordering compares signs before
    subtracting, so opposite-sign bounds near the float32 limit cannot overflow the
    comparison.

    Args:
        values: Float32 array ``(batch, n)`` of intensities.
        residual: Float32 array of the same shape, ``source - float32(source)``.
        valid: Boolean array of the same shape; voxels belonging to each row's set.
        quantiles_high: High parts from ``source_quantiles``, shape ``(batch, 5)``.
        quantiles_low: Corresponding low parts. Membership uses the same compensated
            bounds as the reported percentiles rather than recalculating them.

    Returns:
        Boolean array ``(batch, n)``, True for valid voxels inside the closed range.
    """
    return _robust_kernel()(
        inputs=[values, residual, valid, quantiles_high, quantiles_low],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.bool_],
    )[0]
