"""Scale-safe geometry arithmetic and compensated Metal primitives."""

from dataclasses import dataclass
from functools import lru_cache
from math import frexp, isfinite, ldexp
from typing import cast

from mlxradiomics.numerics.runtime import evaluate, mx


def cross_product(a: mx.array, b: mx.array) -> mx.array:
    """Cross product of 3-vectors along the last axis."""
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class SpacingParts:
    """Binary decomposition of positive spacing into exact power-of-two factors.

    ``spacing[i] == mantissa[i] * 2**exponent[i]`` with each mantissa in ``[0.5, 1)``.
    Boundary measures (2-D perimeter, 3-D area) are accumulated as
    ``boundary_norm * 2**boundary_exponent``, and enclosed measures (2-D area, 3-D
    volume) as ``enclosed_norm * 2**enclosed_exponent``. Keeping the binary exponent
    separate avoids forming squared or cubed spacing when those units overflow or
    underflow float32 even though the requested physical value is representable.
    """

    mantissa: tuple[float, ...]
    exponent: tuple[int, ...]
    boundary_exponent: int
    enclosed_exponent: int


def spacing_parts(spacing: tuple[float, ...]) -> SpacingParts:
    """Return the binary spacing decomposition used by scaled mesh arithmetic."""
    if len(spacing) not in (2, 3):
        raise ValueError("shape spacing must have two or three values")
    mantissa: list[float] = []
    exponent: list[int] = []
    for value in spacing:
        number = float(value)
        if not isfinite(number) or number <= 0:
            raise ValueError("shape spacing values must be finite and positive")
        fraction, power = frexp(number)
        mantissa.append(fraction)
        exponent.append(power)
    if len(spacing) == 3:
        # Triangle areas contain products of two spacing axes; the largest pair sets
        # the natural binary scale for every face in the mesh.
        boundary_exponent = max(exponent[i] + exponent[j] for i, j in ((0, 1), (1, 2), (2, 0)))
    else:
        boundary_exponent = max(exponent)
    return SpacingParts(
        tuple(mantissa),
        tuple(exponent),
        boundary_exponent,
        sum(exponent),
    )


@lru_cache(maxsize=1)
def _pow2_table() -> mx.array:
    """Exact float32 powers of two from ``2**-149`` through ``2**127``."""
    return mx.array([ldexp(1.0, power) for power in range(-149, 128)], dtype=mx.float32)


def scale_pow2(value: mx.array, exponent: int | mx.array) -> mx.array:
    """Multiply ``value`` by ``2**exponent`` without an overflowing intermediate.

    Stages are limited to representable powers of two, so a large positive exponent
    combined with a small mantissa (or the reverse) reaches its representable result
    instead of producing ``inf``, ``0`` or ``nan`` during the scaling itself.
    """
    powers = exponent if isinstance(exponent, mx.array) else mx.array(exponent, dtype=mx.int32)
    powers = powers.astype(mx.int32)
    table = _pow2_table()
    for _ in range(4):
        step = mx.clip(powers, -127, 127)
        value = value * mx.take(table, step + 149)
        powers = powers - step
    step = mx.clip(powers, -149, 127)
    return value * mx.take(table, step + 149)


def scaled_ratio(
    numerator: mx.array,
    numerator_exponent: int,
    denominator: mx.array,
    denominator_exponent: int,
) -> mx.array:
    """Evaluate ``numerator / denominator`` from separately scaled magnitudes."""
    return scale_pow2(numerator / denominator, numerator_exponent - denominator_exponent)


# Double-single ("float-float") arithmetic for the shape spectrum. A bare float32
# covariance cannot represent the 0.25 that distinguishes a thin but full-rank ROI from
# an exactly collinear one at coordinates near 4096. Products are expanded with an FMA
# residual and sums carry a low word, so cancellation in the covariance and Jacobi
# rotations retains about 48 significant bits on Metal.
DS_HEADER = r"""
#include <metal_stdlib>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline float2 ds_add(float2 a, float2 b) {
    volatile float s = a.x + b.x;
    volatile float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    float t = e + (a.y + b.y);
    volatile float h = s + t;
    return float2(h, t - (h - s));
}
inline float2 ds_neg(float2 a) { return float2(-a.x, -a.y); }
inline float2 ds_sub(float2 a, float2 b) { return ds_add(a, ds_neg(b)); }
inline float2 ds_mul(float2 a, float2 b) {
    volatile float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e += a.x * b.y + a.y * b.x;
    e += a.y * b.y;
    return ds_add(float2(p, 0), float2(e, 0));
}
inline float2 ds_div(float2 a, float b) {
    float q = a.x / b;
    float2 r = ds_sub(a, ds_mul(float2(q, 0), float2(b, 0)));
    return ds_add(float2(q, 0), float2((r.x + r.y) / b, 0));
}
inline float2 ds_ratio(float2 a, float2 b) {
    float q = a.x / b.x;
    float2 r = ds_sub(a, ds_mul(float2(q, 0), b));
    return ds_add(float2(q, 0), float2((r.x + r.y) / b.x, 0));
}
inline float2 ds_abs(float2 a) {
    return (a.x < 0 || (a.x == 0 && a.y < 0)) ? ds_neg(a) : a;
}
inline float2 ds_rsqrt(float2 a) {
    float2 r = float2(rsqrt(a.x), 0);
    for (uint iteration = 0; iteration < 2; ++iteration) {
        float2 e = ds_sub(float2(1, 0), ds_mul(a, ds_mul(r, r)));
        r = ds_add(r, ds_mul(ds_mul(float2(0.5f, 0), r), e));
    }
    return r;
}
inline float2 ds_sqrt(float2 a) { return ds_mul(a, ds_rsqrt(a)); }
inline bool ds_gt(float2 a, float2 b) {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}
inline bool ds_le(float2 a, float2 b) { return !ds_gt(a, b); }
inline bool ds_finite(float2 a) { return isfinite(a.x) && isfinite(a.y); }
"""


def checked_measure(value: mx.array, name: str) -> mx.array:
    """Return a dimensional measure, rejecting a float32 overflow or underflow.

    A zero boundary or enclosed measure is impossible for a non-empty mask with positive
    spacing, so zero here means the true positive value is below the smallest float32
    subnormal. Reporting that explicitly is preferable to returning a silent zero.
    """
    evaluate(value)
    number = float(cast(float, value.item()))
    if not isfinite(number):
        raise RuntimeError(f"{name} exceeds the float32 range")
    if number == 0:
        raise RuntimeError(f"{name} underflows the float32 range")
    return value
