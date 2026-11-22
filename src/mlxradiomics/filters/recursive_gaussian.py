"""ITK's recursive Gaussian recurrence (the LoG image type), with the filtering on Metal.

Coefficient equations follow ITK 5.4.4 (Copyright NumFOCUS, Apache-2.0):
Modules/Filtering/Smoothing/include/itkRecursiveGaussianImageFilter.hxx.

The host evaluates the denominator in factored form to avoid cancellation in its DC
normalization. The Metal recurrence filters each row's deviation from its boundary value
and represents coefficients and state as float-float ("double-single") pairs, using the
error-free transforms from :mod:`mlxradiomics.numerics.precision`. The centered form keeps the
analytic constant response exact (order 0 preserves a constant, order 2 maps it to zero),
and pair arithmetic reduces feedback cancellation within the qualified width range.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, cos, exp, fsum, isfinite, prod, sin, sqrt
from sys import float_info
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.precision import PAIR_HEADER
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

# This implementation is qualified through 200 voxels against SimpleITK on
# nonconstant rows. Wider filters have not been qualified; a finite coefficient vector
# alone does not establish a stable recurrence at arbitrary widths.
_MAX_SIGMA_RATIO = 200.0
# Reserve roughly half the host significand after cancellation, keeping the coefficient
# preparation error below float32 precision (sqrt(float64 epsilon)).
_MIN_RELATIVE_SECOND_ORDER_RESPONSE = sqrt(float_info.epsilon)


@dataclass(frozen=True)
class RecursiveCoefficients:
    """Coefficients of the centered causal and anticausal recursions for one sigma and order.

    ``causal_dc`` and ``anticausal_dc`` are the constant gains added after filtering each
    row's deviation from its left or right boundary value. Their sum is the analytic
    constant response: one for order 0 and zero for order 2.
    """

    numerator: tuple[float, ...]
    denominator: tuple[float, ...]
    anticausal: tuple[float, ...]
    causal_dc: float
    anticausal_dc: float


def _numerator(sigma_voxels: float, order: int) -> tuple[float, ...]:
    """Return ITK's Deriche numerator coefficients ``(N0..N3)`` for ``order`` 0 or 2."""
    a1, b1, a2, b2 = {
        0: (1.3530, 1.8151, -0.3531, 0.0902),
        2: (-1.3563, 5.2318, 0.3446, -2.2355),
    }[order]
    theta1, theta2 = 0.6681 / sigma_voxels, 2.0787 / sigma_voxels
    s1, s2 = sin(theta1), sin(theta2)
    c1, c2 = cos(theta1), cos(theta2)
    e1, e2 = exp(-1.3932 / sigma_voxels), exp(-1.3732 / sigma_voxels)
    return (
        a1 + a2,
        e2 * (b2 * s2 - (a2 + 2 * a1) * c2) + e1 * (b1 * s1 - (a1 + 2 * a2) * c1),
        2 * e1 * e2 * ((a1 + a2) * c2 * c1 - b1 * c2 * s1 - b2 * c1 * s2) + a2 * e1**2 + a1 * e2**2,
        e2 * e1**2 * (b2 * s2 - a2 * c2) + e1 * e2**2 * (b1 * s1 - a1 * c1),
    )


def _denominator(sigma_voxels: float) -> tuple[tuple[float, ...], float, float, float]:
    """Return ``(D1..D4, SD, DD, ED)`` with the DC normalization evaluated in factored form.

    In the delay variable ``u``, ``D(u)`` is the product of the two quadratics
    ``1 - 2 e_i cos(theta_i) u + e_i^2 u^2``. ``SD = D(1)``, ``DD = D'(1)`` and
    ``ED = D'(1) + D''(1)`` come from these factors, avoiding the cancellation
    of summing the expanded coefficients when both poles approach one.
    """
    e1, e2 = exp(-1.3932 / sigma_voxels), exp(-1.3732 / sigma_voxels)
    theta1, theta2 = 0.6681 / sigma_voxels, 2.0787 / sigma_voxels
    c1, c2 = cos(theta1), cos(theta2)
    p1, p2 = -2 * e1 * c1, -2 * e2 * c2
    q1, q2 = e1**2, e2**2
    # (1 - e)^2 + 4 e sin^2(theta / 2) == 1 - 2 e cos(theta) + e^2, without cancellation.
    f1 = (1 - e1) ** 2 + 4 * e1 * sin(theta1 / 2) ** 2
    f2 = (1 - e2) ** 2 + 4 * e2 * sin(theta2 / 2) ** 2
    sd = f1 * f2
    dp1, dp2 = p1 + 2 * q1, p2 + 2 * q2
    second = 2 * q1 * f2 + 2 * dp1 * dp2 + 2 * q2 * f1
    dd = dp1 * f2 + f1 * dp2
    ed = dd + second
    d = (p1 + p2, q1 + q2 + p1 * p2, p1 * q2 + p2 * q1, q1 * q2)
    return d, sd, dd, ed


def _coefficients(sigma: float, spacing: float, order: int) -> RecursiveCoefficients:
    """Return stable ITK recurrence coefficients for one physical sigma and voxel spacing.

    Ratios above 200 voxels are outside this recurrence's qualified range. Second-order
    normalization is also rejected when cancellation leaves fewer than half the host
    significand bits. This is a numerical condition, not a Gaussian sampling theorem.

    Raises:
        ValueError: If parameters are invalid, the ratio exceeds 200, or coefficient
            normalization is singular, ill-conditioned or non-finite.
    """
    if type(order) is not int or order not in (0, 2):
        raise ValueError("Gaussian order must be 0 or 2")
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    sigma_voxels = sigma / spacing
    if not isfinite(sigma_voxels) or sigma_voxels <= 0:
        raise ValueError("sigma / spacing must be finite and positive")
    if sigma_voxels > _MAX_SIGMA_RATIO:
        raise ValueError("recursive Gaussian requires sigma / spacing <= 200")
    # exp(-746) underflows even in host float64: both poles vanish and smoothing is
    # exactly the identity in this arithmetic. The derivative normalization is singular.
    if sigma_voxels < 1.3732 / 746.0:
        if order == 2:
            raise ValueError("second-order recursive Gaussian normalization is singular")
        return RecursiveCoefficients(
            (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 1.0, 0.0
        )

    d, sd, dd, ed = _denominator(sigma_voxels)
    if not isfinite(sd) or sd <= 0:
        raise ValueError(
            f"recursive Gaussian coefficient normalization is singular for sigma / spacing "
            f"= {sigma_voxels!r}"
        )
    n0 = _numerator(sigma_voxels, 0)
    if order == 0:
        normalizer = 2 * fsum(n0) / sd - n0[0]
        if not isfinite(normalizer) or normalizer == 0:
            raise ValueError(
                "recursive Gaussian coefficient normalization is singular for "
                f"sigma / spacing = {sigma_voxels!r}"
            )
        n = tuple(value / normalizer for value in n0)
    else:
        n2 = _numerator(sigma_voxels, 2)
        denominator = 2 * fsum(n0) - sd * n0[0]
        if not isfinite(denominator) or denominator == 0:
            raise ValueError(
                "recursive Gaussian coefficient normalization is singular for "
                f"sigma / spacing = {sigma_voxels!r}"
            )
        beta = -(2 * fsum(n2) - sd * n2[0]) / denominator
        n = tuple(a + beta * b for a, b in zip(n2, n0, strict=True))
        scale = max(max(abs(value) for value in n2),
                    abs(beta) * max(abs(value) for value in n0))
        if max(abs(value) for value in n) <= _MIN_RELATIVE_SECOND_ORDER_RESPONSE * scale:
            raise ValueError("second-order recursive Gaussian normalization is ill-conditioned")
        sigma_squared = sigma * sigma
        if not isfinite(sigma_squared) or sigma_squared == 0:
            raise ValueError("second-order recursive Gaussian scale is outside the numeric range")
        sn = fsum(n)
        dn = fsum(k * value for k, value in enumerate(n))
        en = fsum(k * k * value for k, value in enumerate(n))
        sd3 = sd * sd * sd
        if not isfinite(sd3) or sd3 == 0:
            raise ValueError(
                "recursive Gaussian coefficient normalization is singular for "
                f"sigma / spacing = {sigma_voxels!r}"
            )
        alpha = (en * sd * sd - ed * sn * sd - 2 * dn * dd * sd + 2 * dd * dd * sn) / sd3
        if not isfinite(alpha) or alpha == 0:
            raise ValueError(
                "recursive Gaussian coefficient normalization is singular for "
                f"sigma / spacing = {sigma_voxels!r}"
            )
        n = tuple(value * sigma_squared / alpha for value in n)

    m = tuple(n[k] - d[k - 1] * n[0] for k in range(1, 4)) + (-d[3] * n[0],)
    values = (*n, *d, *m)
    if any(not isfinite(value) for value in values):
        raise ValueError(
            f"recursive Gaussian coefficients are not finite for sigma / spacing = {sigma_voxels!r}"
        )
    causal_dc = fsum(n) / sd
    # Use the analytic constant response instead of the difference of two nearly equal
    # double-precision sums; this is the stable normalization of the centered recurrence.
    anticausal_dc = (1.0 if order == 0 else 0.0) - causal_dc
    if not isfinite(causal_dc) or not isfinite(anticausal_dc):
        raise ValueError(
            f"recursive Gaussian coefficients are not finite for sigma / spacing = {sigma_voxels!r}"
        )
    return RecursiveCoefficients(n, d, m, causal_dc, anticausal_dc)


def _pair_arrays(values: tuple[float, ...]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Split float64 coefficients into float32 high and residual low parts."""
    high = np.asarray(values, dtype=np.float32)
    low = np.asarray(
        [value - float(high[index]) for index, value in enumerate(values)], dtype=np.float32
    )
    if (not np.all(np.isfinite(high)) or not np.all(np.isfinite(low))
            or (any(values[:4]) and not np.any(high[:4]) and not np.any(low[:4]))):
        raise ValueError("recursive Gaussian coefficients are outside the float32 range")
    return high, low


@lru_cache(maxsize=128)
def _recursive_kernel(length: int, stride: int, cropped: bool) -> MetalKernel:
    """Compile the pair-precision causal/anticausal recursion on a native array axis."""
    bounds = "int start = interval[0], stop = interval[1];" if cropped else (
        f"constexpr int start = 0, stop = {length};"
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_recursive_gaussian",
            header=PAIR_HEADER,
            input_names=["image", "coeff_hi", "coeff_lo"] + (["interval"] if cropped else []),
            output_names=["causal_lo", "output"],
            source=f"""
        uint t = thread_position_in_grid.x;
        uint total = 1;
        for (int axis = 0; axis < image_ndim; axis++) total *= uint(image_shape[axis]);
        if (t >= total / {length}u) return;
        uint base = (t / {stride}u) * {length * stride}u + t % {stride}u;
        {bounds}
        uint target = (t / {stride}u) * uint(stop - start) * {stride}u + t % {stride}u;
        Pair n[4], d[4], m[4];
        for (int k = 0; k < 4; k++) {{
            n[k] = Pair{{coeff_hi[k], coeff_lo[k]}};
            d[k] = Pair{{coeff_hi[4 + k], coeff_lo[4 + k]}};
            m[k] = Pair{{coeff_hi[8 + k], coeff_lo[8 + k]}};
        }}
        Pair kc = Pair{{coeff_hi[12], coeff_lo[12]}};
        Pair ka = Pair{{coeff_hi[13], coeff_lo[13]}};
        Pair x0 = Pair{{image[base], 0.0f}};
        Pair xn = Pair{{image[base + {(length - 1) * stride}u], 0.0f}};
        Pair zero = Pair{{0.0f, 0.0f}};
        Pair history[4] = {{zero, zero, zero, zero}};

        // Each recurrence reads only its four previous states. Keep them local
        // and retain each causal pair until the reverse pass can sum the result.
        for (int i = 0; i < stop; i++) {{
            Pair numerator = zero, feedback = zero;
            #pragma unroll
            for (int k = 0; k < 4; k++) {{
                Pair sample = Pair{{image[base + uint(max(i - k, 0)) * {stride}u], 0.0f}};
                numerator = add(numerator, multiply(subtract(sample, x0), n[k]));
                if (i > k) {{
                    feedback = add(feedback, multiply(history[k], d[k]));
                }}
            }}
            Pair value = subtract(numerator, feedback);
            history[3] = history[2];
            history[2] = history[1];
            history[1] = history[0];
            history[0] = value;
            if (i >= start) {{
                uint index = target + uint(i - start) * {stride}u;
                output[index] = value.hi;
                causal_lo[index] = value.lo;
            }}
        }}

        Pair offset = add(multiply(kc, x0), multiply(ka, xn));
        for (int i = {length - 1}; i >= start; i--) {{
            Pair numerator = zero, feedback = zero;
            #pragma unroll
            for (int k = 0; k < 4; k++) {{
                uint at = base + uint(min(i + k + 1, {length - 1})) * {stride}u;
                Pair sample = Pair{{image[at], 0.0f}};
                numerator = add(numerator, multiply(subtract(sample, xn), m[k]));
                if (i + k + 1 < {length}) {{
                    feedback = add(feedback, multiply(history[k], d[k]));
                }}
            }}
            Pair anticausal = subtract(numerator, feedback);
            history[3] = history[2];
            history[2] = history[1];
            history[1] = history[0];
            history[0] = anticausal;
            if (i < stop) {{
                uint index = target + uint(i - start) * {stride}u;
                Pair causal = Pair{{output[index], causal_lo[index]}};
                Pair value = add(add(causal, anticausal), offset);
                output[index] = value.hi + value.lo;
            }}
        }}
        """,
        ),
    )


def recursive_gaussian(
    image: mx.array, sigma: float, spacing: float, axis: int, order: int, *,
    interval: tuple[int, int] | None = None,
) -> mx.array:
    """Filter ``image`` along one axis with a recursive Gaussian of order 0 or 2.

    ``sigma`` and ``spacing`` share a unit (mm); the kernel width in voxels is
    ``sigma / spacing``. Order 0 smooths; order 2 applies the ITK second-derivative
    filter, and ``laplacian_gaussian`` divides its result by the squared spacing.
    Boundaries continue the edge value as a constant.

    The recurrence is evaluated in a centered float-float form, so constant rows retain
    their analytic response (order 0: unchanged; order 2: zero). The supported width is
    at most 200 voxels; second-order normalization must retain at least half the host
    float64 significand after cancellation. The result is float32 on Metal.

    Args:
        image: Array with at least four samples along ``axis``.
        sigma: Gaussian sigma in physical units, finite and positive.
        spacing: Voxel spacing along ``axis`` in the same units, finite and positive.
        axis: NumPy axis to filter.
        order: 0 for smoothing or 2 for the second derivative.
        interval: Optional nonempty output interval along the filtered axis. Both
            recurrences still start at the original endpoints.

    Returns:
        A float32 array with the filtered axis restricted to the output interval.

    Raises:
        ValueError: If ``axis`` does not index the image, ``order`` is not 0 or 2,
            ``sigma`` or ``spacing`` is invalid, the filtered axis has fewer than four
            samples, the interval lies outside the axis, or the requested second-order
            ratio is unsupported.
    """
    if type(axis) is not int or not 0 <= axis < image.ndim:
        raise ValueError("axis must index the image's dimensions")
    coefficients = _coefficients(sigma, spacing, order)
    length = image.shape[axis]
    if length < 4:
        raise ValueError("recursive Gaussian requires at least four samples along the axis")
    start, stop = (0, length) if interval is None else interval
    if not 0 <= start < stop <= length:
        raise ValueError("recursive Gaussian output interval must be inside the filtered axis")
    cropped = start != 0 or stop != length
    # Element step along ``axis`` in the row-major layout the kernel receives.
    stride = prod(image.shape[axis + 1:])
    high, low = _pair_arrays(
        coefficients.numerator
        + coefficients.denominator
        + coefficients.anticausal
        + (coefficients.causal_dc, coefficients.anticausal_dc)
    )
    inputs = [image, mx.array(high), mx.array(low)]
    if cropped:
        inputs.append(mx.array((start, stop), dtype=mx.int32))
    return _recursive_kernel(length, stride, cropped)(
        inputs=inputs,
        grid=(image.size // length, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*image.shape[:axis], stop - start, *image.shape[axis + 1:])] * 2,
        output_dtypes=[mx.float32] * 2,
    )[1]


def validate_log_domain(shape: tuple[int, ...], spacing: tuple[float, ...], sigma: float) -> None:
    """Raise ``ValueError`` unless ``laplacian_gaussian`` accepts this grid and sigma.

    Shared with the derived-image pre-flight so an unsupported request fails before
    any image is extracted, with the same message the filter would raise.
    """
    if len(shape) != 3 or any(n < 4 for n in shape):
        raise ValueError("recursive LoG requires a 3D image with every axis at least four voxels")
    if len(spacing) != 3 or any(not isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("LoG spacing must contain three finite positive values")
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("LoG sigma must be finite and positive")
    for axis_spacing in spacing:
        _coefficients(sigma, axis_spacing, 2)
    if any(n < ceil(sigma / s) + 1 for n, s in zip(shape, spacing, strict=True)):
        raise ValueError("LoG sigma exceeds the image's physical support")


def laplacian_gaussian(
    image: mx.array, spacing: tuple[float, ...], sigma: float, *,
    window: tuple[slice, ...] | None = None,
) -> mx.array:
    """Laplacian of Gaussian of a 3-D image (PyRadiomics ``log`` image type).

    The sum over axes of the second derivative along that axis, with Gaussian smoothing
    along the other two, each as a recursive filter.

    Args:
        image: Float32 array in NumPy axis order, every axis at least 4 voxels.
        spacing: Voxel spacing in mm per array axis, three finite positive values.
        sigma: Gaussian sigma in mm, finite and positive. Each axis must satisfy the
            recursive filter's width and second-order normalization conditions.
        window: Optional contiguous output slices in the original image coordinates.

    Returns:
        Full or windowed float32 response.

    Raises:
        ValueError: If the image is not 3-D with every axis at least 4 voxels, ``spacing``
            is not three finite positive values, ``sigma`` is invalid or too large for the
            image's physical extent, or an axis has an unsupported sigma ratio.
    """
    validate_log_domain(tuple(image.shape), spacing, sigma)
    intervals: tuple[tuple[int, int] | None, ...] = (None, None, None)
    output_shape = image.shape
    if window is not None:
        bounds = tuple(
            part.indices(length) for part, length in zip(window, image.shape, strict=True)
        )
        if any(step != 1 for _, _, step in bounds):
            raise ValueError("recursive LoG output window requires contiguous slices")
        output_shape = tuple(max(0, stop - start) for start, stop, _ in bounds)
        if 0 in output_shape:
            return mx.zeros(output_shape, dtype=mx.float32)
        intervals = tuple((start, stop) for start, stop, _ in bounds)
    output = mx.zeros(output_shape, dtype=image.dtype)
    # ITK traverses physical x,y,z axes, opposite the NumPy array-axis order.
    # After a pass, the remaining passes act on different axes and cannot read
    # its discarded output coordinates. Every pass still reads complete input lines.
    # Each pass is evaluated before the next is built, so its input can be released;
    # otherwise one command buffer keeps every pass's buffers alive until it completes.
    for derivative_axis in range(2, -1, -1):
        filtered = recursive_gaussian(
            image, sigma, spacing[derivative_axis], derivative_axis, 2,
            interval=intervals[derivative_axis],
        )
        evaluate(filtered)
        for smooth_axis in range(2, -1, -1):
            if smooth_axis != derivative_axis:
                filtered = recursive_gaussian(
                    filtered, sigma, spacing[smooth_axis], smooth_axis, 0,
                    interval=intervals[smooth_axis],
                )
                evaluate(filtered)
        output = output + filtered / spacing[derivative_axis] ** 2
        evaluate(output)
    return output
