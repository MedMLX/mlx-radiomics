"""Spatial mean, Laws, Gabor and separable-wavelet filters on Metal."""

from math import (
    ceil,
    cos,
    isfinite,
    log2,
    pi,
    prod,
    sin,
)
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.filtering import (
    LAWS_TAPS,
    FilterPadding,
    GaborResponse,
    OrientationPooling,
    validate_laws_kernel,
)
from mlxradiomics.filters._kernels import window_pair_sum
from mlxradiomics.numerics.runtime import evaluate, mx

# Right-angle rotations of a separable 3-D filter, transcribed from Appendix A.2.
# Each entry lists the sources for the k1, k2 and k3 axes; the boolean flips that
# axis. Indices 0, 1 and 2 refer to g1, g2 and g3.
_ROTATIONS_3D: tuple[tuple[tuple[int, bool], ...], ...] = (
    ((0, False), (1, False), (2, False)),
    ((2, True), (1, False), (0, False)),
    ((0, True), (1, False), (2, True)),
    ((2, False), (1, False), (0, True)),
    ((1, False), (2, False), (0, False)),
    ((1, False), (2, True), (0, True)),
    ((1, False), (0, True), (2, False)),
    ((0, True), (1, True), (2, False)),
    ((1, True), (0, False), (2, False)),
    ((2, True), (0, True), (1, False)),
    ((2, True), (1, True), (0, True)),
    ((2, True), (0, False), (1, True)),
    ((1, True), (0, True), (2, True)),
    ((0, False), (1, True), (2, True)),
    ((1, False), (0, False), (2, True)),
    ((2, False), (0, True), (1, True)),
    ((2, False), (1, True), (0, False)),
    ((2, False), (0, False), (1, False)),
    ((0, True), (2, False), (1, False)),
    ((1, True), (2, False), (0, True)),
    ((0, False), (2, False), (1, True)),
    ((0, True), (2, True), (1, True)),
    ((1, True), (2, True), (0, False)),
    ((0, False), (2, True), (1, False)),
)

# The mean filter's unit-weight accumulation may reach the support volume times
# the largest padded value. Switch to a power-of-two downscale before that bound
# exceeds half the float32 range, leaving headroom for the rounded sum.
FLOAT32_SUM_LIMIT = float(np.finfo(np.float32).max) / 2


def _laws_vectors(codes: str) -> tuple[NDArray[np.float64], ...]:
    """Return the validated Laws 1-D kernels of ``codes`` in IBSI axis order (k1, k2, k3)."""
    validate_laws_kernel(codes)
    return tuple(
        np.array(LAWS_TAPS[codes[index:index + 2]], dtype=np.float64)
        for index in range(0, len(codes), 2)
    )


def _kernel_from_vectors(vectors: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    """Form the outer-product Laws kernel from 1-D vectors in IBSI order.

    Array axis order is reversed relative to IBSI: ``(k3, k2, k1)``.
    """
    if len(vectors) == 2:
        return cast(NDArray[np.float64], np.outer(vectors[1], vectors[0]))
    if len(vectors) == 3:
        return cast(
            NDArray[np.float64],
            np.einsum("i,j,k->ijk", vectors[2], vectors[1], vectors[0]),
        )
    raise ValueError("Laws kernels require two or three axes")


def _laws_orientations(codes: str) -> list[tuple[NDArray[np.float64], ...]]:
    """Return every right-angle rotation of the Laws kernel ``codes``."""
    return _right_angle_orientations(_laws_vectors(codes))


def _right_angle_orientations(
    vectors: tuple[NDArray[np.float64], ...],
) -> list[tuple[NDArray[np.float64], ...]]:
    """Return the 4 (2-D) or 24 (3-D) right-angle rotations of a separable kernel.

    ``vectors`` are 1-D kernels in IBSI axis order; each rotation permutes and
    reverses them (manual Appendix A.2).
    """
    if len(vectors) == 2:
        g1, g2 = vectors
        return [
            (g1, g2),
            (g2[::-1], g1),
            (g1[::-1], g2[::-1]),
            (g2, g1[::-1]),
        ]
    return [
        tuple(vectors[index][::-1] if flip else vectors[index] for index, flip in rotation)
        for rotation in _ROTATIONS_3D
    ]


def pad_image(
    array: mx.array,
    widths: list[tuple[int, int]],
    padding: FilterPadding,
    constant: float,
) -> mx.array:
    """Extend ``array`` by ``widths`` with an IBSI boundary condition (manual section 2.2).

    ``MIRROR`` repeats the edge sample (NumPy ``symmetric``), and ``PERIODIC`` wraps
    by modular indexing so widths larger than the axis are supported.
    """
    if all(before == 0 and after == 0 for before, after in widths):
        return array
    if padding is FilterPadding.CONSTANT:
        return mx.pad(array, widths, mode="constant", constant_values=constant)
    if padding is FilterPadding.NEAREST:
        return mx.pad(array, widths, mode="edge")
    if padding is FilterPadding.MIRROR:
        return mx.pad(array, widths, mode="symmetric")
    if padding is FilterPadding.PERIODIC:
        result = array
        for axis, (before, after) in enumerate(widths):
            if before == 0 and after == 0:
                continue
            size = result.shape[axis]
            indexes = mx.arange(-before, size + after, dtype=mx.int32) % size
            result = mx.take(result, indexes, axis=axis)
        return result
    raise ValueError(f"unsupported filter padding: {padding}")


def convolve_nd(
    image: mx.array,
    kernel: NDArray[np.float64],
    padding: FilterPadding,
    constant: float,
) -> mx.array:
    """True convolution of one 2-D or 3-D image with a small kernel."""
    widths = [(size // 2, size // 2) for size in kernel.shape]
    padded = pad_image(image, widths, padding, constant)
    # MLX convolution is cross-correlation; flip the kernel for Eq. 2.2.
    weight = mx.array(np.flip(kernel).astype(np.float32))
    if image.ndim == 2:
        result = mx.conv2d(padded[None, :, :, None], weight[None, :, :, None])
        return result[0, :, :, 0]
    if image.ndim == 3:
        result = mx.conv3d(padded[None, :, :, :, None], weight[None, :, :, :, None])
        return result[0, :, :, :, 0]
    raise ValueError("filters require a 2-D or 3-D image")


def convolve_planes(
    planes: mx.array,
    kernel: NDArray[np.float64],
    padding: FilterPadding,
    constant: float,
) -> mx.array:
    """True 2-D convolution applied independently to a batch of planes."""
    widths = [(0, 0), (kernel.shape[0] // 2, kernel.shape[0] // 2),
              (kernel.shape[1] // 2, kernel.shape[1] // 2)]
    padded = pad_image(planes, widths, padding, constant)
    weight = mx.array(np.flip(kernel).astype(np.float32))
    return mx.conv2d(padded[..., None], weight[None, :, :, None])[..., 0]


# Terms at or above this magnitude are pre-scaled by at most 2**-5 (24 orientations),
# which keeps them normal; smaller terms are summed unscaled so Metal's subnormal flush
# cannot erase them. 2**-96 leaves a wide margin above the float32 normal limit 2**-126.
_MEAN_TINY = 2.0**-96

type _MeanState = tuple[mx.array, mx.array]


def _mean_scale(count: int) -> float:
    """Power of two ``2**-k`` with ``2**k >= count`` used to pre-scale mean terms."""
    return 2.0 ** -max(0, (count - 1).bit_length())


def _mean_add(state: _MeanState | None, value: mx.array, scale: float) -> _MeanState:
    """Add one map to a running orientation mean that cannot overflow when finite.

    Summing before dividing overflowed four identical finite ``1e38`` maps to inf.
    Each term is multiplied by the exact power of two ``scale`` before accumulation, so
    ``count`` terms bounded by the float32 maximum sum to at most that maximum. Terms
    below ``_MEAN_TINY`` keep a separate unscaled sum, so scaling cannot flush them.
    Nonfinite terms propagate unchanged.
    """
    small = mx.abs(value) < _MEAN_TINY
    scaled = mx.where(small, 0, value * scale)
    tiny = mx.where(small, value, 0)
    return (scaled, tiny) if state is None else (state[0] + scaled, state[1] + tiny)


def _mean_finish(state: _MeanState, count: int, scale: float) -> mx.array:
    """Restore the running mean; ``1 / (scale * count)`` lies in ``[1, 2)``."""
    return state[0] * (1.0 / (scale * count)) + state[1] / count


def _mean_maps(maps: list[mx.array]) -> mx.array:
    """Overflow-safe elementwise mean of response maps (see ``_mean_add``)."""
    scale = _mean_scale(len(maps))
    state: _MeanState | None = None
    for value in maps:
        state = _mean_add(state, value, scale)
    assert state is not None
    return _mean_finish(state, len(maps), scale)


def _pool(maps: list[mx.array], pooling: OrientationPooling) -> mx.array:
    """Pool orientation response maps by ``pooling``; ``NONE`` returns the first map."""
    if not maps:
        raise ValueError("orientation pooling requires at least one response map")
    if pooling is OrientationPooling.NONE:
        return maps[0]
    if pooling is OrientationPooling.AVERAGE:
        return _mean_maps(maps)
    if pooling is OrientationPooling.MAX:
        result = maps[0]
        for value in maps[1:]:
            result = mx.maximum(result, value)
        return result
    if pooling is OrientationPooling.MIN:
        result = maps[0]
        for value in maps[1:]:
            result = mx.minimum(result, value)
        return result
    raise ValueError(f"unsupported orientation pooling: {pooling}")


def _atrous_taps(taps: tuple[float, ...], level: int) -> NDArray[np.float64]:
    """Dilate wavelet taps for decomposition ``level`` and make the support odd.

    Follows the manual's worked Haar example: level ``j`` places ``2**(j-1) - 1``
    zeros after every tap, including the last, and one more zero is appended when
    the length is even, so the center sits at ``size // 2``.
    """
    step = 2 ** (level - 1)
    dilated = np.zeros(len(taps) * step, dtype=np.float64)
    dilated[::step] = taps
    if dilated.size % 2 == 0:
        dilated = np.append(dilated, 0.0)
    return dilated


type _SeparablePrefixes = dict[tuple[bytes, ...], tuple[mx.array, float]]


def _convolve_axes(
    image: mx.array,
    vectors: tuple[NDArray[np.float64], ...],
    padding: FilterPadding,
    constant: float,
    prefixes: _SeparablePrefixes | None = None,
) -> tuple[mx.array, float]:
    """Apply one 1-D kernel per IBSI axis ``k1, k2, ...`` by true convolution.

    Fewer vectors than image axes leave the leading native axes unfiltered and
    unpadded, which is how planar filtering of a 3-D image is expressed.

    Returns the response and the value the filtered ``CONSTANT`` extension takes:
    ``constant`` times the tap sum of every applied vector. Each axis pads with the
    value carried by the axes already filtered. Outside the image along a not yet
    filtered axis, every line of the once-extended image is constant, so this
    sequence equals extending the input once by ``constant`` and applying the full
    separable kernel. Reusing ``constant`` at every axis would not: a Haar
    high-pass would respond to a constant image extended by its own value.

    A ``prefixes`` cache may be shared by bands of the same input, padding and
    extension value. Only byte-identical tap prefixes are reused, preserving the
    axis order and every floating-point operation on that path. The cache is
    scoped to one wavelet level and rotation, never reused across inputs.
    """
    result = image
    prefix: tuple[bytes, ...] = ()
    for index, vector in enumerate(vectors):
        if prefixes is not None:
            prefix += (vector.tobytes(),)
            cached = prefixes.get(prefix)
            if cached is not None:
                result, constant = cached
                continue
        shape = [1] * image.ndim
        # IBSI axis k_(index+1) is native array axis ndim - 1 - index.
        shape[image.ndim - 1 - index] = vector.size
        result = convolve_nd(result, vector.reshape(shape), padding, constant)
        constant *= float(vector.sum())
        if prefixes is not None:
            prefixes[prefix] = (result, constant)
    return result, constant


def separable_wavelet_responses(
    image: mx.array,
    low: tuple[float, ...],
    high: tuple[float, ...],
    bands: tuple[str, ...],
    level: int,
    padding: FilterPadding,
    constant: float,
    pooling: OrientationPooling,
    planar: bool = False,
) -> list[mx.array]:
    """Undecimated separable wavelet bands (IBSI ``25BO``, ``CVCQ``).

    ``low`` and ``high`` are the decomposition taps as PyWavelets lists them
    (``dec_lo``/``dec_hi``). Level ``j`` filters the level ``j - 1`` all-low
    response with the à trous taps of :func:`_atrous_taps`, applied along each
    axis with ``padding``. Each band is a string of ``L``/``H`` letters in IBSI
    axis order, so ``"LHL"`` is low on ``k1``, high on ``k2`` and low on ``k3``.
    With pooling, every right-angle rotation of the separable kernel is applied
    through the whole level chain, and the rotated bands are pooled (manual
    section 4.5.2 troubleshooting). This reproduces the IBSI 2 consensus maps
    ``5_a_1`` to ``7_a_2``. Each level pads the previous approximation, as the
    manual does; with ``CONSTANT`` padding its extension value is ``constant``
    carried through the low-pass taps already applied (see :func:`_convolve_axes`),
    so a constant image stays constant and its high-pass bands vanish. A zero
    ``constant``, the IBSI 2 setting, is unchanged. With ``planar`` set,
    two-letter bands filter every plane normal to native array axis 0 of a 3-D
    image with the 2-D transform and its four in-plane rotations; axis 0 is
    neither filtered nor padded.

    Args:
        image: Float32 2-D or 3-D array.
        low: Low-pass decomposition taps.
        high: High-pass decomposition taps.
        bands: Requested band codes, one letter per image axis.
        level: Positive one-based decomposition level of the returned bands.
        padding: Boundary-extension method.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.
        pooling: Pooling across rotations; ``NONE`` keeps the unrotated kernel.
        planar: Filter each axis-0 plane of a 3-D image with the 2-D transform.

    Returns:
        One float32 response map per band, in ``bands`` order.
    """
    if image.ndim not in (2, 3):
        raise ValueError("separable wavelets require a 2-D or 3-D image")
    if planar and image.ndim != 3:
        raise ValueError("planar separable wavelets require a 3-D image")
    if type(level) is not int or level < 1:
        raise ValueError("separable wavelet level must be a positive integer")
    # The filtered axes are the last ``axes`` native axes: IBSI k1, k2 (and k3).
    axes = 2 if planar else image.ndim
    if not bands or any(
        len(band) != axes or set(band) - {"L", "H"} for band in bands
    ):
        raise ValueError("separable wavelet bands need one L or H per filtered axis")
    taps = {"L": low, "H": high}

    def rotated(code: str, scale: int, rotation: int) -> tuple[NDArray[np.float64], ...]:
        vectors = tuple(_atrous_taps(taps[letter], scale) for letter in code)
        return _right_angle_orientations(vectors)[rotation]

    # Rotation 0 is the identity, the only kernel used without pooling.
    rotations = 1 if pooling is OrientationPooling.NONE else (4 if axes == 2 else 24)
    combined: list[mx.array] = []
    # Average pooling keeps an overflow-safe running state (two maps) per band.
    mean_scale = _mean_scale(rotations)
    means: list[_MeanState] = []
    for rotation in range(rotations):
        approximation, extension = image, constant
        for scale in range(1, level):
            approximation, extension = _convolve_axes(
                approximation, rotated("L" * axes, scale, rotation), padding, extension,
            )
        # Every requested band starts from the same approximation. Share exact
        # prefixes: all 3-D bands need 2 + 4 + 8 = 14 axis passes instead of 24.
        prefixes: _SeparablePrefixes = {}
        for position, band in enumerate(bands):
            values, _ = _convolve_axes(
                approximation, rotated(band, level, rotation), padding, extension, prefixes,
            )
            # Running pools keep one resident map per band across the rotations.
            if pooling is OrientationPooling.AVERAGE:
                if rotation == 0:
                    means.append(_mean_add(None, values, mean_scale))
                else:
                    means[position] = _mean_add(means[position], values, mean_scale)
            elif rotation == 0:
                combined.append(values)
            elif pooling is OrientationPooling.MAX:
                combined[position] = mx.maximum(combined[position], values)
            elif pooling is OrientationPooling.MIN:
                combined[position] = mx.minimum(combined[position], values)
            else:
                raise ValueError(f"unsupported orientation pooling: {pooling}")
        # Output graphs retain every shared dependency. Drop the cache's extra
        # roots before evaluation so intermediates can die after their last use.
        prefixes.clear()
        evaluate(*combined, *(part for state in means for part in state))
    if pooling is OrientationPooling.AVERAGE:
        return [_mean_finish(state, rotations, mean_scale) for state in means]
    return combined


def mean_response(
    image: mx.array,
    support: int,
    padding: FilterPadding,
    constant: float,
    planar: bool = False,
) -> mx.array:
    """Mean filter over an isotropic ``support``-voxel cube (IBSI ``S60F``).

    Args:
        image: Float32 2-D or 3-D array.
        support: Odd spatial support ``M`` in voxels, one value for every axis.
        padding: Boundary-extension method.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.
        planar: Filter each axis-0 plane of a 3-D image with the ``M x M`` 2-D
            kernel; axis 0 is neither averaged nor padded.

    Returns:
        Float32 array with the image's shape. The kernel is ``1 / M**D`` over the
        full cube (``D = 2`` when planar), so a constant interior is preserved.

    The cube is separable, so the padded image is summed with unit weights along
    one axis at a time and divided once by ``M**D``. Unit weights avoid the inexact
    float32 tap ``1 / M**D``, and per-axis sums of ``M`` terms replace one
    ``M**D``-term accumulation. Integer-valued sums below ``2**24`` are therefore
    exact before the final division. A dense float32 15-cube kernel lost about
    1.6e-5 relative precision, which is 2.7% of the narrow range of IBSI map 1_a_3.

    That unit-weight sum grows with the support volume, so for large-magnitude
    data (including a ``CONSTANT`` padding value) it can overflow even when the
    mean is representable. The padded maximum is measured first; when the bound
    ``M**D * max|x|`` exceeds half the float32 range, the values are divided by a
    power of two, accumulated, and rescaled after the final division. Power-of-two
    scaling is exact for normal float32 values. Finite ordinary sums retain the
    unit-weight result, preserving weak neighborhoods far from a large value.

    Each axis sum is compensated (see ``window_pair_sum``) and the residual is carried
    into the next axis, so identical neighborhoods give identical means even when
    large terms cancel: every window of ``[1e8, 1, -1e8]`` rows averages to ``1/3``.
    """
    if image.ndim not in (2, 3):
        raise ValueError("filters require a 2-D or 3-D image")
    if planar and image.ndim != 3:
        raise ValueError("planar mean filtering requires a 3-D image")
    if support < 1 or support % 2 == 0:
        raise ValueError("mean filter support must be a positive odd integer")
    half = support // 2
    filtered = range(1, 3) if planar else range(image.ndim)
    result = pad_image(
        image, [(half, half) if axis in filtered else (0, 0) for axis in range(image.ndim)],
        padding, constant,
    )
    full = float(support**len(filtered))
    # The padded array already contains the extension constant, so its maximum
    # bounds every value participating in the accumulation.
    peak = float(cast(float, mx.max(mx.abs(result)).item()))
    unscaled = result
    scale = 1.0
    if isfinite(peak) and peak * full > FLOAT32_SUM_LIMIT:
        # Pick the smallest power of two that keeps the complete sum at or below
        # the limit; logs avoid forming a bound that itself overflows a Python
        # float for an extreme support.
        exponent = ceil(log2(peak) + log2(full) - log2(FLOAT32_SUM_LIMIT))
        scale = 2.0**exponent
        result = result / mx.array(scale, dtype=mx.float32)
    zero = mx.zeros_like(result)
    result_low, unscaled_low = zero, zero
    for axis in filtered:
        result, result_low = window_pair_sum(result, result_low, axis, support)
        if scale != 1.0:
            unscaled, unscaled_low = window_pair_sum(unscaled, unscaled_low, axis, support)
    result = _pair_value(result, result_low) / full
    if scale == 1.0:
        return result
    # Keep finite ordinary sums: a remote large value must not flush a weak
    # neighborhood merely because the bank-wide downscale was needed elsewhere.
    ordinary = _pair_value(unscaled, unscaled_low) / full
    return mx.where(mx.isfinite(ordinary), ordinary, result * mx.array(scale, dtype=mx.float32))


def _pair_value(high: mx.array, low: mx.array) -> mx.array:
    """Round a window pair to float32; an infinite high part (with NaN error) stays."""
    return mx.where(mx.isinf(high), high, high + low)


def laws_response(
    image: mx.array,
    codes: str,
    padding: FilterPadding,
    constant: float,
    pooling: OrientationPooling,
) -> mx.array:
    """Filter an image with a separable Laws kernel (IBSI ``JTXT``).

    A two-name kernel (for example ``"E5L5"``) is applied to each ``(k2, k1)``
    plane when ``image`` is 3-D. A three-name kernel is applied as a 3-D kernel.
    Orientation pooling uses the right-angle rotations of manual Appendix A
    (four in 2-D, twenty-four in 3-D) and is applied before any energy map.

    Args:
        image: Float32 2-D or 3-D array.
        codes: Concatenated kernel names in ``(k1, k2, k3)`` order.
        padding: Boundary-extension method.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.
        pooling: Pooling over rotated kernel orientations.

    Returns:
        Float32 response map with the image's shape.
    """
    vectors = _laws_vectors(codes)
    if len(vectors) == 3 and image.ndim != 3:
        raise ValueError("a three-axis Laws kernel requires a 3-D image")
    orientations = (
        [vectors] if pooling is OrientationPooling.NONE else _laws_orientations(codes)
    )
    responses: list[mx.array] = []
    for orientation in orientations:
        kernel = _kernel_from_vectors(orientation)
        if image.ndim == 2:
            responses.append(convolve_nd(image, kernel, padding, constant))
        elif len(vectors) == 2:
            responses.append(convolve_planes(image, kernel, padding, constant))
        else:
            responses.append(convolve_nd(image, kernel, padding, constant))
    return _pool(responses, pooling)


def laws_energy(
    response: mx.array,
    codes: str,
    distance: int,
    padding: FilterPadding,
    constant: float,
) -> mx.array:
    """Laws texture energy image (IBSI ``PQSD``).

    The absolute response is averaged over a ``(2 * distance + 1)`` cube, or
    over each ``(k2, k1)`` plane when the response came from a two-axis kernel.
    The mean filter normalizes by the number of window voxels ``W``. It is
    :func:`mean_response`, whose compensated unit-weight sums keep the window mean
    correctly rounded; a dense float32 ``1 / W`` kernel lost about 5e-6 relative
    precision over a ``15**3`` window.

    ``CONSTANT`` padding extends the absolute response by the magnitude the Laws
    response takes far outside an image extended by ``constant``: ``|constant|``
    times the product of the kernel's tap sums, the same for every rotation. The
    value is zero for any kernel containing a zero-sum E, S, W or R vector and for
    the IBSI 2 zero extension. Reusing a signed image extension value would make
    the energy of a nonnegative map negative near the boundary.

    Args:
        response: Float32 Laws response map of ``codes``.
        codes: Concatenated kernel names in ``(k1, k2, k3)`` order; a two-name
            kernel on a 3-D response averages each axis-0 plane.
        distance: Chebyshev distance ``delta`` in voxels; nonnegative.
        padding: Boundary-extension method, normally the one used for the
            initial Laws response (manual section 4.4.1).
        constant: Image extension value of the Laws response under
            :attr:`FilterPadding.CONSTANT`.

    Returns:
        Float32 energy map with the response's shape.
    """
    if distance < 0:
        raise ValueError("Laws energy distance must be nonnegative")
    vectors = _laws_vectors(codes)
    extension = abs(constant * prod(float(vector.sum()) for vector in vectors))
    return mean_response(
        mx.abs(response), 2 * distance + 1, padding, extension,
        planar=len(vectors) == 2 and response.ndim == 3,
    )


def gabor_kernel(
    plane_shape: tuple[int, int],
    spacing: tuple[float, float],
    sigma: float,
    wavelength: float,
    gamma: float,
    theta: float,
) -> NDArray[np.complex64]:
    """Sample one complex 2-D Gabor kernel (IBSI Eq. 4.3).

    The kernel spans the plane: each odd support is the plane size, or the plane
    size plus one when that size is even (manual Eq. 4.5). ``spacing`` is the
    physical spacing of the first and last plane axes in mm. ``sigma`` and
    ``wavelength`` are also in mm, so anisotropic voxels are handled in physical
    coordinates. ``theta`` is clockwise from the ``k1`` (last) axis, using the
    manual's rotation matrix.

    Args:
        plane_shape: ``(k2, k1)`` plane dimensions in voxels.
        spacing: ``(k2, k1)`` voxel spacing in mm.
        sigma: Gaussian-envelope standard deviation in mm.
        wavelength: Oscillation wavelength in mm.
        gamma: Ellipticity ``a1 / a2``.
        theta: In-plane orientation in radians.

    Returns:
        Complex64 kernel with odd ``(k2, k1)`` dimensions and the Gaussian
        envelope centered at ``M // 2``.
    """
    sizes = tuple(size + 1 if size % 2 == 0 else size for size in plane_shape)
    offsets2 = np.arange(sizes[0], dtype=np.float64) - sizes[0] // 2
    offsets1 = np.arange(sizes[1], dtype=np.float64) - sizes[1] // 2
    grid2, grid1 = np.meshgrid(offsets2 * spacing[0], offsets1 * spacing[1],
                               indexing="ij")
    radial = cos(theta) * grid1 + sin(theta) * grid2
    orthoradial = sin(theta) * grid1 - cos(theta) * grid2
    envelope = np.exp(
        -(radial * radial + gamma * gamma * orthoradial * orthoradial)
        / (2 * sigma * sigma)
    )
    phase = 2 * pi * radial / wavelength
    return (envelope * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)


def _gabor_convolution(
    planes: mx.array,
    kernel: NDArray[np.complex64],
    padding: FilterPadding,
    constant: float,
    response: GaborResponse,
) -> mx.array:
    """Convolve real/imaginary Gabor taps as two native output channels.

    Spatial convolution preserves small impulse responses that FFT roundoff can
    obscure. Flip both spatial axes because MLX conv2d is cross-correlation.
    """
    widths = [(0, 0), (kernel.shape[0] // 2, kernel.shape[0] // 2),
              (kernel.shape[1] // 2, kernel.shape[1] // 2)]
    padded = pad_image(planes, widths, padding, constant)
    taps = np.stack((kernel.real[::-1, ::-1], kernel.imag[::-1, ::-1]))
    weights = mx.array(taps[..., None].copy())
    convolution = mx.conv2d(padded[..., None], weights)
    real, imaginary = convolution[..., 0], convolution[..., 1]
    if response is GaborResponse.MODULUS:
        return mx.abs(real.astype(mx.complex64) + 1j * imaginary)
    if response is GaborResponse.REAL:
        return real
    if response is GaborResponse.IMAGINARY:
        return imaginary
    raise ValueError(f"unsupported Gabor response: {response}")


def _gabor_family(
    image: mx.array, spacing: tuple[float, ...], axis: int,
) -> tuple[mx.array, tuple[int, int], tuple[float, float]]:
    """Move ``axis`` to the front for in-plane Gabor filtering.

    Returns the moved image, the in-plane shape and the in-plane spacing.
    """
    moved = mx.moveaxis(image, axis, 0)
    remaining = [index for index in range(image.ndim) if index != axis]
    return moved, (moved.shape[1], moved.shape[2]), (
        spacing[remaining[0]], spacing[remaining[1]],
    )


def gabor_responses(
    image: mx.array,
    spacing: tuple[float, ...],
    sigma: float,
    wavelength: float,
    gamma: float,
    thetas: tuple[float, ...],
    response: GaborResponse,
    slice_axis: int | None,
    padding: FilterPadding,
    constant: float,
    pooling: OrientationPooling,
) -> list[mx.array]:
    """Gabor response maps for one scale and a set of orientations.

    A 2-D image is filtered directly. For a 3-D image, ``slice_axis`` selects the
    array axis normal to the filtered planes; ``None`` filters all three
    orthogonal plane families and averages those responses (manual section 4.5).
    Within each plane, ``k1`` is the last and ``k2`` the first remaining array
    axis, so the axial plane keeps the manual's ``(k1, k2)`` convention and the
    other planes use the corresponding local frame. Orientation pooling is
    applied per plane family before that average. With
    :attr:`OrientationPooling.NONE`, one response is returned per orientation;
    otherwise a single pooled response is returned.

    Args:
        image: Float32 2-D or 3-D array.
        spacing: Voxel spacing in mm per array axis.
        sigma: Gaussian-envelope standard deviation in mm.
        wavelength: Oscillation wavelength in mm.
        gamma: Ellipticity ``a1 / a2``.
        thetas: In-plane orientations in radians, clockwise from ``k1``.
        response: Complex response component to return.
        slice_axis: Plane-normal array axis for a 3-D image, or ``None``.
        padding: Boundary-extension method for the plane convolution.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.
        pooling: Pooling over orientations.

    Returns:
        One float32 response map per orientation when ``pooling`` is ``NONE``,
        otherwise one pooled map.
    """
    if not thetas:
        raise ValueError("Gabor filtering requires at least one orientation")
    if image.ndim not in (2, 3):
        raise ValueError("Gabor filtering requires a 2-D or 3-D image")
    if image.ndim == 2 and slice_axis is not None:
        raise ValueError("slice_axis is only valid for a 3-D Gabor image")
    if slice_axis is not None and slice_axis not in range(image.ndim):
        raise ValueError("slice_axis must index a NumPy image axis")
    axes = (slice_axis,) if slice_axis is not None else (
        (0,) if image.ndim == 2 else (0, 1, 2)
    )
    families: list[list[mx.array]] = []
    for axis in axes:
        plane_shape: tuple[int, int]
        plane_spacing: tuple[float, float]
        if image.ndim == 2:
            planes = image[None]
            plane_shape = (image.shape[0], image.shape[1])
            plane_spacing = (spacing[0], spacing[1])
            restore = False
        else:
            planes, plane_shape, plane_spacing = _gabor_family(image, spacing, axis)
            restore = True
        components: list[mx.array] = []
        for theta in thetas:
            kernel = gabor_kernel(
                plane_shape, plane_spacing, sigma, wavelength, gamma, theta,
            )
            values = _gabor_convolution(planes, kernel, padding, constant, response)
            components.append(
                mx.moveaxis(values, 0, axis) if restore else values[0]
            )
        families.append(
            [_pool(components, pooling)] if pooling is not OrientationPooling.NONE
            else components
        )
    if pooling is OrientationPooling.NONE:
        return [
            _mean_maps([family[index] for family in families])
            for index in range(len(thetas))
        ]
    return [_mean_maps([family[0] for family in families])]

