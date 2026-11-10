"""Sampled LoG and guarded Simoncelli/Riesz frequency-domain filters."""

from importlib import metadata
from math import (
    ceil,
    factorial,
    floor,
    isclose,
    isfinite,
    ldexp,
    lgamma,
    log,
    log2,
    pi,
    prod,
    sqrt,
)
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.filtering import FilterPadding, FourierGrid
from mlxradiomics.filters.convolution import (
    FLOAT32_SUM_LIMIT,
    convolve_nd,
    convolve_planes,
    pad_image,
)
from mlxradiomics.numerics.runtime import evaluate, mx

# A batched FFT in MLX releases before 0.32.5 can overrun its destination buffer and
# corrupt adjacent arrays. Simoncelli and Riesz therefore run only on a plain MLX
# release at or above 0.32.5; development builds stay disabled because some predate
# the fix. There is deliberately no runtime override.
_FFT_FIXED_MLX = (0, 32, 5)


def _mlx_release(version: str) -> tuple[int, int, int] | None:
    """Parse a plain ``X.Y.Z`` release; any other version string returns None."""
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def fft_available_on(version: str) -> bool:
    """Whether Simoncelli and Riesz may run on MLX ``version`` (0.32.5 or later releases)."""
    release = _mlx_release(version)
    return release is not None and release >= _FFT_FIXED_MLX


_MLX_VERSION = metadata.version("mlx")
FFT_AVAILABLE = fft_available_on(_MLX_VERSION)
FFT_DISABLED_REASON = (
    f"Simoncelli and Riesz filtering needs MLX 0.32.5 or later; this release of "
    f"mlxradiomics pins MLX 0.32.4 (installed: {_MLX_VERSION}), whose batched FFT can "
    "overrun its buffer and corrupt neighboring arrays."
)


def _require_fft_enabled() -> None:
    """Refuse an FFT-based filter before allocating an image, spectrum or band."""
    if not FFT_AVAILABLE:
        raise ValueError(FFT_DISABLED_REASON)


def _centered_coordinates(shape: tuple[int, ...]) -> tuple[mx.array, ...]:
    """Return endpoint-inclusive coordinates from -1 to 1, in centered array order.

    This is the MIRP/IBSI 2 consensus sampling used by the non-separable
    wavelets and Riesz transforms: ``(k - (N - 1) / 2) / ((N - 1) / 2)``. A
    length-one axis has no well-defined endpoint-inclusive sampling.
    """
    if any(size < 2 for size in shape):
        raise ValueError("the centered Fourier grid requires at least two samples per axis")
    return tuple(
        (mx.arange(size, dtype=mx.float32) - (size - 1) / 2) / ((size - 1) / 2)
        for size in shape
    )


def _simoncelli_band(
    shape: tuple[int, ...], level: int, grid: FourierGrid,
) -> mx.array:
    """Return the real Simoncelli band-pass filter in DFT array order."""
    if type(level) is not int or level < 1:
        raise ValueError("Simoncelli level must be a positive integer")
    if grid is FourierGrid.DFT:
        coordinates = tuple(mx.fft.fftfreq(size) * (2 * pi) for size in shape)
        centered = False
    elif grid is FourierGrid.CENTERED:
        coordinates = tuple(value * pi for value in _centered_coordinates(shape))
        centered = True
    else:
        raise ValueError(f"unsupported Fourier grid: {grid}")
    cutoff = ldexp(pi, 1 - level)
    # Every nonzero sample has norm >= pi / max(shape) on both grids, and the lower
    # edge cutoff / 4 > 0 excludes DC, so a finer cutoff selects nothing. Returning
    # the empty band also keeps float32 from flushing the edges to zero (DC leak).
    if cutoff < pi / max(shape):
        return mx.zeros(shape, dtype=mx.float32)
    grids = mx.meshgrid(*coordinates, indexing="ij")
    squared = grids[0] * grids[0]
    for values in grids[1:]:
        squared = squared + values * values
    norm = mx.sqrt(squared)
    lower = cutoff / 4
    # Keep log2 away from zero in the unselected branch; `where` then zeroes it.
    safe_ratio = mx.maximum(2 * norm / cutoff, 0.5)
    profile = mx.cos(pi / 2 * mx.log2(safe_ratio))
    band = mx.where((norm >= lower) & (norm <= cutoff), profile, mx.zeros_like(profile))
    return mx.fft.ifftshift(band) if centered else band


def simoncelli_response(
    image: mx.array, level: int, grid: FourierGrid = FourierGrid.DFT, planar: bool = False,
) -> mx.array:
    """Radial Simoncelli band-pass response map (IBSI Eq. 4.8, ``PRT7``).

    The band-pass profile is evaluated on the full ``N1 x ... x ND`` Fourier grid
    with ``nu_B = pi`` and cut off at ``nu_B / 2**(level - 1)``. Level one is the
    highest-frequency band, matching the manual's direct B-map construction.
    Because the filter is applied as a full-size Fourier Hadamard product, the
    boundary condition is periodic by construction (manual section 4.6.4); there
    is no spatial padding parameter for this transform.

    :attr:`FourierGrid.DFT` uses the manual's exact ``2*pi/N`` step and has an
    exact zero-frequency bin. :attr:`FourierGrid.CENTERED` reproduces the
    endpoint-inclusive convention of the published IBSI 2 consensus maps: for
    even lengths its samples are half a bin away from the DFT frequencies, so
    the zero-frequency bin is not sampled exactly and a constant image does not
    have an exactly zero response. Native comparisons against the consensus
    maps ``8_a_1`` to ``8_a_3`` agree at float32 precision for levels two and
    three and within the official one-percent range criterion at level one.

    With ``planar`` set, the 2-D band is evaluated on the in-plane grid and each
    plane normal to native array axis 0 is filtered by its own 2-D FFT, so the
    planes are periodic independently and never mix.

    Args:
        image: Float32 2-D or 3-D array.
        level: Positive one-based decomposition level.
        grid: Fourier sampling convention.
        planar: Filter each axis-0 plane of a 3-D image with the 2-D band.

    Returns:
        Float32 real response map with the image's shape.

    Raises:
        ValueError: If the installed MLX is older than 0.32.5 (see
            :data:`FFT_DISABLED_REASON`).
    """
    _require_fft_enabled()
    if image.ndim not in (2, 3):
        raise ValueError("Simoncelli filtering requires a 2-D or 3-D image")
    if planar and image.ndim != 3:
        raise ValueError("planar Simoncelli filtering requires a 3-D image")
    axes = (1, 2) if planar else tuple(range(image.ndim))
    # A 2-D band broadcasts over the leading axis-0 planes.
    band = _simoncelli_band(tuple(image.shape[axis] for axis in axes), level, grid)
    # The band gain is at most one.
    scale = _spectral_scale(image, tuple(image.shape[axis] for axis in axes), 1.0)
    spectrum = mx.fft.fftn(image / scale if scale != 1.0 else image, axes=axes)
    # Evaluate the forward spectrum before the product and inverse FFT: a lazy
    # forward FFT feeding them in one graph can return wrong values on some MLX builds.
    evaluate(spectrum)
    response = mx.real(mx.fft.ifftn(spectrum * band, axes=axes))
    return response * scale if scale != 1.0 else response


def _spectral_scale(image: mx.array, lengths: tuple[int, ...], gain: float) -> float:
    """Power of two that keeps every unnormalized FFT coefficient finite.

    A forward FFT over axes of ``lengths`` sums ``count = prod(lengths)`` samples, so
    its coefficients reach ``count * max|x|`` (the DC term of a constant image) and a
    frequency response of gain at most ``gain`` scales that further, although the
    inverse transform divides by ``count`` again. The filter is applied after the
    forward transform, so an attenuating ``gain < 1`` cannot protect the unfiltered
    spectrum and the gain factor is ``max(1, gain)``.

    That bound holds for the transform's values, not for MLX's GPU partial sums: a
    length-``n`` axis without small radix factors is evaluated as a chirp convolution
    on a power-of-two grid ``m >= 2n - 1``, whose intermediates overflowed for random
    inputs at up to ``2.84 m`` times the ``count * peak`` bound (MLX 0.32.4, lengths
    7 to 8209). Each axis pass starts from the previous pass's bounded spectrum, so
    the growth does not compound and the budget carries ``4 m`` for the longest axis.

    Images whose budget exceeds half the float32 range are divided by an exact power
    of two before the forward transform and the response is multiplied back
    afterwards; smaller images are left unscaled so Metal's subnormal flush cannot
    touch tiny values.
    """
    peak = float(cast(float, mx.max(mx.abs(image)).item()))
    # Smallest power of two >= 2n - 1 for the longest transformed axis.
    chirp = 1 << (2 * max(lengths) - 2).bit_length()
    count = prod(lengths)
    budget = max(1.0, gain) * 4 * chirp
    if not isfinite(peak) or peak == 0 or peak * count * budget <= FLOAT32_SUM_LIMIT:
        return 1.0
    return 2.0 ** ceil(log2(peak) + log2(count) + log2(budget) - log2(FLOAT32_SUM_LIMIT))


def _native_order(order: tuple[int, ...]) -> tuple[int, ...]:
    """Reverse an IBSI ``(k1, ..., kD)`` order into native array-axis order."""
    return tuple(reversed(order))


def riesz_multiplier(
    shape: tuple[int, ...], order: tuple[int, ...], grid: FourierGrid,
) -> mx.array:
    """Return one complex Riesz multiplier in DFT array order (IBSI Eq. 4.13).

    For ``L = sum(order)`` the multiplier is
    ``(-j)**L * sqrt(L! / prod(l_i!)) * prod(nu_i**l_i) / ||nu||**L``.
    The ratio is evaluated from the normalized directions
    ``d_i = nu_i / ||nu||`` and their powers, which is algebraically identical
    but keeps a high-order multiplier finite when separately raising ``nu_i``
    and ``||nu||`` would underflow to zero. The zero-frequency sample is handled
    separately and remains zero.
    A log-domain product combines mixed-order multinomial coefficients with
    directional powers when either factor alone exceeds float32 range.

    This primitive takes powers in native array-axis order. The public
    :func:`riesz_log_responses` and :func:`riesz_simoncelli_responses` accept
    the manual's IBSI ``(k1, ..., kD)`` order and reverse each tuple before
    calling here, because ``k1`` is the last native array axis.

    On the DFT grid, an exact zero frequency has no direction and the multiplier
    is defined as zero for every order with ``L >= 1``. The centered grid has no
    zero sample for even lengths; when every axis has odd length its center
    sample is likewise set to zero.

    Args:
        shape: Native 2-D or 3-D array shape.
        order: Nonnegative integer powers in native array-axis order, one per
            axis, with a positive sum.
        grid: Fourier sampling convention.

    Returns:
        Complex64 array with ``shape``, in DFT array order.
    """
    if len(shape) not in (2, 3) or len(order) != len(shape):
        raise ValueError("a Riesz order needs one integer per 2-D or 3-D image axis")
    if any(type(power) is not int or power < 0 for power in order) or sum(order) < 1:
        raise ValueError("Riesz orders must be nonnegative integers with a positive sum")
    length = sum(order)
    if grid is FourierGrid.DFT:
        # Nyquist-normalized DFT coordinates; the common scale cancels in the ratio.
        coordinates = tuple(mx.fft.fftfreq(size) * 2.0 for size in shape)
        centered = False
    elif grid is FourierGrid.CENTERED:
        coordinates = _centered_coordinates(shape)
        centered = True
    else:
        raise ValueError(f"unsupported Fourier grid: {grid}")
    grids = mx.meshgrid(*coordinates, indexing="ij")
    squared = grids[0] * grids[0]
    for values in grids[1:]:
        squared = squared + values * values
    nonzero = squared > 0
    # Normalize before taking powers. Raising the raw coordinates and the norm
    # separately can underflow both to zero at a representable nonzero frequency,
    # turning an exact ratio of one into 0/0.
    safe_norm = mx.sqrt(mx.where(nonzero, squared, mx.ones_like(squared)))
    directions = [values / safe_norm for values in grids]
    magnitude = mx.ones(shape, dtype=mx.float32)
    for values, power in zip(directions, order, strict=True):
        if power:
            magnitude = magnitude * values ** power
    denominator = 1
    for power in order:
        denominator *= factorial(power)
    log_coefficient = 0.5 * (lgamma(length + 1) - sum(lgamma(power + 1) for power in order))
    # A multinomial coefficient can overflow although its complete directional
    # multiplier is bounded by one. Evaluate that product in logs when needed;
    # also recover directional products flushed to zero before multiplication.
    log_magnitude = mx.full(shape, log_coefficient, dtype=mx.float32)
    sign = mx.ones(shape, dtype=mx.float32)
    for values, power in zip(directions, order, strict=True):
        if power:
            log_magnitude = log_magnitude + power * mx.log(mx.abs(values))
            if power % 2:
                sign = sign * mx.sign(values)
    logarithmic = sign * mx.exp(log_magnitude)
    phase = (-1j) ** (length % 4)
    if log_coefficient > log(float(np.finfo(np.float32).max)):
        magnitude = logarithmic
    else:
        coefficient = sqrt(factorial(length) / denominator)
        magnitude = mx.where(
            mx.abs(magnitude) < float(np.finfo(np.float32).tiny),
            logarithmic, magnitude * coefficient,
        )
    # The zero-frequency sample has no direction and is defined as zero; the
    # normalized directions are also zero there, so this only makes it explicit.
    magnitude = mx.where(nonzero, magnitude, mx.zeros_like(magnitude))
    multiplier = magnitude.astype(mx.complex64) * phase
    return mx.fft.ifftshift(multiplier) if centered else multiplier


def _log_kernel(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    sigma: float,
    cutoff: float,
) -> NDArray[np.float64]:
    """Sample the IBSI 2 LoG kernel (manual Eq. 4.9), shared by LoG and Riesz LoG.

    ``sigma`` is in mm and is converted once to voxel units with the common
    spacing. The manual's and MIRP's LoG definition has a single voxel-unit
    standard deviation, so anisotropic spacing is rejected rather than
    approximated by an invented per-axis kernel. The odd support per axis is
    ``1 + 2*floor(cutoff*sigma_vox + 0.5)`` and the radial distance is measured
    in voxel units.
    """
    if len(shape) not in (2, 3) or len(spacing) != len(shape):
        raise ValueError("sampled LoG requires one spacing value per 2-D or 3-D filter axis")
    require_isotropic_spacing(spacing)
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sampled LoG sigma must be finite and positive")
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("sampled LoG cutoff must be finite and positive")
    sigma_voxel = sigma / spacing[0]
    size = 1 + 2 * floor(cutoff * sigma_voxel + 0.5)
    offsets = tuple(
        np.arange(size, dtype=np.float64) - (size - 1) / 2 for _ in shape
    )
    grids = np.meshgrid(*offsets, indexing="ij")
    squared = np.zeros((size,) * len(shape), dtype=np.float64)
    for values in grids:
        squared = squared + (values / sigma_voxel) ** 2
    return (
        -(1 / sigma_voxel**2)
        * (1 / sqrt(2 * pi * sigma_voxel**2)) ** len(shape)
        * (len(shape) - squared)
        * np.exp(-squared / 2)
    )


def log_response(
    image: mx.array,
    spacing: tuple[float, ...],
    sigma: float,
    cutoff: float,
    planar: bool,
    padding: FilterPadding,
    constant: float,
) -> mx.array:
    """IBSI 2 sampled Laplacian-of-Gaussian response (manual Eq. 4.9, ``L6PA``).

    The kernel has odd support ``1 + 2*floor(cutoff*sigma_vox + 0.5)`` per filtered
    axis and is applied by direct spatial convolution with ``padding``. With
    ``planar`` set, a 2-D kernel (``D = 2`` normalization) filters every plane
    normal to native array axis 0, the manual's in-plane ``(k1, k2)`` filter.
    Direct convolution keeps float32 responses within 3.6e-6 of a float64
    construction on the IBSI 2 maps ``2_a`` to ``2_c``.

    Args:
        image: Float32 2-D or 3-D array.
        spacing: Voxel spacing in mm per native array axis. The filtered axes
            must be isotropic, because the manual defines one voxel-unit sigma.
        sigma: LoG scale in mm.
        cutoff: Kernel-size cutoff as a multiple of sigma.
        planar: Filter each axis-0 plane of a 3-D image with a 2-D kernel.
        padding: Boundary-extension method.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.

    Returns:
        Float32 response map with the image's shape.
    """
    if image.ndim not in (2, 3):
        raise ValueError("LoG filtering requires a 2-D or 3-D image")
    if len(spacing) != image.ndim:
        raise ValueError("LoG spacing must provide one value per image axis")
    if planar:
        if image.ndim != 3:
            raise ValueError("planar LoG filtering requires a 3-D image")
        kernel = _log_kernel(image.shape[1:], spacing[1:], sigma, cutoff)
        return convolve_planes(image, kernel, padding, constant)
    kernel = _log_kernel(image.shape, spacing, sigma, cutoff)
    return convolve_nd(image, kernel, padding, constant)


def riesz_log_responses(
    image: mx.array,
    spacing: tuple[float, ...],
    sigma: float,
    cutoff: float,
    orders: tuple[tuple[int, ...], ...],
    grid: FourierGrid,
    padding: FilterPadding,
    constant: float,
) -> list[mx.array]:
    """Riesz transforms of the IBSI 2 LoG kernel (manual Eq. 4.13).

    The compact spatial kernel is zero-padded to the padded image size and
    transformed with the image, following the convolution convention used for
    the IBSI 2 LoG: pad each axis by ``kernel_size // 2`` with ``padding``,
    multiply the spectra, inverse transform, and crop the valid region with
    offset ``2 * kernel_size // 2``.

    Riesz orders follow the manual's IBSI ``(k1, ..., kD)`` order, so
    ``order[i]`` applies to native array axis ``ndim - 1 - i``; each tuple is
    reversed once before the multiplier is built.

    The real part is taken by definition, as in MIRP and the reference
    construction. On the DFT grid an even-length Nyquist sample is its own
    negative frequency, so an odd-order multiplier is purely imaginary there
    and the product spectrum is not Hermitian; its inverse transform therefore
    has a non-zero imaginary component at that bin. On the centered grid
    even-length axes are sampled off the conjugate frequency pairs, leaving a
    small imaginary residue. Both are discarded by taking the real part.

    Args:
        image: Float32 2-D or 3-D array.
        spacing: Isotropic voxel spacing in mm per native array axis. Riesz LoG
            rejects anisotropic spacing because the manual and MIRP define a
            single voxel-unit sigma.
        sigma: LoG scale in mm.
        cutoff: Kernel-size cutoff multiple of ``sigma``.
        orders: Riesz orders in the manual's IBSI ``(k1, ..., kD)`` order.
        grid: Fourier sampling convention for the Riesz multiplier.
        padding: Boundary-extension method for the spatial kernel.
        constant: Extension value used by :attr:`FilterPadding.CONSTANT`.

    Returns:
        One float32 response map per order, each with the image's shape.

    Raises:
        ValueError: If the installed MLX is older than 0.32.5 (see
            :data:`FFT_DISABLED_REASON`).
    """
    _require_fft_enabled()
    if image.ndim not in (2, 3):
        raise ValueError("Riesz LoG filtering requires a 2-D or 3-D image")
    if not orders:
        raise ValueError("Riesz filtering requires at least one order tuple")
    kernel = _log_kernel(image.shape, spacing, sigma, cutoff)
    widths = [(size // 2, size // 2) for size in kernel.shape]
    padded = pad_image(image, widths, padding, constant)
    axes = tuple(range(padded.ndim))
    # |base| is bounded by the kernel's absolute sum and the Riesz multiplier by one.
    scale = _spectral_scale(padded, tuple(padded.shape), float(np.abs(kernel).sum()))
    spectrum = mx.fft.fftn(padded / scale if scale != 1.0 else padded, axes=axes)
    # The kernel's center is at kernel_size // 2 after end zero-padding, so the
    # circular convolution's valid region starts at 2 * (kernel_size // 2).
    weights = mx.pad(
        mx.array(kernel.astype(np.float32)),
        [(0, padded.shape[axis] - kernel.shape[axis]) for axis in range(padded.ndim)],
    )
    base = mx.fft.fftn(weights, axes=axes)
    # Evaluate both spectra before use; see the MLX FFT note in simoncelli_response.
    evaluate(spectrum, base)
    crop = tuple(
        slice(2 * before, 2 * before + size)
        for (before, _), size in zip(widths, image.shape, strict=True)
    )
    return [
        mx.real(mx.fft.ifftn(spectrum * base * riesz_multiplier(
            tuple(padded.shape), _native_order(order), grid,
        ), axes=axes))[crop] * scale
        for order in orders
    ]


def riesz_simoncelli_responses(
    image: mx.array,
    level: int,
    orders: tuple[tuple[int, ...], ...],
    grid: FourierGrid,
) -> list[mx.array]:
    """Riesz transforms of a radial Simoncelli B-map (manual section 4.7).

    The base filter is the full-image band-pass of
    :func:`simoncelli_response`, so the boundary condition is periodic by
    construction; no spatial padding is applied.

    Riesz orders follow the manual's IBSI ``(k1, ..., kD)`` order, so
    ``order[i]`` applies to native array axis ``ndim - 1 - i``; each tuple is
    reversed once before the multiplier is built.

    The real part is taken by definition, as in MIRP and the reference
    construction. On the DFT grid an even-length Nyquist sample is its own
    negative frequency, so an odd-order multiplier is purely imaginary there
    and the product spectrum is not Hermitian; its inverse transform therefore
    has a non-zero imaginary component at that bin. On the centered grid
    even-length axes are sampled off the conjugate frequency pairs, leaving a
    small imaginary residue. Both are discarded by taking the real part.

    Args:
        image: Float32 2-D or 3-D array.
        level: Positive one-based Simoncelli band level.
        orders: Riesz orders in the manual's IBSI ``(k1, ..., kD)`` order.
        grid: Fourier sampling convention for both the band and the multiplier.

    Returns:
        One float32 response map per order, each with the image's shape.

    Raises:
        ValueError: If the installed MLX is older than 0.32.5 (see
            :data:`FFT_DISABLED_REASON`).
    """
    _require_fft_enabled()
    if image.ndim not in (2, 3):
        raise ValueError("Riesz Simoncelli filtering requires a 2-D or 3-D image")
    if not orders:
        raise ValueError("Riesz filtering requires at least one order tuple")
    band = _simoncelli_band(image.shape, level, grid)
    axes = tuple(range(image.ndim))
    # Band and Riesz multiplier gains are each at most one.
    scale = _spectral_scale(image, tuple(image.shape), 1.0)
    spectrum = mx.fft.fftn(image / scale if scale != 1.0 else image, axes=axes)
    # Evaluate the spectrum before use; see the MLX FFT note in simoncelli_response.
    evaluate(spectrum)
    return [
        mx.real(mx.fft.ifftn(
            spectrum * band * riesz_multiplier(
                image.shape, _native_order(order), grid,
            ),
            axes=axes,
        )) * scale
        for order in orders
    ]


def require_isotropic_spacing(spacing: tuple[float, ...]) -> None:
    """Reject spacing the sampled LoG cannot use: it has one voxel-unit sigma."""
    if any(not isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("sampled LoG spacing must be finite and positive")
    if any(not isclose(value, spacing[0], rel_tol=1e-6) for value in spacing[1:]):
        raise ValueError("sampled LoG requires isotropic voxel spacing over the filtered axes")

