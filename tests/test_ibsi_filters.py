"""Independent response-map checks for the IBSI 2 convolutional filters.

Simoncelli and Riesz tests run only when MLX 0.32.5 or later is installed."""

import dataclasses
from math import cos, factorial, pi, sin, sqrt

import numpy as np
import pytest
from conftest import requires_fft
from numpy.typing import NDArray

from mlxradiomics.config.filtering import (
    FilterPadding,
    FourierGrid,
    GaborResponse,
    LogKernel,
    OrientationPooling,
    RieszBase,
)
from mlxradiomics.config.transforms import (
    GaborSettings,
    ImageType,
    LawsSettings,
    LoGSettings,
    MeanSettings,
    RieszSettings,
    SeparableWaveletSettings,
    SimoncelliSettings,
    TransformSettings,
)
from mlxradiomics.filters import (
    laws_energy,
    mean_response,
    riesz_log_responses,
    riesz_multiplier,
    riesz_simoncelli_responses,
    simoncelli_response,
)
from mlxradiomics.imaging.transforms import feature_image_names, feature_images
from mlxradiomics.numerics.runtime import mx


def _production(
    image: NDArray[np.generic], options: TransformSettings,
    spacing: tuple[float, ...] | None = None,
) -> list[tuple[str, NDArray[np.float64]]]:
    values = np.asarray(image, dtype=np.float32)
    spacing = spacing or (1.0,) * values.ndim
    with mx.stream(mx.gpu):
        return [
            (item.name, np.asarray(item.image, dtype=np.float64))
            for item in feature_images(mx.array(values), spacing, options)
        ]


def _laws_taps(name: str) -> NDArray[np.float64]:
    taps: dict[str, tuple[float, ...]] = {
        "L3": (1.0, 2.0, 1.0),
        "L5": (1.0, 4.0, 6.0, 4.0, 1.0),
        "E3": (-1.0, 0.0, 1.0),
        "E5": (-1.0, -2.0, 0.0, 2.0, 1.0),
        "S3": (-1.0, 2.0, -1.0),
        "S5": (-1.0, 0.0, 2.0, 0.0, -1.0),
        "W5": (-1.0, 2.0, 0.0, -2.0, 1.0),
        "R5": (1.0, -4.0, 6.0, -4.0, 1.0),
    }
    values = np.array(taps[name], dtype=np.float64)
    return values / np.sqrt(np.dot(values, values))


def _laws_kernel(codes: str) -> NDArray[np.float64]:
    vectors = [_laws_taps(codes[index:index + 2]) for index in range(0, len(codes), 2)]
    if len(vectors) == 2:
        return np.outer(vectors[1], vectors[0])
    return np.einsum("i,j,k->ijk", vectors[2], vectors[1], vectors[0])


def _convolve(
    image: NDArray[np.generic], kernel: NDArray[np.generic],
    padding: FilterPadding, constant: float = 0.0,
) -> NDArray[np.complex128]:
    """Direct discrete convolution (Eq. 2.2), independent of production code."""
    values = np.asarray(image, dtype=np.float64)
    # Correlation with the reversed kernel is the convolution of Eq. 2.2.
    taps = np.asarray(kernel, dtype=np.complex128)[
        tuple(slice(None, None, -1) for _ in range(kernel.ndim))
    ]
    widths = [(size // 2, size // 2) for size in taps.shape]
    if padding is FilterPadding.CONSTANT:
        padded = np.pad(values, widths, mode="constant", constant_values=constant)
    elif padding is FilterPadding.NEAREST:
        padded = np.pad(values, widths, mode="edge")
    elif padding is FilterPadding.MIRROR:
        padded = np.pad(values, widths, mode="symmetric")
    elif padding is FilterPadding.PERIODIC:
        padded = np.pad(values, widths, mode="wrap")
    else:
        raise AssertionError(f"no oracle for {padding}")
    output_shape = tuple(
        padded.shape[axis] - taps.shape[axis] + 1 for axis in range(taps.ndim)
    )
    result = np.zeros(output_shape, dtype=np.complex128)
    for index in np.ndindex(taps.shape):
        window = tuple(
            slice(index[axis], index[axis] + output_shape[axis])
            for axis in range(taps.ndim)
        )
        result += taps[index] * padded[window]
    return result


def _gabor_kernel(
    shape: tuple[int, int], spacing: tuple[float, float], sigma: float,
    wavelength: float, gamma: float, theta: float,
) -> NDArray[np.complex128]:
    sizes = tuple(size + 1 if size % 2 == 0 else size for size in shape)
    offsets2 = np.arange(sizes[0], dtype=np.float64) - sizes[0] // 2
    offsets1 = np.arange(sizes[1], dtype=np.float64) - sizes[1] // 2
    grid2, grid1 = np.meshgrid(
        offsets2 * spacing[0], offsets1 * spacing[1], indexing="ij",
    )
    radial = cos(theta) * grid1 + sin(theta) * grid2
    orthoradial = sin(theta) * grid1 - cos(theta) * grid2
    envelope = np.exp(
        -(radial * radial + gamma * gamma * orthoradial * orthoradial)
        / (2 * sigma * sigma)
    )
    return envelope * np.exp(1j * 2 * pi * radial / wavelength)


def _gabor_component(
    convolution: NDArray[np.complex128], response: GaborResponse,
) -> NDArray[np.float64]:
    if response is GaborResponse.MODULUS:
        return np.abs(convolution)
    if response is GaborResponse.REAL:
        return np.real(convolution)
    return np.imag(convolution)


def _simoncelli_band(
    shape: tuple[int, ...], level: int, grid: FourierGrid,
) -> NDArray[np.float64]:
    if grid is FourierGrid.DFT:
        coordinates = [2 * pi * np.fft.fftfreq(size) for size in shape]
        centered = False
    else:
        coordinates = [
            (np.arange(size, dtype=np.float64) - (size - 1) / 2) / ((size - 1) / 2) * pi
            for size in shape
        ]
        centered = True
    grids = np.meshgrid(*coordinates, indexing="ij")
    norm = np.sqrt(sum(grid * grid for grid in grids))
    cutoff = pi / 2 ** (level - 1)
    lower = cutoff / 4
    band = np.zeros_like(norm)
    selected = (norm >= lower) & (norm <= cutoff)
    band[selected] = np.cos(pi / 2 * np.log2(2 * norm[selected] / cutoff))
    return np.fft.ifftshift(band) if centered else band


def _simoncelli(
    image: NDArray[np.float64], level: int, grid: FourierGrid = FourierGrid.DFT,
) -> NDArray[np.float64]:
    band = _simoncelli_band(image.shape, level, grid)
    return np.real(np.fft.ifftn(np.fft.fftn(image) * band))


def _riesz_multiplier(
    shape: tuple[int, ...], order: tuple[int, ...], grid: FourierGrid,
) -> NDArray[np.complex128]:
    length = sum(order)
    denominator = 1
    for power in order:
        denominator *= factorial(power)
    prefactor = (-1j) ** length * sqrt(factorial(length) / denominator)
    if grid is FourierGrid.DFT:
        coordinates = [2 * np.fft.fftfreq(size) for size in shape]
        centered = False
    else:
        coordinates = [
            (np.arange(size, dtype=np.float64) - (size - 1) / 2) / ((size - 1) / 2)
            for size in shape
        ]
        centered = True
    grids = np.meshgrid(*coordinates, indexing="ij")
    numerator = np.ones(shape, dtype=np.float64)
    for values, power in zip(grids, order, strict=True):
        if power:
            numerator = numerator * values**power
    squared = sum(values * values for values in grids)
    safe = np.maximum(squared, np.finfo(np.float64).tiny)
    multiplier = np.where(
        squared > 0, prefactor * numerator / safe ** (length / 2), 0.0,
    ).astype(np.complex128)
    return np.fft.ifftshift(multiplier) if centered else multiplier


def _riesz_log_kernel(
    shape: tuple[int, ...], spacing: tuple[float, ...], sigma: float, cutoff: float,
) -> NDArray[np.float64]:
    # Manual/MIRP LoG: one voxel-unit sigma, defined only for isotropic spacing.
    sigma_voxel = sigma / spacing[0]
    size = 1 + 2 * int(np.floor(cutoff * sigma_voxel + 0.5))
    offsets = tuple(
        np.arange(size, dtype=np.float64) - (size - 1) / 2 for _ in shape
    )
    grids = np.meshgrid(*offsets, indexing="ij")
    squared = np.zeros((size,) * len(shape), dtype=np.float64)
    for values in grids:
        squared = squared + (values / sigma_voxel) ** 2
    return (
        -(1 / sigma_voxel**2)
        * (1 / np.sqrt(2 * pi * sigma_voxel**2)) ** len(shape)
        * (len(shape) - squared)
        * np.exp(-squared / 2)
    )


def _pad_oracle(
    values: NDArray[np.float64],
    widths: list[tuple[int, int]],
    padding: FilterPadding,
    constant: float,
) -> NDArray[np.float64]:
    if padding is FilterPadding.CONSTANT:
        return np.pad(values, widths, mode="constant", constant_values=constant)
    if padding is FilterPadding.NEAREST:
        return np.pad(values, widths, mode="edge")
    if padding is FilterPadding.MIRROR:
        return np.pad(values, widths, mode="symmetric")
    return np.pad(values, widths, mode="wrap")


def _riesz_log(
    image: NDArray[np.float64],
    spacing: tuple[float, ...],
    sigma: float,
    cutoff: float,
    order: tuple[int, ...],
    grid: FourierGrid,
    padding: FilterPadding = FilterPadding.CONSTANT,
    constant: float = 0.0,
) -> NDArray[np.float64]:
    kernel = _riesz_log_kernel(image.shape, spacing, sigma, cutoff)
    widths = [(size // 2, size // 2) for size in kernel.shape]
    padded = _pad_oracle(image, widths, padding, constant)
    spectrum = np.fft.fftn(padded)
    base = np.fft.fftn(kernel, padded.shape, axes=tuple(range(padded.ndim)))
    # Public orders are IBSI k1-first; the multiplier primitive takes native order.
    multiplier = _riesz_multiplier(padded.shape, tuple(reversed(order)), grid)
    response = np.real(np.fft.ifftn(spectrum * base * multiplier))
    crop = tuple(
        slice(2 * before, 2 * before + size)
        for (before, _), size in zip(widths, image.shape, strict=True)
    )
    return response[crop]


def _riesz_simoncelli(
    image: NDArray[np.float64],
    level: int,
    order: tuple[int, ...],
    grid: FourierGrid,
) -> NDArray[np.float64]:
    band = _simoncelli_band(image.shape, level, grid)
    multiplier = _riesz_multiplier(image.shape, tuple(reversed(order)), grid)
    return np.real(np.fft.ifftn(np.fft.fftn(image) * band * multiplier))


def _orders_of_length(ndim: int, length: int) -> list[tuple[int, ...]]:
    """All nonnegative order tuples with the given total length."""
    orders: list[tuple[int, ...]] = []

    def walk(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            orders.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            walk(prefix + [value], remaining - value, slots - 1)

    walk([], length, ndim)
    return orders


@pytest.mark.parametrize("shape", [(7, 8), (5, 6, 7)])
@pytest.mark.parametrize("padding", list(FilterPadding))
def test_mean_filter_matches_independent_padding_convolution(
    shape: tuple[int, ...], padding: FilterPadding,
) -> None:
    image = np.random.default_rng(11).normal(size=shape)
    options = TransformSettings(
        image_types=(ImageType.MEAN,), mean=MeanSettings(support=5, padding=padding, constant=0.25)
    )
    [(name, actual)] = _production(image, options)
    kernel = np.full((5,) * len(shape), 1.0 / 5 ** len(shape))
    expected = _convolve(image, kernel, padding, 0.25)
    assert name == {
        FilterPadding.CONSTANT: "mean-M5-constant0p25",
        FilterPadding.NEAREST: "mean-M5-nearest",
        FilterPadding.MIRROR: "mean-M5-mirror",
        FilterPadding.PERIODIC: "mean-M5-periodic",
    }[padding]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("padding", [FilterPadding.CONSTANT, FilterPadding.MIRROR])
def test_sampled_log_matches_independent_kernel_convolution(
    planar: bool, padding: FilterPadding,
) -> None:
    image = np.random.default_rng(5).normal(size=(6, 9, 10))
    spacing = (2.0, 1.5, 1.5) if planar else (1.5, 1.5, 1.5)
    options = TransformSettings(
        image_types=(ImageType.LOG,),
        log=LoGSettings(
            sigma=(2.0,),
            kernel=LogKernel.SAMPLED,
            cutoff=3.0,
            padding=padding,
            constant=0.5,
            planar=planar,
        ),
    )
    [(name, actual)] = _production(image, options, spacing)
    if planar:
        # A 2-D kernel per axis-0 plane; that axis's spacing is not filtered.
        kernel = _riesz_log_kernel(image.shape[1:], spacing[1:], 2.0, 3.0)
        expected = np.stack([np.real(_convolve(plane, kernel, padding, 0.5)) for plane in image])
    else:
        kernel = _riesz_log_kernel(image.shape, spacing, 2.0, 3.0)
        expected = np.real(_convolve(image, kernel, padding, 0.5))
    padding_tag = "constant0p5" if padding is FilterPadding.CONSTANT else "mirror"
    assert name == f"log-sampled-s2mm-c3-{padding_tag}-{'2D' if planar else '3D'}"
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_mean_filter_preserves_a_large_finite_constant() -> None:
    # M^D times the constant overflows float32 although the mean itself is finite.
    image = np.full((3, 3, 3), 2e37, dtype=np.float32)
    actual = np.asarray(
        mean_response(mx.array(image), 3, FilterPadding.MIRROR, 0.0),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(actual))
    # The same scaled accumulation runs at every voxel, so the result stays constant.
    assert float(np.ptp(actual)) == 0.0
    np.testing.assert_allclose(actual, 2e37, rtol=2e-5, atol=0)


def test_mean_filter_bounds_large_values_and_constant_padding() -> None:
    image = np.full((3, 3, 3), 2e37, dtype=np.float32)
    image[0, 0, 0] = -2e37
    image[2, 1, 0] = 5e36
    constant = 3e37
    actual = np.asarray(
        mean_response(mx.array(image), 3, FilterPadding.CONSTANT, constant),
        dtype=np.float64,
    )
    kernel = np.full((3, 3, 3), 1.0 / 27.0, dtype=np.float64)
    expected = np.real(_convolve(image, kernel, FilterPadding.CONSTANT, constant))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0)


def test_large_remote_mean_value_preserves_weak_neighborhood() -> None:
    image = np.full((9, 9), 1e-37, dtype=np.float32)
    image[0, 0] = 3e38
    actual = np.asarray(
        mean_response(mx.array(image), 3, FilterPadding.MIRROR, 0.0), dtype=np.float64,
    )
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[6, 6], image[6, 6], rtol=2e-6, atol=0)


def test_large_mean_support_keeps_a_narrow_response_range_exact() -> None:
    # IBSI 1_a_3 analogue: a periodic 15-cube mean of a 0/255 checkerboard varies by
    # well under 1% of its ~127.5 level, so float32 tap rounding is visible there.
    indices = np.indices((32, 32, 32)).sum(axis=0)
    image = np.where((indices // 3) % 2 == 0, 255.0, 0.0)
    options = TransformSettings(
        image_types=(ImageType.MEAN,), mean=MeanSettings(support=15, padding=FilterPadding.PERIODIC)
    )
    [(_, actual)] = _production(image, options)
    # Float64 periodic box mean: separable sum of the 15 circular shifts per axis.
    expected = image
    for axis in range(3):
        expected = sum(np.roll(expected, shift, axis=axis) for shift in range(-7, 8)) / 15
    span = float(np.ptp(expected))
    assert span > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * span)


@pytest.mark.parametrize(
    "shape, codes, name",
    [((9, 9), "E5W5", "laws-E5W5-constant0"),
     ((7, 8, 9), "E3W5R5", "laws-E3W5R5-constant0")],
)
def test_laws_impulse_response_is_the_declared_separable_kernel(
    shape: tuple[int, ...], codes: str, name: str,
) -> None:
    image = np.zeros(shape, dtype=np.float64)
    center = tuple(size // 2 for size in shape)
    image[center] = 1.0
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(kernel=codes, padding=FilterPadding.CONSTANT, constant=0.0),
    )
    [(actual_name, actual)] = _production(image, options)
    kernel = _laws_kernel(codes)
    expected = np.real(_convolve(image, kernel, FilterPadding.CONSTANT, 0.0))
    assert actual_name == name
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_mean_and_laws_ramp_responses_are_analytic_in_the_interior() -> None:
    # A symmetric mean filter reproduces a linear ramp away from the boundary.
    cols = np.arange(10, dtype=np.float64)[None, :]
    ramp = 3.0 * cols + 1.0 + np.zeros((9, 1), dtype=np.float64)
    mean_options = TransformSettings(
        image_types=(ImageType.MEAN,), mean=MeanSettings(support=3, padding=FilterPadding.CONSTANT)
    )
    [(_, mean_values)] = _production(ramp, mean_options)
    np.testing.assert_allclose(
        mean_values[1:-1, 1:-1], ramp[1:-1, 1:-1], rtol=1e-6, atol=1e-5,
    )
    constant_options = TransformSettings(
        image_types=(ImageType.MEAN,), mean=MeanSettings(support=3, padding=FilterPadding.NEAREST)
    )
    [(_, constant_values)] = _production(np.full((5, 6), 2.5), constant_options)
    np.testing.assert_allclose(constant_values, 2.5, rtol=1e-6, atol=1e-6)

    # E5 annihilates the constant term and responds to the k1 slope with
    # -a * sum_q q g_E5[q]; L5 sums to 16/sqrt(70) along k2.
    laws_options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(kernel="E5L5", padding=FilterPadding.CONSTANT),
    )
    [(_, laws_values)] = _production(ramp, laws_options)
    slope = 3.0
    expected = -slope * (8.0 / sqrt(10.0)) * (16.0 / sqrt(70.0))
    np.testing.assert_allclose(
        laws_values[2:-2, 2:-2], expected, rtol=1e-5, atol=1e-5,
    )


def test_laws_normalisation_and_zero_mean_kernels_under_periodic_padding() -> None:
    image = np.full((8, 9), 3.0, dtype=np.float64)
    low = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(kernel="L5L5", padding=FilterPadding.PERIODIC),
    )
    zero_mean = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(kernel="L5S5", padding=FilterPadding.PERIODIC),
    )
    [(low_name, low_values)] = _production(image, low)
    [(zero_name, zero_values)] = _production(image, zero_mean)
    assert low_name == "laws-L5L5-periodic"
    assert zero_name == "laws-L5S5-periodic"
    np.testing.assert_allclose(low_values, 3.0 * (16.0 / sqrt(70.0)) ** 2,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(zero_values, 0.0, rtol=0, atol=1e-5)


def test_laws_energy_is_the_absolute_response_mean_window() -> None:
    image = np.random.default_rng(17).normal(size=(7, 8))
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(kernel="E5L5", padding=FilterPadding.NEAREST, energy_distance=2),
    )
    [(name, actual)] = _production(image, options)
    response = _convolve(image, _laws_kernel("E5L5"), FilterPadding.NEAREST)
    expected = _convolve(
        np.abs(response), np.full((5, 5), 1.0 / 25.0), FilterPadding.NEAREST,
    )
    assert name == "laws-E5L5-nearest-energy-d2"
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_laws_energy_window_mean_is_correctly_rounded(planar: bool) -> None:
    # Integer absolute responses below 2**24 / 15**3 have exact window sums, so the
    # energy is the float64 mean rounded once to float32. A dense float32 tap
    # 1 / 15**3 accumulated over 3375 terms returned 2.9999845 for a constant 3.
    codes = "E5L5" if planar else "E5L5S5"
    flat = np.full((4, 5, 6), 3.0, dtype=np.float32)
    energy = laws_energy(mx.array(flat), codes, 7, FilterPadding.MIRROR, 0.0)
    np.testing.assert_array_equal(np.asarray(energy, dtype=np.float32), flat)
    response = np.random.default_rng(23).integers(-50, 51, size=(9, 10, 11))
    window = (1, 15, 15) if planar else (15, 15, 15)
    padded = np.pad(np.abs(response).astype(np.float64),
                    [(size // 2, size // 2) for size in window], mode="symmetric")
    expected = np.lib.stride_tricks.sliding_window_view(padded, window).mean(
        axis=(-3, -2, -1),
    )
    actual = laws_energy(
        mx.array(response.astype(np.float32)), codes, 7, FilterPadding.MIRROR, 0.0,
    )
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected.astype(np.float32))


@pytest.mark.parametrize("codes", ["L5L5", "E5L5"])
def test_laws_energy_extends_the_absolute_response_by_its_far_field(codes: str) -> None:
    # Far outside an image extended by c the response is c times the kernel sum, so
    # the absolute response extends by that magnitude, never by the signed c itself.
    constant = -10.0
    kernel = _laws_kernel(codes)
    far_field = abs(constant * float(np.sum(kernel)))
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(
            kernel=codes, padding=FilterPadding.CONSTANT, constant=constant, energy_distance=1
        ),
    )
    image = np.random.default_rng(19).normal(size=(7, 8))
    [(_, actual)] = _production(image, options)
    response = np.real(_convolve(image, kernel, FilterPadding.CONSTANT, constant))
    expected = np.real(_convolve(
        np.abs(response), np.full((3, 3), 1.0 / 9.0), FilterPadding.CONSTANT, far_field,
    ))
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    # A constant image extended by its own value has a constant energy at every border.
    [(_, flat)] = _production(np.full((6, 7), constant), options)
    np.testing.assert_allclose(flat, far_field, rtol=1e-5, atol=1e-5)


_DB2_LOW = (-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416)
_DB2_HIGH = (-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037)


def _atrous_oracle(taps: tuple[float, ...], level: int) -> NDArray[np.float64]:
    # Manual's Haar example: a zero after every tap at level 2, then odd support.
    step = 2 ** (level - 1)
    values = [0.0] * (len(taps) * step)
    values[::step] = taps
    return np.array(values + ([0.0] if len(values) % 2 == 0 else []), dtype=np.float64)


def _wavelet_band_oracle(
    image: NDArray[np.float64], band: str, level: int, padding: FilterPadding,
) -> NDArray[np.float64]:
    values = image
    for scale in range(1, level + 1):
        code = band if scale == level else "L" * image.ndim
        for index, letter in enumerate(code):
            taps = _atrous_oracle(_DB2_LOW if letter == "L" else _DB2_HIGH, scale)
            shape = [1] * image.ndim
            shape[image.ndim - 1 - index] = taps.size  # k1 is the last array axis
            values = np.real(_convolve(values, taps.reshape(shape), padding))
    return values


@pytest.mark.parametrize("padding", [FilterPadding.CONSTANT, FilterPadding.MIRROR,
                                     FilterPadding.PERIODIC])
def test_separable_wavelet_band_matches_independent_atrous_chain(
    padding: FilterPadding,
) -> None:
    image = np.random.default_rng(3).normal(size=(7, 8, 9))
    options = TransformSettings(
        image_types=(ImageType.SEPARABLE_WAVELET,),
        separable_wavelet=SeparableWaveletSettings(
            name="db2", bands=("HLH", "LLL"), level=2, padding=padding
        ),
    )
    actual = _production(image, options)
    tag = "constant0" if padding is FilterPadding.CONSTANT else padding.value
    assert [name for name, _ in actual] == [
        f"separablewavelet-db2-L2-HLH-{tag}", f"separablewavelet-db2-L2-LLL-{tag}",
    ]
    for (_, values), band in zip(actual, ("HLH", "LLL"), strict=True):
        np.testing.assert_allclose(
            values, _wavelet_band_oracle(image, band, 2, padding), rtol=1e-5, atol=1e-5,
        )


def test_separable_wavelet_constant_extension_is_the_dense_kernel_oracle() -> None:
    # One level equals extending the image once by c and applying the full kernel.
    image = np.random.default_rng(5).normal(size=(7, 8, 9))
    constant = 1.5
    options = TransformSettings(
        image_types=(ImageType.SEPARABLE_WAVELET,),
        separable_wavelet=SeparableWaveletSettings(
            name="db2", bands=("HLH",), level=1, padding=FilterPadding.CONSTANT, constant=constant
        ),
    )
    [(_, actual)] = _production(image, options)
    k1, k2, k3 = (_atrous_oracle(_DB2_LOW if letter == "L" else _DB2_HIGH, 1)
                  for letter in "HLH")
    dense = np.einsum("i,j,k->ijk", k3, k2, k1)  # k1 is the last array axis
    expected = np.real(_convolve(image, dense, FilterPadding.CONSTANT, constant))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # A constant image extended by its own value has zero high-pass response at
    # every level; the all-low approximation is c (sum L)^3 per level.
    flat = dataclasses.replace(
        options,
        separable_wavelet=dataclasses.replace(
            options.separable_wavelet, bands=("HLH", "LLL"), level=2
        ),
    )
    [(_, high), (_, low)] = _production(np.full((7, 8, 9), constant), flat)
    np.testing.assert_allclose(high, 0.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(low, constant * sum(_DB2_LOW) ** 6, rtol=1e-5)


def test_separable_wavelet_average_pooling_averages_the_rotated_image() -> None:
    # Rotating the kernel by a grid rotation equals filtering the inversely rotated
    # cube and rotating back, so the pooled band is the average over all 24 rotations.
    image = np.random.default_rng(4).normal(size=(8, 8, 8))
    rotations: list[NDArray[np.float64]] = []
    for first in range(6):
        oriented = {0: image, 1: np.rot90(image, 1, (0, 1)), 2: np.rot90(image, 2, (0, 1)),
                    3: np.rot90(image, 3, (0, 1)), 4: np.rot90(image, 1, (0, 2)),
                    5: np.rot90(image, 3, (0, 2))}[first]
        rotations.extend(np.rot90(oriented, turn, (1, 2)) for turn in range(4))
    options = TransformSettings(
        image_types=(ImageType.SEPARABLE_WAVELET,),
        separable_wavelet=SeparableWaveletSettings(
            name="db2", bands=("LHH",), level=2, padding=FilterPadding.PERIODIC
        ),
    )
    pooled = dataclasses.replace(
        options,
        separable_wavelet=dataclasses.replace(
            options.separable_wavelet, pooling=OrientationPooling.AVERAGE
        ),
    )
    [(_, actual)] = _production(image, pooled)
    undo: list[NDArray[np.float64]] = []
    for index, rotated in enumerate(rotations):
        [(_, response)] = _production(rotated, options)
        first, turn = divmod(index, 4)
        response = np.rot90(response, -turn, (1, 2))
        response = {0: response, 1: np.rot90(response, -1, (0, 1)),
                    2: np.rot90(response, -2, (0, 1)), 3: np.rot90(response, -3, (0, 1)),
                    4: np.rot90(response, -1, (0, 2)), 5: np.rot90(response, -3, (0, 2))}[first]
        undo.append(response)
    np.testing.assert_allclose(actual, np.mean(undo, axis=0), rtol=1e-5, atol=1e-5)


def test_laws_orientation_pooling_uses_right_angle_rotations() -> None:
    image = np.random.default_rng(19).normal(size=(9, 9))
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(
            kernel="L5E5", padding=FilterPadding.CONSTANT, pooling=OrientationPooling.MAX
        ),
    )
    [(name, actual)] = _production(image, options)
    kernel = _laws_kernel("L5E5")
    expected = np.max(
        [np.real(_convolve(image, np.rot90(kernel, -rotation),
                           FilterPadding.CONSTANT, 0.0))
         for rotation in range(4)],
        axis=0,
    )
    assert name == "laws-L5E5-constant0-pool-max"
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "response, component",
    [(GaborResponse.MODULUS, "modulus"), (GaborResponse.REAL, "real"),
     (GaborResponse.IMAGINARY, "imaginary")],
)
def test_gabor_impulse_matches_complex_kernel_convolution(
    response: GaborResponse, component: str,
) -> None:
    shape = (16, 16)
    image = np.zeros(shape, dtype=np.float64)
    image[8, 8] = 1.0
    spacing = (0.8, 1.2)
    theta = pi / 3
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(3.0,),
            wavelength=(4.0,),
            gamma=0.5,
            theta=(theta,),
            response=response,
            padding=FilterPadding.CONSTANT,
            constant=0.0,
        ),
    )
    [(name, actual)] = _production(image, options, spacing)
    kernel = _gabor_kernel(shape, spacing, 3.0, 4.0, 0.5, theta)
    expected = _gabor_component(
        _convolve(image, kernel, FilterPadding.CONSTANT, 0.0), response,
    )
    assert name == (
        f"gabor-s3mm-l4mm-g0p5-t1p0471975511965976-{component}-constant0"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_gabor_orientation_pooling_precedes_feature_calculation() -> None:
    image = np.random.default_rng(23).normal(size=(9, 10))
    thetas = (0.0, pi / 2)
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(2.0,),
            wavelength=(3.0,),
            theta=thetas,
            pooling=OrientationPooling.MAX,
            padding=FilterPadding.NEAREST,
        ),
    )
    [(name, actual)] = _production(image, options)
    expected = np.max(
        [np.abs(_convolve(image, _gabor_kernel((9, 10), (1.0, 1.0), 2.0, 3.0, 1.0,
                                               theta), FilterPadding.NEAREST))
         for theta in thetas],
        axis=0,
    )
    assert name == "gabor-s2mm-l3mm-g1-o2-modulus-nearest-pool-max"
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_gabor_three_dimensional_plane_selection_and_averaging() -> None:
    image = np.random.default_rng(29).normal(size=(6, 8, 7))
    spacing = (1.0, 0.8, 1.2)
    theta = 0.4
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(2.0,),
            wavelength=(3.0,),
            theta=(theta,),
            slice_axis=1,
            padding=FilterPadding.CONSTANT,
        ),
    )
    [(axis_name, axis_values)] = _production(image, options, spacing)
    plane_shape = (image.shape[0], image.shape[2])
    plane_spacing = (spacing[0], spacing[2])
    kernel = _gabor_kernel(plane_shape, plane_spacing, 2.0, 3.0, 1.0, theta)
    expected = np.stack(
        [_gabor_component(
            _convolve(image[:, index, :], kernel, FilterPadding.CONSTANT, 0.0),
            GaborResponse.MODULUS,
         ) for index in range(image.shape[1])],
        axis=1,
    )
    assert axis_name == "gabor-s2mm-l3mm-g1-t0p4-modulus-constant0-axis1"
    np.testing.assert_allclose(axis_values, expected, rtol=2e-4, atol=2e-5)

    average_options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(2.0,), wavelength=(3.0,), theta=(theta,), padding=FilterPadding.CONSTANT
        ),
    )
    [(planes_name, planes_values)] = _production(image, average_options, spacing)
    families: list[NDArray[np.float64]] = []
    for axis in range(3):
        moved = np.moveaxis(image, axis, 0)
        remaining = [index for index in range(3) if index != axis]
        family_kernel = _gabor_kernel(
            (moved.shape[1], moved.shape[2]),
            (spacing[remaining[0]], spacing[remaining[1]]),
            2.0, 3.0, 1.0, theta,
        )
        families.append(np.stack(
            [_gabor_component(
                _convolve(plane, family_kernel, FilterPadding.CONSTANT, 0.0),
                GaborResponse.MODULUS,
             ) for plane in moved],
            axis=0,
        ))
    expected_planes = np.mean(
        [np.moveaxis(family, 0, axis) for axis, family in enumerate(families)], axis=0,
    )
    assert planes_name == "gabor-s2mm-l3mm-g1-t0p4-modulus-constant0-planes"
    np.testing.assert_allclose(planes_values, expected_planes, rtol=2e-4, atol=2e-5)


def test_gabor_sinusoid_response_follows_frequency_and_orientation() -> None:
    size = 64
    x1 = np.arange(size, dtype=np.float64)[None, :]
    x2 = np.arange(size, dtype=np.float64)[:, None]
    along_k1 = np.cos(2 * pi * x1 / 4.0) * np.ones((size, 1))
    along_k2 = np.cos(2 * pi * x2 / 4.0) * np.ones((1, size))
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(3.0,), wavelength=(4.0,), theta=(0.0,), padding=FilterPadding.PERIODIC
        ),
    )
    [(_, matched)] = _production(along_k1, options)
    [(_, mismatched)] = _production(along_k2, options)
    kernel = _gabor_kernel((size, size), (1.0, 1.0), 3.0, 4.0, 1.0, 0.0)
    expected = np.abs(_convolve(along_k1, kernel, FilterPadding.PERIODIC))
    np.testing.assert_allclose(matched, expected, rtol=2e-4, atol=2e-5)
    assert np.sqrt(np.mean(matched**2)) > 5 * np.sqrt(np.mean(mismatched**2))


@requires_fft
@pytest.mark.parametrize("shape", [(16, 17), (7, 8, 9)])
def test_simoncelli_matches_independent_fourier_bandpass(shape: tuple[int, ...]) -> None:
    image = np.random.default_rng(31).normal(size=shape)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=(1, 2, 3))
    )
    actual = _production(image, options)
    assert [name for name, _ in actual] == [
        "simoncelli-L1", "simoncelli-L2", "simoncelli-L3",
    ]
    for (_, values), level in zip(actual, options.simoncelli.levels, strict=True):
        np.testing.assert_allclose(
            values, _simoncelli(image, level), rtol=2e-4, atol=2e-5,
        )


@requires_fft
@pytest.mark.parametrize("shape", [(7, 8), (7, 8, 9)])
def test_simoncelli_centered_grid_matches_independent_construction(
    shape: tuple[int, ...],
) -> None:
    image = np.random.default_rng(41).normal(size=shape)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,),
        simoncelli=SimoncelliSettings(levels=(1, 2), grid=FourierGrid.CENTERED),
    )
    actual = _production(image, options)
    assert [name for name, _ in actual] == [
        "simoncelli-L1-centered", "simoncelli-L2-centered",
    ]
    for (_, values), level in zip(actual, options.simoncelli.levels, strict=True):
        np.testing.assert_allclose(
            values, _simoncelli(image, level, FourierGrid.CENTERED),
            rtol=2e-4, atol=2e-5,
        )


@requires_fft
@pytest.mark.parametrize("grid", [FourierGrid.DFT, FourierGrid.CENTERED])
def test_simoncelli_level_below_the_grid_resolution_is_empty(grid: FourierGrid) -> None:
    # Once pi / 2**(level - 1) falls below every nonzero sampled frequency the
    # float64 band selects nothing (its lower edge stays positive, so not even DC):
    # the response is exactly zero, also where float32 cutoffs would underflow.
    image = np.random.default_rng(47).normal(size=(6, 7))
    levels = (127, 129, 1000)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=levels, grid=grid)
    )
    for (_, values), level in zip(_production(image, options), levels, strict=True):
        np.testing.assert_array_equal(values, _simoncelli(image, level, grid))


@pytest.mark.parametrize(
    "options",
    [
        TransformSettings(image_types=(ImageType.MEAN,), mean=MeanSettings(support=5, planar=True)),
        pytest.param(
            TransformSettings(
                image_types=(ImageType.SIMONCELLI,),
                simoncelli=SimoncelliSettings(levels=(1, 2), planar=True),
            ),
            marks=requires_fft,
        ),
        pytest.param(
            TransformSettings(
                image_types=(ImageType.SIMONCELLI,),
                simoncelli=SimoncelliSettings(levels=(1,), grid=FourierGrid.CENTERED, planar=True),
            ),
            marks=requires_fft,
        ),
        TransformSettings(
            image_types=(ImageType.SEPARABLE_WAVELET,),
            separable_wavelet=SeparableWaveletSettings(
                name="db3",
                bands=("LH", "HH"),
                level=2,
                pooling=OrientationPooling.AVERAGE,
                planar=True,
            ),
        ),
    ],
    ids=["mean", "simoncelli", "simoncelli-centered", "wavelet"],
)
def test_planar_filters_equal_the_2d_filter_on_every_axis0_plane(
    options: TransformSettings,
) -> None:
    # IBSI 2 configuration A filters slice by slice; planes must never mix.
    image = np.random.default_rng(47).normal(size=(3, 11, 12))
    planar = _production(image, options)
    flat = dataclasses.replace(
        options,
        mean=dataclasses.replace(options.mean, planar=False),
        simoncelli=dataclasses.replace(options.simoncelli, planar=False),
        separable_wavelet=dataclasses.replace(options.separable_wavelet, planar=False),
    )
    slices = [_production(image[index], flat) for index in range(image.shape[0])]
    assert [name for name, _ in planar] == [f"{name}-2D" for name, _ in slices[0]]
    for position, (_, values) in enumerate(planar):
        expected = np.stack([maps[position][1] for maps in slices])
        np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(7, 8), (7, 8, 9), (9, 9, 9)])
@pytest.mark.parametrize("grid", list(FourierGrid))
def test_riesz_multiplier_matches_numpy_frame_property(
    shape: tuple[int, ...], grid: FourierGrid,
) -> None:
    length = 2
    total = np.zeros(shape, dtype=np.float64)
    # The multiplier primitive takes native array-axis powers; the public
    # response functions reverse their IBSI-order tuples before calling it.
    for order in _orders_of_length(len(shape), length):
        actual = np.asarray(riesz_multiplier(shape, order, grid), dtype=np.complex128)
        expected = _riesz_multiplier(shape, order, grid)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        total = total + np.abs(actual) ** 2
    if grid is FourierGrid.DFT:
        grids = np.meshgrid(
            *(2 * np.fft.fftfreq(size) for size in shape), indexing="ij",
        )
        squared = sum(values * values for values in grids)
        zero = squared == 0
    else:
        grids = np.meshgrid(*(
            (np.arange(size) - (size - 1) / 2) / ((size - 1) / 2)
            for size in shape
        ), indexing="ij")
        centered_squared = sum(values * values for values in grids)
        zero = np.fft.ifftshift(centered_squared == 0)
    np.testing.assert_allclose(total[zero], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(
        total[np.logical_not(zero)], 1.0, rtol=2e-5, atol=2e-5,
    )


def test_riesz_multiplier_normalises_high_orders_before_dividing() -> None:
    # Only the last axis is nonzero at [0, 1], so the directional ratio is exactly
    # one. Raising the raw coordinate and the norm separately underflows both to
    # zero at order 16, which would produce 0/0.
    multiplier = np.asarray(
        riesz_multiplier((16, 2048), (0, 16), FourierGrid.DFT), dtype=np.complex128,
    )
    assert np.all(np.isfinite(multiplier))
    np.testing.assert_allclose(multiplier[0, 1], 1.0 + 0.0j, rtol=0, atol=2e-5)
    high = np.asarray(
        riesz_multiplier((16, 512), (0, 20), FourierGrid.DFT), dtype=np.complex128,
    )
    assert np.all(np.isfinite(high))
    np.testing.assert_allclose(high[0, 1], 1.0 + 0.0j, rtol=0, atol=2e-5)


@requires_fft
def test_riesz_log_high_order_impulse_is_finite() -> None:
    image = np.zeros((16, 2048), dtype=np.float64)
    image[0, 0] = 1.0
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(orders=((16, 0),), log_sigma=(1.0,), padding=FilterPadding.PERIODIC),
    )
    [(_, values)] = _production(image, options)
    assert np.all(np.isfinite(values))
    assert np.any(values != 0.0)


def test_mixed_riesz_order_combines_coefficient_before_float32_overflow() -> None:
    # At equal axis frequencies, the multiplier is sqrt(C(400,200))/2**200.
    # The coefficient alone exceeds float32 and the directional power vanishes,
    # but their product is a modest, representable gain.
    coefficient_squared = factorial(400) // (factorial(200) ** 2)
    expected = sqrt(coefficient_squared / (2 ** 400))
    actual = np.asarray(
        riesz_multiplier((16, 16), (200, 200), FourierGrid.DFT), dtype=np.complex128,
    )
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[1, 1], expected, rtol=1e-4, atol=0)


@requires_fft
@pytest.mark.parametrize("grid", list(FourierGrid))
@pytest.mark.parametrize("padding", [FilterPadding.CONSTANT, FilterPadding.PERIODIC])
def test_riesz_log_response_matches_independent_construction(
    grid: FourierGrid, padding: FilterPadding,
) -> None:
    image = np.random.default_rng(43).normal(size=(7, 8, 9))
    spacing = (1.2, 1.2, 1.2)
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            orders=((1, 0, 0), (0, 2, 0)),
            log_sigma=(2.5,),
            log_cutoff=3.0,
            padding=padding,
            grid=grid,
        ),
    )
    actual = _production(image, options, spacing)
    expected = [
        _riesz_log(image, spacing, 2.5, 3.0, order, grid, padding)
        for order in options.riesz.orders
    ]
    padding_tag = "constant0" if padding is FilterPadding.CONSTANT else padding.value
    assert [name for name, _ in actual] == [
        f"riesz-log-s2p5mm-c3-l1-0-0-{padding_tag}"
        + ("-centered" if grid is FourierGrid.CENTERED else ""),
        f"riesz-log-s2p5mm-c3-l0-2-0-{padding_tag}"
        + ("-centered" if grid is FourierGrid.CENTERED else ""),
    ]
    for (_, values), reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(values, reference, rtol=2e-4, atol=2e-5)


@requires_fft
@pytest.mark.parametrize("grid", list(FourierGrid))
def test_riesz_orders_follow_the_manual_axis_order(grid: FourierGrid) -> None:
    # order (1, 0) is the derivative along k1, the last native array axis; the
    # centered periodic impulse response must be odd on that axis and even on k2.
    image = np.zeros((9, 11), dtype=np.float64)
    image[4, 5] = 1.0
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            orders=((1, 0), (0, 1)), log_sigma=(1.5,), padding=FilterPadding.PERIODIC, grid=grid
        ),
    )
    [(_, along_k1), (_, along_k2)] = _production(image, options)
    np.testing.assert_allclose(along_k1, -np.flip(along_k1, axis=1), rtol=0, atol=2e-5)
    np.testing.assert_allclose(along_k1, np.flip(along_k1, axis=0), rtol=0, atol=2e-5)
    np.testing.assert_allclose(along_k2, -np.flip(along_k2, axis=0), rtol=0, atol=2e-5)
    np.testing.assert_allclose(along_k2, np.flip(along_k2, axis=1), rtol=0, atol=2e-5)


@requires_fft
def test_riesz_log_rejects_anisotropic_spacing() -> None:
    options = TransformSettings(
        image_types=(ImageType.RIESZ,), riesz=RieszSettings(orders=((1, 0, 0),), log_sigma=(2.0,))
    )
    with pytest.raises(ValueError, match="isotropic"):
        next(feature_images(mx.zeros((7, 8, 9)), (1.0, 1.2, 1.0), options))


@requires_fft
@pytest.mark.parametrize("grid", list(FourierGrid))
def test_riesz_simoncelli_response_matches_independent_construction(
    grid: FourierGrid,
) -> None:
    image = np.random.default_rng(47).normal(size=(8, 9, 10))
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            base=RieszBase.SIMONCELLI,
            orders=((0, 2, 0),),
            simoncelli_levels=(1, 2),
            padding=FilterPadding.PERIODIC,
            grid=grid,
        ),
    )
    actual = _production(image, options)
    suffix = "-centered" if grid is FourierGrid.CENTERED else ""
    assert [name for name, _ in actual] == [
        f"riesz-simoncelli-L1-l0-2-0{suffix}",
        f"riesz-simoncelli-L2-l0-2-0{suffix}",
    ]
    for (_, values), level in zip(actual, options.riesz.simoncelli_levels, strict=True):
        np.testing.assert_allclose(
            values, _riesz_simoncelli(image, level, (0, 2, 0), grid),
            rtol=2e-4, atol=2e-5,
        )


@requires_fft
def test_riesz_first_order_response_of_a_constant_is_zero_on_the_dft_grid() -> None:
    image = np.full((8, 9, 10), 3.5, dtype=np.float64)
    log_options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(orders=((1, 0, 0),), log_sigma=(2.0,), padding=FilterPadding.PERIODIC),
    )
    [(_, log_values)] = _production(image, log_options)
    np.testing.assert_allclose(log_values, 0.0, rtol=0, atol=1e-6)
    simoncelli_options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            base=RieszBase.SIMONCELLI, orders=((1, 0, 0),), padding=FilterPadding.PERIODIC
        ),
    )
    [(_, simoncelli_values)] = _production(image, simoncelli_options)
    np.testing.assert_allclose(simoncelli_values, 0.0, rtol=0, atol=1e-6)


@requires_fft
def test_simoncelli_constant_is_zero_and_band_selectivity_holds() -> None:
    constant = np.full((16, 17), 3.0, dtype=np.float64)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=(1, 2))
    )
    for _, values in _production(constant, options):
        np.testing.assert_allclose(values, 0.0, rtol=0, atol=5e-4)

    size = 64
    x1 = np.arange(size, dtype=np.float64)[None, :] * np.ones((size, 1))
    passband = np.cos(2 * pi * x1 / 4.0)
    stopband = np.cos(2 * pi * x1 / 32.0)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=(1,))
    )
    [(_, high)] = _production(passband, options)
    [(_, low)] = _production(stopband, options)
    assert np.sqrt(np.mean(high**2)) > 10 * np.sqrt(np.mean(low**2))


def test_filter_names_match_yielded_images() -> None:
    image = np.random.default_rng(37).normal(size=(9, 10))
    # Simoncelli and Riesz cannot be yielded while FFT filtering is disabled; their
    # name planning is covered by the refusal tests.
    options = TransformSettings(
        image_types=(ImageType.MEAN, ImageType.LAWS, ImageType.GABOR),
        mean=MeanSettings(support=3),
        laws=LawsSettings(kernel="L5S5"),
        gabor=GaborSettings(sigma=(3.0,), wavelength=(4.0,), theta=(0.5,)),
    )
    expected = feature_image_names(image.shape, (1.0, 1.0), options)
    actual = tuple(name for name, _ in _production(image, options))
    assert actual == expected


def test_unsupported_filter_options_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="odd"):
        TransformSettings(image_types=(ImageType.MEAN,), mean=MeanSettings(support=4))
    with pytest.raises(ValueError, match="periodic"):
        TransformSettings(
            image_types=(ImageType.SIMONCELLI,),
            simoncelli=SimoncelliSettings(padding=FilterPadding.MIRROR),
        )
    with pytest.raises(ValueError, match="laws_kernel"):
        TransformSettings(image_types=(ImageType.LAWS,), laws=LawsSettings(kernel='X5Y5'))
    with pytest.raises(ValueError, match="Gabor requires"):
        TransformSettings(
            image_types=(ImageType.GABOR,), gabor=GaborSettings(sigma=(3.0,), wavelength=(4.0,))
        )
    with pytest.raises(ValueError, match="explicit riesz_orders"):
        TransformSettings(image_types=(ImageType.RIESZ,), riesz=RieszSettings(log_sigma=(3.0,)))
    with pytest.raises(ValueError, match="riesz_log_sigma"):
        TransformSettings(image_types=(ImageType.RIESZ,), riesz=RieszSettings(orders=((1, 0, 0),)))
    with pytest.raises(ValueError, match="periodic"):
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(base=RieszBase.SIMONCELLI, orders=((1, 0, 0),)),
        )
    for planar in (
        TransformSettings(image_types=(ImageType.MEAN,), mean=MeanSettings(planar=True)),
        TransformSettings(
            image_types=(ImageType.SEPARABLE_WAVELET,),
            separable_wavelet=SeparableWaveletSettings(bands=("LH",), planar=True),
        ),
    ):
        with pytest.raises(ValueError, match="planar .* requires? a 3-D image"):
            feature_image_names((7, 8), (1.0, 1.0), planar)
        with pytest.raises(ValueError, match="planar .* requires? a 3-D image"):
            next(feature_images(mx.zeros((7, 8)), (1.0, 1.0), planar))


@requires_fft
@pytest.mark.parametrize("ndim", [2, 3])
def test_simoncelli_oblique_fourier_mode_has_analytic_gain(ndim: int) -> None:
    # For N=16 and frequency (1,1[,0]), |nu|=pi*sqrt(2)/8.
    # At level 2 the logarithm's argument is 1/sqrt(2), giving gain cos(pi/4).
    axes = np.indices((16,) * ndim, dtype=np.float64)
    image = np.cos(2 * pi * (axes[0] + axes[1]) / 16)
    options = TransformSettings(
        image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=(2,))
    )
    [(_, values)] = _production(image, options)
    np.testing.assert_allclose(values, image / np.sqrt(2), rtol=2e-5, atol=2e-6)


def test_large_gabor_impulse_preserves_small_responses_and_empty_planes() -> None:
    # IBSI4.a.1 scale/support: FFT convolution must keep low-amplitude
    # tails, not only pass the smaller unit-impulse test.
    image = np.zeros((3, 64, 64), dtype=np.float64)
    image[1, 31, 31] = 255.0
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(10.0,),
            wavelength=(4.0,),
            gamma=0.5,
            theta=(pi / 3,),
            slice_axis=0,
            padding=FilterPadding.CONSTANT,
        ),
    )
    [(_, values)] = _production(image, options, (2.0, 2.0, 2.0))
    # An impulse's modulus response is its amplitude times the Gaussian
    # envelope, independent of the complex sinusoid's phase.
    row, col = np.indices((64, 64), dtype=np.float64)
    x, y = 2*(col-31), 2*(row-31)
    radial = np.cos(pi/3)*x + np.sin(pi/3)*y
    orthogonal = np.sin(pi/3)*x - np.cos(pi/3)*y
    expected = 255*np.exp(-(radial**2 + 0.5**2*orthogonal**2)/(2*10**2))
    np.testing.assert_allclose(values[1], expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(values[[0, 2]], 0.0)


def test_average_orientation_pooling_keeps_a_finite_mean_of_finite_maps() -> None:
    # Every rotated L3L3 response of a constant 1e38 image is the finite 8/3 * 1e38;
    # summing the four orientations before dividing overflowed to inf.
    image = np.full((3, 3), 1e38, np.float32)
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        laws=LawsSettings(
            kernel="L3L3", pooling=OrientationPooling.AVERAGE, padding=FilterPadding.NEAREST
        ),
    )
    [(_, pooled)] = _production(image, options)
    single = _production(
        image,
        dataclasses.replace(
            options, laws=dataclasses.replace(options.laws, pooling=OrientationPooling.NONE)
        ),
    )
    np.testing.assert_array_equal(pooled, single[0][1])
    assert np.isfinite(pooled).all()


@pytest.mark.parametrize("row", [(1e8, 1.0, -1e8), (1.0, -1e8, 1e8), (-1e8, 1e8, 1.0)])
@pytest.mark.parametrize("ndim,planar,signal_axis", [
    (2, False, 0), (2, False, 1),
    (3, False, 0), (3, False, 1), (3, False, 2),
    (3, True, 1), (3, True, 2),
])
def test_mean_filter_gives_identical_neighborhoods_the_same_mean(
    row: tuple[float, float, float], ndim: int, planar: bool, signal_axis: int,
) -> None:
    # Every periodic window has equally many copies of all three source values,
    # so its mean is 1/3. Varying the signal axis also catches residuals dropped
    # between separable passes, including planar filtering of a 3-D image.
    shape = [1] * ndim
    shape[signal_axis] = 3
    image = np.broadcast_to(np.array(row, np.float32).reshape(shape), (3,) * ndim).copy()
    with mx.stream(mx.gpu):
        result = np.asarray(mean_response(mx.array(image), 3, FilterPadding.PERIODIC, 0.0, planar),
                            dtype=np.float32)
    np.testing.assert_array_equal(result, np.float32(1 / 3))


@requires_fft
def test_fft_filters_are_amplitude_invariant_beyond_the_spectrum_range() -> None:
    # 64 samples near 3e37 sum past the float32 range in the unnormalized forward FFT
    # although every response is representable; responses are linear in amplitude.
    field = np.random.default_rng(1).normal(size=(8, 8)).astype(np.float32) + 3
    large = field * np.float32(1e37)
    with mx.stream(mx.gpu):
        pairs = [
            (simoncelli_response(mx.array(field), 1, FourierGrid.DFT),
             simoncelli_response(mx.array(large), 1, FourierGrid.DFT)),
            (riesz_simoncelli_responses(mx.array(field), 1, ((1, 0),), FourierGrid.DFT)[0],
             riesz_simoncelli_responses(mx.array(large), 1, ((1, 0),), FourierGrid.DFT)[0]),
        ]
        for unit, scaled in pairs:
            expected = np.asarray(unit, dtype=np.float64) * 1e37
            actual = np.asarray(scaled, dtype=np.float64)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * 1e37)


@requires_fft
@pytest.mark.parametrize("shape", [(23, 23), (53, 2)])
def test_simoncelli_survives_bluestein_growth_on_prime_length_grids(
    shape: tuple[int, int],
) -> None:
    # A prime-length GPU FFT runs as a chirp convolution on a longer power-of-two
    # grid whose partial sums exceed count * peak; at a quarter of the float32 range
    # every response would be NaN if overflowed although the true spectrum stays far below it.
    count = shape[0] * shape[1]
    field = np.random.default_rng(0).normal(size=shape)
    peak = float(np.finfo(np.float32).max) / 4 / count
    image = (field / np.abs(field).max() * peak).astype(np.float32)
    # Float64 oracle of the level-1 DFT band (manual section 4.6, cutoff pi).
    frequencies = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size) for size in shape),
                              indexing="ij")
    norm = np.sqrt(sum(values**2 for values in frequencies))
    selected = (norm >= np.pi / 4) & (norm <= np.pi)
    band = np.zeros(shape)
    band[selected] = np.cos(np.pi / 2 * np.log2(2 * norm[selected] / np.pi))
    expected = np.real(np.fft.ifftn(np.fft.fftn(image.astype(np.float64)) * band))
    with mx.stream(mx.gpu):
        actual = np.asarray(simoncelli_response(mx.array(image), 1, FourierGrid.DFT),
                            dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * peak)


@requires_fft
def test_attenuating_riesz_log_kernel_does_not_let_the_forward_fft_overflow() -> None:
    # The sigma-3 kernel's absolute sum is 0.163, but it multiplies the spectrum only
    # after the forward FFT: budgeting that gain let 361 * 1e38 overflow to NaN.
    image = mx.full((19, 19), 1e38, dtype=mx.float32)
    [response] = riesz_log_responses(image, (1.0, 1.0), 3.0, 4.0, ((1, 0),),
                                     FourierGrid.DFT, FilterPadding.MIRROR, 0.0)
    assert np.isfinite(np.asarray(response, dtype=np.float32)).all()
