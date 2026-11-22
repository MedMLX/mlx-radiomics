"""Metal compensated windows and wavelet bands against independent references.

These tests execute production Metal paths and compare with independent dense
box sums or directly enumerated separable kernels.
"""

from itertools import product
from math import fsum

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_crop

from mlxradiomics.config.filtering import FilterPadding, OrientationPooling
from mlxradiomics.filters import convolution as filters
from mlxradiomics.filters import mean_response
from mlxradiomics.numerics.runtime import mx


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("padding", list(FilterPadding))
def test_fused_mean_matches_independent_dense_box(
    planar: bool, padding: FilterPadding,
) -> None:
    ct, _, _ = hcc_crop(np.s_[105:109, 215:220, 62:69])
    image = ct.astype(np.float32)
    support, constant = 5, 13.0
    axes = tuple(range(1 if planar else 0, image.ndim))
    widths = [(support // 2, support // 2) if axis in axes else (0, 0)
              for axis in range(image.ndim)]
    if padding is FilterPadding.CONSTANT:
        padded = np.pad(image, widths, mode="constant", constant_values=constant)
    elif padding is FilterPadding.MIRROR:
        padded = np.pad(image, widths, mode="symmetric")
    elif padding is FilterPadding.NEAREST:
        padded = np.pad(image, widths, mode="edge")
    else:
        padded = np.pad(image, widths, mode="wrap")
    expected = np.empty(image.shape, dtype=np.float64)
    for coordinate in np.ndindex(*image.shape):
        bounds = tuple(
            slice(value, value + support) if axis in axes else slice(value, value + 1)
            for axis, value in enumerate(coordinate)
        )
        expected[coordinate] = fsum(float(value) for value in padded[bounds].ravel())
    expected /= support ** len(axes)
    with mx.stream(mx.gpu):
        actual = np.asarray(
            mean_response(mx.array(image), support, padding, constant, planar), dtype=np.float32,
        )
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def _direct_wavelet_band(
    image: NDArray[np.float64], code: str, level: int, constant: float,
    low: tuple[float, ...], high: tuple[float, ...],
) -> NDArray[np.float64]:
    """Enumerate sparse tensor-product taps without separable intermediate maps."""
    taps = {"L": low, "H": high}
    step = 2 ** (level - 1)
    expected = np.empty_like(image)
    for coordinate in np.ndindex(*image.shape):
        terms: list[float] = []
        for tap_indexes in product(*(range(len(taps[letter])) for letter in code)):
            point = list(coordinate)
            weight = 1.0
            for k, (letter, tap) in enumerate(zip(code, tap_indexes, strict=True)):
                vector = taps[letter]
                point[image.ndim - 1 - k] += len(vector) * step // 2 - tap * step
                weight *= vector[tap]
            value = constant if any(index < 0 or index >= size
                                    for index, size in zip(point, image.shape, strict=True)) else (
                float(image[tuple(point)])
            )
            terms.append(weight * value)
        expected[coordinate] = fsum(terms)
    return expected


@pytest.mark.parametrize("shape,planar", [((5, 7), False), ((3, 5, 7), False), ((3, 5, 7), True)])
@pytest.mark.parametrize("level", [1, 2])
def test_wavelet_bands_match_independent_dense_kernels(
    shape: tuple[int, ...], planar: bool, level: int,
) -> None:
    crop = np.s_[105:110, 215:222, 64] if len(shape) == 2 else np.s_[105:108, 215:220, 62:69]
    image, _, _ = hcc_crop(crop)
    low, high, constant = (0.5, 1.0, 0.5), (0.5, 0.0, -0.5), 7.0
    axes = 2 if planar else len(shape)
    # Reverse order ensures the cached path is not tied to an all-low-first bank.
    bands = tuple("".join(letters) for letters in product("HL", repeat=axes))
    with mx.stream(mx.gpu):
        responses = filters.separable_wavelet_responses(
            mx.array(image.astype(np.float32)), low, high, bands, level,
            FilterPadding.CONSTANT, constant, OrientationPooling.NONE, planar,
        )
        actual = [np.asarray(response, dtype=np.float64) for response in responses]
    approximation, extension = image, constant
    for prior_level in range(1, level):
        approximation = _direct_wavelet_band(
            approximation, "L" * axes, prior_level, extension, low, high,
        )
        extension *= sum(low) ** axes
    for band, response in zip(bands, actual, strict=True):
        expected = _direct_wavelet_band(approximation, band, level, extension, low, high)
        np.testing.assert_allclose(response, expected, rtol=2e-6, atol=1e-6)
