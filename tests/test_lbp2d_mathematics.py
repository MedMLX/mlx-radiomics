"""LBP bit signs and sampled-ring variance against independent arithmetic."""

from fractions import Fraction
from math import cos, floor, pi, sin, sqrt

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics.config.filtering import LBPMethod
from mlxradiomics.config.transforms import ImageType, LBP2DSettings, TransformSettings
from mlxradiomics.filters.lbp2d import lbp2d
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import mx


def _center(image: NDArray[np.float64], samples: int, radius: float, method: LBPMethod) -> float:
    high = image.astype(np.float32)
    residual = (image - high).astype(np.float32)
    return float(np.asarray(lbp2d(mx.array(high), samples, radius, method, 0,
                                 mx.array(residual)))[1, 1])


def _declared_offsets(samples: int, radius: float) -> list[tuple[float, float]]:
    return [
        (
            round(-radius * sin(2 * pi * k / samples), 5),
            round(radius * cos(2 * pi * k / samples), 5),
        )
        for k in range(samples)
    ]


def _exact_lbp_codes(image: NDArray[np.float64], samples: int, radius: float) -> tuple[int, int]:
    """Derive the raw and uniform codes with exact rational arithmetic."""
    height, width = image.shape
    center = Fraction.from_float(float(image[1, 1]))
    bits: list[bool] = []
    for row_offset, col_offset in _declared_offsets(samples, radius):
        row_floor, col_floor = floor(row_offset), floor(col_offset)
        row_fraction = Fraction(str(row_offset)) - row_floor
        col_fraction = Fraction(str(col_offset)) - col_floor
        difference = Fraction(0)
        for row_corner in range(2):
            for col_corner in range(2):
                row = 1 + row_floor + row_corner
                col = 1 + col_floor + col_corner
                sample = (
                    Fraction.from_float(float(image[row, col]))
                    if 0 <= row < height and 0 <= col < width else Fraction(0)
                )
                row_weight = row_fraction if row_corner else 1 - row_fraction
                col_weight = col_fraction if col_corner else 1 - col_fraction
                difference += (sample - center) * row_weight * col_weight
        bits.append(difference >= 0)
    code = sum(int(bit) << index for index, bit in enumerate(bits))
    changes = sum(bits[index] != bits[(index + 1) % samples] for index in range(samples))
    return code, (sum(bits) if changes <= 2 else samples + 1)


def test_interpolation_coefficient_residual_preserves_ring_sign() -> None:
    # The bit-1 difference is +9.406214334834765e-9 in exact arithmetic on the
    # declared five-decimal sampling coordinates; the bit must follow it, which needs
    # float-float coefficient residuals rather than float32 interpolation weights.
    image = np.zeros((3, 3), dtype=np.float64)
    image[0, 1] = -1.0
    image[0, 2] = 1.8284543250036684
    exact_code, exact_uniform = _exact_lbp_codes(image, samples=8, radius=0.5)
    assert (exact_code, exact_uniform) == (243, 6)
    high = image.astype(np.float32)
    residual = (image - high).astype(np.float32)
    tail = (image - high.astype(np.float64) - residual.astype(np.float64)).astype(np.float32)
    actual: dict[LBPMethod, float] = {}
    with mx.stream(mx.gpu):
        for method in (LBPMethod.DEFAULT, LBPMethod.UNIFORM):
            options = TransformSettings(
                image_types=(ImageType.LBP2D,),
                lbp2d=LBP2DSettings(samples=8, radius=0.5, method=method),
            )
            output = next(feature_images(
                mx.array(high), (1.0, 1.0), options,
                source_residual=mx.array(residual), source_tail=mx.array(tail),
            ))
            actual[method] = float(np.asarray(output.image)[1, 1])
    assert actual == {LBPMethod.DEFAULT: 243, LBPMethod.UNIFORM: 6}


def test_exact_source_ties_set_bits_through_public_transform() -> None:
    image = np.zeros((3, 3), dtype=np.float64)
    high = image.astype(np.float32)
    actual: dict[LBPMethod, float] = {}
    with mx.stream(mx.gpu):
        for method in (LBPMethod.DEFAULT, LBPMethod.UNIFORM):
            options = TransformSettings(
                image_types=(ImageType.LBP2D,),
                lbp2d=LBP2DSettings(samples=8, radius=0.5, method=method),
            )
            output = next(feature_images(
                mx.array(high), (1.0, 1.0), options,
                source_residual=mx.array((image - high).astype(np.float32)),
            ))
            actual[method] = float(np.asarray(output.image)[1, 1])
    # Every sample equals the center, so every bit is set. Uniform code 8 is the
    # number of set bits for a pattern with no transitions.
    assert actual == {LBPMethod.DEFAULT: 255, LBPMethod.UNIFORM: 8}


@pytest.mark.parametrize("delta, expected", [(-3e-9, 0.0), (3e-9, 1.0)])
def test_source_threshold_neighbor_resolves_without_epsilon(
    delta: float, expected: float,
) -> None:
    # A radius-one, four-sample ring places bit 0 exactly on the right neighbor,
    # where the bilinear weight is exactly one. The neighbor differs from the
    # center by less than half a float32 ulp but remains distinct in the source
    # residual. No epsilon may flip the categorical comparison.
    center_source = float(np.float32(0.1))
    image = np.full((3, 3), center_source - 1.0, dtype=np.float64)
    image[1, 1] = center_source
    image[1, 2] = center_source + delta
    high = image.astype(np.float32)
    options = TransformSettings(
        image_types=(ImageType.LBP2D,),
        lbp2d=LBP2DSettings(samples=4, radius=1.0, method=LBPMethod.DEFAULT),
    )
    with mx.stream(mx.gpu):
        output = next(feature_images(
            mx.array(high), (1.0, 1.0), options,
            source_residual=mx.array((image - high).astype(np.float32)),
        ))
    assert float(np.asarray(output.image)[1, 1]) == expected


@pytest.mark.parametrize("method, expected", [(LBPMethod.DEFAULT, 253), (LBPMethod.UNIFORM, 7)])
def test_interpolated_bit_keeps_residual_after_cancellation(
    method: LBPMethod, expected: int,
) -> None:
    image = np.array([[0., 0., -10000.0001], [0., 0., 10000.], [0., 0., 0.]])
    assert _center(image, 8, sqrt(0.5), method) == expected


@pytest.mark.parametrize("baseline", [4000.0, 10000.0, 1e8])
def test_sampled_ring_variance_is_independent_of_center_and_offset(baseline: float) -> None:
    image = np.zeros((3, 3), dtype=np.float64)
    image[1, 2], image[0, 1], image[1, 0], image[2, 1] = baseline + np.arange(4)
    assert _center(image, 4, 1.0, LBPMethod.VARIANCE) == pytest.approx(1.25, rel=1e-6)


def _exact_ring(image: NDArray[np.float64], row: int, col: int, samples: int,
                radius: float) -> list[Fraction]:
    """Bilinear ring samples at the declared offsets, in exact rational arithmetic."""
    height, width = image.shape
    ring: list[Fraction] = []
    for row_offset, col_offset in _declared_offsets(samples, radius):
        row_floor, col_floor = floor(row_offset), floor(col_offset)
        row_fraction = Fraction(str(row_offset)) - row_floor
        col_fraction = Fraction(str(col_offset)) - col_floor
        value = Fraction(0)
        for row_corner in range(2):
            for col_corner in range(2):
                y, x = row + row_floor + row_corner, col + col_floor + col_corner
                sample = (Fraction.from_float(float(image[y, x]))
                          if 0 <= y < height and 0 <= x < width else Fraction(0))
                value += (sample * (row_fraction if row_corner else 1 - row_fraction)
                          * (col_fraction if col_corner else 1 - col_fraction))
        ring.append(value)
    return ring


@pytest.mark.parametrize(("background", "spot", "shape", "radius"), [
    (0.0, 5.0, (7, 7), 2.0),
    (2.0, -7.0, (7, 7), 2.0),
    (0.0, 3.0, (1, 1), 3.0),
    (0.0, 3.0, (9, 1), 1.5),
])
def test_ring_in_uniform_neighbourhood_has_undefined_variance(
    background: float, spot: float, shape: tuple[int, int], radius: float,
) -> None:
    # A voxel unlike its uniform surroundings (or a ring entirely outside a thin
    # slice) samples one value at every ring point. Its exact variance is zero,
    # which the VARIANCE method reports as NaN. Bilinear weights whose float-float
    # products do not sum to exactly one left a ~1e-29 residual variance instead.
    image = np.full(shape, background, dtype=np.float64)
    row, col = shape[0] // 2, shape[1] // 2
    image[row, col] = spot
    ring = _exact_ring(image, row, col, 8, radius)
    mean = sum(ring, Fraction(0)) / len(ring)
    assert sum((value - mean) ** 2 for value in ring) == 0
    output = lbp2d(mx.array(image.astype(np.float32)), 8, radius, LBPMethod.VARIANCE, 0)
    assert np.isnan(np.asarray(output)[row, col])


def test_categorical_interpolation_preserves_exact_cancellation_and_small_signs() -> None:
    # Opposing diagonal ramps cancel exactly. A tiny independently represented
    # corner perturbation must decide the bit even after large terms cancel.
    image = np.array([[-0.0, -1.0, 0.0], [-1.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
    for perturbation in (0.0, 2.0**-70, -(2.0**-70)):
        image[0, 2] = perturbation
        exact, _ = _exact_lbp_codes(image, samples=8, radius=0.5)
        assert _center(image, 8, 0.5, LBPMethod.DEFAULT) == exact


@pytest.mark.parametrize("value", [2.0**-100, 2.0**100])
def test_categorical_interpolation_refuses_unsupported_source_amplitudes(value: float) -> None:
    image = mx.full((3, 3), value)
    with pytest.raises(ValueError, match="source sign is unresolved"):
        lbp2d(image, 8, 0.5, LBPMethod.DEFAULT, 0)


def test_source_third_word_decides_cancellation_through_public_transform() -> None:
    image = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
    options = TransformSettings(
        image_types=(ImageType.LBP2D,),
        lbp2d=LBP2DSettings(method=LBPMethod.DEFAULT, samples=8, radius=0.5),
    )
    for perturbation in (2.0**-70, -(2.0**-70)):
        source = image.copy()
        source[0, 2] = perturbation
        expected, _ = _exact_lbp_codes(source, 8, 0.5)
        tail = np.zeros(image.shape, dtype=np.float32)
        tail[0, 2] = perturbation
        output = next(feature_images(mx.array(image.astype(np.float32)), (1.0, 1.0),
                                     options, source_tail=mx.array(tail)))
        assert float(np.asarray(output.image)[1, 1]) == expected
