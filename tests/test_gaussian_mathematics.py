"""Analytic Gaussian invariants and pinned SimpleITK recurrence comparisons on Metal."""

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.typing import NDArray

from mlxradiomics.filters.recursive_gaussian import laplacian_gaussian, recursive_gaussian
from mlxradiomics.numerics.runtime import mx

RATIOS = (0.1, 0.5, 1.0, 2.0, 20.0, 50.0, 200.0)


def _signals(length: int) -> NDArray[np.float32]:
    coordinate = np.linspace(0.0, 1.0, length)
    return np.stack([
        np.eye(1, length, length // 2).ravel(),
        np.linspace(-1.0, 1.0, length),
        np.sin(3.0 * np.pi * coordinate),
        5.0 + np.sin(3.0 * np.pi * coordinate),
    ]).astype(np.float32)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("order", [0, 2])
def test_constant_line_keeps_analytic_response(ratio: float, order: int) -> None:
    # Constant equilibrium is a stronger independent check than reproducing another
    # finite-precision recursive filter's DC drift, especially at near-unit poles.
    image = np.stack([np.full(250, 7.25), np.full(250, -3.5), np.zeros(250)]).astype(np.float32)
    with mx.stream(mx.gpu):
        values = np.asarray(
            recursive_gaussian(mx.array(image), ratio, 1.0, 1, order), dtype=np.float64,
        )
    expected = image if order == 0 else np.zeros_like(image)
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("order", [0, 2])
def test_recurrence_matches_pinned_simpleitk(ratio: float, order: int) -> None:
    spacing = 0.7
    image = _signals(512)
    source = sitk.GetImageFromArray(image.astype(np.float64))
    source.SetSpacing((spacing, 1.0))
    reference = sitk.RecursiveGaussianImageFilter()
    reference.SetDirection(0)  # SimpleITK x is the final NumPy axis.
    reference.SetOrder(order)
    reference.SetSigma(ratio * spacing)
    reference.SetNormalizeAcrossScale(True)
    expected = np.asarray(sitk.GetArrayFromImage(reference.Execute(source)), dtype=np.float64)
    with mx.stream(mx.gpu):
        values = np.asarray(
            recursive_gaussian(mx.array(image), ratio * spacing, spacing, 1, order),
            dtype=np.float64,
        )
    # Both outputs are float32. This bound catches recurrence drift and derivative
    # normalization errors while allowing host-coefficient and output rounding.
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("constant", [False, True], ids=["nonconstant", "constant"])
def test_anisotropic_laplacian_matches_pinned_simpleitk(constant: bool) -> None:
    sigma, spacing = 1.0, (0.5, 1.0, 2.0)
    shape = (8, 9, 10)
    coordinate = np.indices(shape).sum(axis=0)
    image = (0.1 * np.sin(coordinate / 2.0)).astype(np.float32)
    image[4, 4, 5] += 1.0
    if constant:
        image.fill(-17)
    source = sitk.GetImageFromArray(image.astype(np.float64))
    source.SetSpacing(spacing[::-1])
    reference = sitk.LaplacianRecursiveGaussianImageFilter()
    reference.SetSigma(sigma)
    reference.SetNormalizeAcrossScale(True)
    expected = np.asarray(sitk.GetArrayFromImage(reference.Execute(source)), dtype=np.float64)
    with mx.stream(mx.gpu):
        values = np.asarray(laplacian_gaussian(mx.array(image), spacing, sigma), dtype=np.float64)
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=2e-6)
    if constant:
        np.testing.assert_allclose(values, 0.0, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("sigma", [1e-3, 1e-2])
def test_tiny_smoothing_sigma_is_the_identity_delta_limit(sigma: float) -> None:
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    with mx.stream(mx.gpu):
        result = np.asarray(
            recursive_gaussian(mx.array(values), sigma, 1.0, 0, 0), dtype=np.float64,
        )
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize(
    "sigma,spacing,order",
    [
        pytest.param(np.nan, 1.0, 0, id="nan-sigma"),
        pytest.param(np.inf, 1.0, 0, id="infinite-sigma"),
        pytest.param(0.0, 1.0, 0, id="zero-sigma"),
        pytest.param(-1.0, 1.0, 0, id="negative-sigma"),
        pytest.param(1.0, np.nan, 0, id="nan-spacing"),
        pytest.param(1.0, np.inf, 0, id="infinite-spacing"),
        pytest.param(1.0, 0.0, 0, id="zero-spacing"),
        pytest.param(1.0, -1.0, 0, id="negative-spacing"),
        pytest.param(0.001, 1.0, 2, id="singular-second-order-normalization"),
        pytest.param(0.03, 1.0, 2, id="ill-conditioned-second-order-normalization"),
        pytest.param(201.0, 1.0, 0, id="unqualified-smoothing-width"),
        pytest.param(201.0, 1.0, 2, id="unqualified-derivative-width"),
        pytest.param(1e308, 1.0, 0, id="huge-width"),
        pytest.param(1e200, 1e200, 2, id="overflowing-physical-normalization"),
        pytest.param(1e-200, 1e-200, 2, id="underflowing-physical-normalization"),
        pytest.param(1e-30, 1e-30, 2, id="underflowing-metal-coefficients"),
    ],
)
def test_invalid_or_unresolved_filter_parameters_raise(
    sigma: float, spacing: float, order: int,
) -> None:
    with pytest.raises(ValueError):
        recursive_gaussian(mx.zeros((4, 5)), float(sigma), float(spacing), 0, order)


def test_laplacian_rejects_unresolved_second_derivative() -> None:
    with pytest.raises(ValueError, match="second-order recursive Gaussian"):
        laplacian_gaussian(mx.zeros((4, 4, 4)), (1.0, 1.0, 1.0), 1e-3)


@pytest.mark.parametrize(
    "shape,axis,order",
    [((4, 5), 2, 0), ((4, 5), 0, 1), ((3, 5), 0, 0)],
    ids=["invalid-axis", "invalid-order", "short-axis"],
)
def test_recursive_gaussian_rejects_invalid_topology(
    shape: tuple[int, ...], axis: int, order: int,
) -> None:
    with pytest.raises(ValueError):
        recursive_gaussian(mx.zeros(shape), 1.0, 1.0, axis, order)
