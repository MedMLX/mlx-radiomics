"""Symmetric spectra: scalar float32 contracts and the largest-eigenvalue kernel."""

import numpy as np
import pytest

from mlxradiomics.numerics.eigen import largest_symmetric_eigenvalue, symmetric_eigenvalues
from mlxradiomics.numerics.runtime import mx


def test_singleton_spectra_preserve_signed_zero_and_special_value_words() -> None:
    words = np.array([0, 0x80000000, 0x3F800000, 0x7F800000,
                      0xFF800000, 0x7FC00001], dtype=np.uint32)
    matrices = words.view(np.float32).reshape(-1, 1, 1)
    actual = np.asarray(symmetric_eigenvalues(mx.array(matrices)), dtype=np.float32)
    assert actual.shape == (words.size, 1)
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint32), words)


@pytest.mark.parametrize("size", [2, 3, 17, 64, 83, 89, 90, 128, 200])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e4])
def test_largest_eigenvalue_matches_a_float64_spectrum(size: int, scale: float) -> None:
    # Sizes 89 and 90 straddle the threadgroup-memory limit; larger sizes use the device
    # workspace. Householder reduction and bisection have an absolute error of order
    # n * eps * ||A||, so the bound scales with the spectral radius.
    rng = np.random.default_rng(size)
    general = rng.standard_normal((3, size, size))
    symmetric = (general + np.swapaxes(general, 1, 2)) * scale
    gram = general @ np.swapaxes(general, 1, 2) * scale
    for matrices in (symmetric, gram):
        values = matrices.astype(np.float32)
        actual = np.asarray(largest_symmetric_eigenvalue(mx.array(values)), dtype=np.float64)
        spectra = np.linalg.eigvalsh(values.astype(np.float64))
        radius = np.abs(spectra).max(axis=1)
        bound = 4 * size * float(np.finfo(np.float32).eps) * radius
        assert np.all(np.abs(actual - spectra[:, -1]) <= bound)


def test_largest_eigenvalue_of_trivial_batches() -> None:
    assert np.asarray(largest_symmetric_eigenvalue(mx.zeros((0, 5, 5)))).shape == (0,)
    zero = np.asarray(largest_symmetric_eigenvalue(mx.zeros((2, 7, 7))), dtype=np.float32)
    np.testing.assert_array_equal(zero, np.zeros(2, dtype=np.float32))
    single = np.array([[[3.5]], [[-2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(largest_symmetric_eigenvalue(mx.array(single)), dtype=np.float32),
        np.array([3.5, -2.0], dtype=np.float32),
    )
    diagonal = np.diag(np.array([1.0, 4.0, -9.0, 2.5], dtype=np.float32))[None]
    largest = np.asarray(largest_symmetric_eigenvalue(mx.array(diagonal)), dtype=np.float32)
    assert float(largest[0]) == pytest.approx(4.0, rel=4e-7)
