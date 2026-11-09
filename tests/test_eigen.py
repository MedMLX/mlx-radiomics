"""Small symmetric spectra retain their scalar float32 contracts."""

import numpy as np

from mlxradiomics.numerics.eigen import symmetric_eigenvalues
from mlxradiomics.numerics.runtime import mx


def test_singleton_spectra_preserve_signed_zero_and_special_value_words() -> None:
    words = np.array([0, 0x80000000, 0x3F800000, 0x7F800000,
                      0xFF800000, 0x7FC00001], dtype=np.uint32)
    matrices = words.view(np.float32).reshape(-1, 1, 1)
    actual = np.asarray(symmetric_eigenvalues(mx.array(matrices)), dtype=np.float32)
    assert actual.shape == (words.size, 1)
    np.testing.assert_array_equal(actual.reshape(-1).view(np.uint32), words)
