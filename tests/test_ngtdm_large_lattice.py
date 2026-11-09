"""Independent NGTDM definitions on sparse and fully occupied large gray lattices."""

import numpy as np
import pytest

from mlxradiomics.features.distributions import ngtdm_statistics
from mlxradiomics.numerics.runtime import mx


@pytest.mark.parametrize("selected", [
    None, ("Busyness",), ("Complexity", "Strength"),
])
def test_sparse_ngtdm_preserves_actual_levels_and_undefined_rows(
    selected: tuple[str, ...] | None,
) -> None:
    n = np.zeros((4, 25001), dtype=np.int32)
    s = np.zeros(n.shape, dtype=np.float32)
    n[0, [0, 2, 25000]], s[0, [0, 2, 25000]] = [2, 1, 3], [0.5, 1.5, 2]
    n[1, [1, 24000]], s[1, [1, 24000]] = [4, 2], [1, 2]
    n[3, 20000] = 5
    # Direct rational enumeration over the three/two occupied levels. Row 2 has
    # no eligible centers; row 3 has one constant level and zero difference sum.
    expected = {
        "Coarseness": [12 / 17, 3 / 4, np.nan, np.inf],
        "Contrast": [2812350010 / 81, 575952001 / 9, np.nan, 0],
        "Busyness": [17 / 600008, 2 / 23997, np.nan, np.nan],
        "Complexity": [982475 / 36, 95996 / 9, np.nan, 0],
        "Strength": [468716669, 1151904002 / 3, np.nan, np.nan],
    }
    result = ngtdm_statistics(mx.array(n), mx.array(s), selected)
    assert set(result) == (set(expected) if selected is None else set(selected))
    for name, values in result.items():
        np.testing.assert_allclose(
            np.asarray(values, dtype=np.float32), expected[name], rtol=2e-6, atol=0,
            equal_nan=True, err_msg=name,
        )


@pytest.mark.parametrize("batch", [1, 3])
def test_ngtdm_uniform_lattice_matches_closed_form_across_tiles(batch: int) -> None:
    count = 1025
    levels = 16001 + 2 * np.arange(count)
    n = np.zeros((batch, int(levels[-1])), dtype=np.int32)
    n[:, levels - 1] = 1
    s = n.astype(np.float32)
    # Uniform counts and unit difference sums give p_i=1/K. For equally spaced
    # levels (step 2), sum |i-j| and sum (i-j)^2 have exact polynomial forms.
    expected = {
        "Coarseness": 1,
        "Contrast": 4 * (count + 1) / (6 * count),
        "Busyness": 3 / (2 * (count**2 - 1)),
        "Complexity": 2 * (count**2 - 1) / 3,
        "Strength": 4 * (count**2 - 1) / 3,
    }
    result = ngtdm_statistics(mx.array(n), mx.array(s))
    assert set(result) == set(expected)
    for name, target in expected.items():
        np.testing.assert_allclose(
            np.asarray(result[name], dtype=np.float32), np.full(batch, target),
            rtol=2e-6, atol=0, err_msg=name,
        )
