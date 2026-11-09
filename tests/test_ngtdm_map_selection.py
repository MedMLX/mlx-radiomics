"""Partial NGTDM map selections preserve definitions and singular-value behavior."""

import numpy as np
import pytest
from conftest import settings_for
from radiomics_reference.mathematics import NGTDM_NAMES, ngtdm_features

from mlxradiomics import extract

_SELECTIONS = [
    pytest.param(("Coarseness", "Strength"), id="nonadjacent-columns"),
    pytest.param(NGTDM_NAMES, id="all-five"),
]


@pytest.mark.parametrize("selected", _SELECTIONS)
@pytest.mark.parametrize("masked", [True, False], ids=["masked", "unmasked"])
def test_ngtdm_map_selection_matches_independent_neighborhood_definitions(
    selected: tuple[str, ...], masked: bool,
) -> None:
    # Mixed gray levels at the image edge, a constant component and an isolated
    # center exercise finite, infinite and undefined results in one small image.
    image = np.zeros((5, 7), dtype=np.float64)
    image[:2, :2] = ((0, 1), (3, 2))
    image[3:, :2] = 3
    image[4, 6] = 1
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:2, :2] = 1
    mask[3:, :2] = 1
    mask[4, 6] = 1
    names = tuple(f"ngtdm_{name}" for name in selected)
    result = extract(
        image, mask,
        settings=settings_for("pyradiomics",
            bin_width=1, features=names, masked_kernel=masked),
        mode="voxel",
    )
    assert set(result.features) == {f"original_{name}" for name in names}
    centers = np.argwhere(mask > 0)
    sampling_mask = mask.astype(bool) if masked else np.ones(image.shape, dtype=bool)
    # Bin width one at anchor zero maps source 0,1,2,3 to levels 1,2,3,4.
    bins = image.astype(np.int32) + 1
    expected: dict[str, list[float]] = {name: [] for name in selected}
    for center in centers:
        window = tuple(slice(max(0, c - 1), min(size, c + 2))
                       for c, size in zip(center, image.shape, strict=True))
        local = ngtdm_features(bins[window], sampling_mask[window])
        for name in selected:
            expected[name].append(local[name])
    for name in selected:
        actual = result.features[f"original_ngtdm_{name}"]
        assert isinstance(actual, np.ndarray)
        values = actual[tuple(centers.T)]
        reference = np.asarray(expected[name])
        for singular in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(singular(values), singular(reference), err_msg=name)
        finite = np.isfinite(reference)
        np.testing.assert_allclose(values[finite], reference[finite], rtol=2e-5, atol=2e-6,
                                   err_msg=name)
        np.testing.assert_array_equal(actual[mask == 0], 0, err_msg=name)
