"""Synthetic large-index voxel maps retain local definitions and bounded scheduling."""

from collections import Counter
from dataclasses import replace
from math import fsum

import numpy as np
import pytest
from radiomics_reference.catalogue import glcm, glcm_matrix

from mlxradiomics import Settings, extract
from mlxradiomics.config.settings import DirectionalAggregation


def test_sparse_contrast_maps_match_independent_neighborhoods_across_budgets() -> None:
    # Fractional MRI-scale intensities produce four occupied levels in a 25001-level
    # lattice. Contrast needs occupied pairs; Idmn additionally needs the full Ng scalar.
    levels = np.array([16001, 16002, 16003, 25001], dtype=np.int32)
    bins = levels[np.indices((5, 5, 5)).sum(axis=0) % 4]
    image = (bins - 0.5) * 0.05
    mask = np.ones(image.shape, dtype=np.bool_)
    mask[1, 2:4, 2] = False
    settings = Settings(features=("glcm_Contrast", "glcm_Idmn"),
                        bin_width=0.05, bin_anchor=0, scratch_mib=1)
    low = extract(image, mask, settings=settings, mode="voxel")
    high = extract(image, mask, settings=replace(settings, scratch_mib=64), mode="voxel")
    assert low.metadata["gray_levels"] == high.metadata["gray_levels"] == 25001
    low_batch, high_batch = low.metadata.get("batch_size", 0), high.metadata.get("batch_size", 0)
    assert low_batch < np.count_nonzero(mask) < high_batch
    for name in settings.features:
        np.testing.assert_array_equal(low.features[f"original_{name}"],
                                      high.features[f"original_{name}"])
    expected: dict[str, list[float]] = {name: [] for name in settings.features}
    for center in np.argwhere(mask):
        window = tuple(slice(max(0, int(v) - 1), min(size, int(v) + 2))
                       for v, size in zip(center, image.shape, strict=True))
        # The separate scalar oracle enumerates spatial neighbors and keeps integer
        # coordinate counts; it does not use the runtime's gather/count/reduction paths.
        local = glcm(bins[window], mask[window], settings, 25001)
        for name in expected:
            expected[name].append(local[name.removeprefix("glcm_")])
    for name, values in expected.items():
        actual = np.asarray(low.features[f"original_{name}"])
        np.testing.assert_allclose(actual[mask], values, rtol=2e-6, atol=2e-6, err_msg=name)
        np.testing.assert_array_equal(actual[~mask], 0)


def test_sparse_cluster_shade_maps_preserve_budget_invariance() -> None:
    bins = np.array([16001, 16002, 16003, 16004, 16005, 16006, 24001, 25001],
                    dtype=np.int32).reshape(2, 2, 2)
    image = (bins - 0.5) * 0.05
    mask = np.ones(image.shape, dtype=np.bool_)
    settings = Settings(features=("glcm_Contrast", "glcm_ClusterShade", "glcm_Idmn"),
                        bin_width=0.05, bin_anchor=0)
    # Every center's radius-one neighborhood contains this whole cube. Directional
    # averaging is checked separately from merged-matrix aggregation below.
    expected = glcm(bins, mask, settings, 25001)
    default = extract(image, mask, settings=settings, mode="voxel")
    larger = extract(image, mask, settings=replace(settings, scratch_mib=128), mode="voxel")
    assert default.metadata.get("batch_size", 0) < larger.metadata.get("batch_size", 0)
    for name in settings.features:
        actual = default.features[f"original_{name}"]
        np.testing.assert_array_equal(actual, larger.features[f"original_{name}"])
        np.testing.assert_allclose(actual, expected[name.removeprefix("glcm_")],
                                   rtol=2e-6, atol=2e-6, err_msg=name)
    with pytest.raises(ValueError, match="scratch_mib cannot hold one local histogram"):
        extract(image, mask, settings=replace(settings, scratch_mib=1), mode="voxel")


@pytest.mark.parametrize("mode", ["3d_merge", "2d_slice_merge"])
def test_sparse_explicit_voxel_aggregation_retains_other_family_budget(
    mode: DirectionalAggregation,
) -> None:
    bins = np.array([16001, 16002, 16003, 16004, 16005, 16006, 24001, 25001],
                    dtype=np.int32).reshape(2, 2, 2)
    image = (bins - 0.5) * 0.05
    mask = np.ones(image.shape, dtype=np.bool_)
    settings = Settings(
        features=("glcm_Contrast", "glcm_ClusterShade", "glcm_Idmn",
                  "gldm_SmallDependenceEmphasis"),
        glcm_aggregation=mode, slice_axis=0,
        bin_width=0.05, bin_anchor=0,
    )
    # Every distinct voxel pair in a 2x2(x2) domain is a distance-one neighbor.
    # Merging therefore gives each ordered pair one count. Slice merging computes
    # the two plane distributions independently and then takes their arithmetic mean.
    domains = [bins.reshape(-1)] if mode == "3d_merge" else [plane.reshape(-1) for plane in bins]
    references = [
        glcm_matrix(Counter((int(i), int(j)) for i in domain for j in domain if i != j), 25001)
        for domain in domains
    ]
    result = extract(image, mask, settings=settings, mode="voxel")
    assert result.metadata["gray_levels"] == 25001
    for name in settings.features[:3]:
        feature = name.removeprefix("glcm_")
        expected = fsum(reference[feature] for reference in references) / len(references)
        np.testing.assert_allclose(result.features[f"original_{name}"], expected,
                                   rtol=2e-6, atol=2e-6, err_msg=name)
    # All gray levels are distinct, so the only equal-level dependence is the voxel itself.
    np.testing.assert_array_equal(result.features["original_gldm_SmallDependenceEmphasis"], 1)
    cheap_glcm = replace(settings, features=("glcm_Contrast",), scratch_mib=1)
    cheap = extract(image, mask, settings=cheap_glcm, mode="voxel")
    np.testing.assert_array_equal(cheap.features["original_glcm_Contrast"],
                                  result.features["original_glcm_Contrast"])
    # Sparse GLCM must not erase the separate selected family's dense workspace.
    with pytest.raises(ValueError, match="scratch_mib cannot hold one local histogram"):
        extract(image, mask, settings=replace(cheap_glcm, features=(
            "glcm_Contrast", "gldm_SmallDependenceEmphasis",
        )), mode="voxel")
