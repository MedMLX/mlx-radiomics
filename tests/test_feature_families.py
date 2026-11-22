"""Independent feature-family mathematics at the public extraction boundary.

Expected values come from explicit scalar counts or independent geometry. No expected
value comes from a recorded result or from production feature arithmetic.
"""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.catalogue import dependence_zones, firstorder, glcm
from radiomics_reference.mathematics import glrlm_features, ngtdm_features
from radiomics_reference.real_images import hcc_crop, hcc_patch, hcc_resampled

from mlxradiomics import Settings, extract
from mlxradiomics.voxel.storage import voxel_map_store


def _three_level_image() -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    source = hcc_patch((3, 3))[0]
    image = np.minimum(2, np.floor((source - source.min()) * 3 / np.ptp(source))) * 25
    return image, np.ones(image.shape, dtype=np.uint8)


def test_glszm_matches_independently_counted_ct_zones() -> None:
    image, mask = _three_level_image()
    settings = Settings(features=("glszm_SmallAreaEmphasis", "glszm_LargeAreaEmphasis",
        "glszm_GrayLevelVariance", "glszm_SizeZoneNonUniformity", "glszm_ZoneVariance",
        "glszm_ZoneEntropy", "glszm_ZonePercentage", "glszm_LowGrayLevelZoneEmphasis",
        "glszm_HighGrayLevelZoneEmphasis"), bin_width=25.0)
    bins = (image / 25).astype(np.int32) + 1
    expected = dependence_zones(bins, mask.astype(bool), settings)
    result = extract(image, mask, settings=settings)
    assert set(result.features) == {f"original_{name}" for name in settings.features}
    for name in settings.features:
        np.testing.assert_allclose(result.features[f"original_{name}"], expected[name],
                                   rtol=1e-6, atol=1e-6, err_msg=name)


def test_gray_level_256_preserves_ct_counts() -> None:
    image = hcc_patch((16, 16))[0]
    mask = np.ones(image.shape, dtype=np.uint8)
    names = ("firstorder_Entropy", "firstorder_Uniformity", "glrlm_RunEntropy",
        "glrlm_RunPercentage", "glszm_ZoneEntropy", "glszm_ZonePercentage",
        "gldm_DependenceEntropy", "gldm_DependenceNonUniformity")
    settings = Settings(bin_count=256, features=names)
    bins = np.minimum(256, 1 + np.floor(
        (image - image.min()) * 256 / np.ptp(image),
    )).astype(np.int32)
    assert bins.max() == 256
    expected = {
        **{f"firstorder_{k}": v for k, v in firstorder(image, bins, mask.astype(bool),
            0.0, 1.0, settings.kurtosis, settings.zero_variance_moments).items()},
        **{f"glrlm_{k}": v for k, v in glrlm_features(bins, mask.astype(bool)).items()},
        **dependence_zones(bins, mask.astype(bool), settings),
    }
    result = extract(image, mask, settings=settings)
    for name in names:
        np.testing.assert_allclose(result.features[f"original_{name}"], expected[name],
                                   rtol=1e-6, atol=1e-6, err_msg=name)


def test_ngtdm_large_roi_preserves_fractional_neighbor_sums() -> None:
    source = hcc_crop(np.s_[40:424, 40:424, 64])[0]
    image = np.minimum(2, np.floor((source - source.min()) * 3 / np.ptp(source)))
    mask = np.ones(image.shape, dtype=np.uint8)
    # Two-column strips give five-neighbor interiors and repeated non-binary
    # fractions. Atomic float32 bin accumulation loses their low-order bits.
    mask[:, 2::3] = 0
    settings = Settings(bin_width=1, features=("ngtdm_Complexity",))
    actual = extract(image, mask, settings=settings)
    # Width-one bins for the integer source levels 0..2 are exactly 1..3.
    bins = image.astype(np.int32) + 1
    expected = ngtdm_features(bins, mask.astype(bool), settings.distances)["Complexity"]
    value = actual.features["original_ngtdm_Complexity"]
    assert isinstance(value, float)
    np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "ambiguous", [False, True], ids=["disconnected-anisotropic", "cropped-ambiguous-cells"],
)
def test_mesh_volume_preserves_scale_and_closed_geometry(ambiguous: bool) -> None:
    image = hcc_resampled((8, 11, 14) if ambiguous else (53, 53, 53))
    mask = np.zeros(image.shape, dtype=np.uint8)
    if ambiguous:
        mask[2:6, 3:8, 4:10] = np.random.default_rng(1).random((4, 5, 6)) > 0.35
    else:
        mask[1:4, 1:4, 1:4] = 1
        mask[48:51, 48:51, 48:51] = 1
    spacing = (3.0, 0.977, 0.977)
    actual = extract(image, mask, settings=Settings(features=("shape_MeshVolume",)),
                     spacing=spacing)
    # A complete classic marching-cubes implementation supplies the independent
    # geometry. Require a closed oriented surface before integrating its volume;
    # a complement shortcut opens ambiguous faces.
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        np.pad(mask, 1), level=0.5, spacing=spacing, method="lorensen",
    )
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered_edges = np.sort(edges, axis=1)
    _, counts = np.unique(ordered_edges, axis=0, return_counts=True)
    np.testing.assert_array_equal(counts, 2)
    directed = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    _, inverse = np.unique(ordered_edges, axis=0, return_inverse=True)
    np.testing.assert_array_equal(np.bincount(inverse, weights=directed), 0)
    triangles = vertices[faces].astype(np.float64)
    expected = abs(np.einsum(
        "ij,ij->", np.cross(triangles[:, 0], triangles[:, 1]), triangles[:, 2],
    )) / 6
    np.testing.assert_allclose(actual.features["original_shape_MeshVolume"],
                               cast(float | np.ndarray, expected), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit-mcc"])
@pytest.mark.parametrize("stream", [False, True], ids=["maps", "streamed"])
def test_sparse_voxel_mcc_preserves_defined_centers(explicit: bool, stream: bool) -> None:
    # The solid cube has every direction, the isolated center has none, and the
    # disconnected line has only one. Each center uses its own defined directions.
    image = hcc_patch((8, 8, 8))[0]
    dense = np.zeros(image.shape, dtype=np.uint8)
    dense[1:4, 1:4, 1:4] = 1
    mask = dense.copy()
    mask[6, 6, 6] = 1
    mask[6, 1, 1:3] = 1
    anchor = float(image[mask == 1].min())
    # Four-HU bins resolve the line's two source levels within the 1 MiB budget.
    settings = Settings(scratch_mib=1, bin_anchor=anchor, bin_width=4)
    bins = (1 + np.floor((image - anchor) / 4)).astype(np.int32)
    if explicit:
        settings = replace(settings, features=("glcm_MCC",))
    if stream:
        with voxel_map_store(
            image.shape, tuple(f"original_{name}" for name in settings.features)
        ) as store:
            result = extract(image, mask, settings=settings, mode="voxel", sink=store.append)
            assert not result.features
            actual = {name: values.copy() for name, values in store.maps()}
    else:
        result = extract(image, mask, settings=settings, mode="voxel")
        actual = dict(result.features)
    mcc = np.asarray(actual["original_glcm_MCC"])
    for center in np.ndindex(image.shape):
        if not mask[center]:
            continue
        domain = tuple(slice(max(0, i - 1), min(size, i + 2))
                       for i, size in zip(center, image.shape, strict=True))
        expected = glcm(bins[domain], mask[domain].astype(bool), settings, int(bins.max()))["MCC"]
        np.testing.assert_allclose(mcc[center], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert np.isfinite(mcc[dense == 1]).all()
    assert np.isnan(mcc[6, 6, 6])
    np.testing.assert_array_equal(mcc[mask == 0], 0)
    assert result.metadata["roi_voxels"] == 30
    assert result.metadata["feature_validity"]["original_glcm_MCC"] == {
        "finite": 29, "valid_zero": 0, "nan": 1, "positive_infinity": 0,
        "negative_infinity": 0, "excluded": image.size - 30,
    }
    assert set(actual) == {f"original_{name}" for name in settings.features}
    if not explicit:
        assert np.asarray(actual["original_firstorder_Mean"])[6, 6, 6] == image[6, 6, 6]
        assert result.metadata["feature_validity"]["original_firstorder_Mean"]["valid_zero"] == 0
