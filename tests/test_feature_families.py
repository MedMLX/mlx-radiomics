"""Independent feature-family mathematics at the public extraction boundary.

Expected values come from explicit scalar counts or independent geometry. No expected
value comes from a recorded result or from production feature arithmetic.
"""

from dataclasses import replace
from math import log2
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.mathematics import ngtdm_features

from mlxradiomics import Settings, extract
from mlxradiomics.voxel.storage import voxel_map_store

# Zones of the 3x3 bins [[1, 1, 1], [1, 2, 2], [3, 3, 3]] are level 1 size 4,
# level 2 size 2 and level 3 size 3. The statistics below follow from those three
# (gray level, zone size) counts without a production feature calculation.
_GLSZM_EXPECTED: dict[str, float] = {
    "SmallAreaEmphasis": 61 / 432,
    "LargeAreaEmphasis": 29 / 3,
    "GrayLevelVariance": 2 / 3,
    "SizeZoneNonUniformity": 1.0,
    "ZoneVariance": 2 / 3,
    "ZoneEntropy": log2(3),
    "ZonePercentage": 1 / 3,
    "LowGrayLevelZoneEmphasis": 49 / 108,
    "HighGrayLevelZoneEmphasis": 14 / 3,
}


def _three_level_image() -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    bins = np.array([[1, 1, 1], [1, 2, 2], [3, 3, 3]], dtype=np.int32)
    image = (bins - 1).astype(np.float64) * 25.0
    return image, np.ones(image.shape, dtype=np.uint8)


def test_glszm_matches_hand_counted_zones() -> None:
    image, mask = _three_level_image()
    settings = Settings(features=tuple(f"glszm_{name}" for name in _GLSZM_EXPECTED),
                        bin_width=25.0)
    result = extract(image, mask, settings=settings)

    assert set(result.features) == {f"original_glszm_{name}" for name in _GLSZM_EXPECTED}
    for name, expected in _GLSZM_EXPECTED.items():
        value = result.features[f"original_glszm_{name}"]
        assert isinstance(value, float)
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-6, err_msg=name)


def test_gray_level_256_keeps_a_uniform_distribution() -> None:
    # Fixed-count binning maps the 256 distinct values to gray levels 1..256. Every
    # texture cell then holds one voxel or one size-1 zone, so entropy is log2(256).
    image = np.arange(256, dtype=np.float64).reshape(16, 16)
    mask = np.ones(image.shape, dtype=np.uint8)
    expected: dict[str, float] = {
        "firstorder_Entropy": 8.0,
        "firstorder_Uniformity": 1 / 256,
        "glrlm_RunEntropy": 8.0,
        "glrlm_RunPercentage": 1.0,
        "glszm_ZoneEntropy": 8.0,
        "glszm_ZonePercentage": 1.0,
        "gldm_DependenceEntropy": 8.0,
        "gldm_DependenceNonUniformity": 256.0,
    }
    settings = Settings(bin_count=256, features=tuple(expected))
    result = extract(image, mask, settings=settings)

    for name, target in expected.items():
        value = result.features[f"original_{name}"]
        assert isinstance(value, float)
        np.testing.assert_allclose(value, target, rtol=1e-6, atol=1e-6, err_msg=name)


def test_ngtdm_large_roi_preserves_fractional_neighbor_sums() -> None:
    image = (np.indices((384, 384)).sum(axis=0) % 3).astype(np.float64)
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
    image = np.ones((8, 11, 14) if ambiguous else (53, 53, 53))
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
    image = (np.indices((8, 8, 8)).sum(axis=0) % 2).astype(np.float64) * 25
    dense = np.zeros(image.shape, dtype=np.uint8)
    dense[1:4, 1:4, 1:4] = 1
    mask = dense.copy()
    mask[6, 6, 6] = 1
    mask[6, 1, 1:3] = 1
    settings = Settings(scratch_mib=1)
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
    # Odd-parity offsets pair opposite states; even-parity offsets pair equal states.
    # Every defined two-state GLCM is a permutation with second singular value one.
    np.testing.assert_allclose(mcc[dense == 1], 1.0, rtol=1e-6, atol=1e-6)
    assert np.isfinite(mcc[dense == 1]).all()
    np.testing.assert_allclose(mcc[6, 1, 1:3], 1.0, rtol=1e-6, atol=1e-6)
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
        assert result.metadata["feature_validity"]["original_firstorder_Mean"]["valid_zero"] == 1
