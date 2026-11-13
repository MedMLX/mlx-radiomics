"""Executable acceptance of the intensity and shape features against independent definitions.

The same independent scalar expectations cover ROI, retained maps and streamed
Original maps. Integer HCC-TACE voxels make the fixed-width lattice unambiguous;
the separate discretization suite owns fractional and source-precision edges.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.catalogue import intensity_features, shape_features
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings
from mlxradiomics.config.settings import FEATURES
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract

# These families share one mathematical sampling contract in 2D, 3D and local maps.
# GLDZM, physical local intensity and IVH have their own mask/domain acceptance owners.
INTENSITY = tuple(name for name in FEATURES if name.split("_", 1)[0] in {
    "firstorder", "glcm", "gldm", "ngtdm", "glrlm", "glszm", "intensityhistogram",
})


def _fixture(ndim: int, pattern: str) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Read full-liver, liver-boundary, or constant-background source crops."""
    if pattern == "constant":
        crop = np.s_[0:4, 0:5, 64] if ndim == 2 else np.s_[0:3, 0:4, 60:65]
        image, _, _ = hcc_crop(crop)
        assert np.all(image == -2048)
        return image, np.ones(image.shape, dtype=bool)
    if pattern == "sparse":
        crop = np.s_[103:107, 214:219, 64] if ndim == 2 else np.s_[103:106, 214:218, 60:65]
    else:
        crop = np.s_[105:109, 215:220, 64] if ndim == 2 else np.s_[110:113, 220:224, 55:60]
    image, liver, _ = hcc_crop(crop)
    mask = liver == 1
    assert mask.any() and (not mask.all() if pattern == "sparse" else mask.all())
    return image, mask


def _expected(
    image: NDArray[np.float64], mask: NDArray[np.bool_], settings: Settings,
    spacing: tuple[float, ...], voxel: bool,
) -> dict[str, NDArray[np.float64]]:
    # The explicit HU anchor retains leading empty levels; integer source voxels
    # make the independent fixed-width assignment exact.
    bins = ((image + 2200) // 100 + 1).astype(np.int32)
    sampling = mask if settings.masked_kernel or not voxel else np.ones(mask.shape, dtype=bool)
    ng = int(bins[sampling].max())
    if not voxel:
        expected = intensity_features(image, bins, sampling, settings, spacing, ng)
        return {name: np.asarray(value) for name, value in expected.items()}
    output = {name: np.zeros(image.shape, dtype=np.float64) for name in INTENSITY}
    for index in np.ndindex(image.shape):
        if not mask[index]:
            continue
        window = tuple(
            slice(max(0, i - radius), min(size, i + radius + 1))
            for axis, (i, size) in enumerate(zip(index, image.shape, strict=True))
            for radius in [0 if axis == settings.force2d_axis else settings.kernel_radius]
        )
        values = intensity_features(image[window], bins[window], sampling[window],
                                    settings, spacing, ng)
        for name, value in values.items():
            output[name][index] = value
    return output


# IBSI constant-region conventions differ (for example Coarseness 1e6) and are
# owned by test_ibsi_singular_neighborhood_definitions_are_explicit.
@pytest.mark.parametrize("ndim,pattern,mode,conventions", [
    case for case in product(
        (2, 3), ("dense", "sparse", "constant"), ("roi", "maps", "stream"), ("pyradiomics", "ibsi"),
    ) if case[1:4:2] != ("constant", "ibsi")
])
def test_intensity_catalogue(
    ndim: int, pattern: str, mode: str, conventions: Conventions,
) -> None:
    image, mask = _fixture(ndim, pattern)
    spacing = (0.7, 1.3) if ndim == 2 else (1.7, 0.7, 1.3)
    settings = settings_for(
        conventions,
        features=INTENSITY,
        bin_width=100,
        bin_anchor=-2200,
        voxel_array_shift=2.5,
        symmetrical_glcm=pattern != 'sparse' or conventions == 'ibsi',
        gldm_alpha=1,
        distances=(1, 2) if pattern == 'sparse' and conventions == 'pyradiomics' else (1,),
        masked_kernel=pattern != 'sparse',
        scratch_mib=1 if mode == 'stream' else 64,
    )
    expected = _expected(image, mask, settings, spacing, mode != "roi")
    if mode == "roi":
        # The independent oracle sees the original sample. Extra background must not
        # change a mask-based feature, including when the ROI touches an image edge.
        padding = [(0, 4)] + [(3, 2)] * (ndim - 1)
        image = np.pad(image, padding, mode="edge")
        mask = np.pad(mask, padding, constant_values=False)
    names = tuple(f"original_{name}" for name in INTENSITY)
    if mode == "stream":
        with voxel_map_store(image.shape, names) as store:
            result = extract(image, mask.astype(np.uint8), settings=settings,
                                    spacing=spacing, mode="voxel", sink=store.append)
            actual = {name: values.copy() for name, values in store.maps()}
        assert not result.features
    else:
        result = extract_original(image, mask.astype(np.uint8), settings=settings,
                         spacing=spacing, voxel=mode == "maps")
        actual = {name: np.asarray(value) for name, value in result.features.items()}
    # A newly added output must acquire an independent reference before this gate
    # can pass; the check is coupled to numerical execution, not a copied inventory.
    assert set(actual) == {f"original_{name}" for name in expected} == set(names)
    for name, target in expected.items():
        np.testing.assert_allclose(actual[f"original_{name}"], target, rtol=2e-5, atol=2e-6,
                                   equal_nan=True, err_msg=name)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("mode", ["roi", "maps", "stream"])
def test_force2d_catalogue_axis_permutations(axis: int, mode: str) -> None:
    image, mask = _fixture(2, "sparse")
    spacing = [0.7, 1.3]
    spacing.insert(axis, 2.0)
    volume, volume_mask = np.expand_dims(image, axis), np.expand_dims(mask, axis)
    settings = Settings(features=INTENSITY, bin_width=100, bin_anchor=-2200, force2d_axis=axis)
    expected = _expected(image, mask, replace(settings, force2d_axis=None), (0.7, 1.3),
                         mode != "roi")
    if mode == "stream":
        names = tuple(f"original_{name}" for name in INTENSITY)
        with voxel_map_store(volume.shape, names) as store:
            extract(volume, volume_mask.astype(np.uint8), settings=settings,
                           spacing=tuple(spacing), mode="voxel", sink=store.append)
            actual = {name: np.asarray(values.copy()) for name, values in store.maps()}
    else:
        result = extract_original(volume, volume_mask.astype(np.uint8), settings=settings,
                         spacing=tuple(spacing), voxel=mode == "maps")
        actual = {name: np.asarray(value) for name, value in result.features.items()}
    # TotalEnergy includes the actual voxel volume; a singleton axis is still physical.
    expected["firstorder_TotalEnergy"] *= 2
    for name, target in expected.items():
        value = actual[f"original_{name}"]
        if mode != "roi":
            value = np.squeeze(value, axis=axis)
        np.testing.assert_allclose(value, target, rtol=2e-5, atol=2e-6,
                                   equal_nan=True, err_msg=name)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("ndim,permutation", [(2, (0, 1)), (2, (1, 0)),
                                             (3, (0, 1, 2)), (3, (2, 0, 1))])
def test_shape_catalogue(
    ndim: int, permutation: tuple[int, ...], conventions: Conventions,
) -> None:
    if ndim == 2:
        # A wholly included liver crop retains the analytic rectangle oracle.
        image, mask = _fixture(2, "dense")
    else:
        image, liver, _ = hcc_crop(np.s_[103:108, 214:220, 60:67])
        mask = liver == 1
    spacing = (0.7, 1.3) if ndim == 2 else (1.7, 0.7, 1.3)
    mask = np.transpose(mask, permutation)
    image = np.transpose(image, permutation)
    spacing = tuple(spacing[axis] for axis in permutation)
    expected = shape_features(
        mask, spacing, settings_for(conventions).covariance_divisor,
    )
    # This independent marching-surface oracle owns the original geometric quantities;
    # the extended morphology tests own hull, ellipsoid and intensity-dependent outputs.
    selected = tuple(expected)
    actual = extract_original(image, mask.astype(np.uint8), spacing=spacing,
                     settings=settings_for(conventions, features=selected)).features
    assert set(expected) == set(selected)
    for name, target in expected.items():
        np.testing.assert_allclose(actual[f"original_{name}"], target, rtol=3e-6, atol=2e-6,
                                   err_msg=name)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_force2d_shape_catalogue(axis: int) -> None:
    image, mask = _fixture(2, "dense")
    expected = shape_features(mask, (0.7, 1.3))
    volume = np.expand_dims(mask, axis)
    spacing = [0.7, 1.3]
    spacing.insert(axis, 2.0)
    result = extract_original(
        np.expand_dims(image, axis), volume.astype(np.uint8), spacing=tuple(spacing),
                     settings=settings_for("pyradiomics",
                         features=tuple(expected), force2d_axis=axis))
    for name, target in expected.items():
        np.testing.assert_allclose(result.features[f"original_{name}"], target,
                                   rtol=3e-6, atol=2e-6, err_msg=name)
