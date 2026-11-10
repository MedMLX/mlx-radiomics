"""ROI windowing preserves source validation, global statistics and metadata."""

from dataclasses import asdict, replace
from typing import cast

import numpy as np
import pytest

from mlxradiomics import Settings, extract
from mlxradiomics.workflows.engine import extract_with_shape_mask


@pytest.mark.parametrize("dtype", ["<i8", ">i8", "<u8", ">u8"])
def test_roi_window_checks_integer_admission_over_the_complete_source(dtype: str) -> None:
    image = np.zeros((16, 17, 18), dtype=dtype)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[7:9, 8:10, 9:11] = 1
    signs = (1, -1) if image.dtype.kind == "i" else (1,)
    for sign in signs:
        # Exact large integers outside the window remain admissible, in either endian.
        image[0, 0, 0] = sign * 2**60
        accepted = extract(image, mask, settings=Settings(features=("firstorder_Mean",)))
        assert accepted.features["original_firstorder_Mean"] == 0.0
        image[0, 0, 0] = sign * (2**53 + 1)
        with pytest.raises(ValueError, match="cannot represent exactly"):
            extract(image, mask, settings=Settings(features=("firstorder_Mean",)))


def test_roi_window_keeps_whole_image_normalization_statistics() -> None:
    image = np.arange(8 * 9 * 10, dtype=np.float64).reshape(8, 9, 10) + 1000.0
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[2:4, 3:5, 4:6] = 1
    normalized = (image - np.mean(image)) / np.std(image, ddof=1)
    expected = float(np.mean(normalized[mask == 1]))
    result = extract(
        image, mask,
        settings=Settings(features=("firstorder_Mean",), normalize=True),
    )
    assert result.features["original_firstorder_Mean"] == pytest.approx(expected, rel=1e-6)


def test_explicit_shape_mask_does_not_make_a_missing_label_extract() -> None:
    image = np.zeros((16, 17, 18), dtype=np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    shape_mask = np.zeros(image.shape, dtype=np.bool_)
    shape_mask[4:6, 5:7, 6:8] = True
    # The union would plan a window, but shape morphology never substitutes for the
    # selected intensity label.
    with pytest.raises(ValueError, match="selected label is absent from mask"):
        extract_with_shape_mask(
            image, mask, settings=Settings(features=("firstorder_Mean",)),
            shape_mask=shape_mask,
        )


def test_mixed_feature_crop_keeps_source_metadata_and_explicit_shape_roi() -> None:
    image = (np.arange(20 * 21 * 22).reshape(20, 21, 22) % 37 - 18).astype(np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[9:12, 10:13, 11:14] = 1
    shape_mask = np.zeros(image.shape, dtype=np.bool_)
    shape_mask[8:14, 9:15, 10:16] = True
    settings = Settings(features=(
        "firstorder_Mean", "glcm_Contrast", "gldzm_ZonePercentage",
        "shape_VoxelVolume",
    ))
    spacing = (1.0, 1.5, 2.0)
    full = extract_with_shape_mask(
        image, mask, settings=settings, spacing=spacing, shape_mask=shape_mask,
    )
    # VoxelVolume is the morphology-mask count times the physical voxel volume, so it
    # detects a window that drops explicit shape-mask voxels outside the intensity ROI.
    assert full.features["original_shape_VoxelVolume"] == pytest.approx(
        float(np.count_nonzero(shape_mask)) * 3.0,
    )
    assert full.features["original_firstorder_Mean"] == pytest.approx(
        float(np.mean(image[mask == 1])), rel=1e-6,
    )
    # A window larger than the feature support must produce the same mixed selection.
    crop = (slice(5, 18), slice(6, 19), slice(7, 20))
    cropped = extract_with_shape_mask(
        image[crop], mask[crop], settings=settings, spacing=spacing,
        shape_mask=shape_mask[crop],
    )
    keys = [f"original_{name}" for name in settings.features]
    np.testing.assert_allclose(
        [cast(float, full.features[key]) for key in keys],
        [cast(float, cropped.features[key]) for key in keys],
        rtol=1e-6, atol=1e-6,
    )
    assert full.metadata["array_shape"] == image.shape
    assert full.metadata["array_dtype"] == str(image.dtype)
    assert full.metadata["roi_voxels"] == int(np.count_nonzero(mask == 1))
    assert full.metadata["excluded_voxels"] == image.size - int(np.count_nonzero(mask == 1))


def test_texture_window_still_rejects_unrepresentable_background_integers() -> None:
    image = np.zeros((16, 17, 18), dtype=np.uint64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[7:9, 8:10, 9:11] = 1
    image[0, 0, 0] = 2**53 + 1
    with pytest.raises(ValueError, match="cannot represent exactly"):
        extract(image, mask, settings=Settings(features=("glcm_Contrast",)))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("resegment_shape", [False, True])
def test_public_mixed_roi_preserves_features_validity_and_source_metadata(
    normalize: bool, resegment_shape: bool,
) -> None:
    image = (np.arange(20 * 21 * 22).reshape(20, 21, 22) % 37 - 18).astype(np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    # Touch the source boundary so texture/local-intensity support is truncated
    # at the image edge, independently of any preparation window.
    mask[:3, 8:12, 9:13] = 1
    settings = Settings(
        features=("shape_VoxelVolume", "firstorder_Mean", "glcm_Contrast"),
        normalize=normalize, resegment_shape=resegment_shape,
        resegment_range=(-0.5, 1.0) if normalize else (-10.0, 12.0),
    )
    spacing = (0.8, 1.5, 2.0)
    mixed = extract(image, mask, settings=settings, spacing=spacing)
    singles = [extract(image, mask, settings=replace(settings, features=(name,)), spacing=spacing)
               for name in settings.features]
    assert mixed.features == {
        key: value for item in singles for key, value in item.features.items()
    }
    assert mixed.metadata["feature_validity"] == {
        key: value for item in singles for key, value in item.metadata["feature_validity"].items()
    }
    assert mixed.metadata["settings"] == asdict(settings)
    assert mixed.metadata["array_shape"] == image.shape
    assert mixed.metadata["array_dtype"] == str(image.dtype)
    assert mixed.metadata["spacing"] == spacing
    assert mixed.metadata["roi_voxels"] == singles[-1].metadata["roi_voxels"]
    assert mixed.metadata["excluded_voxels"] == singles[-1].metadata["excluded_voxels"]
    assert mixed.metadata.get("gray_levels_by_image") == {
        "original": singles[-1].metadata["gray_levels"],
    }
