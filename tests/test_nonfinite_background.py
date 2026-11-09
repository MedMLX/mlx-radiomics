"""Original-image extraction ignores nonfinite voxels outside the ROI."""

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import assert_type

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics import ImageType, Result, Settings, TransformSettings, export, extract
from mlxradiomics.imaging.io import read_image
from mlxradiomics.workflows.engine import extract as extract_original

# Two finite ROI voxels over a background that a whole-image finite check rejects.
IMAGE: NDArray[np.float64] = np.array([[1.0, 2.0], [np.nan, np.nan]])
MASK: NDArray[np.uint8] = np.array([[1, 1], [0, 0]], dtype=np.uint8)
MEAN = "original_firstorder_Mean"


def _settings() -> Settings:
    return Settings(features=("firstorder_Mean",))


def test_original_only_roi_matches_direct_extract_over_nonfinite_background() -> None:
    direct = extract_original(IMAGE, MASK, settings=_settings())
    images = extract(IMAGE, MASK, settings=_settings())
    assert_type(images, Result[float])
    shape = extract(IMAGE, MASK, settings=Settings(features=("shape2D_PixelSurface",)))
    # Two unit-spacing ROI pixels occupy area 2.
    assert shape.features["original_shape2D_PixelSurface"] == pytest.approx(2.0)
    # The selected values 1 and 2 have mean 1.5; background NaNs are outside the sample.
    assert direct.features[MEAN] == pytest.approx(1.5, abs=1e-7)
    assert images.features[MEAN] == pytest.approx(1.5, abs=1e-7)
    assert images.metadata.get("gray_levels_by_image") == {"original": 1}


def test_original_only_retained_streamed_and_exported_maps_agree(tmp_path: Path) -> None:
    expected = np.zeros(IMAGE.shape, dtype=np.float32)
    expected[0, :] = 1.5
    retained = extract(IMAGE, MASK, settings=_settings(), mode="voxel")
    assert_type(retained, Result[NDArray[np.float64]])
    np.testing.assert_array_equal(retained.features[MEAN], expected)

    collected = np.zeros(IMAGE.shape, dtype=np.float32)

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        collected[tuple(centers.T)] = batch[MEAN]

    streamed = extract(
        IMAGE, MASK, settings=_settings(), mode="voxel", sink=consume,
    )
    assert not streamed.features
    np.testing.assert_array_equal(collected, expected)

    exported = export(IMAGE, MASK, tmp_path / "export", settings=_settings())
    assert set(exported.maps) == {MEAN}
    written = read_image(exported.maps[MEAN])
    np.testing.assert_array_equal(written.array, expected)


def test_derived_filter_still_requires_a_finite_full_image() -> None:
    transforms = TransformSettings(image_types=(ImageType.ORIGINAL, ImageType.SQUARE))
    # Square is defined on the whole array, so the finite-ROI exception must not reach it.
    with pytest.raises(ValueError, match="finite"):
        extract(IMAGE, MASK, settings=_settings(), transforms=transforms)


@pytest.mark.parametrize("kind", [ImageType.LBP2D, ImageType.LBP3D])
@pytest.mark.parametrize("background", [np.nan, np.inf, 1e39])
def test_lbp_reports_a_nonfinite_float32_background(kind: ImageType, background: float) -> None:
    # 1e39 is finite in float64 but exceeds float32 max (~3.4e38), so the float32 filter
    # input is infinite. The source-precision split must not misreport any of these as
    # an unresolved LBP2D sign or warn about an overflowing cast first.
    image = np.array([[[1.0, 2.0], [background, 3.0]]] * 3)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:, 0, :] = 1
    transforms = TransformSettings(image_types=(kind,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="feature image intensities must be finite"):
            extract(image, mask, spacing=(1.0, 1.0, 1.0), settings=_settings(),
                    transforms=transforms)
