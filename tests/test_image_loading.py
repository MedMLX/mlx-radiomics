"""File loading keeps exact source pixels and rejects lossy integer intensities."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import IbsiInterpolation, ImageSettings
from mlxradiomics.imaging.spatial import load_image_pair


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_loader_preserves_exact_source_pixels_and_large_mask_labels(
    tmp_path: Path, dtype: str,
) -> None:
    # Adjacent representable large intensities must keep their difference, while
    # labels need not be representable in float64 at all on an unchanged grid.
    pixels = np.full((3, 4, 5), 2**53, dtype=dtype)
    pixels[1, 2, 3] = 2**53 + 2
    label = 2**63 + 1
    labels = np.zeros(pixels.shape, dtype=np.uint64)
    labels[1:, 1:3, 1:4] = label
    paths = tmp_path / "image.nrrd", tmp_path / "mask.nrrd"
    for path, array in zip(paths, (pixels, labels), strict=True):
        volume = sitk.GetImageFromArray(array)
        volume.SetSpacing((0.5, 1.5, 3.0))
        volume.SetOrigin((11.0, -7.0, 4.0))
        sitk.WriteImage(volume, str(path))

    pair = load_image_pair(*paths, settings=Settings(label=label))

    np.testing.assert_array_equal(pair.image.array, pixels)
    np.testing.assert_array_equal(pair.mask.array, labels)
    assert pair.image.array.dtype == np.dtype(dtype)
    assert pair.mask.array.dtype == np.dtype(np.uint64)
    assert pair.image.spacing == pair.mask.spacing == (3.0, 1.5, 0.5)
    assert pair.image.origin == pair.mask.origin == (11.0, -7.0, 4.0)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("ibsi", [False, True])
def test_loader_rejects_lossy_integer_background_before_spatial_processing(
    tmp_path: Path, dtype: str, ibsi: bool,
) -> None:
    # File loading validates every pixel, including background outside the ROI.
    # IBSI rounding must not conceal a loss before this integer-domain check.
    pixels = np.zeros((3, 4, 5), dtype=dtype)
    pixels[0, 0, 0] = 2**53 + 1
    labels = np.zeros(pixels.shape, dtype=np.uint8)
    labels[1:, 1:3, 1:4] = 1
    paths = tmp_path / "image.nrrd", tmp_path / "mask.nrrd"
    for path, array in zip(paths, (pixels, labels), strict=True):
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    spatial = ImageSettings(
        ibsi_interpolation=IbsiInterpolation(spacing=(1.0, 1.0, 1.0)) if ibsi else None,
    )

    with pytest.raises(ValueError, match="cannot represent exactly"):
        load_image_pair(*paths, image_settings=spatial)
