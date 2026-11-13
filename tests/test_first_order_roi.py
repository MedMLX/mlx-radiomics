"""Public ROI precision contracts for admitted integer source calculations."""

import numpy as np
import pytest
from radiomics_reference.real_images import hcc_background, hcc_crop

from mlxradiomics import Settings, extract


def test_integer_roi_mean_retains_cancellation_and_original_image_metadata() -> None:
    ct, _, _ = hcc_crop(np.s_[100:111, 212:221, 61:68])
    image = ct.astype(np.float32)
    # Stretch two source intensities to opposite extremes while retaining the
    # actual odd integer at the center. A plain float32 sum loses its low bit.
    endpoints = ct[[4, 6], 4, 3]
    image[[4, 6], 4, 3] = ((endpoints - endpoints.mean()) / np.ptp(endpoints) * 2**25)
    assert int(image[5, 4, 3]) % 2 == 1
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[4:7, 4, 3] = 1
    expected = float(np.float32(sum(float(value) for value in image[mask == 1]) / 3))
    image[mask == 0] = np.nan
    result = extract(
        image, mask, settings=Settings(features=("firstorder_Mean",), bin_count=16),
    )
    assert result.features == {"original_firstorder_Mean": expected}
    assert result.metadata["array_shape"] == (11, 9, 7)
    assert result.metadata["roi_voxels"] == 3
    assert result.metadata["excluded_voxels"] == 690


def test_integer_roi_total_energy_keeps_physical_underflow_error() -> None:
    image = (hcc_background((3, 1, 1)) / -2048).astype(np.float32)
    mask = np.ones(image.shape, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="first-order.*TotalEnergy.*underflow"):
        extract(
            image, mask, spacing=(2.0**-45,) * 3,
            settings=Settings(features=("firstorder_TotalEnergy",)),
        )
