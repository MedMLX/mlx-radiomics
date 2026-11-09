"""Public ROI precision contracts for admitted integer source calculations."""

import numpy as np
import pytest

from mlxradiomics import Settings, extract


def test_integer_roi_mean_retains_cancellation_and_original_image_metadata() -> None:
    # A plain float32 sum loses the middle 1; an integer shortcut must retain the
    # compensated mean while ignoring nonfinite background outside the ROI window.
    image = np.full((11, 9, 7), np.nan, dtype=np.float32)
    image[4:7, 4, 3] = [2**24, 1, -(2**24)]
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[4:7, 4, 3] = 1
    result = extract(
        image, mask, settings=Settings(features=("firstorder_Mean",), bin_count=16),
    )
    assert result.features == {"original_firstorder_Mean": float(np.float32(1 / 3))}
    assert result.metadata["array_shape"] == (11, 9, 7)
    assert result.metadata["roi_voxels"] == 3
    assert result.metadata["excluded_voxels"] == 690


def test_integer_roi_total_energy_keeps_physical_underflow_error() -> None:
    image = np.ones((3, 1, 1), dtype=np.float32)
    mask = np.ones(image.shape, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="first-order.*TotalEnergy.*underflow"):
        extract(
            image, mask, spacing=(2.0**-45,) * 3,
            settings=Settings(features=("firstorder_TotalEnergy",)),
        )
