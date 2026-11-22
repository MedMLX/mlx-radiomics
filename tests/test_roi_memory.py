"""ROI memory admission preserves feature dependencies and original gray indices."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_patch

from mlxradiomics import Settings, extract


def _pair(maximum: float = 1.0) -> NDArray[np.float64]:
    """Scale the contrast of two CT pixels to the lattice admission boundary."""
    ct, _, _ = hcc_patch((1, 2))
    return (ct - ct.min()) / np.ptp(ct) * maximum


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("feature", [
    "firstorder_Entropy", "intensityhistogram_Mean", "glrlm_ShortRunEmphasis",
    "gldm_SmallDependenceEmphasis", "ngtdm_Coarseness", "glcm_Imc1", "glcm_SumEntropy",
])
def test_tiny_roi_refuses_oversized_selected_count_buffer(
    conventions: Conventions, feature: str,
) -> None:
    image = _pair()
    family = feature.split("_")[0]
    with pytest.raises(MemoryError, match=rf"{family} ROI count buffer requires"):
        extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
            conventions,
            features=(feature,),
            bin_width=2.0 ** (-27),
            bin_anchor=0,
        ))


def test_unused_dense_buffers_do_not_block_raw_or_sparse_features() -> None:
    image = _pair()
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=("firstorder_Mean", "glcm_Contrast", "glcm_DifferenceVariance"),
                        bin_width=2.0**-27, bin_anchor=0, scratch_mib=1)
    result = extract(image, mask, settings=settings)
    assert result.features == {
        "original_firstorder_Mean": 0.5,
        "original_glcm_Contrast": float(2**54),
        "original_glcm_DifferenceVariance": 0.0,
    }


def test_explicit_count_budget_preserves_bins_and_entropy_at_allocation_boundary() -> None:
    image = _pair()
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=("firstorder_Entropy",), bin_count=2**18 + 1, scratch_mib=1)
    with pytest.raises(MemoryError, match="ROI count buffer requires"):
        extract(image, mask, settings=settings)
    result = extract(image, mask, settings=replace(settings, scratch_mib=2))
    # The two occupied bins remain equiprobable; increasing memory never rebins them.
    assert result.features["original_firstorder_Entropy"] == 1.0
    assert result.metadata["gray_levels"] == 2**18 + 1


def test_run_buffer_admission_uses_the_actual_cropped_domain() -> None:
    ct, _, _ = hcc_patch((256, 2))
    image = (ct - ct[0, 0]) * 65536
    assert image[0, 0] == 0 and image[0, 1] == 65536
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0] = 1
    result = extract(image, mask, settings=Settings(
        features=("glrlm_ShortRunEmphasis",), bin_width=1, bin_anchor=0, scratch_mib=1,
    ))
    # The two adjacent, different-gray voxels form only length-one runs. Their
    # cropped count matrix fits 1 MiB, while using the 256-row file shape would not.
    assert result.features["original_glrlm_ShortRunEmphasis"] == 1.0


@pytest.mark.parametrize("feature,maximum,bits", [
    ("gldm_SmallDependenceEmphasis", 2.0 - 2.0**-29, 32),
    ("glcm_SumEntropy", 1.0, 31),
])
def test_larger_memory_budget_cannot_overflow_count_indexes(
    feature: str, maximum: float, bits: int,
) -> None:
    image = _pair(maximum)
    with pytest.raises(
        MemoryError, match=rf"ROI count buffer exceeds {bits}-bit indexing capacity",
    ):
        extract(image, np.ones(image.shape, dtype=np.uint8), settings=Settings(
            features=(feature,), bin_width=2.0**-30, bin_anchor=0, scratch_mib=65536,
        ))
