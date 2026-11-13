"""Numerical input-domain regressions using HCC-TACE CT and liver voxels."""

import numpy as np
import pytest
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings, extract

SPACING = (0.859375, 0.859375, 2.5)


def test_ct_resegmentation_preserves_source_statistics() -> None:
    """Resegmentation changes the intensity population without changing the liver ROI."""
    image, mask, _ = hcc_crop(np.s_[105:110, 215:221, 62:69])
    selected = image[(mask != 0) & (image >= 50) & (image <= 100)]
    assert 0 < selected.size < np.count_nonzero(mask)
    expected = {
        "firstorder_Mean": float(selected.mean()),
        "firstorder_Variance": float(selected.var()),
        "firstorder_Minimum": float(selected.min()),
        "firstorder_Maximum": float(selected.max()),
        "shape_VoxelVolume": float(np.count_nonzero(mask) * np.prod(SPACING)),
    }
    result = extract(image, mask, spacing=SPACING, settings=Settings(
        features=tuple(expected), bin_width=10,
        bin_anchor=-200, resegment_range=(50.0, 100.0),
    ))
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)


def test_fractional_ct_weights_have_resolved_morphology() -> None:
    """Closely spaced real-image weights preserve intensity and centroid measurements."""
    ct, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62:64])
    assert np.all(mask == 1)
    image = 2.0 + (ct - ct.min()) * 1e-5
    mean = float(image.mean())
    coordinates = np.indices(image.shape, dtype=np.float64).reshape(3, -1)
    coordinates *= np.asarray(SPACING, dtype=np.float64)[:, None]
    center = (coordinates * image.ravel()).sum(axis=1) / image.sum()
    geometric_center = np.asarray(SPACING, dtype=np.float64) / 2
    expected = {
        "firstorder_Mean": mean,
        "shape_IntegratedIntensity": (17 / 3) * float(np.prod(SPACING)) * mean,
        "shape_CentreOfMassShift": float(np.linalg.norm(center - geometric_center)),
    }
    result = extract(image, mask, spacing=SPACING, settings=Settings(
        features=tuple(expected), bin_width=0.25,
        bin_anchor=0, resegment_range=(0.0,),
    ))
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)


def test_ct_preserves_high_gray_level_indices() -> None:
    """Fine bins preserve actual level indices for real-image histogram and texture values."""
    ct, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62:64])
    assert np.all(mask == 1)
    levels = ct.astype(np.int64) + 20001
    image = (levels - 0.5) * 0.05
    flat = levels.ravel().astype(np.float64)
    first, second = np.meshgrid(flat, flat, indexing="ij")
    distinct = ~np.eye(flat.size, dtype=bool)
    mean = float(flat.mean())
    expected = {
        "intensityhistogram_Minimum": int(flat.min()),
        "intensityhistogram_Maximum": int(flat.max()),
        "intensityhistogram_Mean": mean,
        "intensityhistogram_Variance": float(flat.var()),
        "glcm_JointAverage": mean,
        "glcm_Contrast": float(np.mean((first[distinct] - second[distinct]) ** 2)),
        "glcm_ClusterShade": float(np.mean((first[distinct] + second[distinct] - 2 * mean) ** 3)),
        "glcm_Idmn": float(np.mean(1 / (1 + (first[distinct] - second[distinct]) ** 2
                                                / flat.max() ** 2))),
    }
    # Every distinct voxel pair in a 2x2x2 ROI is a 3D Chebyshev-distance-one pair.
    result = extract(image, mask, spacing=SPACING, settings=Settings(
        features=tuple(expected), bin_width=0.05,
        bin_anchor=0, resegment_range=(0.0,), glcm_aggregation="3d_merge",
    ))
    assert result.metadata["gray_levels"] == int(flat.max())
    assert result.features["original_intensityhistogram_Minimum"] == int(flat.min())
    assert result.features["original_intensityhistogram_Maximum"] == int(flat.max())
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)
