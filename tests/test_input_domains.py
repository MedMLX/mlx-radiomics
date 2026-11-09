"""Synthetic CT/PET/MRI-scale numerical regressions, not patient validation.

Small analytic images exercise the input domains that tiny integer phantoms miss:
CT resegmentation, fractional PET weights and fine-bin MRI texture. Expectations
are derived from the stated geometries and source values, never from runtime output.
"""

from math import sqrt

import numpy as np
import pytest

from mlxradiomics import Settings, extract


def test_ct_resegmentation_preserves_source_statistics() -> None:
    """CT clipping changes the intensity population without changing the shape ROI."""
    image = np.array([-1000, -100, -50, 0, 25, 50, 100, 1200], dtype=np.float64).reshape(2, 2, 2)
    mask = np.ones(image.shape, dtype=np.uint8)
    expected = {
        "firstorder_Mean": 25 / 6,
        "firstorder_Variance": 153125 / 36,
        "firstorder_Minimum": -100,
        "firstorder_Maximum": 100,
        # Eight morphology voxels at 2 * 0.75 * 0.5 mm³, despite two clipped intensities.
        "shape_VoxelVolume": 6,
    }
    result = extract(image, mask, spacing=(2.0, 0.75, 0.5), settings=Settings(
        features=tuple(expected), bin_width=10,
        bin_anchor=-200, resegment_range=(-200.0, 200.0),
    ))
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)


def test_pet_fractional_weights_have_resolved_morphology() -> None:
    """Near-equal finite PET-scale weights still define accurately resolvable outputs."""
    delta = 1e-8
    image = np.array([2, 2 + delta, 2.5, 2.5 + delta,
                      3, 3 + delta, 3.5, 3.5 + delta]).reshape(2, 2, 2)
    mask = np.ones(image.shape, dtype=np.uint8)
    expected = {
        "firstorder_Mean": 2.75 + delta / 2,
        # The marching-cubes mesh volume of this unit-spaced cube is 17/3 mm³.
        "shape_IntegratedIntensity": (17 / 3) * (2.75 + delta / 2),
        # Weighted centroid displacement is (2, 1, 2*delta)/(22 + 4*delta).
        "shape_CentreOfMassShift": sqrt(5 + 4 * delta**2) / (22 + 4 * delta),
    }
    result = extract(image, mask, spacing=(1.0, 1.0, 1.0), settings=Settings(
        features=tuple(expected), bin_width=0.25,
        bin_anchor=0, resegment_range=(0.0,),
    ))
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)


def test_mri_preserves_high_gray_level_indices() -> None:
    """Fine MRI bins preserve actual level indices in histograms and texture features."""
    levels = np.array([16001, 16002, 16003, 16004, 16005, 16006, 24001, 25001])
    # Bin centers avoid boundary ambiguity. The intensity range is about 800–1250.
    image = ((levels - 0.5) * 0.05).reshape(2, 2, 2)
    mask = np.ones(image.shape, dtype=np.uint8)
    expected = {
        "intensityhistogram_Minimum": 16001,
        "intensityhistogram_Maximum": 25001,
        "intensityhistogram_Mean": 145023 / 8,
        "intensityhistogram_Variance": 870490215 / 64,
        "glcm_JointAverage": 145023 / 8,
        "glcm_Contrast": 124355745 / 4,
        "glcm_ClusterShade": 2147415036525 / 32,
        "glcm_Idmn": 0.9554796923530556,
    }
    # A 2x2x2 cube makes every distinct voxel pair a Chebyshev-distance-one pair.
    # The merged symmetric GLCM therefore gives each of the 56 ordered pairs 1/56
    # probability. The exact rational moments above follow from those eight levels;
    # Idmn uses the full maximum level 25001, not eight compacted occupied levels.
    result = extract(image, mask, spacing=(1.0, 1.0, 1.0), settings=Settings(
        features=tuple(expected), bin_width=0.05,
        bin_anchor=0, resegment_range=(0.0,), glcm_aggregation="3d_merge",
    ))
    assert result.metadata["gray_levels"] == 25001
    assert result.features["original_intensityhistogram_Minimum"] == 16001
    assert result.features["original_intensityhistogram_Maximum"] == 25001
    for name, target in expected.items():
        assert result.features[f"original_{name}"] == pytest.approx(target, rel=2e-6, abs=2e-6)
