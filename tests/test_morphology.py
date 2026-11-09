"""Analytic checks for intensity-dependent IBSI morphological features.

Expected values are derived from the manual's definitions with voxel-centre coordinates
and inverse Euclidean distance weights. Public extraction also checks intensity versus
morphological mask selection after resegmentation.
"""

from itertools import permutations
from math import dist, sqrt

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray

from mlxradiomics import Settings, extract
from mlxradiomics.features import morphology as morphology
from mlxradiomics.features.morphology import morphology_statistics


def _values(
    image: NDArray[np.generic],
    intensity_mask: NDArray[np.bool_],
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, float, float],
    names: tuple[str, ...],
) -> dict[str, float]:
    computed = morphology_statistics(
        image, intensity_mask, morphology_mask, spacing, 8, names,
    )
    return {
        name: float(np.asarray(computed[f"shape_{name}"], dtype=np.float64).item())
        for name in names
    }


def test_centre_of_mass_shift_uses_separate_masks() -> None:
    # Geometric centroid of the eight-voxel cube is (1.5, 1.5, 1.5). The intensity
    # mask selects (0, 0, 0), so the physical shift is (-3, -1.5, -1.5) for spacing
    # (2, 1, 1), with length sqrt(13.5).
    morphology = np.zeros((4, 4, 4), dtype=np.bool_)
    morphology[1:3, 1:3, 1:3] = True
    intensity = np.zeros_like(morphology)
    intensity[0, 0, 0] = True
    image = np.zeros((4, 4, 4), dtype=np.float64)
    image[intensity] = 7.0
    values = _values(image, intensity, morphology, (2.0, 1.0, 1.0),
                     ("CentreOfMassShift",))
    assert values["CentreOfMassShift"] == pytest.approx(sqrt(13.5), rel=2e-6)


def test_constant_intensity_centre_of_mass_shift_is_zero() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    image = np.full(mask.shape, 12.5, dtype=np.float64)
    values = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    assert values["CentreOfMassShift"] == pytest.approx(0, abs=1e-7)


def test_integrated_intensity_uses_mesh_volume_and_intensity_mask() -> None:
    # A 2x2x2 voxel cube at spacing (2,1,1) has grid mesh volume 17/3. Physical volume is 34/3;
    # constant intensity 3 gives integrated intensity 34.
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    image = np.full(mask.shape, 3.0, dtype=np.float64)
    values = _values(image, mask, mask, (2.0, 1.0, 1.0), ("IntegratedIntensity",))
    assert values["IntegratedIntensity"] == pytest.approx(34.0, rel=2e-6)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_public_morphology_resegments_intensities_and_retains_the_shape(
    conventions: Conventions,
) -> None:
    """A resegmented half-cube retains the original mesh and geometric centroid."""
    image = np.zeros((4, 4, 4), dtype=np.float64)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    image[1:3, 1:3, 1:3] = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
    points = [(4.0, 1.0, 1.0), (4.0, 1.0, 2.0),
              (4.0, 2.0, 1.0), (4.0, 2.0, 2.0)]
    centered = (-1.5, -0.5, 0.5, 1.5)
    pairs = list(permutations(range(4), 2))
    weight = sum(1 / dist(points[i], points[j]) for i, j in pairs)
    variance_sum = sum(value * value for value in centered)
    expected = {
        # Requesting geometry with IntegratedIntensity exercises their shared lazy
        # mesh. Resegmentation must still leave both features on the original shape.
        "MeshVolume": 34 / 3,
        "CentreOfMassShift": sqrt(681) / 26,
        "IntegratedIntensity": 221 / 3,
        "MoransI": 4 / weight / variance_sum * sum(
            centered[i] * centered[j] / dist(points[i], points[j]) for i, j in pairs
        ),
        "GearysC": 3 / (2 * weight * variance_sum) * sum(
            (centered[i] - centered[j]) ** 2 / dist(points[i], points[j]) for i, j in pairs
        ),
    }
    result = extract(
        image, labels, spacing=(2.0, 1.0, 1.0),
        settings=settings_for(
            conventions,
            features=tuple(f'shape_{name}' for name in expected),
            resegment_range=(5.0, 8.0),
            scratch_mib=1,
        ),
    )
    for name, target in expected.items():
        assert result.features[f"original_shape_{name}"] == pytest.approx(target, rel=2e-6)


def _all_pairs(
    image: NDArray[np.float64], mask: NDArray[np.bool_], spacing: tuple[float, float, float],
) -> tuple[float, float]:
    """Float64 Moran's I and Geary's C summed over every ordered voxel pair."""
    points = np.argwhere(mask) * np.asarray(spacing)
    deviations = image[mask] - image[mask].mean()
    distance = np.sqrt(((points[:, None] - points[None]) ** 2).sum(axis=2))
    weight = np.divide(1, distance, out=np.zeros_like(distance), where=distance > 0)
    variance_sum = float(deviations @ deviations)
    count = deviations.size
    moran = count / weight.sum() * float(deviations @ weight @ deviations) / variance_sum
    geary = (count - 1) / (2 * weight.sum()) * float(
        (weight * (deviations[:, None] - deviations[None]) ** 2).sum()
    ) / variance_sum
    return moran, geary


@pytest.mark.parametrize(
    "spacing", [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1e-25, 1e25, 1e25)],
)
def test_moran_and_geary_match_three_point_line(
    spacing: tuple[float, float, float],
) -> None:
    # Coordinates 0,1,2 along axis 0; intensities 0,1,2. Mean 1, deviations (-1,0,1).
    # Distances along the line scale with spacing, so both ratios are unchanged.
    mask = np.zeros((3, 1, 1), dtype=np.bool_)
    mask[:, 0, 0] = True
    image = np.zeros(mask.shape, dtype=np.float64)
    image[:, 0, 0] = (0.0, 1.0, 2.0)
    values = _values(image, mask, mask, spacing, ("MoransI", "GearysC"))
    assert values["MoransI"] == pytest.approx(-0.3, rel=2e-6)
    assert values["GearysC"] == pytest.approx(0.8, rel=2e-6)


def test_moran_and_geary_handle_anisotropic_distance_weights() -> None:
    # Three voxels at (0,0,0), (0,1,0), (1,0,0), spacing (2,1,1), intensities
    # (0,1,2). Pair distances are 1, 2, sqrt(5); the expected values follow directly
    # from the weighted definitions with mean 1.
    mask = np.zeros((2, 2, 1), dtype=np.bool_)
    mask[0, 0, 0] = mask[1, 0, 0] = mask[0, 1, 0] = True
    image = np.zeros(mask.shape, dtype=np.float64)
    image[0, 0, 0], image[0, 1, 0], image[1, 0, 0] = 0.0, 1.0, 2.0
    distance_01, distance_02, distance_12 = 1.0, 2.0, sqrt(5)
    weight_sum = 2 * (1 / distance_01 + 1 / distance_02 + 1 / distance_12)
    moran_numerator = 2 * (1 / distance_02) * (-1)
    geary_numerator = 2 * (
        (1 / distance_01) * 1**2 + (1 / distance_02) * 2**2
        + (1 / distance_12) * 1**2
    )
    expected_moran = (3 / weight_sum) * moran_numerator / 2
    expected_geary = ((3 - 1) / (2 * weight_sum)) * geary_numerator / 2
    values = _values(
        image, mask, mask, (2.0, 1.0, 1.0), ("MoransI", "GearysC"),
    )
    assert values["MoransI"] == pytest.approx(expected_moran, rel=2e-6)
    assert values["GearysC"] == pytest.approx(expected_geary, rel=2e-6)


@pytest.mark.parametrize("dense", [True, False])
def test_tiled_spatial_autocorrelation_matches_all_pairs(dense: bool) -> None:
    # 792 voxels fill the box (dense) or about 60% of it with holes (sparse); the pair
    # kernel spans several 256-voxel threadgroup tiles.
    rng = np.random.default_rng(4)
    shape = (9, 11, 8)
    mask = np.ones(shape, dtype=np.bool_) if dense else rng.random(shape) < 0.6
    image = rng.integers(-50, 80, size=shape).astype(np.float64)
    spacing = (2.0, 0.7, 0.9)
    computed = morphology_statistics(image, mask, mask, spacing, 1, ("MoransI", "GearysC"))
    moran, geary = _all_pairs(image, mask, spacing)
    observed = {name: float(np.asarray(value, dtype=np.float64).item())
                for name, value in computed.items()}
    assert observed["shape_MoransI"] == pytest.approx(moran, rel=2e-6)
    assert observed["shape_GearysC"] == pytest.approx(geary, rel=2e-6)


def test_pairwise_autocorrelation_refuses_oversized_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(morphology, "MAX_PAIRWISE_VOXELS", 7)
    image = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:2, :2, :2] = 1
    with pytest.raises(ValueError, match="8 ROI voxels, above the 7-voxel limit"):
        extract(image, mask, settings=Settings(features=("firstorder_Mean", "shape_MoransI")))
    mask[1, 1, 1] = 0
    assert extract(image, mask, settings=Settings(features=("shape_GearysC",))).features


def test_spatial_autocorrelation_is_invariant_to_a_large_intensity_offset() -> None:
    mask = np.zeros((3, 1, 1), dtype=np.bool_)
    mask[:, 0, 0] = True
    baseline = np.zeros(mask.shape, dtype=np.float64)
    baseline[:, 0, 0] = (0.0, 1.0, 2.0)
    shifted = baseline + 1e8
    expected = _values(baseline, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    observed = _values(shifted, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    assert observed["MoransI"] == pytest.approx(expected["MoransI"], rel=2e-6)
    assert observed["GearysC"] == pytest.approx(expected["GearysC"], rel=2e-6)


def test_constant_intensity_autocorrelation_is_undefined() -> None:
    mask = np.ones((2, 2, 1), dtype=np.bool_)
    image = np.full(mask.shape, 5.0, dtype=np.float64)
    values = _values(image, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    assert np.isnan(values["MoransI"])
    assert np.isnan(values["GearysC"])


@pytest.mark.parametrize("offset, expected", [(0.0, 1 / 4), (1.0, 1 / 6)])
def test_weighted_centroid_retains_absolute_intensities(offset: float, expected: float) -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = np.array([1.0 + offset, 3.0 + offset], dtype=np.float64).reshape(mask.shape)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    assert actual["CentreOfMassShift"] == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("separation", [1, 2**20])
@pytest.mark.parametrize("base", [1.5, 3.0, 1e6])
def test_adjacent_float32_weights_give_the_exact_centroid_shift(
    separation: int, base: float,
) -> None:
    # Two voxels whose weights differ by one float32 ulp: the shift is
    # separation * (w1 - w0) / (2 (w0 + w1)). Dividing by a non-power-of-two maximum
    # made it about 25% low.
    w0 = float(np.float32(base))
    w1 = float(np.nextafter(np.float32(base), np.float32(np.inf)))
    image = np.zeros((separation + 1, 1, 1), dtype=np.float64)
    image[0], image[separation] = w0, w1
    mask = image != 0
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    expected = separation * (w1 - w0) / (2 * (w0 + w1))
    assert actual["CentreOfMassShift"] == pytest.approx(expected, rel=1e-6)


def test_constant_weights_on_a_long_asymmetric_mask_have_zero_shift() -> None:
    # The geometric centroid (2**20 + 1) / 3 is not a float32 value; centering on its
    # rounded copy reported a 0.0208 shift for identical weights.
    image = np.zeros((2**20 + 1, 1, 1), dtype=np.float64)
    image[[0, 1, 2**20]] = 5.0
    mask = image != 0
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    assert actual["CentreOfMassShift"] == 0


def test_centroid_preserves_weights_carried_by_source_residuals() -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = np.array([1e8, 1e8 + 1], dtype=np.float64).reshape(mask.shape)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    # Two weights a,b one voxel apart have shift (b-a)/(2*(a+b)). Their
    # high words coincide, but the residual represents their difference exactly.
    assert actual["CentreOfMassShift"] == pytest.approx(1 / 400000002, rel=1e-6, abs=0)


def test_centroid_refuses_source_differences_lost_by_both_words() -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = np.array([0.1, np.nextafter(0.1, np.inf)]).reshape(mask.shape)
    with pytest.raises(RuntimeError, match="centroid is too ill-conditioned"):
        _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))


def test_centroid_preserves_a_resolved_small_signed_total() -> None:
    mask = np.ones((3, 1, 1), dtype=np.bool_)
    image = np.array([1e8, 1.0, -1e8], dtype=np.float64).reshape(mask.shape)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    # Exact signed mass is one; weighted centroid is 1-2e8, geometric centroid 1.
    assert actual["CentreOfMassShift"] == 2e8


@pytest.mark.parametrize("line, expected", [
    ([1e8, 1.0, -1e8, 0, 0, 0, 0, 0], 17 / 24),
    ([-1e8, -1.0, 1e8, 0, 0, 0, 0, 0], -17 / 24),
    ([1.0, -1.0] * 4, 0.0),
])
def test_integrated_intensity_retains_signed_and_zero_totals(
    line: list[float], expected: float,
) -> None:
    image = np.asarray(line).reshape(2, 2, 2)
    mask = np.ones(image.shape, dtype=np.bool_)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("IntegratedIntensity",))
    # The cube's independent mesh volume is 17/3. Its mean is +1/8, -1/8 or zero.
    assert actual["IntegratedIntensity"] == pytest.approx(expected, rel=2e-6, abs=0)


def test_zero_signed_mass_does_not_define_a_weighted_centroid() -> None:
    image = np.array([1.0, -1.0]).reshape(2, 1, 1)
    mask = np.ones(image.shape, dtype=np.bool_)
    with pytest.raises(RuntimeError, match="total intensity is too ill-conditioned"):
        _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))


def test_spatial_autocorrelation_preserves_colliding_high_words() -> None:
    image = np.array([2.0, 2.0 + 1e-8, 2.5, 2.5 + 1e-8,
                      3.0, 3.0 + 1e-8, 3.5, 3.5 + 1e-8]).reshape(2, 2, 2)
    mask = np.ones(image.shape, dtype=np.bool_)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    moran, geary = _all_pairs(image, mask, (1.0, 1.0, 1.0))
    assert actual["MoransI"] == pytest.approx(moran, rel=2e-6, abs=0)
    assert actual["GearysC"] == pytest.approx(geary, rel=2e-6, abs=0)


def test_centre_of_mass_shift_below_float32_range_is_not_reported_as_zero() -> None:
    # Weights 1 and 3 on two voxels along axis 0 put the weighted centroid at 0.75 and
    # the geometric centroid at 0.5: a 0.25-voxel shift, i.e. 2.5e-41 for spacing
    # 1e-40. That nonzero length is below the normal float32 range, so it must be
    # refused like an IntegratedIntensity underflow rather than returned as zero.
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = np.array([1.0, 3.0], dtype=np.float64).reshape(mask.shape)
    with pytest.raises(RuntimeError, match="CentreOfMassShift underflows the float32 range"):
        _values(image, mask, mask, (1e-40, 1.0, 1.0), ("CentreOfMassShift",))
