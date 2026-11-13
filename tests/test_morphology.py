"""Analytic checks for intensity-dependent IBSI morphological features.

Expected values are derived from the manual's definitions with voxel-center coordinates
and inverse Euclidean distance weights. Public extraction also checks intensity versus
morphological mask selection after resegmentation.
"""

from itertools import permutations
from math import dist, fsum, sqrt

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_background, hcc_patch, hcc_resampled

from mlxradiomics import Settings, extract
from mlxradiomics.features import morphology as morphology
from mlxradiomics.features.morphology import morphology_statistics


def _pair(first: float, second: float) -> NDArray[np.float64]:
    """Calibrate the contrast of two adjacent CT voxels to a numeric stress range."""
    image = hcc_patch((2, 1, 1))[0]
    fraction = (image - image[0, 0, 0]) / (image[1, 0, 0] - image[0, 0, 0])
    return first + fraction * (second - first)


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
    image = hcc_patch((4, 4, 4))[0]
    values = _values(image, intensity, morphology, (2.0, 1.0, 1.0),
                     ("CentreOfMassShift",))
    assert values["CentreOfMassShift"] == pytest.approx(sqrt(13.5), rel=2e-6)


def test_constant_intensity_centre_of_mass_shift_is_zero() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    image = hcc_background(mask.shape) * (-12.5 / 2048)
    values = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    assert values["CentreOfMassShift"] == pytest.approx(0, abs=1e-7)


def test_integrated_intensity_uses_mesh_volume_and_intensity_mask() -> None:
    # A 2x2x2 voxel cube at spacing (2,1,1) has grid mesh volume 17/3. Physical volume is 34/3;
    # constant intensity 3 gives integrated intensity 34.
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    image = hcc_background(mask.shape) * (-3 / 2048)
    values = _values(image, mask, mask, (2.0, 1.0, 1.0), ("IntegratedIntensity",))
    assert values["IntegratedIntensity"] == pytest.approx(34.0, rel=2e-6)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_public_morphology_resegments_intensities_and_retains_the_shape(
    conventions: Conventions,
) -> None:
    """A resegmented source ROI retains its original mesh and geometric centroid."""
    image = hcc_patch((4, 4, 4))[0]
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    threshold = float(np.median(image[labels == 1]))
    intensity = (labels == 1) & (image >= threshold)
    points = np.argwhere(intensity) * np.asarray([2.0, 1.0, 1.0])
    values = image[intensity]
    centered = values - values.mean()
    count = len(values)
    pairs = list(permutations(range(count), 2))
    weight = sum(1 / dist(points[i], points[j]) for i, j in pairs)
    variance_sum = float(centered @ centered)
    geometric = np.argwhere(labels == 1).mean(axis=0) * [2, 1, 1]
    weighted = np.sum(points * values[:, None], axis=0) / values.sum()
    expected = {
        "MeshVolume": 34 / 3,
        "CentreOfMassShift": float(np.linalg.norm(weighted - geometric)),
        "IntegratedIntensity": float(values.mean()) * 34 / 3,
        "MoransI": count / weight / variance_sum * sum(
            centered[i] * centered[j] / dist(points[i], points[j]) for i, j in pairs
        ),
        "GearysC": (count - 1) / (2 * weight * variance_sum) * sum(
            (centered[i] - centered[j]) ** 2 / dist(points[i], points[j]) for i, j in pairs
        ),
    }
    result = extract(
        image, labels, spacing=(2.0, 1.0, 1.0),
        settings=settings_for(
            conventions,
            features=tuple(f'shape_{name}' for name in expected),
            resegment_range=(threshold,),
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
    mask = np.ones((3, 1, 1), dtype=np.bool_)
    image = hcc_patch(mask.shape)[0]
    expected_moran, expected_geary = _all_pairs(image, mask, spacing)
    values = _values(image, mask, mask, spacing, ("MoransI", "GearysC"))
    assert values["MoransI"] == pytest.approx(expected_moran, rel=2e-6)
    assert values["GearysC"] == pytest.approx(expected_geary, rel=2e-6)


def test_moran_and_geary_handle_anisotropic_distance_weights() -> None:
    mask = np.zeros((2, 2, 1), dtype=np.bool_)
    mask[0, 0, 0] = mask[1, 0, 0] = mask[0, 1, 0] = True
    image = hcc_patch(mask.shape)[0]
    expected_moran, expected_geary = _all_pairs(image, mask, (2.0, 1.0, 1.0))
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
    image = hcc_patch(shape)[0]
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
    image = hcc_patch((3, 3, 3))[0]
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:2, :2, :2] = 1
    with pytest.raises(ValueError, match="8 ROI voxels, above the 7-voxel limit"):
        extract(image, mask, settings=Settings(features=("firstorder_Mean", "shape_MoransI")))
    mask[1, 1, 1] = 0
    assert extract(image, mask, settings=Settings(features=("shape_GearysC",))).features


def test_spatial_autocorrelation_is_invariant_to_a_large_intensity_offset() -> None:
    mask = np.zeros((3, 1, 1), dtype=np.bool_)
    mask[:, 0, 0] = True
    baseline = hcc_patch(mask.shape)[0]
    shifted = baseline + 1e8
    expected = _values(baseline, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    observed = _values(shifted, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    assert observed["MoransI"] == pytest.approx(expected["MoransI"], rel=2e-6)
    assert observed["GearysC"] == pytest.approx(expected["GearysC"], rel=2e-6)


def test_constant_intensity_autocorrelation_is_undefined() -> None:
    mask = np.ones((2, 2, 1), dtype=np.bool_)
    image = hcc_background(mask.shape) * (-5 / 2048)
    values = _values(image, mask, mask, (1.0, 1.0, 1.0), ("MoransI", "GearysC"))
    assert np.isnan(values["MoransI"])
    assert np.isnan(values["GearysC"])


@pytest.mark.parametrize("offset, expected", [(0.0, 1 / 4), (1.0, 1 / 6)])
def test_weighted_centroid_retains_absolute_intensities(offset: float, expected: float) -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = _pair(1.0 + offset, 3.0 + offset)
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
    image = hcc_resampled((separation + 1, 1, 1))
    image[0], image[separation] = _pair(w0, w1)
    mask = np.zeros(image.shape, dtype=np.bool_)
    mask[[0, separation]] = True
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    expected = separation * (w1 - w0) / (2 * (w0 + w1))
    assert actual["CentreOfMassShift"] == pytest.approx(expected, rel=1e-6)


def test_constant_weights_on_a_long_asymmetric_mask_have_zero_shift() -> None:
    # The geometric centroid (2**20 + 1) / 3 is not a float32 value; centering on its
    # rounded copy reported a 0.0208 shift for identical weights.
    image = np.repeat(hcc_background((1, 1, 1)) * (-5 / 2048), 2**20 + 1, axis=0)
    mask = np.zeros(image.shape, dtype=np.bool_)
    mask[[0, 1, 2**20]] = True
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    assert actual["CentreOfMassShift"] == 0


def test_centroid_preserves_weights_carried_by_source_residuals() -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = _pair(1e8, 1e8 + 1)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    # Two weights a,b one voxel apart have shift (b-a)/(2*(a+b)). Their
    # high words coincide, but the residual represents their difference exactly.
    assert actual["CentreOfMassShift"] == pytest.approx(1 / 400000002, rel=1e-6, abs=0)


def test_centroid_refuses_source_differences_lost_by_both_words() -> None:
    mask = np.ones((2, 1, 1), dtype=np.bool_)
    image = _pair(0.1, float(np.nextafter(0.1, np.inf)))
    with pytest.raises(RuntimeError, match="centroid is too ill-conditioned"):
        _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))


def test_centroid_preserves_a_resolved_small_signed_total() -> None:
    mask = np.ones((3, 1, 1), dtype=np.bool_)
    image = hcc_patch(mask.shape)[0]
    image[0], image[-1] = 1e8, -1e8
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))
    # The source middle voxel survives the cancelling endpoint masses.
    assert actual["CentreOfMassShift"] == pytest.approx(2e8 / image[1, 0, 0], rel=2e-6)


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_integrated_intensity_retains_signed_and_zero_totals(sign: int) -> None:
    image = hcc_patch((2, 2, 2))[0]
    if sign:
        image.reshape(-1)[[0, -1]] = 1e8, -1e8
        image *= sign
    else:
        image -= image.mean()
    mask = np.ones(image.shape, dtype=np.bool_)
    actual = _values(image, mask, mask, (1.0, 1.0, 1.0), ("IntegratedIntensity",))
    # The independently defined cube mesh has volume 17/3.
    expected = fsum(float(v) for v in image.reshape(-1)) / image.size * 17 / 3
    assert actual["IntegratedIntensity"] == pytest.approx(expected, rel=2e-6, abs=0)


def test_zero_signed_mass_does_not_define_a_weighted_centroid() -> None:
    image = hcc_patch((2, 1, 1))[0]
    image -= image.mean()
    mask = np.ones(image.shape, dtype=np.bool_)
    with pytest.raises(RuntimeError, match="total intensity is too ill-conditioned"):
        _values(image, mask, mask, (1.0, 1.0, 1.0), ("CentreOfMassShift",))


def test_spatial_autocorrelation_preserves_colliding_high_words() -> None:
    image = 2 + hcc_patch((2, 2, 2))[0] * 1e-9
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
    image = _pair(1.0, 3.0)
    with pytest.raises(RuntimeError, match="CentreOfMassShift underflows the float32 range"):
        _values(image, mask, mask, (1e-40, 1.0, 1.0), ("CentreOfMassShift",))
