"""Independent physical-neighborhood and intensity-volume acceptance cases."""

from math import fsum, pi

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray

from mlxradiomics import Settings
from mlxradiomics.config.settings import IVH, LOCAL_INTENSITY
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract


def _local_reference(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
) -> tuple[float, float]:
    """Enumerate world-coordinate spheres without convolution or production helpers."""
    radius = (3000 / (4 * pi)) ** (1 / 3)
    maximum = float(image[mask].max())
    local: list[float] = []
    global_: list[float] = []
    for center in np.ndindex(image.shape):
        if not mask[center]:
            continue
        samples = [
            float(image[other])
            for other in np.ndindex(image.shape)
            if sum(((a - b) * s) ** 2 for a, b, s in zip(center, other, spacing, strict=True))
            <= radius**2
        ]
        average = fsum(samples) / len(samples)
        global_.append(average)
        if image[center] == maximum:
            local.append(average)
    return max(local), max(global_)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("spacing", [(3.0, 2.0, 4.0), (10.0, 1.0, 2.0)])
def test_physical_peaks_include_background_and_truncate_at_image_boundary(
    spacing: tuple[float, ...], conventions: Conventions,
) -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape((4, 5, 6)) % 17 - 3
    mask = np.zeros(image.shape, dtype=bool)
    mask[0, 0, 0] = mask[2, 2, 2] = mask[3, 4, 5] = True
    # Tied source maxima must evaluate both neighborhoods before selecting the peak.
    image[mask] = 13
    names = tuple(f"localintensity_{name}" for name in LOCAL_INTENSITY)
    result = extract_original(image, mask, spacing=spacing, settings=settings_for(
        conventions,
        features=names,
        scratch_mib=1,
    ))
    expected = _local_reference(image, mask, spacing)
    for name, target in zip(names, expected, strict=True):
        np.testing.assert_allclose(
            result.features[f"original_{name}"], target, rtol=3e-6, atol=2e-6
        )


@pytest.mark.parametrize("case", ["cancellation", "signed_zero"])
def test_local_peak_preserves_source_residuals_and_equal_signed_zero_candidates(case: str) -> None:
    if case == "cancellation":
        image = np.array([[[1e8 + 1, -1e8]]], dtype=np.float64)
        mask = np.ones(image.shape, dtype=np.bool_)
        spacing = (1.0, 1.0, 1.0)
        target = 0.5
    else:
        image = np.array([[[-0.0, -4.0, -8.0, 0.0]]], dtype=np.float64)
        mask = np.array([[[True, False, False, True]]])
        spacing = (1.0, 1.0, 5.0)
        target = -2.0
    result = extract_original(
        image,
        mask,
        spacing=spacing,
        settings=Settings(
            features=("localintensity_LocalIntensityPeak",),
            scratch_mib=1,
        ),
    )
    np.testing.assert_allclose(
        result.features["original_localintensity_LocalIntensityPeak"], target, rtol=3e-6, atol=0
    )


def _ivh_reference(values: list[int], low: int, high: int) -> dict[str, float]:
    levels = list(range(low, high + 1))
    survival = {level: sum(value >= level for value in values) / len(values) for level in levels}
    if low == high:
        result = dict.fromkeys(IVH, float("nan"))
        result["AreaUnderIVHCurve"] = 0.0
        return result
    result: dict[str, float] = {}
    for fraction in (10, 90):
        at_intensity = [
            survival[level] for level in levels if 100 * (level - low) >= fraction * (high - low)
        ]
        at_volume = [
            level
            for level in levels
            if 100 * sum(value >= level for value in values) <= fraction * len(values)
        ]
        result[f"VolumeAtIntensityFraction{fraction}"] = max(at_intensity)
        result[f"IntensityAtVolumeFraction{fraction}"] = (
            min(at_volume) if at_volume else float("nan")
        )
    result["VolumeFractionDifference"] = (
        result["VolumeAtIntensityFraction10"] - result["VolumeAtIntensityFraction90"]
    )
    result["IntensityFractionDifference"] = (
        result["IntensityAtVolumeFraction10"] - result["IntensityAtVolumeFraction90"]
    )
    gamma = [(level - low) / (high - low) for level in levels]
    if len(set(values)) == 1:
        result["AreaUnderIVHCurve"] = 0.0
    else:
        result["AreaUnderIVHCurve"] = float(
            np.trapezoid(y=[survival[level] for level in levels], x=gamma)
        )
    return result


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mode", ["roi", "maps", "stream"])
def test_ivh_preserves_missing_levels_and_independent_texture_lattice(
    mode: str, ndim: int, conventions: Conventions,
) -> None:
    image = np.array([[0, 0, 2, 2], [3, 3, 5, 9], [0, 2, 5, 9]], dtype=np.float64)
    mask = np.ones(image.shape, dtype=bool)
    mask[1, 1] = False
    if ndim == 3:
        # A second, shifted plane makes every 3D window differ from its planar one.
        image = np.stack([image, np.roll(image, 1, axis=1)])
        mask = np.stack([mask, np.roll(mask, 2, axis=1)])
    names = tuple(f"ivh_{name}" for name in IVH)
    settings = settings_for(
        conventions,
        features=names,
        bin_width=10000,
        ivh_intensity_range=(0.0, 9.0),
        scratch_mib=1,
    )
    if mode == "stream":
        keys = tuple(f"original_{name}" for name in names)
        with voxel_map_store(image.shape, keys) as store:
            result = extract(
                image, mask, settings=settings, mode="voxel", sink=store.append
            )
            actual = {name: values.copy() for name, values in store.maps()}
        assert not result.features
    else:
        result = extract_original(image, mask, settings=settings, voxel=mode == "maps")
        actual = {name: np.asarray(values) for name, values in result.features.items()}
    centers = [None] if mode == "roi" else list(np.ndindex(image.shape))
    for center in centers:
        if center is not None and not mask[center]:
            continue
        domain = (
            tuple(
                slice(max(0, p - 1), min(s, p + 2))
                for p, s in zip(center, image.shape, strict=True)
            )
            if center is not None
            else (slice(None),) * ndim
        )
        expected = _ivh_reference([int(v) for v in image[domain][mask[domain]]], 0, 9)
        for name, value in expected.items():
            observed = actual[f"original_ivh_{name}"]
            scalar = observed[center] if center is not None else observed
            np.testing.assert_allclose(scalar, value, rtol=2e-6, atol=1e-7, equal_nan=True)
    assert result.metadata.get("ivh_lattice") == {"origin": 0.0, "step": 1.0, "levels": 10}


@pytest.mark.parametrize("discretizer", ["width", "count"])
def test_ivh_calibrated_centers_and_arbitrary_units(discretizer: str) -> None:
    image = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float64)
    names = tuple(f"ivh_{name}" for name in IVH)
    settings = (
        Settings(features=names, ivh_bin_width=0.25, ivh_intensity_range=(0.0, 1.0))
        if discretizer == "width"
        else Settings(features=names, ivh_bin_count=4)
    )
    result = extract_original(image, np.ones(image.shape, np.uint8), settings=settings)
    expected = _ivh_reference([1, 2, 3, 4, 4], 1, 4)
    for name, value in expected.items():
        if name.startswith("IntensityAt") and discretizer == "width":
            value = (value - 0.5) * 0.25
        elif name == "IntensityFractionDifference" and discretizer == "width":
            value *= 0.25
        np.testing.assert_allclose(
            result.features[f"original_ivh_{name}"], value, rtol=2e-6, atol=1e-7, equal_nan=True
        )


def test_ivh_area_under_curve_uses_all_lattice_levels_and_single_intensity() -> None:
    # Bins 1, 1, 2, 4 over the four-level lattice 1..4. The survival curve is
    # [1, 0.5, 0.25, 0.25] at gamma [0, 1/3, 2/3, 1], whose trapezoidal area is
    # 11/24. Constant intensities within a wider lattice must report zero instead.
    image = np.array([[0.0, 0.0, 1.0, 3.0]], dtype=np.float64)
    settings = Settings(
        features=("ivh_AreaUnderIVHCurve",), ivh_intensity_range=(0.0, 3.0),
    )
    result = extract_original(image, np.ones(image.shape, np.uint8), settings=settings)
    assert result.features["original_ivh_AreaUnderIVHCurve"] == pytest.approx(11 / 24, rel=2e-6)

    constant = extract_original(
        np.full((2, 2), 5.0), np.ones((2, 2), np.uint8),
        settings=Settings(
            features=("ivh_AreaUnderIVHCurve",), ivh_intensity_range=(0.0, 9.0),
        ),
    )
    assert constant.features["original_ivh_AreaUnderIVHCurve"] == 0.0


def test_ivh_integer_lattice_calibration_keeps_small_reported_intensities() -> None:
    # The calibrated integer lattice runs from -1e8 to 9. Volume fractions select the
    # 10th and 90th percentiles at physical intensities 9 and 1; evaluating
    # ``-1e8 + 99999999`` in float32 would truncate the result to 8 and 0.
    image = np.array([[-1e8, *range(10)]], dtype=np.float64)
    result = extract_original(
        image,
        np.ones(image.shape, dtype=np.uint8),
        settings=Settings(
            features=(
                "ivh_IntensityAtVolumeFraction10",
                "ivh_IntensityAtVolumeFraction90",
            )
        ),
    )
    assert result.features["original_ivh_IntensityAtVolumeFraction10"] == 9.0
    assert result.features["original_ivh_IntensityAtVolumeFraction90"] == 1.0


def test_ivh_absolute_resegmentation_uses_sorted_effective_bounds() -> None:
    # Resegmentation sorts reversed absolute thresholds, so IVH must read the same
    # effective (1, 4) interval rather than rejecting the accepted request.
    image = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    mask = np.ones(image.shape, dtype=np.uint8)
    names = tuple(f"ivh_{name}" for name in IVH)
    result = extract_original(
        image, mask, settings=Settings(features=names, resegment_range=(4.0, 1.0))
    )
    expected = _ivh_reference([1, 2, 3, 4], 1, 4)
    for name, value in expected.items():
        np.testing.assert_allclose(
            result.features[f"original_ivh_{name}"], value, rtol=2e-6, atol=1e-7,
            equal_nan=True, err_msg=name,
        )
    assert result.metadata.get("ivh_lattice") == {"origin": 1.0, "step": 1.0, "levels": 4}


def test_local_peak_isolated_tile_extreme_does_not_flush_small_sphere() -> None:
    # The 1e30 corner is outside the central physical sphere but in the same processing
    # tile. Its whole-tile power-of-two scale would flush the 1e-30 sphere values to zero.
    def peak(extreme: bool) -> float:
        image = np.full((21, 21, 21), 1e-30, dtype=np.float64)
        if extreme:
            image[0, 0, 0] = 1e30
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[10, 10, 10] = 1
        result = extract_original(
            image, mask, settings=Settings(features=("localintensity_LocalIntensityPeak",))
        )
        return float(result.features["original_localintensity_LocalIntensityPeak"])

    isolated = peak(True)
    # The remote extreme must not change the sphere mean at all.
    assert isolated == peak(False)
    # The sphere sums ~4,200 copies of the non-dyadic float32 constant; 1e-5 is well
    # inside that float32 accumulation bound (n * 2**-24 ~ 2.5e-4).
    np.testing.assert_allclose(isolated, float(np.float32(1e-30)), rtol=1e-5, atol=0)


@pytest.mark.parametrize("remote", [0.0, 3e38])
def test_local_peaks_keep_residuals_when_a_remote_value_sets_the_tile_scale(
    remote: float,
) -> None:
    # Both 6 mm spheres hold only the first two voxels, whose float32 values cancel and
    # whose residuals sum to 2. The remote 3e38 voxel is outside both spheres but sets
    # the tile scale to 2**127, which flushes the residuals but not their values.
    image = np.zeros((1, 1, 10))
    image[0, 0, :2] = 1e8 + 1, -1e8 + 1
    image[0, 0, -1] = remote
    mask = np.zeros(image.shape, np.uint8)
    mask[0, 0, :2] = 1
    names = ("localintensity_LocalIntensityPeak", "localintensity_GlobalIntensityPeak")
    result = extract_original(image, mask, spacing=(1.0, 1.0, 6.0),
        settings=Settings(features=names))
    for name in names:
        assert float(result.features[f"original_{name}"]) == 1.0, name


def test_local_peak_common_large_intensities_do_not_overflow_sphere_sum() -> None:
    # A sphere of ~1000 voxels at 2e38 overflows an unscaled sum; the power-of-two scale
    # must still recover the representable constant peak.
    image = np.full((21, 21, 21), 2e38, dtype=np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[10, 10, 10] = 1
    result = extract_original(
        image, mask, settings=Settings(features=("localintensity_LocalIntensityPeak",))
    )
    np.testing.assert_allclose(
        result.features["original_localintensity_LocalIntensityPeak"],
        float(np.float32(2e38)),
        rtol=3e-6,
        atol=0,
    )


def test_ivh_tiny_lattice_coefficients_still_give_normal_thresholds() -> None:
    # Origin and step lie below the normal float32 range, but every source intensity
    # and every expected output is a normal float32 value.
    image = np.array([[2.0**-125, 2.0**-124, 3 * 2.0**-125]])
    names = ("IntensityAtVolumeFraction10", "IntensityAtVolumeFraction90",
             "IntensityFractionDifference")
    result = extract_original(image, np.ones(image.shape, np.uint8), settings=Settings(
        features=tuple(f"ivh_{name}" for name in names), ivh_bin_width=2.0**-140,
        ivh_intensity_range=(0.0, 2.0**-122)))
    # Each threshold is the bin center one level above the last excluded sample's bin:
    # (k + 3/2) * 2**-140 for a sample at k * 2**-140.
    expected = (3 * 2.0**-125 + 3 * 2.0**-141, 2.0**-125 + 3 * 2.0**-141, 2.0**-124)
    for name, value in zip(names, expected, strict=True):
        assert result.features[f"original_ivh_{name}"] == float(np.float32(value)), name


def test_ivh_fixed_count_compares_exact_source_rational_boundaries() -> None:
    image = np.array([[0.0, float(1 / 3), np.nextafter(float(1 / 3), np.inf), 1.0]])
    result = extract_original(
        image,
        np.ones(image.shape, np.uint8),
        settings=Settings(
            features=("ivh_VolumeAtIntensityFraction10",),
            ivh_bin_count=3,
        ),
    )
    # Binary64 float(1/3) lies below the exact first-third boundary; its upper
    # representable neighbor lies above. The bins are therefore 1,1,2,3.
    assert result.features["original_ivh_VolumeAtIntensityFraction10"] == 0.5


@pytest.mark.parametrize("width,levels", [(0.1, 10), (0.3, 4), (0.25, 4)])
def test_ivh_fixed_width_covers_partial_bins_without_snapping(width: float, levels: int) -> None:
    from mlxradiomics.features.ivh import prepare_ivh

    image = np.array([[0.0, np.nextafter(width, -np.inf), width, np.nextafter(width, np.inf), 1.0]])
    mask = np.ones(image.shape, dtype=np.bool_)
    settings = Settings(
        features=("ivh_VolumeAtIntensityFraction10",),
        ivh_bin_width=width,
        ivh_intensity_range=(0.0, 1.0),
    )
    lattice = prepare_ivh(image, mask, settings)
    assert lattice.levels == levels
    assert lattice.origin == width / 2
    assert lattice.step == width
    # Adjacent values straddle the supplied exact binary64 width. The explicit
    # upper edge is assigned to the last bin even when it is an exact bin edge.
    np.testing.assert_array_equal(lattice.bins, [[1, 1, 2, 2, levels]])
    result = extract_original(image, mask, settings=settings)
    np.testing.assert_allclose(
        result.features["original_ivh_VolumeAtIntensityFraction10"], 3 / 5, rtol=1e-6, atol=0
    )
    assert result.metadata.get("ivh_lattice") == {
        "origin": width / 2,
        "step": width,
        "levels": levels,
    }


def test_ivh_fractional_lattice_calibration_survives_a_cancelling_origin() -> None:
    # Bin width 0.1 from -1000 puts the first centre at -999.95. Ten percent of the 11
    # samples is one sample, so the 10% threshold is the centre of the lattice level after
    # the one holding 0.5: -999.95 + 10005 * 0.1 = 0.55. Float32 affine evaluation would round
    # both terms near 1e3 and report 0.5499878.
    from fractions import Fraction

    image = np.array([[-1000.0, *([0.5] * 9), 0.9]], dtype=np.float64)
    result = extract_original(
        image,
        np.ones(image.shape, dtype=np.uint8),
        settings=Settings(
            features=("ivh_IntensityAtVolumeFraction10",), ivh_bin_width=0.1,
        ),
    )
    width = Fraction(0.1)
    index = int((Fraction(0.5) - Fraction(-1000.0)) // width) + 1
    expected = float(np.float32(float(Fraction(-1000.0) + width / 2 + index * width)))
    assert result.features["original_ivh_IntensityAtVolumeFraction10"] == expected


def test_ivh_batch_thresholds_preserve_empty_rows_and_missing_levels() -> None:
    from mlxradiomics.features.ivh import ivh_statistics
    from mlxradiomics.numerics.runtime import mx

    # The first population occupies bins [1, 1, 2, 4] on a four-level lattice;
    # its missing level remains part of the physical domain. The second population
    # is constant, and the third is empty. Prefix counts must not cross row boundaries.
    bins = mx.array([[1, 1, 2, 4], [3, 3, 3, 3], [0, 0, 0, 0]], dtype=mx.int32)
    valid = mx.array([[True] * 4, [True] * 4, [False] * 4], dtype=mx.bool_)
    actual = ivh_statistics(bins, valid, 4, -3.0, 2.0)
    expected = {
        "VolumeAtIntensityFraction10": [0.5, 1.0, float("nan")],
        "VolumeAtIntensityFraction90": [0.25, 0.0, float("nan")],
        "IntensityAtVolumeFraction10": [float("nan"), 3.0, float("nan")],
        "IntensityAtVolumeFraction90": [-1.0, 3.0, float("nan")],
        "VolumeFractionDifference": [0.25, 1.0, float("nan")],
        "IntensityFractionDifference": [float("nan"), 0.0, float("nan")],
        "AreaUnderIVHCurve": [float(np.float32(11 / 24)), 0.0, float("nan")],
    }
    assert actual.keys() == expected.keys()
    for name, target in expected.items():
        np.testing.assert_array_equal(
            np.asarray(actual[name], dtype=np.float32), target, err_msg=name,
        )


def test_ivh_fixed_width_edges_retain_exact_integer_cancellation() -> None:
    from mlxradiomics.features.ivh import prepare_ivh

    # Every final edge fits binary64, but 3 * width exceeds its exact integer range.
    # Rounding that product before adding the origin turns the true edge 1 into 0,
    # moving a sampled zero from bin 3 into bin 4.
    low, width, high = -9007199254740992.0, 3002399751580331.0, 6004799503160663.0
    image = np.array([[low, 0.0, high]], dtype=np.float64)
    mask = np.ones(image.shape, dtype=np.bool_)
    lattice = prepare_ivh(image, mask, Settings(
        features=("ivh_VolumeAtIntensityFraction10",),
        ivh_bin_width=width, ivh_intensity_range=(low, high),
    ))
    np.testing.assert_array_equal(lattice.bins, np.array([[1, 3, 5]], dtype=np.int32))


def test_convention_fields_are_independent() -> None:
    constant = np.full((3, 4), 0.1)
    mask = np.ones(constant.shape, np.uint8)
    names = ("firstorder_Skewness", "firstorder_Kurtosis")
    # Raw kurtosis keeps the IBSI zero for a constant ROI; excess kurtosis keeps NaN.
    raw = extract_original(constant, mask, settings=Settings(features=names, kurtosis="raw"))
    assert [raw.features[f"original_{n}"] for n in names] == [0.0, 0.0]
    nan = extract_original(
        constant, mask, settings=Settings(features=names, zero_variance_moments="nan"),
    )
    assert all(np.isnan(nan.features[f"original_{n}"]) for n in names)
    # NGTDM singular caps do not depend on the neighborhood support, and vice versa.
    ngtdm = ("ngtdm_Coarseness", "ngtdm_Busyness", "ngtdm_Strength")
    shells = extract_original(
        np.ones((3, 4)), mask, settings=Settings(features=ngtdm, neighborhood_support="distances"),
    )
    assert tuple(shells.features[f"original_{n}"] for n in ngtdm) == (1e6, 0.0, 0.0)
    unadjusted = extract_original(
        np.ones((3, 4)), mask, settings=Settings(features=ngtdm, ngtdm_singular_cases="unadjusted"),
    )
    assert np.isposinf(unadjusted.features["original_ngtdm_Coarseness"])
    capped = extract_original(
        np.ones((3, 4)), mask, settings=settings_for("pyradiomics",
            features=ngtdm, ngtdm_singular_cases="ibsi-caps"),
    )
    assert tuple(capped.features[f"original_{n}"] for n in ngtdm) == (1e6, 0.0, 0.0)
