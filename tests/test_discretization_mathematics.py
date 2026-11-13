"""Exact binary64 lattice and histogram endpoint contracts for host discretization."""

from fractions import Fraction
from math import floor

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_crop, hcc_patch

from mlxradiomics import Settings, extract
from mlxradiomics.imaging.preprocessing import prepare


def _ct_pair(low: float, high: float) -> NDArray[np.float64]:
    """Calibrate two source CT pixels to the numerical domain under test."""
    ct, _, _ = hcc_patch((1, 2))
    return low + (ct - ct.min()) / np.ptp(ct) * (high - low)


def _fractional_ct() -> NDArray[np.float64]:
    """Quantize source CT contrast into three fractional-width test levels."""
    ct, _, _ = hcc_crop(np.s_[106:107, 215:218, 62])
    return -1000 + np.rint(2 * (ct - ct.min()) / np.ptp(ct)) * 0.2


@pytest.mark.parametrize("anchor", [None, -1000.0])
def test_fixed_width_boundary_neighbors_follow_exact_lattice(anchor: float | None) -> None:
    centers = _fractional_ct()[0]
    image = np.stack([np.nextafter(centers, -np.inf), centers,
                      np.nextafter(centers, np.inf)])
    if anchor is not None:
        image[0, 0] = anchor
    width = Fraction(0.1)
    origin = Fraction(0 if anchor is None else anchor)
    indexes = [floor((Fraction(float(value)) - origin) / width) for value in image.flat]
    first = min(indexes) if anchor is None else 0
    expected = np.array(indexes).reshape(image.shape) - first + 1
    result = prepare(image, np.ones(image.shape, np.uint8),
                     settings_for("pyradiomics", bin_width=0.1, bin_anchor=anchor), voxel=False)
    np.testing.assert_array_equal(result.bins, expected)


def test_fractional_width_contrast_uses_uniform_gray_level_steps() -> None:
    image = _fractional_ct()
    result = extract(image, np.ones(image.shape, np.uint8),
                     settings=settings_for("pyradiomics",
                         bin_width=0.1, features=("glcm_Contrast",)))
    assert result.features["original_glcm_Contrast"] == pytest.approx(4)


def test_fixed_bin_count_includes_large_maximum_in_last_bin() -> None:
    image = _ct_pair(2.0**53, 2.0**53 + 4)
    result = prepare(image, np.ones(image.shape, np.uint8), Settings(bin_count=2), voxel=False)
    np.testing.assert_array_equal(result.bins, [[1, 2]])
    assert result.gray_levels == 2


def test_ibsi_fixed_width_anchors_at_absolute_resegmentation_lower_bound() -> None:
    image, _, _ = hcc_patch((1, 4))
    # IBSI X_gl,min is the declared lower bound, which precedes this CT crop's
    # lowest sample. Leading empty lattice intervals must remain present.
    lower, upper = 0.0, float(image.max())
    settings = Settings(bin_width=25.0, resegment_range=(lower, upper))
    result = prepare(image, np.ones(image.shape, np.uint8), settings, voxel=False)
    expected = np.floor((image - lower) / 25).astype(np.int32) + 1
    np.testing.assert_array_equal(result.bins, expected)
    explicit = prepare(image, np.ones(image.shape, np.uint8),
                       Settings(bin_width=25.0, resegment_range=(lower, upper), bin_anchor=lower),
                       voxel=False)
    np.testing.assert_array_equal(result.bins, explicit.bins)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("width", [1.0, 25.0, 2.0**16, 2.0**31 - 1])
@pytest.mark.parametrize("anchor", [None, -1500.0])
def test_integral_fixed_width_bins_follow_rational_intervals(
    conventions: Conventions, width: float, anchor: float | None,
) -> None:
    image, _, _ = hcc_patch((2, 5))
    origin = Fraction(float(image.min()) if conventions == "ibsi" and anchor is None
                      else anchor or 0)
    indexes = [floor((Fraction(float(value)) - origin) / Fraction(width)) for value in image.flat]
    first = min(indexes) if conventions == "pyradiomics" and anchor is None else 0
    expected = np.asarray(indexes, dtype=np.int32).reshape(image.shape) - first + 1
    settings = settings_for(
        conventions,
        bin_width=width,
        bin_anchor=anchor,
        max_gray_levels=4096,
    )
    result = prepare(image, np.ones(image.shape, dtype=np.uint8), settings, voxel=False)
    np.testing.assert_array_equal(result.bins, expected)
    assert result.gray_levels == int(expected.max())


def test_integral_edge_neighbors_keep_the_exact_source_thresholds() -> None:
    ct, _, _ = hcc_patch((1, 5))
    boundaries = (np.floor(ct[0] / 25) - np.floor(ct[0, 2] / 25)) * 25
    image = np.stack([
        np.nextafter(boundaries, -np.inf), boundaries, np.nextafter(boundaries, np.inf),
    ])
    indexes = [floor(Fraction(float(value)) / 25) for value in image.flat]
    expected = np.asarray(indexes, dtype=np.int32).reshape(image.shape) - min(indexes) + 1
    result = prepare(
        image, np.ones(image.shape, dtype=np.uint8), settings_for("pyradiomics"),
        voxel=False)
    np.testing.assert_array_equal(result.bins, expected)
    assert result.integer_source_bounds is None


def test_integral_ibsi_unmasked_samples_preserve_the_anchor_error() -> None:
    image, _, _ = hcc_crop(np.s_[105:106, 215:220, 64])
    mask = np.array([[0, 1, 1, 1, 1]], dtype=np.uint8)
    settings = Settings(
        bin_width=25.0, resegment_range=(70.0, float(image.max())),
        masked_kernel=False,
    )
    with pytest.raises(ValueError, match="below the resegmentation lower bound"):
        prepare(image, mask, settings, voxel=True)


@pytest.mark.parametrize("fixed_count", [False, True])
def test_large_lattice_keeps_exact_boundary_neighbors_and_closed_maximum(
    fixed_count: bool,
) -> None:
    ct, _, _ = hcc_patch((2, 3))
    image = (ct - ct.min()) / np.ptp(ct)
    # Keep the actual source extrema; perturb three interior values around an
    # exactly representable large-lattice edge to test its adjacent floats.
    selected = np.flatnonzero((image > 0) & (image < 1))[:3]
    edge = floor(float(image.reshape(-1)[int(selected[0])]) * 2**30) * 2.0**-30
    image.flat[selected] = (edge, np.nextafter(edge, -np.inf), np.nextafter(edge, np.inf))
    settings = (Settings(bin_count=2**30) if fixed_count else
                Settings(bin_width=2.0**-30, bin_anchor=0))
    result = prepare(image, np.ones(image.shape, dtype=np.uint8), settings, voxel=False)
    expected_max = 2**30 if fixed_count else 2**30 + 1
    expected = np.asarray([
        min(expected_max, floor(Fraction(float(value)) * 2**30) + 1) for value in image.flat
    ], dtype=np.int32).reshape(image.shape)
    np.testing.assert_array_equal(result.bins, expected)
    assert result.gray_levels == expected_max


def test_large_fixed_width_lattice_retains_the_actual_int32_boundary() -> None:
    image = _ct_pair(0.0, 2.0 - 2.0**-29)
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(bin_width=2.0**-30, bin_anchor=0)
    result = prepare(image, mask, settings, voxel=False)
    np.testing.assert_array_equal(result.bins, [[1, 2**31 - 1]])
    image[0, 1] = 2.0 - 2.0**-30
    with pytest.raises(ValueError, match="int32 indexing capacity"):
        prepare(image, mask, settings, voxel=False)


def test_finite_source_can_have_a_bin_upper_edge_beyond_binary64() -> None:
    image = _ct_pair(1e308, float(np.finfo(np.float64).max))
    result = prepare(image, np.ones(image.shape, dtype=np.uint8),
                     Settings(bin_width=1e308, bin_anchor=0), voxel=False)
    np.testing.assert_array_equal(result.bins, [[2, 2]])
    assert result.gray_levels == 2


@pytest.mark.parametrize("large_offset", [False, True])
def test_exact_sparse_discretization_reaches_public_texture_features(large_offset: bool) -> None:
    image = (_ct_pair(1e20, float(np.nextafter(1e20, np.inf))) if large_offset else
             _ct_pair(0.0, 1.0))
    settings = Settings(
        features=("glcm_Contrast",), bin_width=1.0 if large_offset else 2.0**-20,
        bin_anchor=1e20 if large_offset else 0,
    )
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings)
    # Contrast of a single pair is the squared difference of its original indices.
    assert result.features["original_glcm_Contrast"] == (2**28 if large_offset else 2**40)
