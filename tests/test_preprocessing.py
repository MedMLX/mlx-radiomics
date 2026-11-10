"""Host preprocessing contracts with independently derived expectations."""

from dataclasses import replace
from fractions import Fraction
from typing import Literal, cast

import numpy as np
import pytest
from conftest import settings_for
from numpy.typing import NDArray
from radiomics_reference.mathematics import gradient_magnitude, masked_neighborhood_mean

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.imaging.preprocessing import (
    normalize_values,
    prepare,
    prepare_sampled_input,
)
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract


def _resegmented_mask(
    values: NDArray[np.float64],
    roi: NDArray[np.bool_],
    thresholds: tuple[float, ...],
    mode: Literal["absolute", "relative", "sigma"],
) -> NDArray[np.bool_]:
    """Transform the thresholds, sort them, and apply both inclusive bounds directly."""
    selected = values[roi]
    transformed = np.asarray(thresholds, dtype=np.float64)
    if mode == "relative":
        transformed = transformed * np.max(selected)
    elif mode == "sigma":
        transformed = np.mean(selected) + np.std(selected) * transformed
    transformed = np.sort(transformed)
    retained = roi & (values >= float(transformed[0]))
    if len(transformed) == 2:
        retained &= values <= float(transformed[1])
    return retained


@pytest.mark.parametrize("scale,outliers", [(1.0, None), (25.0, 0.8)])
def test_normalization_uses_whole_image_sample_variance(
    scale: float, outliers: float | None,
) -> None:
    image = np.random.default_rng(25).normal(35, 500, (8, 7, 6))
    options = Settings(normalize=True, normalize_scale=scale, remove_outliers=outliers)
    values = image.astype(np.float64)
    expected = (values - np.mean(values)) / np.std(values, ddof=1)
    if outliers is not None:
        expected = np.clip(expected, -outliers, outliers)
    np.testing.assert_allclose(normalize_values(image, options), expected * scale,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("value", [0.1, -0.1, 1e20])
def test_normalization_rejects_exactly_constant_values(value: float) -> None:
    with pytest.raises(ValueError, match="nonzero finite whole-image variance"):
        normalize_values(np.full((1, 3), value), Settings(normalize=True))


@pytest.mark.parametrize("value", [1.0, 1e8])
def test_normalization_preserves_adjacent_source_values(value: float) -> None:
    image = np.array([[value, np.nextafter(value, np.inf)]])
    actual = normalize_values(image, Settings(normalize=True))
    np.testing.assert_allclose(actual, [[-1 / np.sqrt(2), 1 / np.sqrt(2)]],
                               rtol=1e-15, atol=0)


@pytest.mark.parametrize("constant", [False, True])
def test_sigma_resegmentation_compares_centered_source_values(constant: bool) -> None:
    image = (np.full((1, 3), 0.1) if constant else
             np.array([[1., 1., np.nextafter(1., np.inf), np.nextafter(1., np.inf)]]))
    actual = prepare(image, np.ones(image.shape, np.uint8), Settings(
        resegment_mode="sigma", resegment_range=(0.,), bin_count=2), voxel=False)
    expected = np.ones(image.shape, bool) if constant else np.array([[False, False, True, True]])
    np.testing.assert_array_equal(actual.roi, expected)


def test_sigma_outlier_filter_preserves_adjacent_value_boundary() -> None:
    image = np.array([[1., 1., np.nextafter(1., np.inf), np.nextafter(1., np.inf)]])
    with pytest.raises(ValueError, match="retain at least one"):
        prepare(image, np.ones(image.shape, np.uint8), Settings(
            resegment_range=(0., 2.), resegment_outlier_sigma=0.5, bin_count=2), voxel=False)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize(
    "thresholds,kept", [((-0.5, 0.5), [2, 3]), ((-2.0, 2.0), [0, 1, 2, 3, 4, 5])],
)
def test_sigma_resegmentation_is_invariant_to_positive_intensity_scale(
    scale: float, thresholds: tuple[float, float], kept: list[int],
) -> None:
    # Mean 1 and population sigma sqrt(2/3); the result must not depend on the intensity scale.
    image = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]) * scale
    mask = np.ones(image.shape, np.uint8)
    for settings in (
        Settings(resegment_range=thresholds, resegment_mode="sigma", bin_count=2),
        Settings(resegment_range=(-1e300, 1e300), resegment_outlier_sigma=thresholds[1],
                 bin_count=2),
    ):
        data = prepare(image, mask, settings, voxel=False)
        assert np.flatnonzero(data.roi[0]).tolist() == kept


@pytest.mark.parametrize("count", [1, 7, 8, 32, 64])
@pytest.mark.parametrize("flat", [False, True])
def test_fixed_bin_count_uses_closed_form_bins(count: int, flat: bool) -> None:
    image = np.ones((4, 6)) if flat else np.linspace(-1024, 257.5, 24).reshape(4, 6)
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[0, 1] = 0
    selected = image[mask.astype(bool)]
    if flat:
        # A constant signal lies at the midpoint of the equal-width lattice; NumPy
        # widens the degenerate range by half a unit on each side before counting.
        expected = np.full(selected.shape, count // 2 + 1, dtype=np.int32)
    else:
        low, high = Fraction(float(selected.min())), Fraction(float(selected.max()))
        expected = np.array([
            min(count, int(count * (Fraction(float(value)) - low) // (high - low)) + 1)
            for value in selected
        ], dtype=np.int32)
    actual = prepare(image, mask, settings_for("pyradiomics", bin_count=count), voxel=False)
    assert actual.bins is not None
    np.testing.assert_array_equal(actual.bins[mask.astype(bool)], expected)
    np.testing.assert_array_equal(actual.bins[~mask.astype(bool)], 0)
    assert actual.gray_levels == int(expected.max())


@pytest.mark.parametrize("dtype", ["<i8", ">i8", "<u8", ">u8"])
def test_lossy_64_bit_integer_intensities_are_refused(dtype: str) -> None:
    # float64 rounds 2**53 + 1 to 2**53, which would merge distinct bins silently.
    # The check is byte-order independent.
    mask = np.ones((1, 3), np.uint8)
    with pytest.raises(ValueError, match="cannot represent exactly"):
        prepare(np.array([[2**53, 2**53 + 1, 2**53 + 2]], dtype), mask,
                Settings(bin_count=3), voxel=False)
    # Large integers with few significant bits are exact and stay accepted.
    data = prepare(np.array([[2**60, 2**60 + 2**8, 2**60 + 2**9]], dtype), mask,
                   Settings(bin_count=3), voxel=False)
    assert data.bins is not None
    assert data.bins.tolist() == [[1, 2, 3]]
    if dtype[1] == "i":
        signed = prepare(np.array([[-1000, 0, 1000]], dtype), mask,
                         Settings(bin_count=3), voxel=False)
        assert signed.bins is not None
        assert signed.bins.tolist() == [[1, 2, 3]]


def test_unmasked_kernel_below_resegmentation_anchor_is_refused() -> None:
    # Extending the IBSI lattice to sampled background below the lower bound renumbered
    # every ROI gray level: [-480, -460, -420, 390] gave [1, 2, 4, 36] alone but
    # [5, 6, 8, 40] beside a -600 background voxel.
    image = np.array([[-480.0, -460.0, -420.0, 390.0, -600.0]])
    mask = np.array([[1, 1, 1, 1, 0]], np.uint8)
    settings = Settings(bin_width=25.0,
                        resegment_range=(-500.0, 400.0), masked_kernel=False)
    masked = prepare(image, mask, replace(settings, masked_kernel=True), voxel=True)
    assert masked.bins is not None
    assert masked.bins[0, :4].tolist() == [1, 2, 4, 36]
    with pytest.raises(ValueError, match="resegmentation lower bound"):
        prepare(image, mask, settings, voxel=True)


@pytest.mark.parametrize("scale", [1e-162, 1e-161, 1e300])
def test_normalization_is_scale_invariant_at_float64_extremes(scale: float) -> None:
    # Squaring unscaled offsets would underflow or overflow; the result must be scale-free.
    image = np.array([[0.0, 1.0, 2.0, 5.0]]) * scale
    actual = normalize_values(image, Settings(normalize=True))
    reference = normalize_values(np.array([[0.0, 1.0, 2.0, 5.0]]), Settings(normalize=True))
    # Decimal scaling is inexact, so the zero-mean entry may differ by an ulp of 1.
    np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-15)
    assert float(np.std(actual, ddof=1)) == pytest.approx(1.0, rel=1e-14)


def test_sigma_resegmentation_scales_before_subtracting_opposite_extremes() -> None:
    # 1e308 - (-1e308) overflows, so the extremes are scaled before subtracting.
    image = np.array([[1e308, -1e308, 0.0, 0.0]])
    data = prepare(image, np.ones(image.shape, np.uint8), Settings(
        resegment_mode="sigma", resegment_range=(-1.0, 1.0), bin_count=2), voxel=False)
    np.testing.assert_array_equal(data.roi, [[False, False, True, True]])


def test_fixed_width_bins_keep_float64_boundary_neighbors() -> None:
    width = 25.0
    edges = np.arange(-2, 3, dtype=np.float64) * width
    image = np.stack((np.nextafter(edges, -np.inf), edges, np.nextafter(edges, np.inf)))
    mask = np.ones(image.shape, dtype=np.uint8)
    actual = prepare(image, mask, settings_for("pyradiomics", bin_width=width), voxel=False)
    # These width-25 edges are exactly representable. Each edge belongs to the upper
    # bin, while its adjacent representable predecessor belongs to the lower bin.
    expected = ((1, 2, 3, 4, 5), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6))
    np.testing.assert_array_equal(actual.bins, np.asarray(expected, dtype=np.int32))


def test_default_bin_anchor_uses_zero_lattice_edges() -> None:
    values = np.array([[-1001.0, -1000.0, -939.0, -900.0, 393.0, 400.0]])
    mask = np.ones(values.shape, dtype=np.uint8)
    default = prepare(values, mask, settings_for("pyradiomics", bin_width=25.0), voxel=False)
    explicit_none = prepare(values, mask,
                            settings_for("pyradiomics", bin_width=25.0, bin_anchor=None),
                            voxel=False)
    expected = np.array([[1, 2, 4, 6, 57, 58]], dtype=np.int32)
    np.testing.assert_array_equal(default.bins, expected)
    np.testing.assert_array_equal(explicit_none.bins, expected)


def test_explicit_bin_anchor_uses_declared_origin() -> None:
    values = np.array([[-939.0, 393.0]])
    mask = np.ones(values.shape, dtype=np.uint8)
    options = Settings(bin_width=25.0, bin_anchor=-1000.0)
    actual = prepare(values, mask, options, voxel=False)
    # Fixed bin size: floor((x - anchor) / width) + 1. Bins 1-2 stay empty, so the ROI
    # keeps gray levels 3..56 instead of being renumbered to 1..54.
    expected = np.floor((values + 1000.0) / 25.0).astype(np.int32) + 1
    np.testing.assert_array_equal(actual.bins, expected)
    assert actual.gray_levels == 56


def test_bin_anchor_rejects_selected_values_below_origin() -> None:
    values = np.array([[-1001.0, -939.0]])
    mask = np.ones(values.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="bin_anchor"):
        prepare(values, mask, Settings(bin_width=25.0, bin_anchor=-1000.0), voxel=False)


def test_bin_anchor_rejects_bin_count() -> None:
    with pytest.raises(ValueError, match="bin_anchor.*bin_count"):
        Settings(bin_count=32, bin_anchor=-1000.0).validate(2)


@pytest.mark.parametrize("anchor", [float("nan"), float("inf"), -float("inf")])
def test_bin_anchor_requires_finite_value(anchor: float) -> None:
    with pytest.raises(ValueError, match="bin_anchor"):
        Settings(bin_anchor=anchor).validate(2)


def test_bin_anchor_rejects_excessive_gray_level_range() -> None:
    values = np.array([[0.0]])
    mask = np.ones(values.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="max_gray_levels"):
        prepare(values, mask, Settings(
            bin_width=1.0, bin_anchor=-1_000_000.0, max_gray_levels=1024,
        ), voxel=False)


def test_bin_anchor_rejects_overflowing_gray_level_range() -> None:
    values = np.array([[1e308]])
    mask = np.ones(values.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="int32 indexing capacity"):
        prepare(values, mask, Settings(bin_width=1.0, bin_anchor=-1e308), voxel=False)


def test_bin_anchor_preserves_indices_when_float64_edges_coincide() -> None:
    values = np.array([[1e20, np.nextafter(1e20, np.inf)]])
    mask = np.ones(values.shape, dtype=np.uint8)
    actual = prepare(values, mask, Settings(bin_width=1.0, bin_anchor=1e20), voxel=False)
    # Adjacent source floats are exactly 16384 units apart. Empty intervals remain
    # in the index even though their edges cannot all be distinct binary64 values.
    np.testing.assert_array_equal(actual.bins, [[1, 16385]])


@pytest.mark.parametrize(
    "voxel,masked_kernel,normalize,sampled,rejected", [
        (False, True, False, False, False),
        (True, True, False, False, False),
        (True, False, False, False, True),
        (False, True, True, False, True),
        (False, True, False, True, True),
    ], ids=["roi-background", "masked-background", "unmasked-background",
            "normalization-background", "sampled-values"],
)
def test_array_nonfinite_values_follow_sampling_contract(
    voxel: bool, masked_kernel: bool, normalize: bool, sampled: bool, rejected: bool,
) -> None:
    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:4, 1:4] = 1
    image[2 if sampled else 0, 1:4] = [np.nan, np.inf, -np.inf]
    settings = Settings(features=("firstorder_Mean",), masked_kernel=masked_kernel,
                        normalize=normalize)
    if rejected:
        with pytest.raises(ValueError, match="finite"):
            extract_original(image, mask, settings=settings, voxel=voxel)
        return
    actual = extract_original(image, mask, settings=settings, voxel=voxel)
    expected = (masked_neighborhood_mean(image, mask == 1)
                if voxel else float(np.mean(image[mask == 1])))
    np.testing.assert_allclose(actual.features["original_firstorder_Mean"], expected,
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode,thresholds", [
    ("absolute", (-150.0, 500.0)), ("absolute", (-150.0,)),
    ("relative", (0.2, 0.8)), ("sigma", (1.0, -1.0)),
])
@pytest.mark.parametrize("resegment_shape", [False, True])
def test_resegmentation_preserves_shape_mask(
    mode: Literal["absolute", "relative", "sigma"], thresholds: tuple[float, ...],
    resegment_shape: bool,
) -> None:
    image = np.linspace(-350, 820, 120).reshape(4, 5, 6)
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[0] = 3
    options = Settings(resegment_range=thresholds, resegment_mode=mode,
                       resegment_shape=resegment_shape)
    actual = prepare(image, mask, options, voxel=False)
    roi = mask == 1
    expected = _resegmented_mask(image, roi, thresholds, mode)
    np.testing.assert_array_equal(actual.roi, expected)
    np.testing.assert_array_equal(actual.shape_roi, expected if resegment_shape else roi)


@pytest.mark.parametrize("thresholds", [(0.5, 1.0), (1.0, 0.5), (1.0,)])
def test_relative_resegmentation_orders_transformed_negative_bounds(
    thresholds: tuple[float, ...],
) -> None:
    # Maximum -2 maps the pair to intensity interval [-2, -1]; both maxima survive.
    # A single threshold still means values >= -2 rather than a reversed inequality.
    image = np.array([[-6.0, -4.0, -2.0, -2.0]])
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(resegment_range=thresholds, resegment_mode="relative")
    actual = prepare(image, mask, settings, voxel=False)
    np.testing.assert_array_equal(actual.roi, [[False, False, True, True]])
    np.testing.assert_array_equal(actual.shape_roi, mask.astype(bool))


@pytest.mark.parametrize("resegment_shape", [False, True])
def test_combined_resegmentation_intersects_independent_filters(
    resegment_shape: bool,
) -> None:
    # Fifty low values and ten high values lie inside the absolute range; one outlier
    # outside it pulls the pre-range mean and spread up. A sigma filter recomputed
    # after the absolute range would discard the ten high values, but the contract
    # applies both filters independently to the original ROI and intersects the masks.
    values = np.concatenate((np.zeros(50), np.full(10, 10.0), [1000.0])).reshape(1, -1)
    mask = np.ones(values.shape, dtype=np.uint8)
    thresholds, sigma = (0.0, 10.0), 1.0
    options = Settings(resegment_range=thresholds, resegment_outlier_sigma=sigma,
                       resegment_shape=resegment_shape)
    actual = prepare(values, mask, options, voxel=False)
    roi = mask == 1
    selected = values[roi]
    absolute = roi & (values >= thresholds[0]) & (values <= thresholds[1])
    center, spread = float(np.mean(selected)), float(np.std(selected))
    sigma_mask = (roi & (values >= center - sigma * spread)
                  & (values <= center + sigma * spread))
    expected = absolute & sigma_mask
    retained = values[absolute]
    post_absolute = absolute & (
        values >= float(np.mean(retained)) - sigma * float(np.std(retained))
    ) & (
        values <= float(np.mean(retained)) + sigma * float(np.std(retained))
    )
    # This fixture separates the two orders: clipping before sigma drops every 10, so a
    # post-absolute implementation would fail the mask comparison below.
    assert not np.array_equal(expected, post_absolute)
    np.testing.assert_array_equal(actual.roi, expected)
    np.testing.assert_array_equal(
        actual.shape_roi, expected if resegment_shape else mask == 1
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True])
def test_resegment_outlier_sigma_requires_finite_positive_number(value: object) -> None:
    with pytest.raises(ValueError, match="resegment_outlier_sigma"):
        Settings(resegment_range=(0.0, 10.0),
                 resegment_outlier_sigma=cast(float, value)).validate(2)


@pytest.mark.parametrize("thresholds,mode", [
    (None, "absolute"), ((0.0, 10.0), "relative"), ((0.0, 10.0), "sigma"),
])
def test_resegment_outlier_sigma_requires_absolute_range(
    thresholds: tuple[float, ...] | None,
    mode: Literal["absolute", "relative", "sigma"],
) -> None:
    with pytest.raises(ValueError, match="resegment_outlier_sigma"):
        Settings(resegment_range=thresholds, resegment_mode=mode,
                 resegment_outlier_sigma=1).validate(2)


def test_combined_resegmentation_filters_once_before_derived_images() -> None:
    image = np.linspace(-20.0, 20.0, 64).reshape(8, 8)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:-1, 1:-1] = 1
    options = Settings(resegment_range=(-20.0, 20.0), resegment_outlier_sigma=1,
                       bin_width=2.0, features=("firstorder_Mean",))
    actual = extract(
        image, mask, settings=options,
        transforms=TransformSettings(image_types=(ImageType.SQUARE,)),
    )
    roi = mask == 1
    selected = image[roi]
    center, spread = float(np.mean(selected)), float(np.std(selected))
    filtered = (roi & (image >= -20.0) & (image <= 20.0)
                & (image >= center - spread) & (image <= center + spread))
    assert not np.array_equal(filtered, roi)
    maximum = float(np.max(np.abs(image)))
    squared = np.asarray(np.square(image[filtered] / np.sqrt(maximum)), dtype=np.float64)
    expected = float(np.mean(squared))
    # The derived image is filtered only by the original combined ROI, not again with
    # the transform's own value distribution.
    np.testing.assert_allclose(actual.features["square_firstorder_Mean"], expected,
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode,thresholds,expected", [
    ("absolute", (-4.0, 4.0), [-4.0, 0.0, 4.0]),
    ("relative", (0.0, 0.5), [0.0, 4.0]),
    ("sigma", (-1.0, 1.0), [-4.0, 0.0, 4.0]),
])
@pytest.mark.parametrize("unmasked", [False, True])
def test_resegmented_samples_keep_grid_order_and_the_full_unmasked_finite_contract(
    mode: Literal["absolute", "relative", "sigma"], thresholds: tuple[float, ...],
    expected: list[float], unmasked: bool,
) -> None:
    image = np.full((7, 9), np.nan, dtype=np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    coordinates = ([1, 2, 3, 5, 6], [7, 2, 5, 0, 8])
    image[coordinates] = [-8.0, -4.0, 0.0, 4.0, 8.0]
    mask[coordinates] = 1
    settings = Settings(
        resegment_mode=mode, resegment_range=thresholds, masked_kernel=not unmasked,
    )
    if unmasked:
        with pytest.raises(ValueError, match="sampled image intensities must be finite"):
            prepare_sampled_input(image, mask, settings, voxel=True)
        return
    data, sampled = prepare_sampled_input(image, mask, settings, voxel=True)
    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(sampled, data.image[data.sampling_mask])
    np.testing.assert_array_equal(data.shape_roi, mask == 1)


def test_filtered_images_do_not_discretize_unused_original() -> None:
    image = np.arange(16, dtype=np.float64)[:, None] * 10000 + np.arange(17)[None] * 1000
    mask = np.ones(image.shape, dtype=np.uint8)
    result = extract(image, mask, settings=Settings(features=("firstorder_Mean",)),
                            transforms=TransformSettings(image_types=(ImageType.GRADIENT,)))
    assert set(result.features) == {"gradient_firstorder_Mean"}
    expected = gradient_magnitude(image, (1.0, 1.0))
    np.testing.assert_allclose(result.features["gradient_firstorder_Mean"],
                               np.mean(expected[mask == 1]), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("binned_feature", [
    "firstorder_Entropy", "firstorder_Uniformity", "glcm_Contrast",
])
def test_mixed_raw_and_binned_features_honor_explicit_lattice_limit(
    binned_feature: str,
) -> None:
    image = np.array([[0.0, 25.0, 50.0]])
    mask = np.ones(image.shape, dtype=np.uint8)
    features = ("firstorder_Mean", binned_feature)
    settings = Settings(features=features, bin_width=25.0, max_gray_levels=2)
    with pytest.raises(ValueError, match="max_gray_levels"):
        extract(image, mask, settings=settings)
    for voxel in (False, True):
        result = extract(image, mask, settings=replace(settings, max_gray_levels=3),
                                mode=("voxel" if voxel else "roi"))
        assert result.metadata["gray_levels"] == 3
        np.testing.assert_allclose(
            result.features["original_firstorder_Mean"],
            [[12.5, 25.0, 37.5]] if voxel else 25.0,
            rtol=0, atol=1e-6,
        )


@pytest.mark.parametrize("options", [
    pytest.param({"bin_width": 0.05, "bin_anchor": 0.0}, id="fine-lattice"),
    pytest.param({"bin_width": 25.0, "bin_anchor": 1000.0}, id="unused-anchor"),
    pytest.param({"bin_count": 2048}, id="unused-bin-count"),
])
@pytest.mark.parametrize("mode", ["roi", "voxel"])
def test_raw_statistics_are_independent_of_unused_discretization(
    options: dict[str, float | int], mode: Literal["roi", "voxel"],
) -> None:
    image = np.array([[300.0, 400.0, 500.0]])
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(
        features=("firstorder_Mean", "firstorder_Variance", "firstorder_Median",
                  "firstorder_Energy"),
        max_gray_levels=1,
    )
    result = extract(image, mask, settings=replace(settings, **options), mode=mode)
    expected = (
        {"Mean": 400.0, "Variance": 20000 / 3, "Median": 400.0, "Energy": 500000.0}
        if mode == "roi" else
        {"Mean": [[350.0, 400.0, 450.0]],
         "Variance": [[2500.0, 20000 / 3, 2500.0]],
         "Median": [[350.0, 400.0, 450.0]], "Energy": [[250000.0, 500000.0, 410000.0]]}
    )
    for name, value in expected.items():
        np.testing.assert_allclose(result.features[f"original_firstorder_{name}"], value,
                                   rtol=1e-6, atol=1e-6)
    assert result.metadata["gray_levels"] == 1


def test_fixed_width_bins_preserve_large_anchored_lattice() -> None:
    image = np.array([[0.0, 2500.0, 5000.0]])
    prepared = prepare(image, np.ones(image.shape, dtype=np.uint8), Settings(
        features=("intensityhistogram_Mean",), bin_width=0.25, bin_anchor=0.0,
    ), voxel=False)
    np.testing.assert_array_equal(prepared.bins, [[1, 10001, 20001]])
    assert prepared.gray_levels == 20001


@pytest.mark.parametrize("features,binned", [
    (("firstorder_Mean",), False),
    (("shape2D_PixelSurface", "localintensity_LocalIntensityPeak"), False),
    (("ivh_AreaUnderIVHCurve",), False),
    (("firstorder_Entropy",), True),
    (("firstorder_Mean", "firstorder_Entropy", "ivh_AreaUnderIVHCurve", "glcm_Contrast"), True),
])
def test_main_lattice_is_absent_when_selected_features_do_not_read_it(
    features: tuple[str, ...], binned: bool,
) -> None:
    image = np.array([[1.0, 2.0, 3.0]])
    data = prepare(image, np.ones(image.shape, dtype=np.uint8),
                   Settings(features=features, bin_width=1), voxel=False)
    assert (data.bins is not None) == binned
    assert data.gray_levels == (3 if binned else 1)


def test_identity_resegmentation_preserves_singleton_statistics() -> None:
    image = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[1, 0, 0]], dtype=np.uint8)
    settings = Settings(features=("firstorder_Mean", "firstorder_Variance"))
    for bounds in (None, (0.0,), (1.0, 1.0)):
        selected = replace(settings, resegment_range=bounds)
        result = extract(image, mask, settings=selected)
        assert result.features["original_firstorder_Mean"] == 1.0
        assert result.features["original_firstorder_Variance"] == 0.0
        with pytest.raises(ValueError, match="voxel extraction requires at least two"):
            prepare(image, mask, selected, voxel=True)
