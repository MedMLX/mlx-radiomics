"""Source-precision first-order moments at the public extraction boundary.

Expected values are computed independently in float64 from the source intensities, not
from the production moment arithmetic: this suite protects the mathematical definitions,
including variation that float32 alone collapses near large offsets.
"""

import math

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray

from mlxradiomics import Settings, extract
from mlxradiomics.contracts.types import HostArray
from mlxradiomics.workflows.engine import Result

MOMENT_NAMES = (
    "Mean",
    "Variance",
    "Skewness",
    "Kurtosis",
    "Minimum",
    "Maximum",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "10Percentile",
    "90Percentile",
    "Median",
    "InterquartileRange",
)
FEATURES = tuple(f"firstorder_{name}" for name in MOMENT_NAMES)
_INVARIANT_NAMES = (
    "Variance",
    "Skewness",
    "Kurtosis",
    "Range",
    "InterquartileRange",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
)


@pytest.mark.parametrize("scale", [1e-30, 1e-12, 1., 1e10, 1e13, 1e30, 1.5e38])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_standardized_moments_preserve_amplitude_scale(scale: float, asymmetric: bool) -> None:
    image = np.array([[-1., -1., 2.]] if asymmetric else [[-1., 1.]]) * scale
    result = extract(image, np.ones(image.shape, np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Skewness", "firstorder_Kurtosis"), bin_count=2))
    assert _roi_value(result, "Skewness") == pytest.approx(
        1 / math.sqrt(2) if asymmetric else 0, rel=2e-6, abs=1e-7)
    assert _roi_value(result, "Kurtosis") == pytest.approx(1.5 if asymmetric else 1, rel=2e-6)


def test_quantiles_keep_finite_opposite_sign_extremes() -> None:
    amplitude = float(np.float32(2e38))
    image = np.array([[-amplitude, amplitude]])
    result = extract(image, np.ones(image.shape, np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_10Percentile", "firstorder_Median", "firstorder_90Percentile",
                  "firstorder_InterquartileRange", "firstorder_Skewness", "firstorder_Kurtosis",
                  "firstorder_MeanAbsoluteDeviation"), bin_count=2))
    for name, expected in (("10Percentile", -0.8 * amplitude), ("Median", 0.),
                           ("90Percentile", 0.8 * amplitude), ("InterquartileRange", amplitude),
                           ("Skewness", 0.), ("Kurtosis", 1.),
                           ("MeanAbsoluteDeviation", amplitude)):
        assert _roi_value(result, name) == pytest.approx(expected, rel=1e-6, abs=0), name


@pytest.mark.parametrize("outlier, step", [(-10000.0, 0.0001), (-1e9, 1.0)])
@pytest.mark.parametrize("outlier_first", [True, False])
def test_robust_deviation_recenters_on_retained_values(
    outlier: float, step: float, outlier_first: bool,
) -> None:
    line = np.concatenate([[outlier], np.arange(10) * step])
    if not outlier_first:
        line = line[::-1].copy()
    image = line[None, :]
    result = extract(image, np.ones(image.shape, np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_RobustMeanAbsoluteDeviation",), bin_count=2))
    assert _roi_value(result, "RobustMeanAbsoluteDeviation") == pytest.approx(20 / 9 * step,
                                                                            rel=1e-6)


@pytest.mark.parametrize("line", [[3e38, 2e38], [-3e38, 1e38, 1e38]])
def test_mean_and_rms_stay_finite_when_energy_overflows(line: list[float]) -> None:
    # A representable mean and RMS must survive overflowing raw sums and squares.
    image = np.asarray([line], dtype=np.float64)
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Mean", "firstorder_RootMeanSquared"), bin_count=2))
    expected_mean = float(np.mean(image))
    expected_rms = float(np.sqrt(np.mean(image**2)))
    assert _roi_value(result, "Mean") == pytest.approx(expected_mean, rel=1e-6)
    assert _roi_value(result, "RootMeanSquared") == pytest.approx(expected_rms, rel=1e-6)


@pytest.mark.parametrize("permutation", [0, 1, 2], ids=["ordered", "reversed", "rolled"])
def test_robust_deviation_excludes_overflowing_opposite_sign_outlier(
    permutation: int,
) -> None:
    # The percentile bounds equal the repeated positive value, so the negative
    # outlier is outside the retained subset even though -3e38 - 1e38 overflows.
    line = np.array([-3e38] + [1e38] * 10, dtype=np.float32).astype(np.float64)
    image = line[None, :]
    if permutation == 1:
        image = image[:, ::-1].copy()
    elif permutation == 2:
        image = np.roll(image, 3, axis=1)
    result = extract(
        image,
        np.ones(image.shape, dtype=np.uint8),
        settings=settings_for(
        "pyradiomics",
            features=("firstorder_RobustMeanAbsoluteDeviation",), bin_count=2
        ),
    )
    assert _roi_value(result, "RobustMeanAbsoluteDeviation") == 0.0


@pytest.mark.parametrize("permutation", [0, 1, 2], ids=["ordered", "reversed", "rolled"])
def test_robust_deviation_retains_representable_extreme_mad(permutation: int) -> None:
    # The retained subset [-2e38, -1e38, 0, 1e38, 2e38] has a representable
    # deviation 1.2e38. Centering on -2e38 without scaling first formed +4e38.
    line = np.arange(-3, 4, dtype=np.float64) * 1e38
    image = line[None, :]
    if permutation == 1:
        image = image[:, ::-1].copy()
    elif permutation == 2:
        image = np.roll(image, 2, axis=1)
    result = extract(
        image,
        np.ones(image.shape, dtype=np.uint8),
        settings=settings_for(
        "pyradiomics",
            features=("firstorder_RobustMeanAbsoluteDeviation",), bin_count=2
        ),
    )
    assert _roi_value(result, "RobustMeanAbsoluteDeviation") == pytest.approx(
        1.2e38, rel=1e-6
    )


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            np.array([-3e38] + [1e38] * 10, dtype=np.float32).astype(np.float64),
            id="excluded-outlier",
        ),
        pytest.param(np.arange(-3, 4, dtype=np.float64) * 1e38, id="retained-spread"),
    ],
)
def test_robust_deviation_voxel_uses_scaled_subset(
    line: NDArray[np.float64],
) -> None:
    # The radius covers the full line at its middle centers, so the retained extremes
    # and a representable deviation are present. The scaled center and reduction must
    # agree with independent float64 evaluation at every voxel center, without an
    # overflowing opposite-sign span.
    image = line[None, :]
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = settings_for("pyradiomics",
        features=("firstorder_RobustMeanAbsoluteDeviation",),
        bin_count=2,
        kernel_radius=5,
    )
    result = extract(image, mask, settings=settings, mode="voxel")
    expected = _voxel_expected(image, radius=5)["RobustMeanAbsoluteDeviation"]
    actual = _voxel_map(result, "RobustMeanAbsoluteDeviation")
    _assert_map("RobustMeanAbsoluteDeviation", actual, expected)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("scale", [1000.0, 1e7])
@pytest.mark.parametrize("sign, name", [(1, "90Percentile"), (-1, "10Percentile")])
def test_percentile_coefficient_preserves_opposite_sign_cancellation(
    scale: float, sign: int, name: str,
) -> None:
    image = np.array([[-9 * scale, scale]]) * sign
    result = extract(image, np.ones(image.shape, np.uint8), settings=settings_for(
        "pyradiomics",
        features=(f"firstorder_{name}",), bin_count=2))
    # Float-float arithmetic has about 48 significant bits; cancellation leaves only
    # its low-part rounding, not the float32 error of the interpolation coefficient.
    assert abs(_roi_value(result, name)) <= scale * 1e-13


def _quantile(source: NDArray[np.float64], q: float) -> float:
    """Return the linear-interpolation quantile of an ordered-independent source."""
    ordered = np.sort(source)
    position = (ordered.size - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, ordered.size - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _first_order(values: HostArray) -> dict[str, float]:
    """Evaluate the first-order definitions with compensated float64 accumulation."""
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    count = source.size
    mean = math.fsum(source.tolist()) / count
    deviations = source - mean
    second = math.fsum((deviations * deviations).tolist()) / count
    third = math.fsum((deviations**3).tolist()) / count
    fourth = math.fsum((deviations**4).tolist()) / count
    lower = _quantile(source, 0.1)
    upper = _quantile(source, 0.9)
    robust = source[(source >= lower) & (source <= upper)]
    robust_mean = math.fsum(robust.tolist()) / robust.size
    return {
        "Mean": mean,
        "Variance": second,
        # Standardized moments are undefined for a constant set.
        "Skewness": math.nan if second == 0 else third / second**1.5,
        "Kurtosis": math.nan if second == 0 else fourth / second**2,
        "Minimum": float(source.min()),
        "Maximum": float(source.max()),
        "Range": float(source.max() - source.min()),
        "MeanAbsoluteDeviation": math.fsum(np.abs(deviations).tolist()) / count,
        "RobustMeanAbsoluteDeviation": (
            math.fsum(np.abs(robust - robust_mean).tolist()) / robust.size
        ),
        "10Percentile": lower,
        "90Percentile": upper,
        "Median": _quantile(source, 0.5),
        "InterquartileRange": _quantile(source, 0.75) - _quantile(source, 0.25),
    }


def _neighborhood(
    image: NDArray[np.float64],
    center: tuple[int, ...],
    radius: int,
) -> NDArray[np.float64]:
    """Return the in-bounds neighborhood values used by masked voxel extraction."""
    slices = tuple(
        slice(max(0, index - radius), min(size, index + radius + 1))
        for index, size in zip(center, image.shape, strict=True)
    )
    return image[slices]


def _voxel_expected(
    image: NDArray[np.float64],
    radius: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate every ROI center's neighborhood independently."""
    expected = {name: np.zeros(image.shape, dtype=np.float64) for name in MOMENT_NAMES}
    for index in np.ndindex(image.shape):
        values = _neighborhood(image, index, radius)
        for name, value in _first_order(values).items():
            expected[name][index] = value
    return expected


def _roi_value(result: Result, name: str) -> float:
    return float(np.asarray(result.features[f"original_firstorder_{name}"], np.float64).item())


def _voxel_map(result: Result, name: str) -> NDArray[np.float32]:
    return np.asarray(result.features[f"original_firstorder_{name}"], dtype=np.float32)


def _settings() -> Settings:
    return settings_for("pyradiomics", features=FEATURES, bin_width=1.0)


def _assert_scalar(name: str, actual: float, expected: float) -> None:
    if np.isnan(expected):
        assert np.isnan(actual), name
    elif name in {"Mean", "Minimum", "Maximum"}:
        # These features are reported as float32; the source expectation is quantized.
        assert actual == float(np.float32(expected)), f"{name}: {actual} != fl32({expected})"
    elif name == "Skewness":
        # Dimensionless; allow the float64 reference's representation-level residual.
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0, err_msg=name)


@pytest.mark.parametrize("name", MOMENT_NAMES)
def test_selected_moment_uses_only_sparse_roi_sources(name: str) -> None:
    # Excluded NaNs and large outliers must not enter compact samples or their sort.
    image = np.full((11, 13), np.nan)
    image[4:7, 4:7] = -1e30
    mask = np.zeros(image.shape, dtype=np.uint8)
    coordinates = ([1, 3, 5, 7, 9], [1, 4, 6, 8, 11])
    source = 1e8 + np.array([-5.0, -2.0, 1.0, 4.0, 7.0])
    image[coordinates] = source
    mask[coordinates] = 3
    result = extract(image.T, mask.T, settings=settings_for(
        "pyradiomics",
        label=3, features=(f"firstorder_{name}",), bin_count=2,
    ))
    _assert_scalar(name, _roi_value(result, name), _first_order(source)[name])


def _assert_map(name: str, actual: NDArray[np.float32], expected: NDArray[np.float64]) -> None:
    assert actual.dtype == np.float32
    if name in {"Mean", "Minimum", "Maximum"}:
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32), err_msg=name)
    elif name == "Skewness":
        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=name
        )
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=0.0, equal_nan=True, err_msg=name
        )


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_high_offset_small_variation_preserves_moments(voxel: bool) -> None:
    # Every source value rounds to the float32 value 1e8. The source population variance
    # is 2/3 and its standardized moments are defined; centering before reduction must
    # retain that variation instead of collapsing the row to a constant.
    line = np.array([1e8, 1e8 + 1, 1e8 + 2], dtype=np.float64)
    image = np.tile(line, (3, 1)) if voxel else line[None, :]
    mask = np.ones(image.shape, dtype=np.uint8)
    result = extract(image, mask, settings=_settings(), mode=("voxel" if voxel else "roi"))

    if voxel:
        expected = _voxel_expected(image)
        for name in MOMENT_NAMES:
            values = _voxel_map(result, name)
            _assert_map(name, values, expected[name])
            assert np.isfinite(values).all(), name
    else:
        source = image[mask == 1]
        expected = _first_order(source)
        for name in MOMENT_NAMES:
            value = _roi_value(result, name)
            _assert_scalar(name, value, expected[name])
            assert np.isfinite(value), name
        # The correctly rounded float32 mean is 1e8; the exact float64 mean is 1e8 + 1.
        assert _roi_value(result, "Mean") == float(np.float32(1e8 + 1))
        assert _roi_value(result, "Range") == 2.0
        assert _roi_value(result, "InterquartileRange") == 1.0


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_residual_only_variation_preserves_moments(voxel: bool) -> None:
    # These source values all convert to the float32 value 10000. Their entire spread is
    # carried by the source residual; variance, range and interquartile range must retain it.
    line = np.array([9999.99999, 10000.0, 10000.00001], dtype=np.float64)
    image = np.tile(line, (3, 1)) if voxel else line[None, :]
    mask = np.ones(image.shape, dtype=np.uint8)
    result = extract(image, mask, settings=_settings(), mode=("voxel" if voxel else "roi"))

    if voxel:
        expected = _voxel_expected(image)
        for name in MOMENT_NAMES:
            _assert_map(name, _voxel_map(result, name), expected[name])
        variance = _voxel_map(result, "Variance")
        ranges = _voxel_map(result, "Range")
        assert np.all(variance > 0)
        assert np.all(ranges > 0)
    else:
        expected = _first_order(image[mask == 1])
        for name in MOMENT_NAMES:
            _assert_scalar(name, _roi_value(result, name), expected[name])
        assert _roi_value(result, "Variance") > 0
        assert _roi_value(result, "Range") > 0
        assert _roi_value(result, "InterquartileRange") > 0


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_translation_does_not_change_dispersion_moments(voxel: bool) -> None:
    # Adding a constant changes the mean but not variance or standardized shape. The
    # translated values are not exactly representable in float32 at this offset, so the
    # residual must carry the original differences.
    line = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    base = np.tile(line, (5, 1)) if voxel else line[None, :]
    translated = base + 1e6
    mask = np.ones(base.shape, dtype=np.uint8)
    base_result = extract(base, mask, settings=_settings(), mode=("voxel" if voxel else "roi"))
    translated_result = extract(translated, mask, settings=_settings(),
        mode=("voxel" if voxel else "roi"))

    if voxel:
        expected = _voxel_expected(base)
        for name in _INVARIANT_NAMES:
            base_map = _voxel_map(base_result, name)
            _assert_map(name, base_map, expected[name])
            np.testing.assert_allclose(
                _voxel_map(translated_result, name),
                base_map,
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=name,
            )
    else:
        expected = _first_order(base[mask == 1])
        for name in _INVARIANT_NAMES:
            base_value = _roi_value(base_result, name)
            _assert_scalar(name, base_value, expected[name])
            translated_value = _roi_value(translated_result, name)
            np.testing.assert_allclose(
                translated_value, base_value, rtol=1e-6, atol=1e-9, err_msg=name
            )


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_zero_variance_standardized_moments_are_undefined(voxel: bool) -> None:
    image = np.full((3, 3), 42.5, dtype=np.float64)
    mask = np.ones(image.shape, dtype=np.uint8)
    result = extract(image, mask, settings=_settings(), mode=("voxel" if voxel else "roi"))

    if voxel:
        for name in (
            "Variance",
            "MeanAbsoluteDeviation",
            "RobustMeanAbsoluteDeviation",
            "Range",
            "InterquartileRange",
        ):
            assert np.all(_voxel_map(result, name) == 0), name
        assert np.isnan(_voxel_map(result, "Skewness")).all()
        assert np.isnan(_voxel_map(result, "Kurtosis")).all()
        assert result.metadata["feature_validity"]["original_firstorder_Skewness"]["nan"] == 9
        assert result.metadata["feature_validity"]["original_firstorder_Kurtosis"]["nan"] == 9
        assert (
            result.metadata["feature_validity"]["original_firstorder_Variance"]["valid_zero"] == 9
        )
    else:
        assert _roi_value(result, "Variance") == 0.0
        assert _roi_value(result, "MeanAbsoluteDeviation") == 0.0
        assert _roi_value(result, "RobustMeanAbsoluteDeviation") == 0.0
        assert _roi_value(result, "Range") == 0.0
        assert _roi_value(result, "InterquartileRange") == 0.0
        assert math.isnan(_roi_value(result, "Skewness"))
        assert math.isnan(_roi_value(result, "Kurtosis"))
        assert result.metadata["feature_validity"]["original_firstorder_Skewness"]["nan"] == 1
        assert result.metadata["feature_validity"]["original_firstorder_Kurtosis"]["nan"] == 1
        assert (
            result.metadata["feature_validity"]["original_firstorder_Variance"]["valid_zero"] == 1
        )


def test_quantiles_interpolate_source_values_before_float32_rounding() -> None:
    # Source median 1e8+4.5 rounds upward; interpolating the already rounded endpoints
    # (1e8 and 1e8+8) instead gives the tie 1e8+4 and rounds downward.
    image = np.array([[1e8 + 3, 1e8 + 6]], dtype=np.float64)
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=_settings())
    for name, q in (("10Percentile", 0.1), ("Median", 0.5), ("90Percentile", 0.9)):
        expected = float(np.float32(_quantile(image.reshape(-1), q)))
        assert _roi_value(result, name) == expected, name


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
@pytest.mark.parametrize("shift", [-1e8, -1e8 - 0.5])
def test_shifted_energy_preserves_source_and_shift_residuals(voxel: bool, shift: float) -> None:
    line = np.array([1e8, 1e8 + 1, 1e8 + 2], dtype=np.float64)
    image = np.tile(line, (3, 1)) if voxel else line[None, :]
    mask = np.ones(image.shape, dtype=np.uint8)
    names = ("Energy", "TotalEnergy", "RootMeanSquared")
    spacing = (2.0, 3.0)
    settings = settings_for("pyradiomics",
        features=tuple(f"firstorder_{name}" for name in names),
        bin_width=1.0,
        voxel_array_shift=shift,
    )
    result = extract(image, mask, spacing=spacing, settings=settings,
        mode=("voxel" if voxel else "roi"))
    centers = np.ndindex(image.shape) if voxel else [None]
    for center in centers:
        source = image if center is None else _neighborhood(image, center, 1)
        shifted = source + shift
        energy = math.fsum((shifted**2).reshape(-1).tolist())
        expected = (energy, energy * math.prod(spacing), math.sqrt(energy / source.size))
        for name, target in zip(names, expected, strict=True):
            actual = (
                _roi_value(result, name) if center is None else _voxel_map(result, name)[center]
            )
            np.testing.assert_allclose(actual, target, rtol=1e-6, atol=0, err_msg=name)


@pytest.mark.parametrize("count", [0, 1000])
def test_energy_keeps_squares_below_the_normal_float32_range(count: int) -> None:
    # Each 1e-19 square (1e-38) is below 2**-126 ~ 1.18e-38 and is flushed to zero on Metal,
    # but their sum is a normal float32: Energy = 4e-38 + count * 1e-38 by hand.
    image = np.array([[2e-19] + [1e-19] * count], dtype=np.float64)
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Energy",), bin_count=1,
    ))
    expected = 4e-38 + count * 1e-38
    np.testing.assert_allclose(_roi_value(result, "Energy"), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
@pytest.mark.parametrize("shift", [-1e39, 1e39])
def test_rms_preserves_infinity_for_unrepresentable_shift(voxel: bool, shift: float) -> None:
    image = np.zeros((2, 2), dtype=np.float64)
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_RootMeanSquared",), bin_count=1, voxel_array_shift=shift,
    ), mode=("voxel" if voxel else "roi"))
    values = _voxel_map(result, "RootMeanSquared") if voxel else _roi_value(
        result, "RootMeanSquared"
    )
    assert np.isposinf(values).all()


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
@pytest.mark.parametrize("name", ["Energy", "TotalEnergy", "RootMeanSquared"])
def test_shifted_energy_refuses_pair_unresolved_cancellation(voxel: bool, name: str) -> None:
    # The two float64 sources differ by one ulp but share one float32 value-plus-residual
    # pair; the shift cancels that pair exactly. Their true Energy is the representable
    # 1.925929944387236e-34, but it cannot be recovered from the rounded pair, so reporting
    # zero (or any inferred value) is refused.
    line = np.array([0.1, np.nextafter(0.1, np.inf)], dtype=np.float64)
    image = np.tile(line, (2, 1)) if voxel else line[None, :]
    with pytest.raises(RuntimeError, match="unresolved"):
        extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
            features=(f"firstorder_{name}",), bin_count=1, voxel_array_shift=-0.1,
        ), mode=("voxel" if voxel else "roi"))


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_shifted_energy_and_rms_classify_finite_sum_overflow_as_infinity(
    voxel: bool,
) -> None:
    # Both operands are finite float32 values, but source + shift overflows the float32
    # range. The high-part difference is already infinite, so the residual compensation
    # must preserve the positive-infinite classification instead of producing NaN.
    x = float(np.float32(2e38))
    line = np.array([x, x], dtype=np.float64)
    image = np.tile(line, (2, 1)) if voxel else line[None, :]
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Energy", "firstorder_RootMeanSquared"),
        bin_count=1,
        voxel_array_shift=x,
    ), mode=("voxel" if voxel else "roi"))
    for name in ("Energy", "RootMeanSquared"):
        values = _voxel_map(result, name) if voxel else _roi_value(result, name)
        assert np.isposinf(values).all(), name


@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "voxel"])
def test_coefficient_of_variation_survives_cancelling_mean(voxel: bool) -> None:
    # The exact mean is 1/3 and the population standard deviation is sqrt(2/3)*1e8.
    # A plain float32 tree sums 1e8 + 1 - 1e8 to zero and reports infinite CV.
    line = np.array([1e8, 1.0, -1e8], dtype=np.float64)
    image = np.tile(line, (3, 1)) if voxel else line[None, :]
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Mean", "firstorder_CoefficientOfVariation"), bin_count=2,
    ), mode=("voxel" if voxel else "roi"))
    expected_cv = math.sqrt(2 / 3) * 1e8 / (1 / 3)
    if voxel:
        # Only the centre column's radius-1 neighbourhoods hold the three columns in
        # equal proportion; edge neighbourhoods are clipped to two columns.
        np.testing.assert_allclose(
            _voxel_map(result, "Mean")[:, 1], float(np.float32(1 / 3)), rtol=1e-6, atol=0,
        )
        np.testing.assert_allclose(
            _voxel_map(result, "CoefficientOfVariation")[:, 1], expected_cv, rtol=1e-6,
        )
    else:
        assert _roi_value(result, "Mean") == float(np.float32(1 / 3))
        np.testing.assert_allclose(
            _roi_value(result, "CoefficientOfVariation"), expected_cv, rtol=1e-6,
        )


def test_true_zero_mean_keeps_infinite_coefficient_of_variation() -> None:
    # Exactly opposite representable sources still mean zero; the compensated reduction
    # must not turn that valid classification into an error or a finite value.
    image = np.array([[-1e8, 1e8]], dtype=np.float64)
    result = extract(image, np.ones(image.shape, dtype=np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Mean", "firstorder_CoefficientOfVariation"), bin_count=2,
    ))
    assert _roi_value(result, "Mean") == 0.0
    assert np.isposinf(_roi_value(result, "CoefficientOfVariation"))


def _direct_first_order(
    line: NDArray[np.float64],
    selected: tuple[str, ...],
    *,
    conventions: Conventions = "pyradiomics",
    kurtosis: str | None = None,
    zero_variance_moments: str | None = None,
) -> dict[str, float]:
    """Evaluate ``first_order`` on one line without the extraction engine."""
    from mlxradiomics.features.first_order import first_order
    from mlxradiomics.numerics.precision import pair_representation_error
    from mlxradiomics.numerics.runtime import array_from_bool, mx

    settings = settings_for(conventions)
    source = np.asarray(line, dtype=np.float64).reshape(1, -1)
    values = source.astype(np.float32)
    residual = (source - values.astype(np.float64)).astype(np.float32)
    result = first_order(
        mx.array(values),
        array_from_bool(np.ones(source.shape, dtype=np.bool_)),
        mx.array(np.ones(source.shape, dtype=np.int32)),
        1,
        shift=0.0,
        voxel_volume=1.0,
        residual=mx.array(residual),
        source_bits=mx.array(source.view(np.uint64)),
        representation_error=mx.array(pair_representation_error(source)),
        selected=selected,
        kurtosis=settings.kurtosis if kurtosis is None else kurtosis,
        zero_variance_moments=(
            settings.zero_variance_moments if zero_variance_moments is None
            else zero_variance_moments
        ),
    )
    return {
        name: float(np.asarray(value, dtype=np.float32).item())
        for name, value in result.items()
    }


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_quartile_dispersion_preserves_representable_ratio_at_extreme_scale(sign: float) -> None:
    observed = _direct_first_order(
        sign * np.array([[2e38, 3e38]]), ("QuartileCoefficientOfDispersion",)
    )
    # Ordered quartiles have a gap 0.5e38 and a sum +-5e38. The sum overflows
    # float32, but the ratio is finite and independent of the intensity scale.
    assert observed["QuartileCoefficientOfDispersion"] == pytest.approx(sign * 0.1, rel=1e-6)


def test_high_offset_median_absolute_deviation() -> None:
    observed = _direct_first_order(
        np.array([[1e8, 1e8 + 1, 1e8 + 2]]), ("MedianAbsoluteDeviation",)
    )
    # Median offset 1; distances 1, 0, 1.  Absolute indices alone round to 1e8.
    assert observed["MedianAbsoluteDeviation"] == pytest.approx(2 / 3, rel=1e-6)


def test_first_order_conventions_change_kurtosis_and_constant_moments() -> None:
    native = _direct_first_order(np.array([[1.0, 2.0, 3.0]]), ("Kurtosis",))
    ibsi = _direct_first_order(
        np.array([[1.0, 2.0, 3.0]]), ("Kurtosis",), conventions="ibsi"
    )
    assert native["Kurtosis"] == pytest.approx(1.5, rel=1e-6)
    assert ibsi["Kurtosis"] == pytest.approx(-1.5, rel=1e-6)

    constant_native = _direct_first_order(
        np.array([[2.0, 2.0, 2.0]]), ("Skewness", "Kurtosis")
    )
    constant_ibsi = _direct_first_order(
        np.array([[2.0, 2.0, 2.0]]), ("Skewness", "Kurtosis"), conventions="ibsi"
    )
    assert math.isnan(constant_native["Skewness"])
    assert math.isnan(constant_native["Kurtosis"])
    assert constant_ibsi == {"Skewness": 0.0, "Kurtosis": 0.0}


def test_requested_spread_refuses_collapsed_source_pair() -> None:
    # The two source values differ by one float64 ulp, but both upload as the same
    # float32 value plus residual pair.  Range and variance are representable in
    # float32, so silently reporting zero is not acceptable.
    line = np.array([[0.1, np.nextafter(0.1, np.inf)]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="unresolved"):
        _direct_first_order(line, ("Range",))
    with pytest.raises(RuntimeError, match="unresolved"):
        _direct_first_order(line, ("Variance",))

    observed = _direct_first_order(line, ("Mean", "Minimum", "Entropy"))
    expected = float(np.float32(0.1))
    assert observed == {"Mean": expected, "Minimum": expected, "Entropy": 0.0}


@pytest.mark.parametrize("line", [
    [-0.1, np.nextafter(-0.1, -np.inf)],
    [0.0, 1e-100],
    [-1e-100, 1e-100],
])
def test_spread_refusal_covers_negative_and_underflow_collisions(line: list[float]) -> None:
    with pytest.raises(RuntimeError, match="unresolved"):
        _direct_first_order(np.array([line]), ("Skewness",))


def test_shared_pair_inside_a_resolved_spread_is_accepted() -> None:
    from radiomics_reference.catalogue import firstorder

    # Two sources share one float32 pair, but their one-ulp difference is negligible
    # against the 0.1 spread. Large float ROIs hit such collisions by chance, so they
    # must not refuse an accurately computed spread.
    line = np.array([[0.1, np.nextafter(0.1, np.inf), 0.2]])
    names = ("Variance", "Skewness", "Kurtosis", "Range", "MeanAbsoluteDeviation")
    observed = _direct_first_order(line, names)
    ones = np.ones(line.shape, dtype=np.bool_)
    expected = firstorder(line, ones.astype(np.int32), ones, 0.0, 1.0)
    for name in names:
        assert observed[name] == pytest.approx(expected[name], rel=1e-6), name


def test_dimensional_underflow_raises_instead_of_reporting_zero() -> None:
    # Variance and Energy of 1e-20-scale intensities lie near 1e-40, below the normal
    # float32 range; Metal would report 0.0, which reads as constant data.
    line = 1e-20 * np.array([[0.0, 1.0, 2.0, 3.0]])
    for name in ("Variance", "Energy"):
        with pytest.raises(RuntimeError, match="underflow"):
            _direct_first_order(line, (name,))
    # Dimensionless and representable outputs of the same data remain available.
    observed = _direct_first_order(line, ("Range", "Skewness"))
    assert observed["Range"] == pytest.approx(3e-20, rel=1e-6)
    assert observed["Skewness"] == pytest.approx(0.0, abs=1e-6)


def test_float32_subnormal_sources_are_refused() -> None:
    # Metal flushes these to zero, so even a Mean-only request would report 0.0.
    line = np.array([[1e-39, 2e-39, 3e-39]])
    with pytest.raises(RuntimeError, match="flushed"):
        _direct_first_order(line, ("Mean",))
    assert _direct_first_order(line, ("Entropy",)) == {"Entropy": 0.0}


@pytest.mark.parametrize("line, name, expected", [
    ([1e-40, 1.0, 2.0], "Maximum", 2.0),
    ([-2.0, -1.0, -1e-40], "Minimum", -2.0),
    ([-1e-40, 0.0], "Maximum", 0.0),
    ([0.0, 1e-40], "Minimum", 0.0),
])
def test_extrema_ignore_unrelated_subnormal_sources(
    line: list[float], name: str, expected: float,
) -> None:
    image = np.array([line])
    result = extract(image, np.ones(image.shape, np.uint8),
                     settings=settings_for("pyradiomics", features=(f"firstorder_{name}",)))
    assert _roi_value(result, name) == expected


@pytest.mark.parametrize("line, name", [
    ([1e-40, 1.0], "Minimum"),
    ([-1.0, -1e-40], "Maximum"),
])
def test_selected_subnormal_extremum_is_not_reported_as_zero(
    line: list[float], name: str,
) -> None:
    with pytest.raises(RuntimeError, match="flushed"):
        _direct_first_order(np.array([line]), (name,))


def test_spread_refuses_inexact_pairs_without_collision() -> None:
    # Near 1e12 the float32 residual (up to 2**15) needs more than 24 bits to hold
    # source differences of 1e-3, so the pairs stay distinct but round those differences.
    line = 1e12 + 1e-3 * np.array([[0.0, 3.0, 1.0, 4.0, 1.0, 2.0, 0.0, 4.0]])
    with pytest.raises(RuntimeError, match="unresolved"):
        _direct_first_order(line, ("Variance",))
    # Equal sources share one inexact pair; their zero spread is exact and accepted.
    constant = _direct_first_order(np.array([[0.1, 0.1, 0.1]]), ("Variance", "Range"))
    assert constant == {"Variance": 0.0, "Range": 0.0}


def test_rejections_keep_their_precedence_when_several_apply() -> None:
    # All checks complete in one readback; a sample failing several still reports the
    # first in order: subnormal source, unresolved spread, unresolved shifted energy.
    inexact = 1e12 + 1e-3 * np.array([[0.0, 3.0, 1.0, 4.0, 1.0, 2.0]])
    with pytest.raises(RuntimeError, match="flushed"):
        _direct_first_order(np.concatenate([[[1e-40]], inexact], axis=1), ("Variance",))
    with pytest.raises(RuntimeError, match="flushed"):
        _direct_first_order(1e-20 * np.array([[1e-20, 1.0, 2.0]]), ("Variance",))
    mask = np.ones(inexact.shape, np.uint8)
    for names, message in [(("Variance", "Energy"), "source spread"), (("Energy",), "shifted")]:
        settings = settings_for(
            "pyradiomics", features=tuple(f"firstorder_{name}" for name in names),
            bin_count=2, voxel_array_shift=-1e12)
        with pytest.raises(RuntimeError, match=message):
            extract(inexact, mask, settings=settings)


def test_signed_source_zeros_do_not_trigger_spread_refusal() -> None:
    observed = _direct_first_order(
        np.array([[-0.0, 0.0]]), ("Variance", "Range", "Skewness"), conventions="ibsi"
    )
    assert observed == {"Variance": 0.0, "Range": 0.0, "Skewness": 0.0}


def test_first_order_selected_and_convention_validation() -> None:
    observed = _direct_first_order(np.array([[1.0, 2.0, 3.0]]), ("Mean",))
    assert observed == {"Mean": 2.0}
    with pytest.raises(ValueError, match="unknown first-order"):
        _direct_first_order(np.array([[1.0]]), ("NotAFeature",))
    with pytest.raises(ValueError, match="kurtosis"):
        _direct_first_order(np.array([[1.0]]), ("Mean",), kurtosis="other")
    with pytest.raises(ValueError, match="zero_variance_moments"):
        _direct_first_order(np.array([[1.0]]), ("Mean",), zero_variance_moments="other")


def test_extended_first_order_features_through_public_extract() -> None:
    image = np.array([[1.0, 2.0, 10.0]])
    result = extract(
        image,
        np.ones(image.shape, dtype=np.uint8),
        settings=settings_for(
        "pyradiomics",
            features=(
                "firstorder_MedianAbsoluteDeviation",
                "firstorder_CoefficientOfVariation",
                "firstorder_QuartileCoefficientOfDispersion",
            ),
            bin_width=1.0,
        ),
    )
    assert _roi_value(result, "MedianAbsoluteDeviation") == pytest.approx(3.0, rel=1e-6)
    assert _roi_value(result, "CoefficientOfVariation") == pytest.approx(
        math.sqrt(146) / 13, rel=1e-6
    )
    assert _roi_value(result, "QuartileCoefficientOfDispersion") == pytest.approx(
        0.6, rel=1e-6
    )


def test_range_and_iqr_beyond_float32_range_are_positive_infinity() -> None:
    # Exact Range and IQR are 2 * float32(3e38), beyond the float32 range: they report
    # +inf rather than NaN from inf - inf in the compensation. A finite opposite-sign
    # span that fits stays exact.
    a = float(np.float32(3e38))
    names = ("firstorder_Range", "firstorder_InterquartileRange")
    image = np.array([[-a, -a, a, a]])
    result = extract(image, np.ones(image.shape, np.uint8),
                     settings=settings_for("pyradiomics", features=names, bin_count=4))
    for name in ("Range", "InterquartileRange"):
        assert np.isposinf(_roi_value(result, name)), name
    image = np.array([[-a / 2, -a / 2, a / 2, a / 2]])
    result = extract(image, np.ones(image.shape, np.uint8),
                     settings=settings_for("pyradiomics", features=names, bin_count=4))
    for name in ("Range", "InterquartileRange"):
        assert _roi_value(result, name) == a, name


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (1, 2, 0, 3), (3, 0, 2, 1)])
def test_mean_keeps_a_small_residual_beside_cancelling_residuals(
    order: tuple[int, ...],
) -> None:
    # Every source is an exact float32 value plus residual; the exact mean is 2**-28 and
    # CV = sqrt(variance) / mean. Reducing values and residuals separately would give zero.
    line = np.array([1e8 + 1, 1e8 + 2.0**-26, -1e8 - 1, -1e8])[list(order)]
    result = extract(line[None, :], np.ones((1, 4), np.uint8), settings=settings_for(
        "pyradiomics",
        features=("firstorder_Mean", "firstorder_CoefficientOfVariation"), bin_count=2))
    assert _roi_value(result, "Mean") == 2.0**-28
    assert _roi_value(result, "CoefficientOfVariation") == pytest.approx(
        float(np.std(line)) / 2.0**-28, rel=1e-6)


@pytest.mark.parametrize("value,spacing", [(1e20, 0.001), (1e-20, 1000.0)])
def test_total_energy_is_finite_when_only_energy_leaves_the_float32_range(
    value: float, spacing: float,
) -> None:
    sample = float(np.float32(value))
    image = np.array([[value, value]])
    result = extract(image, np.ones(image.shape, np.uint8), spacing=(spacing, spacing),
                     settings=settings_for(
                         "pyradiomics", features=("firstorder_TotalEnergy",), bin_count=1))
    expected = 2 * sample * sample * spacing * spacing
    assert _roi_value(result, "TotalEnergy") == pytest.approx(expected, rel=1e-6)
