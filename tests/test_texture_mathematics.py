"""Analytic GLRLM, GLDZM, GLDM and NGTDM mathematics including sparse cases."""

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.mathematics import (
    GLRLM_NAMES,
    NGTDM_NAMES,
    glrlm_features,
    ngtdm_features,
)
from radiomics_reference.real_images import hcc_background, hcc_patch

from mlxradiomics import Settings, extract

_GLRLM_FEATURES = tuple(f"glrlm_{name}" for name in GLRLM_NAMES)
_NGTDM_FEATURES = tuple(f"ngtdm_{name}" for name in NGTDM_NAMES)
_SINGLETON_3D = np.zeros((3, 3, 3), dtype=bool)
_SINGLETON_3D[1, 1, 1] = True


@pytest.mark.parametrize("radius", [1, 3])
@pytest.mark.parametrize("anchor", [0., -254.])
def test_ngtdm_neighborhood_difference_is_gray_shift_invariant(radius: int, anchor: float) -> None:
    image = hcc_background((4 * radius + 1,) * 3) / -2048
    image[(2 * radius,) * 3] = 0
    result = extract(image, np.ones(image.shape, np.uint8), settings=Settings(
        features=("ngtdm_Coarseness", "ngtdm_Strength"), bin_width=1,
        bin_anchor=anchor, distances=(radius,)))
    # The low center contributes one; K high neighbors each contribute exactly 1/K.
    for name in ("Coarseness", "Strength"):
        assert result.features[f"original_ngtdm_{name}"] == pytest.approx(1, rel=2e-6)


@pytest.mark.parametrize("large", [False, True])
def test_busyness_zero_classification_uses_integer_occurrences(large: bool) -> None:
    if large:
        # Nearest-neighbor expansion of adjacent source levels gives unequal
        # populations whose weighted occurrence products nearly cancel.
        source = hcc_patch((1, 2))[0] - 70 + 254
        image = np.repeat(source, [128001, 127501], axis=1)
    else:
        image = (hcc_patch((1, 6))[0] > 90).astype(np.float64) * 4
    result = extract(image, np.ones(image.shape, np.uint8), settings=Settings(
        features=("ngtdm_Busyness",), bin_width=1, bin_anchor=0))
    value = result.features["original_ngtdm_Busyness"]
    if large:
        assert value == pytest.approx(63875.5, rel=1e-6)
    else:
        assert np.isposinf(value)


def test_busyness_retains_occurrences_above_float32_integer_precision() -> None:
    from mlxradiomics.features.distributions import ngtdm_statistics
    from mlxradiomics.numerics.runtime import mx

    # 1*n1 and 2*n2 differ by exactly one, although float32 rounds n1 down.
    n = mx.array([[2**25 + 1, 2**24]], dtype=mx.uint32)
    s = mx.array([[0.5, 0.5]], dtype=mx.float32)
    actual = ngtdm_statistics(n, s)["Busyness"].item()
    assert actual == pytest.approx((3 * 2**24 + 1) / 4, rel=1e-6)


def _compare(
    features: Mapping[str, float | NDArray[np.float64]],
    family: str,
    expected: Mapping[str, float],
) -> dict[str, float]:
    actual: dict[str, float] = {}
    for name, expected_value in expected.items():
        value = features[f"original_{family}_{name}"]
        assert isinstance(value, float)
        actual[name] = value
        if np.isnan(expected_value):
            assert np.isnan(value), name
        elif np.isposinf(expected_value):
            assert np.isposinf(value), name
        elif np.isneginf(expected_value):
            assert np.isneginf(value), name
        else:
            np.testing.assert_allclose(
                value, expected_value, rtol=1e-5, atol=1e-6, err_msg=name
            )
    return actual


def _ngtdm_settings(*, bin_width: float = 1.0) -> Settings:
    return settings_for("pyradiomics", bin_width=bin_width, features=_NGTDM_FEATURES)


@pytest.mark.parametrize(
    ("mask", "sre"),
    [
        pytest.param(np.eye(2, dtype=bool), 13 / 16, id="2x2-diagonal"),
        pytest.param(np.eye(3, dtype=bool), 7 / 9, id="3x3-diagonal"),
        pytest.param(np.ones((1, 3), dtype=bool), 7 / 9, id="length-one-axis"),
        pytest.param(np.ones((1, 1), dtype=bool), 1.0, id="one-voxel-image"),
        pytest.param(_SINGLETON_3D, 1.0, id="3d-singleton"),
        pytest.param(
            np.array([[True, False, True], [False, False, False], [False, True, False]]),
            1.0,
            id="scattered-singletons",
        ),
    ],
)
def test_glrlm_singleton_runs_keep_every_available_direction(
    mask: NDArray[np.bool_], sre: float,
) -> None:
    image = hcc_background(mask.shape) * 0
    bins = np.ones(mask.shape, dtype=np.int32)
    settings = Settings(features=_GLRLM_FEATURES)
    result = extract(image, mask.astype(np.uint8), settings=settings)
    assert set(result.features) == {f"original_{name}" for name in _GLRLM_FEATURES}
    actual = _compare(result.features, "glrlm", glrlm_features(bins, mask))
    np.testing.assert_allclose(actual["ShortRunEmphasis"], sre, rtol=1e-6, atol=0)
    assert result.metadata["feature_validity"]["original_glrlm_ShortRunEmphasis"] == {
        "finite": 1, "valid_zero": 0, "nan": 0, "positive_infinity": 0,
        "negative_infinity": 0, "excluded": 0,
    }


def test_glrlm_voxel_singleton_runs_match_local_enumeration() -> None:
    mask = np.eye(2, dtype=bool)
    image = hcc_background(mask.shape) * 0
    settings = Settings(features=_GLRLM_FEATURES)
    result = extract(image, mask.astype(np.uint8), settings=settings, mode="voxel")
    # Each 3x3 clipped neighborhood holds the same two diagonal valid voxels.
    local_mask = np.zeros((3, 3), dtype=bool)
    local_mask[np.arange(2), np.arange(2)] = True
    expected = glrlm_features(np.ones((3, 3), dtype=np.int32), local_mask)
    for name, expected_value in expected.items():
        value = result.features[f"original_glrlm_{name}"]
        assert isinstance(value, np.ndarray)
        background = ~mask
        np.testing.assert_allclose(value[mask], expected_value, rtol=1e-5, atol=1e-6,
                                   err_msg=name)
        assert np.all(value[background] == 0), name


def test_ngtdm_busyness_positive_over_zero_is_infinite() -> None:
    image = (hcc_patch((1, 3))[0] > 90).astype(np.float64)
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = _ngtdm_settings()
    result = extract(image, mask, settings=settings)
    expected = ngtdm_features(np.array([[1, 1, 2]], dtype=np.int32), mask.astype(bool))
    actual = _compare(result.features, "ngtdm", expected)
    # n=[2,1], s=[0.5,1]: both occupied levels have equal i*p_i, so Busyness is
    # positive-over-zero, while the other features stay finite.
    assert actual["Coarseness"] == pytest.approx(1.5)
    assert actual["Contrast"] == pytest.approx(1 / 9)
    assert np.isposinf(actual["Busyness"])
    assert actual["Complexity"] == pytest.approx(4 / 9)
    assert actual["Strength"] == pytest.approx(4 / 3)
    validity = result.metadata["feature_validity"]
    assert validity["original_ngtdm_Busyness"]["positive_infinity"] == 1
    assert validity["original_ngtdm_Busyness"]["finite"] == 0
    for name in ("Coarseness", "Contrast", "Complexity", "Strength"):
        assert validity[f"original_ngtdm_{name}"]["finite"] == 1


def test_ngtdm_excludes_isolated_centers_while_gldm_counts_them() -> None:
    image = (hcc_patch((1, 6))[0] > 90).astype(np.float64)
    mask = np.array([[1, 1, 1, 0, 0, 1]], dtype=np.uint8)
    settings = Settings(
        bin_width=1.0,
        features=(
            "gldm_DependenceNonUniformity", "gldm_SmallDependenceEmphasis",
            *_NGTDM_FEATURES,
        ),
    )
    result = extract(image, mask, settings=settings)
    bins = np.array([[1, 1, 2, 0, 0, 1]], dtype=np.int32)
    expected = ngtdm_features(bins, mask.astype(bool))
    actual = _compare(result.features, "ngtdm", expected)
    # GLDM keeps the isolated level-1 center as dependence 1; NGTDM drops it, so
    # Busyness remains positive-over-zero instead of becoming finite.
    gldm = result.features
    dnu = gldm["original_gldm_DependenceNonUniformity"]
    sde = gldm["original_gldm_SmallDependenceEmphasis"]
    assert isinstance(dnu, float)
    assert isinstance(sde, float)
    np.testing.assert_allclose(dnu, 2.0, rtol=0, atol=0)
    np.testing.assert_allclose(sde, 0.625, rtol=0, atol=0)
    assert np.isposinf(actual["Busyness"])


def test_ngtdm_constant_region_singularities() -> None:
    image = hcc_background((2, 2)) * 0
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = _ngtdm_settings(bin_width=25.0)
    result = extract(image, mask, settings=settings)
    expected = ngtdm_features(np.ones((2, 2), dtype=np.int32), mask.astype(bool))
    actual = _compare(result.features, "ngtdm", expected)
    assert np.isposinf(actual["Coarseness"])
    assert actual["Contrast"] == 0
    assert np.isnan(actual["Busyness"])
    assert actual["Complexity"] == 0
    assert np.isnan(actual["Strength"])
    validity = result.metadata["feature_validity"]
    assert validity["original_ngtdm_Coarseness"] == {
        "finite": 0, "valid_zero": 0, "nan": 0, "positive_infinity": 1,
        "negative_infinity": 0, "excluded": 0,
    }
    assert validity["original_ngtdm_Contrast"] == {
        "finite": 1, "valid_zero": 1, "nan": 0, "positive_infinity": 0,
        "negative_infinity": 0, "excluded": 0,
    }
    assert validity["original_ngtdm_Busyness"]["nan"] == 1
    assert validity["original_ngtdm_Complexity"] == {
        "finite": 1, "valid_zero": 1, "nan": 0, "positive_infinity": 0,
        "negative_infinity": 0, "excluded": 0,
    }
    assert validity["original_ngtdm_Strength"]["nan"] == 1


def test_ngtdm_disjoint_constant_pairs_strength_is_infinite() -> None:
    image = (hcc_patch((3, 3))[0] < 65).astype(np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, :2] = 1
    mask[2, :2] = 1
    bins = np.zeros(image.shape, dtype=np.int32)
    bins[0, :2] = 1
    bins[2, :2] = 2
    settings = _ngtdm_settings()
    result = extract(image, mask, settings=settings)
    actual = _compare(result.features, "ngtdm", ngtdm_features(bins, mask.astype(bool)))
    assert np.isposinf(actual["Coarseness"])
    assert actual["Contrast"] == 0
    assert actual["Busyness"] == 0
    assert actual["Complexity"] == 0
    assert np.isposinf(actual["Strength"])
    validity = result.metadata["feature_validity"]
    assert validity["original_ngtdm_Coarseness"]["positive_infinity"] == 1
    assert validity["original_ngtdm_Strength"]["positive_infinity"] == 1


def test_ngtdm_no_eligible_centers_is_undefined() -> None:
    image = hcc_patch((3, 3))[0]
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1, 1] = 1
    settings = _ngtdm_settings()
    result = extract(image, mask, settings=settings)
    expected = ngtdm_features(np.ones((1, 1), dtype=np.int32), np.ones((1, 1), dtype=bool))
    actual = _compare(result.features, "ngtdm", expected)
    assert all(np.isnan(value) for value in actual.values())
    validity = result.metadata["feature_validity"]
    for name in _NGTDM_FEATURES:
        assert validity[f"original_{name}"]["nan"] == 1
        assert validity[f"original_{name}"]["finite"] == 0


def test_ngtdm_voxel_mixes_defined_and_undefined_centers() -> None:
    image = np.clip(hcc_patch((3, 5))[0] - 70, 0, 1)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[2, 4] = 1
    local_bins = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]], dtype=np.int32)
    local_mask = np.zeros((3, 3), dtype=bool)
    local_mask[1, 1] = True
    local_mask[1, 2] = True
    expected = ngtdm_features(local_bins, local_mask)
    result = extract(image, mask, settings=_ngtdm_settings(), mode="voxel")
    for name, expected_value in expected.items():
        value = result.features[f"original_ngtdm_{name}"]
        assert isinstance(value, np.ndarray)
        for center in ((0, 0), (0, 1)):
            np.testing.assert_allclose(value[center], expected_value, rtol=1e-5, atol=1e-6,
                                       err_msg=name)
        assert np.isnan(value[2, 4]), name
        assert np.count_nonzero(np.isnan(value)) == 1, name
        assert np.all(value[mask == 0] == 0), name
        assert result.metadata["feature_validity"][f"original_ngtdm_{name}"] == {
            "finite": 2, "valid_zero": 2 if expected_value == 0 else 0, "nan": 1,
            "positive_infinity": 0, "negative_infinity": 0, "excluded": 12,
        }


def test_ngtdm_voxel_no_eligible_centers_is_undefined() -> None:
    image = hcc_patch((3, 5))[0]
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 4] = 1
    settings = _ngtdm_settings()
    result = extract(image, mask, settings=settings, mode="voxel")
    for name in _NGTDM_FEATURES:
        value = result.features[f"original_{name}"]
        assert isinstance(value, np.ndarray)
        assert np.isnan(value[mask.astype(bool)]).all(), name
        assert np.all(value[mask == 0] == 0), name
        assert result.metadata["feature_validity"][f"original_{name}"] == {
            "finite": 0, "valid_zero": 0, "nan": 2, "positive_infinity": 0,
            "negative_infinity": 0, "excluded": 13,
        }


def _count_entropy_bits(values: NDArray[np.int64]) -> float:
    total = float(values.sum())
    return -math.fsum(
        float(count) / total * math.log2(float(count) / total)
        for count in values.reshape(-1)
        if count > 0
    )


@pytest.mark.parametrize(
    ("family", "name"),
    [
        pytest.param("glrlm", "RunVariance", id="run"),
        pytest.param("glszm", "ZoneVariance", id="size-zone"),
        pytest.param("gldzm", "ZoneDistanceVariance", id="distance-zone"),
    ],
)
def test_size_variance_preserves_large_integer_offsets(family: str, name: str) -> None:
    from mlxradiomics.features.distributions import size_statistics
    from mlxradiomics.numerics.runtime import mx

    # Sizes 524288 (twice) and 524289 (once): exact population variance 2/9.  Both
    # absolute indices have the same float32 representation, so variance must be formed
    # from integer offsets before conversion.
    counts = np.zeros((1, 1, 524_289), dtype=np.uint32)
    counts[0, 0, 524_287] = 2
    counts[0, 0, 524_288] = 1
    options = {"voxel_count": mx.array([3], dtype=mx.int32)} if family == "gldzm" else {}
    actual = size_statistics(mx.array(counts), family, (name,), **options)[name]
    value = float(np.asarray(actual, dtype=np.float32).item())
    assert value == pytest.approx(2 / 9, rel=1e-6)


def test_gray_level_variance_preserves_large_integer_offsets() -> None:
    from mlxradiomics.features.distributions import size_statistics
    from mlxradiomics.numerics.runtime import mx

    # The same correction applies to the gray-level marginal: levels 524288 and 524289
    # collapse to one float32 index, but their exact variance is 2/9.
    counts = np.zeros((1, 524_289, 1), dtype=np.uint32)
    counts[0, 524_287, 0] = 2
    counts[0, 524_288, 0] = 1
    actual = size_statistics(mx.array(counts), "gldm", ("GrayLevelVariance",))[
        "GrayLevelVariance"
    ]
    assert float(np.asarray(actual, dtype=np.float32).item()) == pytest.approx(
        2 / 9, rel=1e-6
    )


def test_gldzm_matches_independent_matrix_formulas() -> None:
    from radiomics_reference.catalogue import size_features

    from mlxradiomics.features.distributions import size_statistics
    from mlxradiomics.numerics.runtime import mx

    matrix = np.zeros((1, 3, 4), dtype=np.uint32)
    matrix[0, 0, 0] = 2
    matrix[0, 0, 2] = 1
    matrix[0, 1, 1] = 1
    matrix[0, 2, 0] = 1
    matrix[0, 2, 3] = 2
    counts = Counter({
        (1, 1): 2, (1, 3): 1, (2, 2): 1, (3, 1): 1, (3, 4): 2,
    })
    expected = size_features(counts, "gldzm", voxel_count=37)
    actual = size_statistics(
        mx.array(matrix), "gldzm", tuple(expected), voxel_count=mx.array([37], mx.int64)
    )
    assert set(actual) == set(expected)
    for name, target in expected.items():
        value = float(np.asarray(actual[name], dtype=np.float32).item())
        np.testing.assert_allclose(value, target, rtol=1e-5, atol=1e-7, err_msg=name)
    # Seven zones over 37 voxels, not seven over sum(distance * zone count) = 16.
    percentage = float(np.asarray(actual["ZonePercentage"], dtype=np.float32).item())
    assert percentage == pytest.approx(7 / 37, rel=1e-6)
    assert percentage != pytest.approx(7 / 16, rel=1e-3)


@pytest.mark.parametrize(
    ("family", "name"),
    [
        pytest.param("gldm", "DependenceEntropy", id="gldm"),
        pytest.param("glrlm", "RunEntropy", id="glrlm"),
        pytest.param("glszm", "ZoneEntropy", id="glszm"),
        pytest.param("gldzm", "ZoneDistanceEntropy", id="gldzm"),
    ],
)
def test_size_entropy_keeps_dominant_integer_count(family: str, name: str) -> None:
    from mlxradiomics.features.distributions import size_statistics
    from mlxradiomics.numerics.runtime import mx

    counts = np.array([[[100_000_000, 1]]], dtype=np.uint32)
    options = {"voxel_count": mx.array([2], dtype=mx.int32)} if family == "gldzm" else {}
    actual = size_statistics(mx.array(counts), family, (name,), **options)[name]
    value = float(np.asarray(actual, dtype=np.float32).item())
    expected = _count_entropy_bits(np.array([100_000_000, 1], dtype=np.int64))
    assert value == pytest.approx(expected, rel=1e-5)


def test_gldm_additions_match_independent_definitions() -> None:
    from radiomics_reference.catalogue import size_features

    from mlxradiomics.features.distributions import size_statistics
    from mlxradiomics.numerics.runtime import mx

    counts = Counter({(1, 1): 2, (1, 2): 1, (2, 2): 1, (2, 3): 1})
    expected = size_features(counts, "gldm")
    matrix = np.zeros((1, 2, 3), dtype=np.uint32)
    for (level, dependence), count in counts.items():
        matrix[0, level - 1, dependence - 1] = count
    actual = size_statistics(
        mx.array(matrix), "gldm",
        ("GrayLevelNonUniformityNormalized", "DependenceCountEnergy", "DependencePercentage"),
    )
    for name in (
        "GrayLevelNonUniformityNormalized", "DependenceCountEnergy", "DependencePercentage"
    ):
        value = float(np.asarray(actual[name], dtype=np.float32).item())
        np.testing.assert_allclose(value, expected[name], rtol=1e-6, atol=1e-7, err_msg=name)
    assert float(np.asarray(actual["DependencePercentage"], dtype=np.float32).item()) == 1.0


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("with_gldzm", [False, True])
def test_voxel_zone_maps_do_not_depend_on_batching(
    conventions: Conventions, with_gldzm: bool,
) -> None:
    # The size axis must not depend on the largest zone in a batch: every center gets
    # the same bits however scratch_mib batches the centers.
    rng = np.random.default_rng(1)
    source = hcc_patch((4, 6, 6))[0]
    image = np.minimum(3, np.floor((source - source.min()) * 4 / np.ptp(source)))
    mask = (rng.random(image.shape) < 0.8).astype(np.uint8)
    features = ("glszm_SmallAreaEmphasis", "glszm_ZoneEntropy", "glszm_LargeAreaEmphasis")
    if with_gldzm:
        features += ("gldzm_SmallDistanceEmphasis", "gldzm_ZoneDistanceEntropy")
    settings = settings_for(conventions, bin_width=1.0, features=features, kernel_radius=2)
    one = extract(image, mask, settings=settings, mode="voxel")
    many = extract(image, mask, settings=replace(settings, scratch_mib=1), mode="voxel")
    assert many.metadata.get("batch_size", 0) < one.metadata.get("batch_size", 0)
    for name in one.features:
        np.testing.assert_array_equal(one.features[name], many.features[name])


@pytest.mark.parametrize("family", ["glcm", "glrlm"])
def test_roi_direction_averages_preserve_bits_across_scratch_budgets(family: str) -> None:
    from radiomics_reference.catalogue import glcm

    if family == "glcm":
        source = hcc_patch((13, 17))[0]
        image = np.floor((source - source.min()) * 128 / np.ptp(source))
        features = tuple(f"glcm_{name}" for name in (
            "Contrast", "JointEntropy", "ClusterShade", "Correlation",
        ))
        anchor = 0.0
    else:
        source = hcc_patch((300, 2))[0]
        image = (source > np.median(source)).astype(np.float64)
        features, anchor = _GLRLM_FEATURES, -1022.0
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[4:8, 0] = 0
    bins = image.astype(np.int32) - int(anchor) + 1
    settings = Settings(features=features, bin_width=1, bin_anchor=anchor)
    expected = (glcm(bins, mask.astype(bool), settings, 129) if family == "glcm"
                else glrlm_features(bins, mask.astype(bool)))
    expected = {name: value for name, value in expected.items()
                if f"{family}_{name}" in features}
    # The GLRLM count buffer is 1.17 MiB; its smaller admitted budget is 2 MiB.
    # Both small budgets force smaller directional groups. Means must retain their
    # addition order, as well as agreeing with the independent scalar counts.
    large = extract(image, mask, settings=settings)
    small = extract(
        image, mask, settings=replace(settings, scratch_mib=1 if family == "glcm" else 2),
    )
    _compare(large.features, family, expected)
    _compare(small.features, family, expected)
    assert large.features == small.features


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_zone_convergence_retains_a_long_connected_turn(conventions: Conventions) -> None:
    image = np.repeat(hcc_background((1, 5)) / -2048, 129, axis=0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:, (0, 4)] = 1
    mask[-1, :] = 1
    mask[:3, 2] = 1
    # A long U is one 261-voxel zone. Three isolated upper-middle voxels form a
    # second zone. Stopping propagation early splits the U and changes every size.
    expected = {
        "ZoneEntropy": 1.0, "ZonePercentage": 1 / 132,
        "LargeAreaEmphasis": 34065.0, "SmallAreaEmphasis": (1 / 261**2 + 1 / 3**2) / 2,
        "ZoneVariance": 16641.0,
    }
    result = extract(image, mask, settings=settings_for(
        conventions,
        features=tuple(f'glszm_{name}' for name in expected),
        bin_width=1,
    ))
    _compare(result.features, "glszm", expected)


@pytest.mark.parametrize("name,expected", [
    ("Coarseness", 1.0), ("Contrast", 0.125), ("Busyness", 1.0),
    ("Complexity", 0.5), ("Strength", 1.0),
])
def test_selected_ngtdm_reuses_exact_population_and_preserves_undefined_rows(
    name: str, expected: float,
) -> None:
    from mlxradiomics.features.distributions import ngtdm_statistics
    from mlxradiomics.numerics.runtime import mx

    # Counts [2, 2] and difference sums [1, 1] give p=[1/2, 1/2]. The two
    # off-diagonal pairs have unit level separation; the second row is undefined.
    actual = ngtdm_statistics(
        mx.array([[2, 2], [0, 0]], dtype=mx.int64),
        mx.array([[1, 1], [0, 0]], dtype=mx.float32),
        (name,), total_count=mx.array([4, 0], dtype=mx.int64),
    )
    assert set(actual) == {name}
    np.testing.assert_array_equal(
        np.asarray(actual[name], dtype=np.float32), [expected, float("nan")],
    )
