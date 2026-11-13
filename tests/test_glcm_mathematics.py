"""Mathematically defined GLCM features at explicit and public boundaries.

Expected values come from explicit local GLCM counts. They do not use a PyRadiomics or
IBSI oracle: a matrix with fewer than two occupied row or column gray levels has no second
eigenvalue, so this package reports NaN instead of copying an upstream numerical convention.
Information-theoretic features use the zero-probability limit explicitly, and MCC uses
the centered standardized residual so weak dependence is not erased by the unit component.
"""

import math
from dataclasses import replace
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_background, hcc_patch, hcc_resampled

from mlxradiomics import Settings, extract
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.engine import Result

GLCM_FEATURES = ("glcm_JointAverage", "glcm_MCC")
GLCM_DEPENDENCE = ("Imc1", "Imc2", "MCC")


def _count_features(counts: NDArray[np.uint32], names: tuple[str, ...]) -> dict[str, float]:
    """Evaluate mathematical contingency-table controls without creating an image."""
    from mlxradiomics.features.glcm import glcm_statistics
    from mlxradiomics.numerics.runtime import mx

    values, _ = glcm_statistics(mx.array(counts[None, ...]), mx.float32, names)
    return {name: float(np.asarray(value, dtype=np.float32).item())
            for name, value in values.items()}


def test_correlation_detects_constant_marginal_from_occupied_counts() -> None:
    counts = np.zeros((7, 7), dtype=np.uint32)
    counts[6, 0], counts[6, 1] = 2, 1
    assert _count_features(counts, ("Correlation",))["Correlation"] == 1


def test_cluster_shade_is_zero_for_symmetric_gray_counts() -> None:
    counts = np.zeros((253, 253), dtype=np.uint32)
    counts[[0, 126, 252], [0, 126, 252]] = 4, 6, 4
    actual = _count_features(counts, ("ClusterShade", "JointAverage", "ClusterProminence"))
    assert actual["ClusterShade"] == 0
    assert actual["JointAverage"] == 127
    assert actual["ClusterProminence"] == 8 / 14 * 252**4


@pytest.mark.parametrize("levels", [(1, 128, 255), (2, 129, 256)])
def test_cluster_shade_is_translation_invariant(levels: tuple[int, int, int]) -> None:
    counts = np.zeros((levels[-1] + 1,) * 2, dtype=np.uint32)
    indices = np.asarray(levels)
    counts[indices, indices] = 4
    assert _count_features(counts, ("ClusterShade",))["ClusterShade"] == 0


def test_cluster_shade_cancels_exactly_after_unequal_probabilities() -> None:
    # Exact mean 151 and offsets [-300, -100, 100, 200] give a zero third moment.
    # Rounded probability-weighted cubes previously summed to 0.0859375.
    counts = np.zeros((251, 251), dtype=np.uint32)
    counts[[0, 100, 200, 250], [0, 100, 200, 250]] = 2, 12, 2, 8
    assert _count_features(counts, ("ClusterShade",))["ClusterShade"] == 0


def _ct_profiles() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Quantize one CT profile at two scales and reflect its large-range contrast."""
    source = hcc_patch((1, 14))[0].reshape(-1)
    unit = (source - source.min()) / np.ptp(source)
    large = np.floor(unit * 254)
    return large, 254 - large, np.floor(unit * 3)


def _profile_cluster_shade(profile: NDArray[np.float64]) -> Fraction:
    """Enumerate symmetric adjacent pairs and their exact rational third moment."""
    counts: dict[int, int] = {}
    for left, right in zip(profile[:-1], profile[1:], strict=True):
        level_sum = int(left + right) + 2
        counts[level_sum] = counts.get(level_sum, 0) + 2
    total = sum(counts.values())
    mean = Fraction(sum(level * count for level, count in counts.items()), total)
    return sum((count * (level - mean) ** 3 for level, count in counts.items()),
               Fraction(0)) / total


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_signed_slice_average_does_not_depend_on_slice_order(
    order: tuple[int, int, int],
) -> None:
    profiles = _ct_profiles()
    image = np.stack([profiles[index] for index in order])[:, None, :]
    mask = np.ones(image.shape, np.uint8)
    result = extract(image, mask, settings=Settings(
        features=("glcm_ClusterShade",), bin_width=1, bin_anchor=0,
        glcm_aggregation="2d_average"))
    expected = sum(map(_profile_cluster_shade, profiles), Fraction(0)) / 3
    assert abs(_profile_cluster_shade(profiles[0])) > 1e5
    assert expected != 0
    assert _roi_value(result, "glcm_ClusterShade") == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_signed_direction_average_does_not_depend_on_direction_order(
    order: tuple[int, int, int],
) -> None:
    # Relocate and orient the same CT profiles on isolated controlled label lines.
    # Each contributes to one direction; the other directions have no valid pairs.
    profiles = _ct_profiles()
    steps = ((0, 1), (1, 0), (1, 1))
    image = hcc_patch((60, 20))[0]
    mask = np.zeros(image.shape, np.uint8)
    for row, step, index in zip((1, 20, 40), steps, order, strict=True):
        for t, value in enumerate(profiles[index]):
            image[row + t * step[0], 1 + t * step[1]] = value
            mask[row + t * step[0], 1 + t * step[1]] = 1
    result = extract(image, mask, settings=Settings(
        features=("glcm_ClusterShade",), bin_width=1, bin_anchor=0))
    expected = sum(map(_profile_cluster_shade, profiles), Fraction(0)) / 3
    assert _roi_value(result, "glcm_ClusterShade") == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize("exponent", [27, 28])
def test_mcc_preserves_perfect_dependence_for_rare_states(exponent: int) -> None:
    from mlxradiomics.numerics.eigen import maximal_correlation
    from mlxradiomics.numerics.runtime import mx

    counts = mx.array([[[2**exponent, 0], [0, 1]]], dtype=mx.uint32)
    actual = np.asarray(maximal_correlation(counts), dtype=np.float32).item()
    assert actual == pytest.approx(1, rel=1e-6)


def test_mcc_of_an_empty_batch_is_empty() -> None:
    # A zero-row batch has no matrices, so there is no MCC to report. Its Gram matrix comes
    # from an empty matmul; dispatching the Jacobi kernel on that buffer crashed the process.
    from mlxradiomics.numerics.eigen import maximal_correlation
    from mlxradiomics.numerics.runtime import mx

    actual = np.asarray(
        maximal_correlation(mx.zeros((0, 4, 4), dtype=mx.int32)), dtype=np.float32
    )
    assert actual.shape == (0,)


def _map(result: Result, name: str) -> NDArray[np.float64]:
    return np.asarray(result.features[f"original_{name}"], dtype=np.float64)


def _roi_value(result: Result, name: str) -> float:
    return float(np.asarray(result.features[f"original_{name}"], dtype=np.float64).item())


def _points(
    values: NDArray[np.float64],
    points: tuple[tuple[int, int], ...],
) -> NDArray[np.float64]:
    rows = np.array([point[0] for point in points], dtype=np.intp)
    columns = np.array([point[1] for point in points], dtype=np.intp)
    return values[rows, columns]


def _count_entropy_bits(counts: NDArray[np.int64]) -> float:
    total = float(counts.sum())
    return -math.fsum(
        float(count) / total * math.log2(float(count) / total)
        for count in counts.reshape(-1)
        if count > 0
    )


def _reference_glcm(counts: NDArray[np.int64]) -> dict[str, float]:
    """Evaluate Imc1, Imc2 and 2x2 MCC from explicit integer counts."""
    if counts.shape != (2, 2):
        raise ValueError("the reference dependence fixture is two-state")
    total = float(counts.sum())
    rows = counts.sum(axis=1)
    columns = counts.sum(axis=0)
    information = math.fsum(
        float(counts[i, j]) / total
        * math.log2(float(counts[i, j]) * total / (float(rows[i]) * float(columns[j])))
        for i in range(2)
        for j in range(2)
        if counts[i, j] > 0
    )
    hx, hy = _count_entropy_bits(rows), _count_entropy_bits(columns)
    determinant = float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0])
    denominator = math.sqrt(
        float(rows[0] * rows[1] * columns[0] * columns[1])
    )
    return {
        "Imc1": -information / max(hx, hy) if max(hx, hy) > 0 else 0.0,
        "Imc2": math.sqrt(-math.expm1(-2 * max(information, 0.0))),
        "MCC": abs(determinant) / denominator,
    }



def _local_glcm_case() -> tuple[
    NDArray[np.float64],
    NDArray[np.uint8],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Return a sparse 2-D ROI whose first alternating line center is index 255.

    The 253 isolated centers plus the two-voxel horizontal pair give 255 centers before
    the line, so the first line center is center 256 while the pair contributes a
    direction the line never has.
    """
    image = (hcc_patch((40, 40))[0] > 70).astype(np.float64) * 25
    mask = np.zeros((40, 40), dtype=np.uint8)
    isolated = [(row, column) for row in range(0, 30, 2) for column in range(0, 40, 2)][:253]
    pair = ((26, 0), (26, 1))
    line = ((30, 0), (31, 0), (32, 0))
    for row, column in isolated:
        mask[row, column] = 1
    for row, column in (*pair, *line):
        mask[row, column] = 1
    image[26, :2] = (hcc_patch((1, 2))[0] - 70).reshape(-1) * 25
    profile = hcc_patch((1, 3))[0].reshape(-1)
    image[30:33, 0] = (np.abs(profile - 71) < 0.5) * 25
    return image, mask, line, tuple(isolated)


def test_mcc_matches_manual_glcm_value() -> None:
    # One direction over bins [1, 1, 2] gives directed counts [[1, 1], [0, 0]];
    # symmetrization makes [[2, 1], [1, 0]], p = [[0.5, 0.25], [0.25, 0]], row
    # marginal [0.75, 0.25] and Q eigenvalues 1 and 1/9.
    image = (hcc_patch((3, 1))[0] < 50).astype(np.float64)
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=GLCM_FEATURES, bin_width=1.0, bin_anchor=0.0)
    result = extract(image, mask, settings=settings)

    assert _roi_value(result, "glcm_JointAverage") == pytest.approx(1.25, rel=1e-6)
    assert _roi_value(result, "glcm_MCC") == pytest.approx(1 / 3, rel=1e-6)
    assert result.metadata["feature_validity"]["original_glcm_MCC"]["finite"] == 1
    assert result.metadata["feature_validity"]["original_glcm_MCC"]["valid_zero"] == 0


@pytest.mark.parametrize("reverse", [False, True], ids=["one-row-state", "one-column-state"])
def test_degenerate_row_or_column_marginals_are_nan(reverse: bool) -> None:
    column = (hcc_patch((3, 1))[0] < 50).astype(np.float64)
    if reverse:
        column = column[::-1].copy()
    joint_average = 1.5 if reverse else 1.0
    mask = np.ones(column.shape, dtype=np.uint8)
    settings = Settings(
        features=GLCM_FEATURES, bin_width=1.0, bin_anchor=0.0, symmetrical_glcm=False
    )
    result = extract(column, mask, settings=settings)

    assert _roi_value(result, "glcm_JointAverage") == pytest.approx(joint_average, rel=1e-6)
    mcc = np.asarray(result.features["original_glcm_MCC"])
    assert np.isnan(mcc).all()
    assert result.metadata["feature_validity"]["original_glcm_MCC"]["nan"] == 1


def test_flat_region_mcc_is_undefined() -> None:
    image = hcc_background((8, 8)) * 0
    block = np.zeros(image.shape, dtype=np.uint8)
    block[1:4, 1:4] = 1
    settings = Settings(features=GLCM_FEATURES, bin_width=1.0, bin_anchor=0.0)

    # ROI: every direction is the one-cell GLCM [[N]], so JointAverage has one
    # occupied row value but MCC has no second eigenvalue, so MCC is undefined.
    roi = extract(image, block, settings=settings)
    assert _roi_value(roi, "glcm_JointAverage") == pytest.approx(1.0, rel=1e-6)
    assert np.isnan(np.asarray(roi.features["original_glcm_MCC"])).all()
    assert roi.metadata["feature_validity"]["original_glcm_JointAverage"] == {
        "finite": 1,
        "valid_zero": 0,
        "nan": 0,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "excluded": 0,
    }
    assert roi.metadata["feature_validity"]["original_glcm_MCC"] == {
        "finite": 0,
        "valid_zero": 0,
        "nan": 1,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "excluded": 0,
    }

    # Voxel: the 3x3 flat block and the isolated center are both undefined for MCC.
    # The block still defines JointAverage, while the isolated center has no pair.
    mask = block.copy()
    mask[7, 7] = 1
    voxel = extract(image, mask, settings=settings, mode="voxel")
    joint = _map(voxel, "glcm_JointAverage")
    mcc = _map(voxel, "glcm_MCC")
    block_points = ((row, column) for row in range(1, 4) for column in range(1, 4))
    assert np.allclose(_points(joint, tuple(block_points)), 1.0, rtol=1e-6, atol=1e-6)
    assert np.isnan(_points(joint, ((7, 7),))).all()
    assert np.isnan(mcc[mask == 1]).all()
    assert voxel.metadata["gray_levels"] == 1
    assert voxel.metadata["feature_validity"]["original_glcm_MCC"] == {
        "finite": 0,
        "valid_zero": 0,
        "nan": 10,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "excluded": 54,
    }
    assert voxel.metadata["feature_validity"]["original_glcm_JointAverage"] == {
        "finite": 9,
        "valid_zero": 0,
        "nan": 1,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "excluded": 54,
    }


def test_one_level_region_stays_undefined_with_a_second_global_level() -> None:
    # The distant voxel makes the global gray-level count 2, but every local 3x3 block
    # matrix still occupies only the dark state. Validity follows the local matrix, not
    # the global gray count, and a genuine second eigenvalue is not manufactured.
    image = hcc_background((8, 8)) * 0
    image[7, 7] = 25.0
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[7, 7] = 1
    settings = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)

    result = extract(image, mask, settings=settings, mode="voxel")
    mcc = _map(result, "glcm_MCC")
    joint = _map(result, "glcm_JointAverage")
    block_points = tuple((row, column) for row in range(1, 4) for column in range(1, 4))

    assert result.metadata["gray_levels"] == 2
    assert np.isnan(_points(mcc, block_points)).all()
    assert np.allclose(_points(joint, block_points), 1.0, rtol=1e-6, atol=1e-6)
    assert np.isnan(_points(joint, ((7, 7),))).all()
    assert result.metadata["feature_validity"]["original_glcm_MCC"]["nan"] == 10
    assert result.metadata["feature_validity"]["original_glcm_MCC"]["finite"] == 0


def test_two_level_pair_has_unit_mcc() -> None:
    # One horizontal direction over bins 1 and 2 gives p = [[0, 0.5], [0.5, 0]] after
    # symmetrization, whose Q has the repeated eigenvalue 1 and therefore MCC = 1.
    image = (hcc_patch((1, 2))[0] - 70) * 25
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)

    result = extract(image, mask, settings=settings)
    assert _roi_value(result, "glcm_JointAverage") == pytest.approx(1.5, rel=1e-6)
    assert _roi_value(result, "glcm_MCC") == pytest.approx(1.0, rel=1e-6)


def test_zero_second_eigenvalue_is_valid_zero_not_nan() -> None:
    # CT thresholding gives bins [1, 2, 2, 1, 1], one count in every cell, so the symmetric
    # GLCM is uniform: both marginals occupy two states and the second eigenvalue is a
    # genuine zero. This must remain a finite 0, unlike an absent second eigenvalue.
    image = (hcc_patch((1, 5))[0] > 70).astype(np.float64) * 25
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)

    result = extract(image, mask, settings=settings)
    assert _roi_value(result, "glcm_JointAverage") == pytest.approx(1.5, rel=1e-6)
    assert _roi_value(result, "glcm_MCC") == pytest.approx(0.0, abs=1e-6)
    validity = result.metadata["feature_validity"]["original_glcm_MCC"]
    assert validity["finite"] == 1
    assert validity["valid_zero"] == 1
    assert validity["nan"] == 0


def test_mcc_averages_only_directions_with_a_second_eigenvalue() -> None:
    # Bins [[1, 1], [1, 2]] over a 2x2 ROI. Three directions have a defined MCC
    # (1/3, 1/3 and 1) and the diagonal direction has only the one occupied cell
    # (1, 1). The absent eigenvalue is omitted from MCC's directional average rather
    # than erasing the three defined values; JointAverage still includes all four
    # nonempty directions and averages to 1.25.
    image = (hcc_patch((2, 2))[0] < 61).astype(np.float64) * 25
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)

    result = extract(image, mask, settings=settings)
    assert _roi_value(result, "glcm_JointAverage") == pytest.approx(1.25, rel=1e-6)
    assert _roi_value(result, "glcm_MCC") == pytest.approx(5 / 9, rel=1e-6)
    validity = result.metadata["feature_validity"]["original_glcm_MCC"]
    assert validity["finite"] == 1
    assert validity["nan"] == 0


def test_local_glcm_ignores_distant_centers() -> None:
    image, mask, line, isolated = _local_glcm_case()
    settings = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)
    result = extract(image, mask, settings=settings, mode="voxel")
    joint = _map(result, "glcm_JointAverage")
    mcc = _map(result, "glcm_MCC")

    # The alternating vertical line has one two-level pair per center, so every line
    # center has the same manually derived directional GLCM and average.
    assert np.allclose(_points(joint, line), 1.5, rtol=1e-6, atol=1e-6)
    assert np.allclose(_points(mcc, line), 1.0, rtol=1e-6, atol=1e-6)
    # The horizontal pair is defined by its own two-level direction.
    pair = ((26, 0), (26, 1))
    assert np.allclose(_points(joint, pair), 1.5, rtol=1e-6, atol=1e-6)
    assert np.allclose(_points(mcc, pair), 1.0, rtol=1e-6, atol=1e-6)
    # The isolated centers have no pair in any direction.
    assert np.isnan(_points(joint, isolated)).all()
    assert np.isnan(_points(mcc, isolated)).all()

    # Removing every distant center and direction leaves each line center unchanged.
    line_mask = np.zeros_like(mask)
    for row, column in line:
        line_mask[row, column] = 1
    control = extract(image, line_mask, settings=settings, mode="voxel")
    np.testing.assert_allclose(
        _points(_map(control, "glcm_JointAverage"), line),
        _points(joint, line),
        rtol=1e-6,
        atol=1e-6,
        err_msg="JointAverage",
    )
    np.testing.assert_allclose(
        _points(_map(control, "glcm_MCC"), line),
        _points(mcc, line),
        rtol=1e-6,
        atol=1e-6,
        err_msg="MCC",
    )


def test_voxel_glcm_is_invariant_to_batch_budget_and_streaming() -> None:
    image, mask, line, _ = _local_glcm_case()
    base = Settings(features=GLCM_FEATURES, bin_width=25.0, bin_anchor=0.0)
    retained_small = extract(image, mask, settings=replace(base, scratch_mib=1), mode="voxel")
    retained_large = extract(image, mask, settings=replace(base, scratch_mib=64), mode="voxel")
    assert retained_small.metadata.get("batch_size", 0) < retained_large.metadata.get(
        "batch_size", 0
    )

    names = tuple(f"original_{name}" for name in GLCM_FEATURES)
    with voxel_map_store(image.shape, names) as store:
        streamed_result = extract(
            image,
            mask,
            settings=replace(base, scratch_mib=1),
            mode="voxel",
            sink=store.append,
        )
        assert not streamed_result.features
        streamed = {name: values.copy() for name, values in store.maps()}

    for name in names:
        small = np.asarray(retained_small.features[name], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(retained_large.features[name], dtype=np.float64),
            small,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=name,
        )
        np.testing.assert_allclose(
            streamed[name],
            small,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=name,
        )
    assert (
        retained_small.metadata["feature_validity"] == retained_large.metadata["feature_validity"]
    )
    assert (
        streamed_result.metadata["feature_validity"] == retained_small.metadata["feature_validity"]
    )

    # Manually derived local values remain present after both scheduling changes.
    small_joint = _map(retained_small, "glcm_JointAverage")
    small_mcc = _map(retained_small, "glcm_MCC")
    assert np.allclose(_points(small_joint, line), 1.5, rtol=1e-6, atol=1e-6)
    assert np.allclose(_points(small_mcc, line), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("symmetric", [False, True], ids=["directed", "symmetric"])
def test_mcc_ignores_leading_empty_bins_in_asymmetric_local_counts(symmetric: bool) -> None:
    from radiomics_reference.mathematics import glcm_features

    # This line's directed matrix is [[1,1,1], [1,0,0], [0,1,0]]. Padding its
    # gray-level axes with empty states must preserve MCC and shift JointAverage.
    source = hcc_patch((1, 6))[0]
    image = np.floor((source - source.min()) * 2 / np.ptp(source))
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(features=GLCM_FEATURES, bin_width=1.0, bin_anchor=0.0,
                        symmetrical_glcm=symmetric)
    for gap in (0, 2):
        shifted = replace(settings, bin_anchor=-float(gap))
        expected = glcm_features(image.astype(np.int32) + gap + 1, mask.astype(bool), shifted)
        result = extract(image, mask, settings=shifted)
        for name, value in expected.items():
            assert _roi_value(result, f"glcm_{name}") == pytest.approx(value, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("total", [10_000_000, 100_000_000, 1_000_000_000, 2_000_000_000])
def test_imc1_reaches_minus_one_for_rare_diagonal(total: int) -> None:
    from mlxradiomics.features.glcm import glcm_statistics
    from mlxradiomics.numerics.runtime import mx

    # A diagonal two-state GLCM is perfectly dependent; the zero-probability limit
    # therefore gives Imc1 = -1 at every positive imbalance; no epsilon may enter px*py.
    counts = mx.array([[[1, 0], [0, total]]], dtype=mx.uint32)
    values, _ = glcm_statistics(counts, mx.float32, ("Imc1",))
    actual = float(np.asarray(values["Imc1"], dtype=np.float32).item())
    assert actual == pytest.approx(-1.0, rel=1e-5)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param([[9, 24], [24, 64]], id="exact-independence"),
        pytest.param([[2000, 1999], [1999, 2000]], id="weak-dependence"),
        pytest.param([[4000, 3999], [3999, 4000]], id="weaker-dependence"),
    ],
)
def test_glcm_dependence_uses_exact_integer_residuals(
    matrix: list[list[int]],
) -> None:
    from mlxradiomics.features.glcm import glcm_statistics
    from mlxradiomics.numerics.runtime import mx

    counts = np.asarray(matrix, dtype=np.int64)
    expected = _reference_glcm(counts)
    values, _ = glcm_statistics(
        mx.array([matrix], dtype=mx.uint32), mx.float32, GLCM_DEPENDENCE
    )
    for name, target in expected.items():
        actual = float(np.asarray(values[name], dtype=np.float32).item())
        assert actual == pytest.approx(target, rel=1e-5, abs=1e-12), name


def test_imc2_keeps_float32_accuracy_for_moderate_dependence() -> None:
    # Every asymmetric two-state directed GLCM with entries up to 8 puts 2*MI across
    # (0, 1.4], including [0.1, 1] where MLX's GPU expm1 is ~3.5e-6 relative off
    # (([[5, 8], [0, 6]]) gave Imc2 1.8e-6 high). Budget: MI is within ~5e-7 relative,
    # and Imc2 = sqrt(1 - exp(-y)) halves the relative error of 1 - exp(-y), whose
    # sensitivity to y is y e^-y / (1 - e^-y) <= 1, so a few float32 ulps of 1 - exp(-y)
    # leave Imc2 within 1e-6.
    from mlxradiomics.features.glcm import glcm_statistics
    from mlxradiomics.numerics.runtime import mx

    matrices = [
        np.array([[a, b], [c, d]], dtype=np.int64)
        for a in range(1, 9) for b in range(9) for c in range(9) for d in range(1, 9)
        if b != c
    ]
    values, _ = glcm_statistics(
        mx.array(np.stack(matrices).astype(np.uint32)), mx.float32, ("Imc2",)
    )
    actual = np.asarray(values["Imc2"], dtype=np.float64)
    expected = np.array([_reference_glcm(matrix)["Imc2"] for matrix in matrices])
    # Exactly independent tables such as [[8, 8], [5, 5]] must give exactly zero.
    assert np.all(np.abs(actual - expected) <= 1e-6 * expected)


@pytest.mark.parametrize("length", [121, 7998], ids=["short-profile", "long-profile"])
def test_glcm_dependence_at_roi_boundary_and_rotations(
    length: int,
) -> None:
    # Close a binary CT profile into a cycle. Its directed transition multiset is
    # preserved by reversal and cyclic rotation, including unequal state populations.
    profile = hcc_resampled((1, length))
    profile = (profile > np.median(profile)).astype(np.float64) * 25
    image = np.concatenate((profile, profile[:, :1]), axis=1)
    counts = np.zeros((2, 2), dtype=np.int64)
    for left, right in zip(image[0, :-1], image[0, 1:], strict=True):
        counts[int(left / 25), int(right / 25)] += 1
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(
        features=tuple(f"glcm_{name}" for name in GLCM_DEPENDENCE),
        bin_width=25.0,
        bin_anchor=0.0,
        symmetrical_glcm=False,
    )
    expected = _reference_glcm(counts)
    rotated_cycle = np.roll(image[:, :-1], 137, axis=1)
    rotated = np.concatenate((rotated_cycle, rotated_cycle[:, :1]), axis=1)
    for transformed in (image, image[:, ::-1].copy(), rotated):
        result = extract(transformed, mask, settings=settings)
        for name, target in expected.items():
            actual = _roi_value(result, f"glcm_{name}")
            assert actual == pytest.approx(target, rel=1e-5, abs=1e-12), name


def test_sum_entropy_matches_joint_entropy_for_rare_diagonal() -> None:
    from mlxradiomics.features.glcm import glcm_statistics
    from mlxradiomics.numerics.runtime import mx

    # On a diagonal table the sum index uniquely identifies the joint state, so the
    # exact count-based entropies are equal.  The dominant probability rounds to one in
    # float32, so the entropy must come from counts to keep the rare state.
    counts = np.array([[[100_000_000, 0], [0, 1]]], dtype=np.uint32)
    values, _ = glcm_statistics(
        mx.array(counts), mx.float32, ("SumEntropy", "JointEntropy", "DifferenceEntropy")
    )
    expected = _count_entropy_bits(np.array([100_000_000, 1], dtype=np.int64))
    for name in ("SumEntropy", "JointEntropy"):
        actual = float(np.asarray(values[name], dtype=np.float32).item())
        assert actual == pytest.approx(expected, rel=1e-5), name
    assert float(np.asarray(values["DifferenceEntropy"], dtype=np.float32).item()) == 0.0


def test_imc_voxel_two_state_pair_matches_closed_form() -> None:
    # Every local neighborhood of this line holds one two-state pair direction, whose
    # probability matrix [[0,.5],[.5,0]] has one bit of mutual information.
    image = (np.abs(hcc_patch((1, 3))[0] - 71) < 0.5).astype(np.float64) * 25
    mask = np.ones(image.shape, dtype=np.uint8)
    settings = Settings(
        features=tuple(f"glcm_{name}" for name in GLCM_DEPENDENCE),
        bin_width=25.0,
        bin_anchor=0.0,
    )
    result = extract(image, mask, settings=settings, mode="voxel")
    expected = {
        "Imc1": -1.0,
        "Imc2": math.sqrt(-math.expm1(-2.0)),
        "MCC": 1.0,
    }
    for name, target in expected.items():
        np.testing.assert_allclose(
            _map(result, f"glcm_{name}"), target, rtol=1e-5, atol=1e-12, err_msg=name
        )
