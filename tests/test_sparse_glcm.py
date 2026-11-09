"""Large-lattice co-occurrence against independent occupied-cell definitions."""

from collections import Counter
from decimal import Decimal, localcontext

import numpy as np
import pytest
from radiomics_reference.catalogue import glcm, glcm_matrix

from mlxradiomics import Settings, extract
from mlxradiomics.config.settings import FAMILIES
from mlxradiomics.features.sparse_glcm import sparse_glcm_statistics
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.numerics.sparse_counts import coordinate_counts, merge_counts


@pytest.mark.parametrize("symmetric", [False, True])
def test_sparse_pair_counts_preserve_large_indices_and_empty_rows(symmetric: bool) -> None:
    left = np.array([[1, 25001, 25001, 7, 8], [1, 2, 3, 4, 5]], np.int32)
    right = np.array([[25001, 1, 1, 8, 9], [2, 3, 4, 5, 6]], np.int32)
    valid = np.array([[True, True, True, False, False], [False] * 5])
    keys, counts = coordinate_counts(mx.array(left), mx.array(right), mx.array(valid),
                                     25001, symmetric=symmetric)
    expected: Counter[int] = Counter()
    for i, j in zip(left[0, valid[0]], right[0, valid[0]], strict=True):
        expected[(int(i) - 1) * 25001 + int(j) - 1] += 1
        if symmetric:
            expected[(int(j) - 1) * 25001 + int(i) - 1] += 1
    actual_keys = np.asarray(keys, dtype=np.int64)
    actual_counts = np.asarray(counts, dtype=np.int32)
    assert dict(zip(actual_keys[0, actual_counts[0] > 0],
                    actual_counts[0, actual_counts[0] > 0], strict=True)) == dict(expected)
    assert not actual_counts[1].any()
    merged_keys, merged_counts = merge_counts([keys, keys], [counts, counts])
    np.testing.assert_array_equal(merged_keys, keys)
    np.testing.assert_array_equal(merged_counts, actual_counts * 2)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("symmetric", [False, True])
def test_native_sparse_glcm_matches_independent_counting(ndim: int, symmetric: bool) -> None:
    shape = (4, 5) if ndim == 2 else (3, 4, 5)
    bins = np.array([1, 16001, 16002, 25001], np.int32)[
        np.random.default_rng(417).integers(0, 4, size=shape)
    ]
    mask = np.indices(shape).sum(axis=0) % 4 != 0
    settings = Settings(features=tuple(f"glcm_{name}" for name in FAMILIES["glcm"]),
                        bin_width=1, bin_anchor=1, symmetrical_glcm=symmetric)
    expected = glcm(bins, mask, settings, 25001)
    result = extract(bins, mask.astype(np.uint8), settings=settings)
    for name, value in expected.items():
        np.testing.assert_allclose(result.features[f"original_glcm_{name}"], value,
                                   rtol=2e-5, atol=2e-6, equal_nan=True, err_msg=name)


def test_sparse_information_is_independent_of_batch_padding() -> None:
    # Both tables are exactly independent. The smaller row needs five padding
    # cells beside the 3x3 row; those cells must contribute no information.
    ng = 25001
    keys = np.zeros((2, 9), np.int64)
    counts = np.zeros((2, 9), np.int32)
    keys[0, :4] = [0, 1, ng, ng + 1]
    counts[0, :4] = 1
    keys[1] = [i * ng + j for i in range(3) for j in range(3)]
    counts[1] = 1
    values, active = sparse_glcm_statistics(mx.array(keys), mx.array(counts), ng,
                                            ("Imc1", "Imc2", "MCC"), 64)
    np.testing.assert_array_equal(np.asarray(active, dtype=np.bool_), [True, True])
    for name in values:
        np.testing.assert_allclose(np.asarray(values[name], dtype=np.float64), [0, 0],
                                   atol=2e-7, err_msg=name)


@pytest.mark.parametrize("extra_count", [0, 1])
def test_large_cluster_shade_preserves_exact_integer_cancellation(extra_count: int) -> None:
    # Exact central offsets [-300000,-100000,100000,200000] with weights
    # [2,12,2,8] have zero third moment. Large populations overflow an int64
    # weighted square, so the result depends on extending the arithmetic,
    # not just increasing an admission constant.
    levels = np.array([1, 100001, 200001, 250001], np.int64)
    counts = np.array([2, 12, 2, 8], np.int32) * 2**25
    counts[-1] += extra_count
    ng = 250001
    keys = (levels - 1) * ng + levels - 1
    values, _ = sparse_glcm_statistics(mx.array(keys[None]), mx.array(counts[None]),
                                       ng, ("ClusterShade", "JointAverage"), 64)
    expected_shade = 0 if not extra_count else -9934107.462565104
    expected_mean = 150001 if not extra_count else 120796760756369 / 805306369
    assert float(np.asarray(values["ClusterShade"], dtype=np.float32)[0]) == pytest.approx(
        expected_shade, rel=2e-6, abs=0,
    )
    assert float(np.asarray(values["JointAverage"], dtype=np.float32)[0]) == pytest.approx(
        expected_mean, rel=2e-6,
    )


def test_sparse_glcm_preserves_rare_state_dependence() -> None:
    ng = 25001
    table = {(1, 1): 2**28, (25001, 25001): 1}
    keys = [(i - 1) * ng + j - 1 for i, j in table]
    counts = list(table.values())
    expected = glcm_matrix(table, ng)
    names = ("Imc1", "Imc2", "MCC", "JointEntropy")
    values, _ = sparse_glcm_statistics(mx.array([keys], dtype=mx.int64),
                                       mx.array([counts], dtype=mx.int32), ng, names, 64)
    for name in names:
        np.testing.assert_allclose(np.asarray(values[name])[0], expected[name],
                                   rtol=2e-5, atol=1e-10, err_msg=name)


def test_sparse_glcm_resolves_near_independence() -> None:
    # Exact integer perturbations about independence carry information far below
    # float32 epsilon. Decimal evaluates the defining entropy expression directly.
    count = 2**26
    with localcontext() as context:
        context.prec = 60
        population = Decimal(4 * count + 2)
        information = sum(
            2 * Decimal(c) / population * (4 * Decimal(c) / population).ln()
            for c in (count, count + 1)
        ) / Decimal(2).ln()
        expected = {
            "Imc1": -float(information),
            "Imc2": float((1 - (-2 * information).exp()).sqrt()),
            "MCC": 1 / (2 * count + 1),
        }
    ng = 25001
    keys = mx.array([[0, ng - 1, (ng - 1) * ng, ng * ng - 1]], dtype=mx.int64)
    counts = mx.array([[count + 1, count, count, count + 1]], dtype=mx.int32)
    values, _ = sparse_glcm_statistics(keys, counts, ng, tuple(expected), 64)
    for name, target in expected.items():
        assert float(np.asarray(values[name], dtype=np.float32)[0]) == pytest.approx(
            target, rel=2e-5, abs=1e-24,
        ), name


def test_sparse_difference_variance_centers_before_float_conversion() -> None:
    # Differences 100000000 and 100000001 each occur once. Their population
    # variance is exactly 1/4 even though both round to the same float32 value.
    # Only occupied coordinates are needed for this output.
    ng = 100000002
    keys = mx.array([[100000000, 100000001]], dtype=mx.int64)
    counts = mx.array([[1, 1]], dtype=mx.int32)
    values, active = sparse_glcm_statistics(keys, counts, ng, ("DifferenceVariance",), 64)
    assert bool(np.asarray(active)[0])
    assert float(np.asarray(values["DifferenceVariance"])[0]) == 0.25
