"""Public ROI, voxel-map and streamed values from hand or scalar mathematics.

No expected value comes from a recorded result, from a second extraction, or from
production feature arithmetic.
"""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.fixtures import Case, cases

from mlxradiomics import Settings, extract
from mlxradiomics.contracts.types import HostArray
from mlxradiomics.features.cooccurrence import roi_pairs
from mlxradiomics.numerics.histogram import histogram
from mlxradiomics.numerics.runtime import array_from_bool, mx
from mlxradiomics.voxel.storage import voxel_map_store

# A 2x2 image is small enough that every voxel center's clipped 3x3 kernel contains the
# whole ROI, so ROI and voxel mode share the same hand mean (0 + 25 + 50 + 25) / 4.
_HAND_IMAGE = np.array([[0.0, 25.0], [50.0, 25.0]])
_HAND_MEAN = 25.0
_OFFSETS = ((1, 1), (1, 0), (1, -1), (0, 1))


@pytest.mark.parametrize("stream", [False, True], ids=["maps", "streamed"])
def test_roi_and_voxel_maps_match_hand_mean(stream: bool) -> None:
    mask = np.ones(_HAND_IMAGE.shape, dtype=np.uint8)
    settings = Settings(features=("firstorder_Mean",), bin_width=25.0)
    roi = extract(_HAND_IMAGE, mask, settings=settings)
    assert roi.metadata["roi_voxels"] == _HAND_IMAGE.size
    mean = roi.features["original_firstorder_Mean"]
    assert isinstance(mean, float)
    np.testing.assert_allclose(mean, _HAND_MEAN, rtol=0, atol=0)

    names = ("original_firstorder_Mean",)
    if stream:
        with voxel_map_store(_HAND_IMAGE.shape, names) as store:
            result = extract(_HAND_IMAGE, mask, settings=settings, mode="voxel",
                             sink=store.append)
            assert not result.features
            maps: dict[str, NDArray[np.float32]] = {
                name: values.copy() for name, values in store.maps()
            }
    else:
        result = extract(_HAND_IMAGE, mask, settings=settings, mode="voxel")
        maps = {name: np.asarray(value, dtype=np.float32)
                for name, value in result.features.items()}

    assert result.metadata["roi_voxels"] == _HAND_IMAGE.size
    assert set(maps) == set(names)
    np.testing.assert_allclose(
        np.asarray(maps["original_firstorder_Mean"], dtype=np.float64),
        _HAND_MEAN,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_first_order_values_follow_source_and_label(case: Case) -> None:
    roi = np.asarray(case.mask) == case.settings.label
    source = np.asarray(case.image, dtype=np.float64)[roi]
    result = extract(case.image, case.mask, settings=case.settings, spacing=case.spacing)

    assert result.metadata["roi_voxels"] == int(roi.sum())
    expected = {
        "Mean": float(source.mean()),
        "Minimum": float(np.float32(source.min())),
        "Maximum": float(np.float32(source.max())),
        "Range": float(np.float32(source.max() - source.min())),
    }
    for name, target in expected.items():
        value = result.features[f"original_firstorder_{name}"]
        assert isinstance(value, float)
        if name in {"Minimum", "Maximum"}:
            assert value == target, name
        else:
            np.testing.assert_allclose(value, target, rtol=1e-5, atol=1e-5, err_msg=name)


def test_voxel_mode_refuses_a_single_voxel_roi() -> None:
    case = next(case for case in cases() if case.name == "singleton")
    with pytest.raises(ValueError, match="at least two"):
        extract(case.image, case.mask, settings=case.settings,
                spacing=case.spacing, mode="voxel")


@pytest.mark.parametrize(
    "masked_kernel,expected",
    [
        pytest.param(
            True,
            np.array([[50.0, 0.0, 0.0], [0.0, 36.0, 0.0], [0.0, 0.0, 54.0]]),
            id="masked",
        ),
        pytest.param(
            False,
            np.array([[26.0, 0.0, 0.0], [0.0, 132.0 / 9.0, 0.0], [0.0, 0.0, 30.0]]),
            id="unmasked",
        ),
    ],
)
def test_voxel_neighborhood_sampling_matches_scalar_means(
    masked_kernel: bool, expected: NDArray[np.float64],
) -> None:
    image = np.array([[0.0, 1.0, 2.0], [3.0, 100.0, 5.0], [6.0, 7.0, 8.0]])
    mask = np.zeros(image.shape, dtype=np.uint8)
    np.fill_diagonal(mask, 1)
    settings = Settings(features=("firstorder_Mean",), kernel_radius=1,
                        masked_kernel=masked_kernel)
    result = extract(image, mask, settings=settings, mode="voxel")

    assert result.metadata["roi_voxels"] == 3
    actual = np.asarray(result.features["original_firstorder_Mean"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["mlx", "metal"])
def test_integer_pair_counts_match_scalar_enumeration(method: str) -> None:
    bins = np.array([[1, 1, 2], [2, 3, 1], [3, 2, 1]], dtype=np.int32)
    mask = np.array([[True, True, False], [True, True, True], [False, True, True]])
    settings = Settings(features=("glcm_Contrast",), distances=(1,))

    expected: list[NDArray[np.uint32]] = []
    for offset in _OFFSETS:
        counts = np.zeros((3, 3), dtype=np.uint32)
        for index in np.ndindex(bins.shape):
            neighbor = tuple(index[axis] + offset[axis] for axis in range(2))
            inside = all(0 <= neighbor[axis] < bins.shape[axis] for axis in range(2))
            if inside and bool(mask[index]) and bool(mask[neighbor]):
                counts[int(bins[index]) - 1, int(bins[neighbor]) - 1] += 1
        expected.append(counts)

    with mx.stream(mx.gpu):
        pairs = list(roi_pairs(mx.array(bins), array_from_bool(mask), settings))
        assert len(pairs) == len(_OFFSETS)
        actual: list[NDArray[np.uint32]] = []
        for left, right, valid in pairs:
            counts = cast(
                HostArray, np.asarray(histogram(left, right, valid, 3, method=method))
            )
            assert counts.dtype == np.uint32
            assert counts.shape == (1, 3, 3)
            actual.append(cast(NDArray[np.uint32], counts.reshape(3, 3)))

    # Direction order is not a feature contract; compare the per-direction count multiset.
    assert sorted(array.tobytes() for array in actual) == sorted(
        array.tobytes() for array in expected
    )


@pytest.mark.parametrize("method", ["mlx", "metal"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (3, 0)])
def test_pair_counts_of_empty_inputs_are_empty(method: str, shape: tuple[int, int]) -> None:
    # Empty inputs hold no pairs and give all-zero counts for both methods.
    pairs = mx.zeros(shape, dtype=mx.int32)
    with mx.stream(mx.gpu):
        counts = cast(HostArray, np.asarray(histogram(pairs, pairs, pairs > 0, 3, method=method)))
    assert counts.dtype == np.uint32
    assert counts.shape == (shape[0], 3, 3)
    assert not np.any(counts)


@pytest.mark.parametrize("method", ["mlx", "metal"])
@pytest.mark.parametrize("constant", [False, True])
def test_rectangular_pair_counts_preserve_rows_and_mask(method: str, constant: bool) -> None:
    rng = np.random.default_rng(81)
    left = rng.integers(1, 5, (3, 1026), dtype=np.int32)
    right = rng.integers(1, 10, left.shape, dtype=np.int32)
    if constant:
        left.fill(4)
        right.fill(9)
    valid = rng.random(left.shape) > 0.2
    valid[1] = False
    left[~valid] = right[~valid] = 0
    expected = np.zeros((3, 4, 9), dtype=np.uint32)
    for row in range(3):
        for column in range(0, left.shape[1], 2):
            if valid[row, column]:
                expected[row, left[row, column] - 1, right[row, column] - 1] += 1
    actual = histogram(
        mx.array(left)[:, ::2], mx.array(right)[:, ::2], array_from_bool(valid)[:, ::2],
        4, columns=9, method=method,
    )
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.uint32), expected)


def test_resident_roi_counts_preserve_masked_signed_offsets() -> None:
    from mlxradiomics.numerics.histogram import roi_histograms

    bins = np.array([[1, 2, 0], [3, 1, 2]], dtype=np.int32)
    mask = bins != 0
    offsets = ((0, 1), (1, 0), (1, 1), (1, -1))
    # Enumerated transitions: horizontal 1->2 twice and 3->1; vertical 1->3
    # and 2->1; down-right 1->1 and 2->2; down-left only 2->3.
    expected = np.array([
        [[0, 2, 0], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    ], dtype=np.uint32)
    with mx.stream(mx.gpu):
        actual = roi_histograms(mx.array(bins), mx.array(mask), offsets, 3)
        np.testing.assert_array_equal(np.asarray(actual, dtype=np.uint32), expected)
