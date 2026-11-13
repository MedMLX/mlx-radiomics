"""Public ROI, voxel-map and streamed values from hand or scalar mathematics.

No expected value comes from a recorded result, from a second extraction, or from
production feature arithmetic.
"""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.fixtures import Case, cases
from radiomics_reference.mathematics import masked_neighborhood_mean
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings, extract
from mlxradiomics.contracts.types import HostArray
from mlxradiomics.features.cooccurrence import roi_pairs
from mlxradiomics.numerics.histogram import histogram
from mlxradiomics.numerics.runtime import array_from_bool, mx
from mlxradiomics.voxel.storage import voxel_map_store

_OFFSETS = ((1, 1), (1, 0), (1, -1), (0, 1))


@pytest.mark.parametrize("stream", [False, True], ids=["maps", "streamed"])
def test_roi_and_voxel_maps_match_source_mean(stream: bool) -> None:
    image, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    # Every clipped radius-one neighborhood contains all four real liver pixels.
    assert np.all(mask == 1)
    expected_mean = sum(float(value) for value in image.flat) / image.size
    settings = Settings(features=("firstorder_Mean",), bin_width=25.0)
    roi = extract(image, mask, settings=settings)
    assert roi.metadata["roi_voxels"] == image.size
    mean = roi.features["original_firstorder_Mean"]
    assert isinstance(mean, float)
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=0)

    names = ("original_firstorder_Mean",)
    if stream:
        with voxel_map_store(image.shape, names) as store:
            result = extract(image, mask, settings=settings, mode="voxel",
                             sink=store.append)
            assert not result.features
            maps: dict[str, NDArray[np.float32]] = {
                name: values.copy() for name, values in store.maps()
            }
    else:
        result = extract(image, mask, settings=settings, mode="voxel")
        maps = {name: np.asarray(value, dtype=np.float32)
                for name, value in result.features.items()}

    assert result.metadata["roi_voxels"] == image.size
    assert set(maps) == set(names)
    np.testing.assert_allclose(
        np.asarray(maps["original_firstorder_Mean"], dtype=np.float64),
        expected_mean,
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


@pytest.mark.parametrize("masked_kernel", [True, False], ids=["masked", "unmasked"])
def test_voxel_neighborhood_sampling_matches_scalar_means(masked_kernel: bool) -> None:
    image, mask, _ = hcc_crop(np.s_[105:108, 215:218, 62])
    assert mask.any() and not mask.all()
    expected = (masked_neighborhood_mean(image, mask == 1) if masked_kernel else
                np.asarray([
                    [image[max(0, row - 1):row + 2, max(0, column - 1):column + 2].mean()
                     if mask[row, column] else 0 for column in range(image.shape[1])]
                    for row in range(image.shape[0])
                ], dtype=np.float64))
    settings = Settings(features=("firstorder_Mean",), kernel_radius=1,
                        masked_kernel=masked_kernel)
    result = extract(image, mask, settings=settings, mode="voxel")

    assert result.metadata["roi_voxels"] == int(np.count_nonzero(mask))
    actual = np.asarray(result.features["original_firstorder_Mean"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["mlx", "metal"])
def test_integer_pair_counts_match_scalar_enumeration(method: str) -> None:
    image, liver, _ = hcc_crop(np.s_[105:108, 215:218, 62])
    bins = ((image - image.min()) // 25 + 1).astype(np.int32)
    mask = liver == 1
    levels = int(bins.max())
    settings = Settings(features=("glcm_Contrast",), distances=(1,))

    expected: list[NDArray[np.uint32]] = []
    for offset in _OFFSETS:
        counts = np.zeros((levels, levels), dtype=np.uint32)
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
                HostArray, np.asarray(histogram(left, right, valid, levels, method=method))
            )
            assert counts.dtype == np.uint32
            assert counts.shape == (1, levels, levels)
            actual.append(cast(NDArray[np.uint32], counts.reshape(levels, levels)))

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

    image, liver, _ = hcc_crop(np.s_[105:108, 215:218, 62])
    bins = ((image - image.min()) // 25 + 1).astype(np.int32)
    mask = liver == 1
    bins[~mask] = 0
    levels = int(bins.max())
    offsets = ((0, 1), (1, 0), (1, 1), (1, -1))
    expected = np.zeros((len(offsets), levels, levels), dtype=np.uint32)
    for direction, (dr, dc) in enumerate(offsets):
        for row, column in np.argwhere(mask):
            nr, nc = row + dr, column + dc
            if 0 <= nr < bins.shape[0] and 0 <= nc < bins.shape[1] and mask[nr, nc]:
                expected[direction, bins[row, column] - 1, bins[nr, nc] - 1] += 1
    with mx.stream(mx.gpu):
        actual = roi_histograms(mx.array(bins), mx.array(mask), offsets, levels)
        np.testing.assert_array_equal(np.asarray(actual, dtype=np.uint32), expected)
