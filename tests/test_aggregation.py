"""Public texture aggregation against independent integer enumeration."""

from collections import Counter
from itertools import product
from math import fsum, isnan
from typing import Literal

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray
from radiomics_reference.catalogue import glcm_matrix, size_features
from radiomics_reference.mathematics import ngtdm_features

from mlxradiomics import Settings
from mlxradiomics.config.settings import (
    FAMILIES,
    DirectionalAggregation,
    MatrixAggregation,
)
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract

type Counts = Counter[tuple[int, int]]


def _directions(ndim: int) -> list[tuple[int, ...]]:
    return [
        offset
        for offset in product((-1, 0, 1), repeat=ndim)
        if any(offset) and next(v for v in offset if v) > 0
    ]


def _shift(index: tuple[int, ...], offset: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(index, offset, strict=True))


def _inside(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= value < size for value, size in zip(index, shape, strict=True))


def _direction_counts(
    bins: NDArray[np.int32],
    mask: NDArray[np.bool_],
    family: str,
) -> list[Counts]:
    matrices: list[Counts] = []
    for delta in _directions(bins.ndim):
        counts: Counts = Counter()
        for index in np.ndindex(bins.shape):
            if not mask[index]:
                continue
            level = int(bins[index])
            if family == "glcm":
                other = _shift(index, delta)
                if _inside(other, bins.shape) and mask[other]:
                    partner = int(bins[other])
                    counts[level, partner] += 1
                    counts[partner, level] += 1
            else:
                previous = _shift(index, tuple(-v for v in delta))
                if _inside(previous, bins.shape) and mask[previous] and bins[previous] == level:
                    continue
                length = 1
                following = _shift(index, delta)
                while (
                    _inside(following, bins.shape) and mask[following] and bins[following] == level
                ):
                    length += 1
                    following = _shift(following, delta)
                counts[level, length] += 1
        matrices.append(counts)
    return matrices


def _run_features(counts: Counts) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in size_features(counts, "glszm").items():
        for old, new in (
            ("SmallArea", "ShortRun"),
            ("LargeArea", "LongRun"),
            ("SizeZone", "RunLength"),
            ("ZoneVariance", "RunVariance"),
            ("ZoneEntropy", "RunEntropy"),
            ("ZonePercentage", "RunPercentage"),
            ("GrayLevelZone", "GrayLevelRun"),
        ):
            name = name.replace(old, new)
        result[name] = value
    return result


def _reference_directional(
    bins: NDArray[np.int32],
    mask: NDArray[np.bool_],
    family: str,
    mode: str,
    axis: int,
) -> dict[str, float]:
    if mode.startswith("3d"):
        grouped = [_direction_counts(bins, mask, family)]
    else:
        grouped = [
            _direction_counts(np.take(bins, i, axis=axis), np.take(mask, i, axis=axis), family)
            for i in range(bins.shape[axis])
        ]
    if mode in {"2d_average", "3d_average"}:
        matrices = [counts for row in grouped for counts in row]
    elif mode == "2d_slice_merge":
        matrices = [sum(row, Counter[tuple[int, int]]()) for row in grouped]
    elif mode == "2.5d_direction_merge":
        matrices = [
            sum((row[i] for row in grouped), Counter[tuple[int, int]]())
            for i in range(len(grouped[0]))
        ]
    else:
        matrices = [sum((counts for row in grouped for counts in row), Counter[tuple[int, int]]())]
    values = [
        glcm_matrix(dict(counts), int(bins.max())) if family == "glcm" else _run_features(counts)
        for counts in matrices
        if counts
    ]
    result: dict[str, float] = {}
    for name in values[0]:
        samples = [value[name] for value in values if name != "MCC" or not isnan(value[name])]
        result[name] = fsum(samples) / len(samples) if samples else float("nan")
    return result


def _fixture() -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    image = np.ones((4, 4, 5), dtype=np.int32)
    image[1] = 1 + np.indices((4, 5)).sum(axis=0) % 3
    image[3] = 3
    mask = np.ones(image.shape, dtype=bool)
    mask[1, 1:3, 2:4] = False
    mask[2] = False
    mask[3, :2] = False
    return image, mask


@pytest.mark.parametrize("family,large_lattice", [
    ("glcm", False), ("glcm", True), ("glrlm", False),
])
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize(
    "mode",
    [
        "2d_average",
        "2d_slice_merge",
        "2.5d_direction_merge",
        "2.5d_merge",
        "3d_average",
        "3d_merge",
    ],
)
def test_directional_aggregation(
    family: str,
    large_lattice: bool,
    axis: int,
    mode: DirectionalAggregation,
) -> None:
    image, mask = _fixture()
    if large_lattice:
        image = np.array([16001, 17001, 25001], dtype=np.int32)[image - 1]
    image, mask = np.moveaxis(image, 0, axis), np.moveaxis(mask, 0, axis)
    settings = Settings(
        features=tuple(f"{family}_{name}" for name in FAMILIES[family]),
        bin_width=1,
        bin_anchor=1,
        slice_axis=axis,
        glcm_aggregation=mode,
        glrlm_aggregation=mode,
    )
    actual = extract_original(image, mask, settings=settings).features
    expected = _reference_directional(image, mask, family, mode, axis)
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[f"original_{family}_{name}"],
            value,
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
            err_msg=name,
        )


def test_2d_average_weights_each_valid_direction_matrix_equally() -> None:
    image = np.zeros((2, 3, 3), dtype=np.float64)
    image[0, 1] = (0, 1, 0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 1] = 1
    mask[1] = 1
    actual = extract_original(
        image, mask,
        settings=Settings(features=("glcm_Contrast",), bin_width=1,
                          glcm_aggregation="2d_average", slice_axis=0),
    )
    # The first slice has one direction with contrast one; the second has four
    # directions with contrast zero. The declared policy averages all five matrices.
    np.testing.assert_allclose(actual.features["original_glcm_Contrast"], 1 / 5,
                               rtol=1e-6, atol=1e-7)


def _zone_counts(
    bins: NDArray[np.int32],
    intensity: NDArray[np.bool_],
    morphology: NDArray[np.bool_],
    distance_zone: bool,
) -> Counts:
    remaining = {index for index in np.ndindex(bins.shape) if intensity[index]}
    neighbors = [d for d in product((-1, 0, 1), repeat=bins.ndim) if any(d)]
    outside = [
        tuple(v - 1 for v in index)
        for index in np.ndindex(tuple(s + 2 for s in bins.shape))
        if not all(1 <= v <= size for v, size in zip(index, bins.shape, strict=True))
        or not morphology[tuple(v - 1 for v in index)]
    ]
    counts: Counts = Counter()
    while remaining:
        seed = remaining.pop()
        zone, queue = {seed}, [seed]
        while queue:
            index = queue.pop()
            for delta in neighbors:
                other = _shift(index, delta)
                if other in remaining and bins[other] == bins[seed]:
                    remaining.remove(other)
                    zone.add(other)
                    queue.append(other)
        column = (
            min(
                sum(abs(a - b) for a, b in zip(index, out, strict=True))
                for index in zone
                for out in outside
            )
            if distance_zone
            else len(zone)
        )
        counts[int(bins[seed]), column] += 1
    return counts


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("family", ["glszm", "gldzm"])
@pytest.mark.parametrize("mode", ["2d_average", "2.5d_merge", "3d", "planar_input"])
def test_zone_aggregation_and_morphological_distance(
    family: str, mode: MatrixAggregation | Literal["planar_input"], conventions: Conventions,
) -> None:
    image, morphology = _fixture()
    # The low-valued border remains in the morphology mask but is resegmented out of
    # the intensity mask. Its boundary distance must still be used by the GLDZM.
    image[1, 0, :] = 0
    intensity = morphology & (image >= 1)
    # "planar_input" extracts one 2D plane with the default aggregation.
    aggregation: MatrixAggregation = "native" if mode == "planar_input" else mode
    settings = settings_for(
        conventions,
        features=tuple(f'{family}_{name}' for name in FAMILIES[family]),
        bin_width=1,
        bin_anchor=1,
        resegment_range=(1.0,),
        glszm_aggregation=aggregation,
        gldzm_aggregation=aggregation,
    )
    if mode == "planar_input":
        image, intensity, morphology = image[1], intensity[1], morphology[1]
        domains = [(image, intensity, morphology)]
    elif mode == "3d":
        domains = [(image, intensity, morphology)]
    else:
        domains = [(image[i], intensity[i], morphology[i]) for i in range(image.shape[0])]
    matrices = [
        (_zone_counts(b, m, morph, family == "gldzm"), int(m.sum()))
        for b, m, morph in domains
        if m.any()
    ]
    if mode == "2.5d_merge":
        matrices = [
            (sum((c for c, _ in matrices), Counter[tuple[int, int]]()), sum(n for _, n in matrices))
        ]
    expected = [size_features(c, family, voxel_count=n) for c, n in matrices]
    actual = extract_original(image, morphology, settings=settings).features
    for name in expected[0]:
        target = fsum(row[name] for row in expected) / len(expected)
        np.testing.assert_allclose(
            actual[f"original_{family}_{name}"], target, rtol=2e-5, atol=2e-6, err_msg=name
        )


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mode", ["maps", "stream"])
def test_aggregated_maps_use_the_same_local_matrices(
    mode: str, ndim: int, conventions: Conventions,
) -> None:
    image, mask = _fixture()
    zones = tuple(f"gldzm_{name}" for name in FAMILIES["gldzm"])
    if ndim == 2:
        # One plane with default aggregation: the local zones are the window itself.
        image, mask = image[1], mask[1]
        settings = settings_for(
            conventions,
            features=zones,
            bin_width=1,
            bin_anchor=1,
            scratch_mib=1,
        )
    else:
        settings = settings_for(
            conventions,
            features=('glcm_JointEntropy', *zones),
            bin_width=1,
            bin_anchor=1,
            glcm_aggregation='2d_slice_merge',
            gldzm_aggregation='2.5d_merge',
            scratch_mib=1,
        )
    names = tuple(f"original_{name}" for name in settings.features)
    if mode == "stream":
        with voxel_map_store(image.shape, names) as store:
            result = extract(
                image, mask, settings=settings, mode="voxel", sink=store.append
            )
            actual = {name: array.copy() for name, array in store.maps()}
        assert not result.features
    else:
        result = extract_original(image, mask, settings=settings, voxel=True)
        actual = {name: np.asarray(value) for name, value in result.features.items()}
    for index in np.ndindex(image.shape):
        if not mask[index]:
            assert all(values[index] == 0 for values in actual.values())
            continue
        slices = tuple(
            slice(max(0, i - 1), min(s, i + 2)) for i, s in zip(index, image.shape, strict=True)
        )
        b, m = image[slices], mask[slices]
        planes = [(b, m)] if ndim == 2 else [(b[i], m[i]) for i in range(b.shape[0])]
        counts = sum(
            (_zone_counts(plane, valid, valid, True) for plane, valid in planes),
            Counter[tuple[int, int]](),
        )
        expected = size_features(counts, "gldzm", voxel_count=int(m.sum()))
        for name, target in expected.items():
            np.testing.assert_allclose(actual[f"original_gldzm_{name}"][index], target,
                                       rtol=2e-5, atol=2e-6, equal_nan=True, err_msg=name)
        if ndim == 3:
            entropy = _reference_directional(b, m, "glcm", "2d_slice_merge", 0)["JointEntropy"]
            np.testing.assert_allclose(actual["original_glcm_JointEntropy"][index], entropy,
                                       rtol=2e-5, atol=2e-6)
    if ndim == 3:
        assert result.metadata.get("aggregation") == {
            "glcm": "2d_slice_merge", "gldzm": "2.5d_merge",
        }


def test_ibsi_neighborhood_radius_includes_inner_shells() -> None:
    image, mask = _fixture()
    names = ("gldm_DependenceVariance", "ngtdm_Contrast")
    ibsi = Settings(
        features=names,
        bin_width=1,
        bin_anchor=1,
        distances=(2,),
        neighborhood_radius=2,
    )
    offsets = [delta for delta in product(range(-2, 3), repeat=3) if any(delta)]
    dependences: Counts = Counter()
    for index in np.ndindex(image.shape):
        if not mask[index]:
            continue
        size = 1
        for delta in offsets:
            other = _shift(index, delta)
            if _inside(other, image.shape) and mask[other] and image[other] == image[index]:
                size += 1
        dependences[int(image[index]), size] += 1
    expected = {
        "original_gldm_DependenceVariance": size_features(dependences, "gldm")[
            "DependenceVariance"
        ],
        "original_ngtdm_Contrast": ngtdm_features(image, mask, distances=(1, 2))["Contrast"],
    }
    actual = extract_original(image, mask, settings=ibsi).features
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-5, atol=2e-6)


def _dependence_counts(bins: NDArray[np.int32], mask: NDArray[np.bool_]) -> Counts:
    counts: Counts = Counter()
    adjacent = [d for d in product((-1, 0, 1), repeat=bins.ndim) if any(d)]
    for index in np.ndindex(bins.shape):
        if not mask[index]:
            continue
        count = 1
        for delta in adjacent:
            other = _shift(index, delta)
            if _inside(other, bins.shape) and mask[other] and bins[index] == bins[other]:
                count += 1
        counts[int(bins[index]), count] += 1
    return counts


@pytest.mark.parametrize("mode", ["2d_average", "2.5d_merge", "3d"])
def test_neighborhood_aggregation_against_independent_domains(
    mode: MatrixAggregation,
) -> None:
    image, mask = _fixture()
    # Unequal masks and an empty slice distinguish averaging from merging.
    if mode == "3d":
        domains = [(image, mask)]
    else:
        domains = [(image[i], mask[i]) for i in range(image.shape[0]) if mask[i].any()]
    matrices = [_dependence_counts(b, m) for b, m in domains]
    if mode == "2.5d_merge":
        matrices = [sum(matrices, Counter[tuple[int, int]]())]
        # Empty separators independently realize the sum of slice NGTDM matrices:
        # no radius-one neighborhood can cross between the retained slices.
        separated = np.zeros((2 * len(domains) - 1, *image.shape[1:]), dtype=np.int32)
        separated_mask = np.zeros(separated.shape, dtype=np.bool_)
        for i, (b, m) in enumerate(domains):
            separated[2 * i], separated_mask[2 * i] = b, m
        ngtdm = [ngtdm_features(separated, separated_mask)]
    else:
        ngtdm = [ngtdm_features(b, m) for b, m in domains]
    gldm = [size_features(counts, "gldm") for counts in matrices]
    # Nonlinear features depend on both merged counts and sample denominators, so every
    # family member distinguishes averaging from merging.
    expected = {
        **{f"gldm_{name}": fsum(row[name] for row in gldm) / len(gldm)
           for name in FAMILIES["gldm"]},
        **{f"ngtdm_{name}": fsum(row[name] for row in ngtdm) / len(ngtdm)
           for name in FAMILIES["ngtdm"]},
    }
    result = extract_original(
        image,
        mask,
        settings=settings_for("pyradiomics",
            features=tuple(expected),
            bin_width=1,
            bin_anchor=1,
            gldm_aggregation=mode,
            ngtdm_aggregation=mode,
        ),
    )
    for name, target in expected.items():
        np.testing.assert_allclose(
            result.features[f"original_{name}"], target, rtol=2e-5, atol=2e-6,
            equal_nan=True, err_msg=name,
        )
