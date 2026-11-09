"""Sparse zone storage preserves original gray and size coordinates."""

from collections import Counter

import numpy as np
import pytest
from conftest import Conventions, settings_for
from radiomics_reference.catalogue import size_features

from mlxradiomics import Settings, extract
from mlxradiomics.features.matrices import zone_matrix
from mlxradiomics.features.sparse_size import sparse_size_statistics
from mlxradiomics.numerics.runtime import mx


def _run_reference(counts: Counter[tuple[int, int]]) -> dict[str, float]:
    """Run and zone definitions differ in the counted objects and feature names."""
    result: dict[str, float] = {}
    for name, value in size_features(counts, "glszm").items():
        for zone, run in (
            ("SmallArea", "ShortRun"), ("LargeArea", "LongRun"),
            ("SizeZone", "RunLength"), ("ZoneVariance", "RunVariance"),
            ("ZoneEntropy", "RunEntropy"), ("ZonePercentage", "RunPercentage"),
            ("GrayLevelZone", "GrayLevelRun"),
        ):
            name = name.replace(zone, run)
        result[name] = value
    return result


@pytest.mark.parametrize("family", ["gldm", "glrlm", "glszm", "gldzm"])
def test_occupied_size_cells_match_independent_integer_counts(family: str) -> None:
    # This virtual matrix would contain over 800 billion cells. The adjacent
    # zone sizes also exceed float32's consecutive-integer range: rounding the
    # coordinates before centering would erase their nonzero variance.
    large = 2**25
    first = Counter({(25000, large): 2, (25001, large + 1): 1, (25001, large): 3})
    second = Counter({(9, 3): 1})
    columns = large + 1
    keys = mx.array([
        [(i - 1) * columns + j - 1 for i, j in first],
        [(9 - 1) * columns + 3 - 1, 0, 0],
    ], dtype=mx.int64)
    counts = mx.array([list(first.values()), [1, 0, 0]], dtype=mx.int32)
    voxels = [sum(j * c for (_, j), c in table.items()) for table in (first, second)]
    actual = sparse_size_statistics(
        keys, counts, columns, family,
        voxel_count=mx.array(voxels, dtype=mx.int64) if family == "gldzm" else None,
    )
    for row, table in enumerate((first, second)):
        expected = (
            _run_reference(table)
            if family == "glrlm" else size_features(table, family, voxel_count=voxels[row])
        )
        assert set(actual) == set(expected)
        for name, value in expected.items():
            assert actual[name][row].item() == pytest.approx(value, rel=2e-6, abs=1e-30), name
    assert actual["GrayLevelVariance"][0].item() == pytest.approx(2 / 9, rel=2e-6)
    variance = {
        "gldm": "DependenceVariance", "glrlm": "RunVariance",
        "glszm": "ZoneVariance", "gldzm": "ZoneDistanceVariance",
    }[family]
    assert actual[variance][0].item() == pytest.approx(5 / 36, rel=2e-6)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_high_gray_single_zone_uses_bounded_storage_and_exact_indices(
    conventions: Conventions,
) -> None:
    shape = (4, 16, 32)
    width, gray = 2048, 25001
    expected = size_features(Counter({(gray, width): 1}), "glszm")
    settings = settings_for(
        conventions,
        features=tuple(f'glszm_{name}' for name in expected),
        bin_width=1,
        bin_anchor=0,
    )
    matrix = zone_matrix(
        mx.full((1, width), gray, dtype=mx.int32),
        mx.ones((1, width), dtype=mx.bool_), shape, gray, settings,
    )
    # A constant ROI needs one occupied count, independent of its 51-million-cell
    # virtual lattice. This guards the allocation, not merely successful output.
    assert matrix.column_count == width
    assert matrix.cell_keys is not None
    assert matrix.cell_keys.tolist() == [[gray * width - 1]]
    assert matrix.counts.tolist() == [[1]]
    result = extract(
        np.full(shape, gray - 1, dtype=np.int32), np.ones(shape, dtype=np.uint8),
        settings=settings,
    )
    assert set(result.features) == {f"original_glszm_{name}" for name in expected}
    for name, value in expected.items():
        assert result.features[f"original_glszm_{name}"] == pytest.approx(
            value, rel=2e-6, abs=1e-30,
        ), name


def test_merged_slices_preserve_different_zone_sizes_and_original_gray_levels() -> None:
    image = np.full((3, 32, 32), 25000, dtype=np.int32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, :16] = 1
    mask[2] = 1
    # The singleton slice fits the dense policy; the larger slices use sparse
    # counts with different column widths. Merging must preserve all three sizes.
    expected = size_features(Counter({(25001, 1): 1, (25001, 512): 1, (25001, 1024): 1}), "glszm")
    result = extract(image, mask, settings=Settings(
        features=tuple(f"glszm_{name}" for name in expected),
        bin_width=1, bin_anchor=0,
        force2d_axis=0, glszm_aggregation="2.5d_merge",
    ))
    for name, value in expected.items():
        assert result.features[f"original_glszm_{name}"] == pytest.approx(
            value, rel=2e-6, abs=1e-30,
        ), name
