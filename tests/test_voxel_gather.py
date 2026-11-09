"""Exact voxel-neighborhood bounds, mask selection and source-bit lookup."""

import numpy as np

from mlxradiomics.numerics.runtime import mx
from mlxradiomics.voxel.gather import NeighborhoodPlan


def test_2d_clamped_values_keep_outside_positions_invalid() -> None:
    plan = NeighborhoodPlan((2, 3))
    positions = plan.index(
        mx.array([[0, 0], [1, 2]], dtype=mx.int32),
        mx.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]], dtype=mx.int32),
    )
    expected = np.array([[0, 0, 0, 1, 3], [2, 4, 5, 5, 5]], dtype=np.int32)
    inside = np.array([[False, False, True, True, True],
                       [True, True, True, False, False]], dtype=np.bool_)
    sampling = mx.array([True, False, True, True, False, True], dtype=mx.bool_)
    valid = np.array([[False, False, True, False, True],
                      [True, False, True, False, False]], dtype=np.bool_)
    np.testing.assert_array_equal(np.asarray(positions.flat, dtype=np.int32), expected)
    np.testing.assert_array_equal(np.asarray(positions.inside, dtype=np.bool_), inside)
    np.testing.assert_array_equal(np.asarray(positions.valid(sampling), dtype=np.bool_), valid)
    np.testing.assert_array_equal(
        np.asarray(positions.gather(mx.arange(6, dtype=mx.int32)), dtype=np.int32), expected,
    )


def test_3d_flattening_preserves_all_source_bits_at_edges() -> None:
    positions = NeighborhoodPlan((2, 2, 2)).index(
        mx.array([[0, 1, 1], [1, 0, 0]], dtype=mx.int32),
        mx.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, -1, -1]],
                 dtype=mx.int32),
    )
    expected = np.array([[3, 3, 3, 7, 0], [4, 5, 6, 4, 0]], dtype=np.int32)
    bits = np.array([0, 0x8000000000000000, 1, 0x000FFFFFFFFFFFFF,
                     0x3FB999999999999A, 0x7FEFFFFFFFFFFFFF,
                     0x7FF8000000000001, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    gathered = positions.gather(mx.array(bits, dtype=mx.uint64))
    assert gathered.dtype == mx.uint64
    assert positions.flat.dtype == mx.int32
    np.testing.assert_array_equal(np.asarray(gathered, dtype=np.uint64), bits[expected])
    np.testing.assert_array_equal(
        np.asarray(positions.inside, dtype=np.bool_),
        np.array([[True, False, False, True, False],
                  [True, True, True, False, False]], dtype=np.bool_),
    )

