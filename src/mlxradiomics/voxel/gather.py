"""Reusable Metal indexing for one batch of voxel neighborhoods.

Coordinate addition, bounds tests and edge clamping are shared by every resident
input sampled at the same centers. Feature arithmetic and batch scheduling belong
to the caller; this module only constructs exact integer lookup positions.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx


@dataclass(frozen=True)
class NeighborhoodIndex:
    """Clamped flat positions and the original, unclamped image-bounds mask."""

    flat: mx.array
    inside: mx.array

    def gather(self, source: mx.array) -> mx.array:
        """Sample a flat resident buffer, preserving its dtype and edge values."""
        return mx.take(source, self.flat)

    def valid(self, mask: mx.array) -> mx.array:
        """Require both image bounds and membership in the flat sampling mask."""
        return self.inside & self.gather(mask)


@lru_cache(maxsize=4)
def _index_kernel(ndim: int) -> MetalKernel:
    """Compile the ``ndim``-D neighborhood index kernel.

    One thread per ``(center, offset)`` performs an exact int32 bounds test, edge
    clamp and row-major index.
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_neighborhood_index",
        input_names=["centers", "offsets", "shape"], output_names=["flat", "inside"],
        source=f"""
        uint t = thread_position_in_grid.x;
        uint size = offsets_shape[0];
        if (t >= centers_shape[0] * size) return;
        uint center = t / size, offset = t % size;
        bool in_image = true;
        int index = 0;
        for (uint axis = 0; axis < {ndim}; ++axis) {{
            int extent = shape[axis];
            int coordinate = centers[center * {ndim} + axis] + offsets[offset * {ndim} + axis];
            in_image = in_image && coordinate >= 0 && coordinate < extent;
            index = index * extent + min(max(coordinate, 0), extent - 1);
        }}
        flat[t] = index;
        inside[t] = in_image;
        """,
    ))


class NeighborhoodPlan:
    """Retain a grid's small integer constants during one voxel extraction.

    The caller constructs this plan inside its explicit GPU stream, once per input
    grid, then passes int32 centers ``(C,D)`` and offsets ``(K,D)`` to :meth:`index`.
    The plan retains no per-batch arrays. Each returned index can serve values,
    discretized bins, masks, source bits, residuals and error bounds until that
    batch completes. One Metal pass per index performs the int32 index arithmetic.
    """

    def __init__(self, image_shape: tuple[int, ...]) -> None:
        """Upload the grid ``image_shape`` as int32 constants."""
        self._ndim = len(image_shape)
        self._shape = mx.array(image_shape, dtype=mx.int32)

    def index(self, centers: mx.array, offsets: mx.array) -> NeighborhoodIndex:
        """Build one ``(C,K)`` lookup, marking out-of-image positions invalid.

        One Metal pass forms each position's coordinates, bounds test, edge clamp and
        row-major index in int32. Indexes stay below the 2**30-voxel image limit, so
        this equals the native coordinate, comparison, clamp and stride operations
        without their ``(C,K,D)`` intermediates.
        """
        flat, inside = _index_kernel(self._ndim)(
            inputs=[centers, offsets, self._shape],
            grid=(centers.shape[0] * offsets.shape[0], 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(centers.shape[0], offsets.shape[0])] * 2,
            output_dtypes=[mx.int32, mx.bool_],
        )
        return NeighborhoodIndex(flat, inside)
