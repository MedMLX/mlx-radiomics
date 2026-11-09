"""Marching mesh construction and reusable geometry for shape features."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import (
    SpacingParts,
    cross_product,
    scale_pow2,
    spacing_parts,
)
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import array_from_bool, evaluate, mx


@dataclass(frozen=True)
class MeshTables:
    """Lookup tables from ``mesh_tables.json``, named after the tables in ``cshape.c``.

    ``corners`` are the cell-corner offsets (``gridAngles``), ``facets`` the full
    triangle table (``triTable``, 256 codes) or segment table (``lineTable2D``), and
    ``vertices`` the edge-midpoint offsets (``vertList``).
    """

    corners: mx.array
    facets: mx.array
    vertices: mx.array


@lru_cache(maxsize=2)
def _tables(ndim: int) -> MeshTables:
    """Load the 3-D marching-cubes or 2-D marching-squares tables from ``mesh_tables.json``."""
    data = cast(
        Mapping[str, list[list[int | float]]],
        json.loads(Path(__file__).with_name("mesh_tables.json").read_text()),
    )
    suffix = "2D" if ndim == 2 else ""
    return MeshTables(
        mx.array(data["gridAngles" + suffix]),
        mx.array(data["lineTable2D" if ndim == 2 else "triTable"]),
        mx.array(data["vertList" + suffix]),
    )


@dataclass(frozen=True)
class MeshGeometry:
    """Scaled mesh measures plus the grid-space boundary vertices.

    ``boundary`` (perimeter/area) and ``enclosed`` (area/volume) are the physical
    float32 values for callers that request them. The ``*_norm`` fields are the
    corresponding mantissas; they remain finite even when the physical value is outside
    float32, so dimensionless densities and sphericity can still be evaluated. Vertices
    stay in grid coordinates; the diameter kernel applies spacing per component.
    """

    boundary: mx.array
    boundary_norm: mx.array
    boundary_exponent: int
    enclosed: mx.array
    enclosed_norm: mx.array
    enclosed_exponent: int
    grid_enclosed: mx.array
    vertices: mx.array


@lru_cache(maxsize=4)
def _compact_kernel(ndim: int) -> MetalKernel:
    """Compile the kernel that scatters valid ``ndim``-D points to their compacted rows."""
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_compact_mesh",
            input_names=["points", "valid", "indexes"],
            output_names=["compact"],
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= valid_shape[0] || !valid[t]) return;
        for (int d = 0; d < {ndim}; d++)
            compact[(indexes[t] - 1) * {ndim} + d] = points[t * {ndim} + d];
        """,
        ),
    )


def _compact(points: mx.array, valid: mx.array, ndim: int) -> mx.array:
    """Return the rows of ``points`` where ``valid`` is set, in their original order."""
    points, valid = mx.reshape(points, (-1, ndim)), mx.reshape(valid, (-1,))
    indexes = mx.cumsum(valid.astype(mx.int32))
    count = int(cast(int, indexes[-1].item()))
    if count < 1:
        return mx.zeros((0, ndim), dtype=mx.float32)
    return _compact_kernel(ndim)(
        inputs=[points, valid, indexes],
        grid=(valid.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(count, ndim)],
        output_dtypes=[mx.float32],
    )[0]


# Largest padded cropped extent per axis whose half-grid vertices are exact in float32.
_MAX_MESH_EXTENT = 2**22


def _mesh_chunk(
    padded: mx.array,
    coords: mx.array,
    parts: SpacingParts,
    *,
    include_vertices: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Process the mesh cells whose origins in the padded mask are ``coords``.

    Returns the summed boundary measure normalized by ``2**boundary_exponent``, the
    summed signed enclosed measure in grid units, and the boundary vertices in grid
    coordinates.
    """
    ndim = len(parts.mantissa)
    tables = _tables(ndim)
    corners, facets, vertex = tables.corners, tables.facets, tables.vertices
    strides = mx.array([prod(padded.shape[i + 1 :]) for i in range(ndim)], dtype=mx.int32)
    indexes = mx.sum((coords[:, None, :] + corners[None, :, :]) * strides, axis=2)
    occupancy = mx.take(mx.reshape(padded, (-1,)), indexes)
    code = mx.sum(occupancy.astype(mx.int32) * mx.array([1 << i for i in range(2**ndim)]), 1)
    selected = mx.take(facets, code, axis=0)
    selected = selected[:, : 15 if ndim == 3 else 4]
    valid = selected[:, ::ndim] >= 0
    points = coords[:, None, :] + mx.take(vertex, mx.maximum(selected, 0), axis=0)
    points = mx.reshape(points.astype(mx.float32), (-1, 5 if ndim == 3 else 2, ndim, ndim))
    mantissa = mx.array(parts.mantissa, dtype=mx.float32)
    e0, e1, e2 = (parts.exponent + (0, 0, 0))[:3]
    if ndim == 3:
        # Signed tetrahedron volumes, 6V = cross_product(a-c, b-c).c. The full 256-row table
        # orients every triangle outward, so no complement sign correction is needed.
        # Evaluate the enclosed measure on the exact half-index grid and apply the
        # spacing determinant after summation; physical-coordinate products lose
        # precision through cancellation on large or disconnected anisotropic ROIs.
        ia, ib, ic = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        measure = mx.sum(cross_product(ia - ic, ib - ic) * ic, axis=2)
        # Cross components separate into grid factors and one spacing-axis pair:
        # c0 = s1*s2*C0, c1 = s2*s0*C1, c2 = s0*s1*C2. Evaluate each component from its
        # pair exponent so a face whose naturally scaled area is representable does not
        # overflow or underflow merely because another spacing axis is extreme.
        edge_a, edge_b = (ia - ic) * mantissa, (ib - ic) * mantissa
        c0 = edge_a[..., 1] * edge_b[..., 2] - edge_a[..., 2] * edge_b[..., 1]
        c1 = edge_a[..., 2] * edge_b[..., 0] - edge_a[..., 0] * edge_b[..., 2]
        c2 = edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0]
        n0 = scale_pow2(c0, e1 + e2 - parts.boundary_exponent)
        n1 = scale_pow2(c1, e2 + e0 - parts.boundary_exponent)
        n2 = scale_pow2(c2, e0 + e1 - parts.boundary_exponent)
        largest = mx.maximum(mx.maximum(mx.abs(n0), mx.abs(n1)), mx.abs(n2))
        safe = mx.where(largest > 0, largest, 1)
        normalized = mx.stack([n0 / safe, n1 / safe, n2 / safe], axis=-1)
        boundary = 0.5 * safe * mx.sqrt(mx.sum(normalized * normalized, axis=-1))
        boundary = mx.where(valid, boundary, 0)
        edge_ends, edges = ((7, 6), (7, 4), (7, 3)), (6, 7, 11)
    else:
        # Difference first on the exact half-index grid: this avoids both subtracting
        # large physical products and losing edge lengths between distant components.
        ia, ib = points[:, :, 0], points[:, :, 1]
        delta = ib - ia
        measure = ia[:, :, 0] * delta[:, :, 1] - ia[:, :, 1] * delta[:, :, 0]
        edge = delta * mantissa
        n0 = scale_pow2(edge[..., 0], e0 - parts.boundary_exponent)
        n1 = scale_pow2(edge[..., 1], e1 - parts.boundary_exponent)
        largest = mx.maximum(mx.abs(n0), mx.abs(n1))
        safe = mx.where(largest > 0, largest, 1)
        boundary = safe * mx.sqrt((n0 / safe) * (n0 / safe) + (n1 / safe) * (n1 / safe))
        boundary = mx.where(valid, boundary, 0)
        edge_ends, edges = ((3, 0), (3, 2)), (3, 2)
    if not include_vertices:
        return (
            mx.sum(mx.where(valid, boundary, 0)),
            mx.sum(mx.where(valid, measure, 0)),
            mx.zeros((0, ndim), dtype=mx.float32),
        )
    starts = mx.array([start for start, _ in edge_ends], dtype=mx.int32)
    ends = mx.array([end for _, end in edge_ends], dtype=mx.int32)
    vertex_valid = ((code[:, None] >> starts) & 1) != ((code[:, None] >> ends) & 1)
    edge_points = coords[:, None, :] + mx.take(vertex, mx.array(edges), axis=0)[None]
    return (
        mx.sum(mx.where(valid, boundary, 0)),
        mx.sum(mx.where(valid, measure, 0)),
        _compact(edge_points.astype(mx.float32), vertex_valid, ndim),
    )


def mesh_geometry(
    mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    scratch_mib: int,
    *,
    include_vertices: bool = True,
    coordinates: NDArray[np.int64] | None = None,
) -> MeshGeometry:
    """Build the mesh of a boolean mask and return its scaled measures and vertices.

    The mask is cropped to the ROI bounding box and padded by one voxel. The 3-D volume
    integral is evaluated in that cropped frame; the closed surface makes it independent
    of the ROI's position in the source array. Cells are processed in chunks of
    ``scratch_mib * 1024**2 // 1024`` cells. Spacing is decomposed into binary mantissa
    and exponent fields so boundary and enclosed measures can be restored at the end
    without squaring or cubing an unrepresentable intermediate. ``coordinates`` optionally
    supplies the precomputed ``np.argwhere(mask)`` rows.

    Raises:
        ValueError: If the mesh has no boundary vertices, spacing is invalid, or the
            padded cropped ROI extent exceeds ``2**22`` voxels on an axis.
    """
    parts = spacing_parts(spacing)
    roi_coords = np.argwhere(mask) if coordinates is None else coordinates
    offset = tuple(int(roi_coords[:, i].min()) for i in range(mask.ndim))
    bounds = tuple(slice(start, int(roi_coords[:, i].max()) + 1) for i, start in enumerate(offset))
    # Mesh vertices are float32 half-grid coordinates in the cropped frame. Above 2**23
    # half-integers are no longer representable and distant components would get
    # distorted local edges, so the cropped extent between selected voxels is bounded.
    if max(stop.stop - stop.start for stop in bounds) + 2 > _MAX_MESH_EXTENT:
        raise ValueError(
            f"shape mesh requires a cropped ROI extent below {_MAX_MESH_EXTENT} voxels per "
            "axis for exact float32 half-grid vertices"
        )
    padded = array_from_bool(np.pad(mask[bounds], 1))
    shape = tuple(s - 1 for s in padded.shape)
    width = prod(shape)
    strides = [prod(shape[i + 1 :]) for i in range(mask.ndim)]
    batch_size = max(1, scratch_mib * 1024**2 // 1024)
    boundary_norm = mx.array(0, dtype=mx.float32)
    enclosed = mx.array(0, dtype=mx.float32)
    boundary_points: list[mx.array] = []
    for start in range(0, width, batch_size):
        indexes = mx.arange(start, min(width, start + batch_size), dtype=mx.int32)
        coords = mx.stack(
            [(indexes // stride) % size for stride, size in zip(strides, shape, strict=True)],
            axis=1,
        )
        part_boundary, part_enclosed, vertices = _mesh_chunk(
            padded,
            coords,
            parts,
            include_vertices=include_vertices,
        )
        boundary_norm, enclosed = boundary_norm + part_boundary, enclosed + part_enclosed
        evaluate(boundary_norm, enclosed, vertices)
        if include_vertices:
            boundary_points.append(vertices)
    points = (
        mx.concatenate(boundary_points, axis=0)
        if include_vertices
        else mx.zeros((0, mask.ndim), dtype=mx.float32)
    )
    if include_vertices and points.shape[0] == 0:
        raise ValueError("mask has no mesh boundary vertices")
    grid_enclosed = mx.abs(enclosed)
    divisor = 6 if mask.ndim == 3 else 2
    enclosed_norm = grid_enclosed * (prod(parts.mantissa) / divisor)
    boundary = scale_pow2(boundary_norm, parts.boundary_exponent)
    physical_enclosed = scale_pow2(enclosed_norm, parts.enclosed_exponent)
    evaluate(boundary, physical_enclosed)
    return MeshGeometry(
        boundary=boundary,
        boundary_norm=boundary_norm,
        boundary_exponent=parts.boundary_exponent,
        enclosed=physical_enclosed,
        enclosed_norm=enclosed_norm,
        enclosed_exponent=parts.enclosed_exponent,
        grid_enclosed=grid_enclosed,
        vertices=points,
    )


@dataclass
class MeshGeometryContext:
    """Lazily share one mask's coordinates and mesh within an extraction call.

    The bound mask, spacing and budget must remain unchanged while the context is in
    use. Nothing is retained across public calls. A scalar-only consumer can omit
    boundary-vertex compaction; a later vertex consumer upgrades the cached geometry.
    """

    mask: NDArray[np.bool_]
    spacing: tuple[float, ...]
    scratch_mib: int
    _coordinates: NDArray[np.int64] | None = field(default=None, init=False, repr=False)
    _geometry: MeshGeometry | None = field(default=None, init=False, repr=False)
    _has_vertices: bool = field(default=False, init=False, repr=False)

    def coordinates(self) -> NDArray[np.int64]:
        """Return the ``(n, ndim)`` mask coordinates, computed once."""
        if self._coordinates is None:
            self._coordinates = np.argwhere(self.mask).astype(np.int64, copy=False)
        return self._coordinates

    def get(self, *, include_vertices: bool = False) -> MeshGeometry:
        """Return the mask's mesh geometry, building it on first use.

        A cached geometry built without vertices is rebuilt when ``include_vertices`` is
        requested.
        """
        if self._geometry is None or include_vertices and not self._has_vertices:
            self._geometry = mesh_geometry(
                self.mask,
                self.spacing,
                self.scratch_mib,
                include_vertices=include_vertices,
                coordinates=self.coordinates(),
            )
            self._has_vertices = include_vertices
        return self._geometry
