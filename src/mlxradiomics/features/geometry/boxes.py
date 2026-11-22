"""Oriented bounding boxes from convex-hull candidate orientations."""

from typing import cast

import numpy as np
import scipy.spatial as spatial
from numpy.typing import NDArray

from mlxradiomics.features.geometry.hull import hull_vertex_simplices, scipy_hull
from mlxradiomics.numerics.runtime import evaluate, mx

# Float32 projection elements retained per OMBB candidate block (16 MiB).
_OMBB_BLOCK_ELEMENTS = 4 * 1024**2


# Float64 silhouette products retained per OMBB face block (8 MiB per product).
_OMBB_SILHOUETTE_ELEMENTS = 1024**2


def _row_dots(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return broadcast 3-vector dot products summed in index order from ``+0.0``.

    This is the per-row arithmetic of NumPy's ``(n, 3) @ (3,)`` matrix-vector product
    with this BLAS, including the sign of exact zeros, so batching the per-normal
    products over many normals does not change a silhouette decision or a candidate.
    """
    total = 0.0 + left[..., 0] * right[..., 0]
    total = total + left[..., 1] * right[..., 1]
    return total + left[..., 2] * right[..., 2]


def _ombb_candidate_axes(
    points: NDArray[np.float64],
    simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Enumerate face-flush OMBB orientations of a 3-D convex hull.

    For every hull face normal ``u``, each silhouette edge ``e`` supplies a second axis
    ``v = normalize(e - (e·u)u)`` and ``w = u x v``. For a fixed normal ``u``, the
    minimum-area rectangle enclosing the projected hull has a side collinear with a
    projected hull edge (Freeman and Shapira), and projected hull edges are projections
    of silhouette edges, so the search is exact over boxes with one face flush with a
    hull face. O'Rourke showed that the unrestricted 3-D optimum need only have two
    adjacent faces flush with hull edges, so this is the face-flush approximation the
    IBSI manual permits, not O'Rourke's exact algorithm. Edges whose projection on the
    face plane is zero cannot define ``v`` and are skipped. ``simplices`` index the
    already-extracted hull ``points``.
    """
    triangles = points[simplices]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):
        raise RuntimeError("convex hull contains a zero-area face")
    normals = normals / lengths[:, None]
    # Qhull's triangle winding is not consistently outward. Silhouette signs are only
    # meaningful with outward normals, so orient each face against the hull centroid.
    centroids = triangles.mean(axis=1)
    interior = points.mean(axis=0)
    inward = np.einsum("ij,ij->i", normals, centroids - interior) < 0
    normals[inward] *= -1
    # Edge keys in first-occurrence order over faces and their (b, c), (c, a), (a, b)
    # edges, with the faces of the first and second occurrence.
    starts = simplices[:, [1, 2, 0]].reshape(-1)
    ends = simplices[:, [2, 0, 1]].reshape(-1)
    owners = np.repeat(np.arange(simplices.shape[0], dtype=np.int64), 3)
    ordered = np.stack((np.minimum(starts, ends), np.maximum(starts, ends)), axis=1)
    unique_keys, first_seen, inverse, counts = np.unique(
        ordered, axis=0, return_index=True, return_inverse=True, return_counts=True,
    )
    if unique_keys.shape[0] == 0:
        raise RuntimeError("convex hull has no edges")
    if np.any(counts != 2):
        raise RuntimeError("convex hull is not a closed two-manifold")
    occurrences = np.argsort(inverse.reshape(-1), kind="stable")
    insertion = np.argsort(first_seen, kind="stable")
    key_array: NDArray[np.int64] = unique_keys[insertion].astype(np.int64, copy=False)
    first: NDArray[np.int64] = owners[occurrences[0::2]][insertion]
    second: NDArray[np.int64] = owners[occurrences[1::2]][insertion]
    edges = points[key_array[:, 1]] - points[key_array[:, 0]]
    tolerance = max(float(np.max(np.linalg.norm(edges, axis=1))), 1.0) * 1e-12
    first_normals = normals[first]
    second_normals = normals[second]
    # Face blocks bound the (faces, edges) silhouette products. Candidates keep the
    # face-major, edge-minor order of one pass per face normal.
    block = max(1, _OMBB_SILHOUETTE_ELEMENTS // edges.shape[0])
    face_parts: list[NDArray[np.int64]] = []
    edge_parts: list[NDArray[np.int64]] = []
    for begin in range(0, normals.shape[0], block):
        face_normals = normals[begin:begin + block, None, :]
        # One adjacency dot product is zero when the edge lies in a face parallel to
        # the candidate direction; opposite signs select the usual silhouette.
        silhouette = (
            _row_dots(first_normals[None, :, :], face_normals)
            * _row_dots(second_normals[None, :, :], face_normals)
        ) <= 0
        faces, edge_indexes = np.nonzero(silhouette)
        face_parts.append(faces + begin)
        edge_parts.append(edge_indexes)
    candidate_faces = np.concatenate(face_parts)
    selected: NDArray[np.float64] = edges[np.concatenate(edge_parts)]
    normal: NDArray[np.float64] = normals[candidate_faces]
    projected: NDArray[np.float64] = _row_dots(selected, normal)
    perpendicular: NDArray[np.float64] = selected - projected[:, None] * normal
    perpendicular_lengths: NDArray[np.float64] = np.linalg.norm(perpendicular, axis=1)
    # Edges whose projection on the face plane vanishes cannot define v.
    usable: NDArray[np.bool_] = perpendicular_lengths > tolerance
    if not bool(np.any(usable)):
        raise RuntimeError("convex hull has no face-flush box orientations")
    v: NDArray[np.float64] = perpendicular[usable] / perpendicular_lengths[usable, None]
    u: NDArray[np.float64] = normal[usable]
    w: NDArray[np.float64] = np.cross(u, v)
    return np.stack((u, v, w), axis=1)


def _ombb_volume_error(
    extents: NDArray[np.float64],
    radius: float,
    unit: float,
) -> NDArray[np.float64]:
    """Return a bound on the rounding error of box volumes from projected extents.

    With unit roundoff ``unit`` and point radius ``radius``, the rounded points and the
    rounded unit axis each add at most ``unit * radius`` and the three-term dot product
    at most ``3 * unit * radius`` per projection, so an extent is within
    ``10 * unit * radius + unit * extent`` and the two volume products add
    ``2 * unit * volume``. Twice this first-order bound covers the dropped terms. The
    sum of pairwise extent products is the volume times the sum of reciprocal extents,
    written without dividing by a zero extent.
    """
    x, y, z = extents[..., 0], extents[..., 1], extents[..., 2]
    return 2 * unit * (10 * radius * (y * z + z * x + x * y) + 5 * x * y * z)


def ombb_measures(
    host: NDArray[np.float64],
    hull: spatial.ConvexHull,
) -> tuple[mx.array, mx.array]:
    """Return the minimum-volume face-flush box volume and its surface area.

    ``host`` holds the centered hull vertices in float64. Candidate projections,
    extents, volumes and areas are evaluated on Metal in blocks sized to a bounded
    float32 scratch budget.
    Voxel hulls often have several boxes of exactly equal volume (a full box mask is
    enclosed equally by its axis-aligned box and by tilted boxes), so float32 rounding of
    the volumes must not choose the box: every candidate whose float32 volume is within
    the rounding bound of the minimum is re-evaluated in float64 on the host, and the
    boxes whose float64 volume is the minimum within its own bound are resolved to the
    smallest surface area, which is the IBSI area-density companion to the same box
    family.
    """
    points = mx.array(host, dtype=mx.float32)
    host_axes = _ombb_candidate_axes(host, hull_vertex_simplices(hull))
    axes = mx.array(host_axes, dtype=mx.float32)
    vertices = points.shape[0]
    block = max(1, min(256, _OMBB_BLOCK_ELEMENTS // max(1, 3 * vertices)))
    # Blocks are evaluated in groups that together hold at most the block budget of
    # projections. Left as one lazy graph, the scheduler keeps many groups' projection
    # blocks alive at once.
    group = max(1, _OMBB_BLOCK_ELEMENTS // (block * 3 * vertices))
    volumes: list[mx.array] = []
    areas: list[mx.array] = []
    extent_blocks: list[mx.array] = []
    for index, start in enumerate(range(0, axes.shape[0], block)):
        projections = mx.matmul(
            axes[start : start + block],
            mx.swapaxes(points, 0, 1),
        )
        extents = mx.max(projections, axis=2) - mx.min(projections, axis=2)
        x, y, z = extents[:, 0], extents[:, 1], extents[:, 2]
        volumes.append(x * y * z)
        areas.append(2 * (x * y + y * z + z * x))
        extent_blocks.append(extents)
        if (index + 1) % group == 0:
            evaluate(*volumes[-group:], *areas[-group:], *extent_blocks[-group:])
    volume_all = mx.concatenate(volumes)
    area_all = mx.concatenate(areas)
    extent_all = mx.concatenate(extent_blocks)
    evaluate(volume_all, area_all, extent_all)
    if not bool(mx.all(mx.isfinite(volume_all)).item()) or not bool(
        mx.all(mx.isfinite(area_all)).item()
    ):
        raise RuntimeError("oriented minimum bounding box is not finite")
    radius = float(np.max(np.linalg.norm(host, axis=1)))
    volume32 = np.asarray(volume_all, dtype=np.float64)
    error32 = _ombb_volume_error(
        np.asarray(extent_all, dtype=np.float64),
        radius,
        2.0**-24,
    )
    near = np.flatnonzero(volume32 - error32 <= np.min(volume32 + error32))
    exact_volumes: list[NDArray[np.float64]] = []
    exact_areas: list[NDArray[np.float64]] = []
    exact_errors: list[NDArray[np.float64]] = []
    for start in range(0, near.size, block):
        projected: NDArray[np.float64] = host_axes[near[start : start + block]] @ host.T
        extents64: NDArray[np.float64] = np.max(projected, axis=2) - np.min(projected, axis=2)
        x64, y64, z64 = extents64[:, 0], extents64[:, 1], extents64[:, 2]
        exact_volumes.append(x64 * y64 * z64)
        exact_areas.append(2 * (x64 * y64 + y64 * z64 + z64 * x64))
        exact_errors.append(_ombb_volume_error(extents64, radius, 2.0**-53))
    volume64 = np.concatenate(exact_volumes)
    area64 = np.concatenate(exact_areas)
    error64 = np.concatenate(exact_errors)
    tied = volume64 - error64 <= np.min(volume64 + error64)
    best_volume = mx.array(float(np.min(volume64)), dtype=mx.float32)
    best_area = mx.array(float(np.min(area64[tied])), dtype=mx.float32)
    if float(cast(float, best_volume.item())) <= 0:
        raise RuntimeError("oriented minimum bounding box volume underflows float32")
    return best_volume, best_area


def oriented_bounding_box_measures(points: mx.array) -> tuple[mx.array, mx.array]:
    """Return the minimum-volume face-flush box volume and area of a point cloud.

    ``points`` is a float32 ``(n, 3)`` set of physical coordinates. The hull is built
    with SciPy; candidate projections and box measures run on Metal. This public wrapper
    isolates the geometry from shape-feature extraction so it can be checked against an
    independent oriented-box implementation.

    Raises:
        ValueError: If ``points`` is not an ``(n, 3)`` array.
        RuntimeError: If the convex hull is degenerate or the box measures are non-finite.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("oriented bounding box requires an (n, 3) point array")
    evaluate(points)
    host = np.asarray(points, dtype=np.float64)
    hull = scipy_hull(host)
    hull_points = host[hull.vertices]
    return ombb_measures(hull_points - hull_points.mean(axis=0), hull)
