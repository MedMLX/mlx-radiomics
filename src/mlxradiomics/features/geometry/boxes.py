"""Oriented bounding boxes from convex-hull candidate orientations."""

from typing import cast

import numpy as np
import scipy.spatial as spatial
from numpy.typing import NDArray

from mlxradiomics.features.geometry.hull import hull_vertex_simplices, scipy_hull
from mlxradiomics.numerics.runtime import evaluate, mx

# Float32 projection elements retained per OMBB candidate block (16 MiB).
_OMBB_BLOCK_ELEMENTS = 4 * 1024**2


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
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face, triangle in enumerate(simplices):
        a, b, c = (int(value) for value in triangle)
        for start, end in ((b, c), (c, a), (a, b)):
            key = (start, end) if start < end else (end, start)
            edge_faces.setdefault(key, []).append(face)
    keys: list[tuple[int, int]] = []
    adjacency: list[tuple[int, int]] = []
    for key, faces in edge_faces.items():
        if len(faces) != 2:
            raise RuntimeError("convex hull is not a closed two-manifold")
        keys.append(key)
        adjacency.append((faces[0], faces[1]))
    if not keys:
        raise RuntimeError("convex hull has no edges")
    key_array: NDArray[np.int64] = np.asarray(keys, dtype=np.int64)
    edges = points[key_array[:, 1]] - points[key_array[:, 0]]
    first: NDArray[np.int64] = np.asarray(
        [pair[0] for pair in adjacency],
        dtype=np.int64,
    )
    second: NDArray[np.int64] = np.asarray(
        [pair[1] for pair in adjacency],
        dtype=np.int64,
    )
    tolerance = max(float(np.max(np.linalg.norm(edges, axis=1))), 1.0) * 1e-12
    candidates: list[NDArray[np.float64]] = []
    for normal in normals:
        # One adjacency dot product is zero when the edge lies in a face parallel to
        # the candidate direction; opposite signs select the usual silhouette.
        silhouette: NDArray[np.bool_] = (normals[first] @ normal) * (normals[second] @ normal) <= 0
        if not bool(np.any(silhouette)):
            continue
        selected: NDArray[np.float64] = edges[silhouette]
        projected: NDArray[np.float64] = selected @ normal
        perpendicular: NDArray[np.float64] = selected - projected[:, None] * normal[None, :]
        perpendicular_lengths: NDArray[np.float64] = np.linalg.norm(
            perpendicular,
            axis=1,
        )
        usable: NDArray[np.bool_] = perpendicular_lengths > tolerance
        if not bool(np.any(usable)):
            continue
        v: NDArray[np.float64] = perpendicular[usable] / perpendicular_lengths[usable, None]
        u: NDArray[np.float64] = np.broadcast_to(normal, v.shape)
        w: NDArray[np.float64] = np.cross(u, v)
        candidates.append(np.stack((u, v, w), axis=1))
    if not candidates:
        raise RuntimeError("convex hull has no face-flush box orientations")
    return np.concatenate(candidates, axis=0)


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
    volumes: list[mx.array] = []
    areas: list[mx.array] = []
    extent_blocks: list[mx.array] = []
    for start in range(0, axes.shape[0], block):
        projections = mx.matmul(
            axes[start : start + block],
            mx.swapaxes(points, 0, 1),
        )
        extents = mx.max(projections, axis=2) - mx.min(projections, axis=2)
        x, y, z = extents[:, 0], extents[:, 1], extents[:, 2]
        volumes.append(x * y * z)
        areas.append(2 * (x * y + y * z + z * x))
        extent_blocks.append(extents)
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
