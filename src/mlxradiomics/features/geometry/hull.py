"""Convex-hull topology and Metal area/volume measurements."""

import numpy as np
import scipy.spatial as spatial
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import SpacingParts, cross_product, scale_pow2
from mlxradiomics.numerics.runtime import mx


def scipy_hull(host: NDArray[np.float64]) -> spatial.ConvexHull:
    """Build the SciPy convex hull of host points and return its connectivity.

    SciPy provides only the hull connectivity; geometry is evaluated from the returned
    vertex and triangle indices with MLX. The caller chooses the coordinate unit: the
    hull's vertex/facet incidence is unchanged by an invertible diagonal scaling, so a
    grid or physical point set yields the same connectivity.

    Raises:
        RuntimeError: If Qhull cannot build a three-dimensional hull, including a planar
            or collinear point set.
    """
    try:
        hull = spatial.ConvexHull(host)
    except spatial.QhullError as error:
        raise RuntimeError("convex hull is degenerate for the requested shape features") from error
    return hull


def hull_measures(
    hull: spatial.ConvexHull,
    vertices: mx.array,
    parts: SpacingParts,
) -> tuple[mx.array, mx.array]:
    """Return convex-hull area normalized by the mesh boundary exponent and grid volume.

    Areas and tetrahedral volumes are evaluated from the hull triangle indices with MLX.
    ``vertices`` are the float32 grid-coordinate points the hull was built from; grid
    coordinates are an invertible diagonal scaling of physical coordinates and therefore
    have the same vertex/facet incidence.
    """
    hull_points = mx.take(vertices, mx.array(hull.vertices, dtype=mx.int32), axis=0)
    triangles = mx.take(vertices, mx.array(hull.simplices, dtype=mx.int32), axis=0)
    edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = cross_product(edge_a * mx.array(parts.mantissa), edge_b * mx.array(parts.mantissa))
    e0, e1, e2 = parts.exponent
    positive = parts.boundary_exponent
    n0 = scale_pow2(cross[:, 0], e1 + e2 - positive)
    n1 = scale_pow2(cross[:, 1], e2 + e0 - positive)
    n2 = scale_pow2(cross[:, 2], e0 + e1 - positive)
    largest = mx.maximum(mx.maximum(mx.abs(n0), mx.abs(n1)), mx.abs(n2))
    safe = mx.where(largest > 0, largest, 1)
    normalized = mx.stack([n0 / safe, n1 / safe, n2 / safe], axis=-1)
    hull_area = mx.sum(0.5 * safe * mx.sqrt(mx.sum(normalized * normalized, axis=-1)))
    # An interior point partitions the convex polyhedron into disjoint tetrahedra, so the
    # sum of absolute face volumes is independent of facet winding.
    origin = mx.mean(hull_points, axis=0)
    a = triangles[:, 0] - origin
    b = triangles[:, 1] - origin
    c = triangles[:, 2] - origin
    hull_volume = mx.sum(mx.abs(mx.sum(a * cross_product(b, c), axis=1))) / 6
    return hull_area, hull_volume


def hull_vertex_simplices(hull: spatial.ConvexHull) -> NDArray[np.int64]:
    """Return hull triangles remapped from full-point to hull-vertex indices."""
    vertices = np.asarray(hull.vertices, dtype=np.int64)
    mapping = np.full(int(np.max(vertices)) + 1, -1, dtype=np.int64)
    mapping[vertices] = np.arange(vertices.size)
    return mapping[np.asarray(hull.simplices, dtype=np.int64)]
