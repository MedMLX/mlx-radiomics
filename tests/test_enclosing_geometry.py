"""Independent geometry for oriented boxes and minimum-volume enclosing ellipsoids.

The oracles use float64 NumPy/trimesh hull topology and either a brute-force
face-flush box family or Khachiyan's published barycentric ascent. They do not call
the production candidate, projection or MVEE code.
"""

from fractions import Fraction
from itertools import product
from math import pi, prod, sqrt
from typing import cast

import numpy as np
import pytest
from conftest import Conventions, convention_shape_statistics
from numpy.typing import NDArray
from skimage.measure import marching_cubes
from trimesh.convex import convex_hull

from mlxradiomics.features.geometry.boxes import oriented_bounding_box_measures
from mlxradiomics.features.geometry.ellipsoids import minimum_volume_ellipsoid_axes
from mlxradiomics.numerics.runtime import mx

GEOMETRY_NAMES = (
    "VolumeDensityOrientedMinimumBoundingBox",
    "AreaDensityOrientedMinimumBoundingBox",
    "VolumeDensityMinimumVolumeEnclosingEllipsoid",
    "AreaDensityMinimumVolumeEnclosingEllipsoid",
)


def _ombb_reference(points: NDArray[np.float64]) -> tuple[float, float]:
    """Return the minimum face-flush box volume and that box's surface area."""
    hull = convex_hull(np.asarray(points, dtype=np.float64))
    vertices = np.asarray(hull.vertices, dtype=np.float64)
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    adjacency = np.asarray(hull.face_adjacency, dtype=np.int64)
    edges = np.asarray(hull.face_adjacency_edges, dtype=np.int64)
    candidates: list[tuple[float, float]] = []
    for normal in normals:
        side_first = normals[adjacency[:, 0]] @ normal
        side_second = normals[adjacency[:, 1]] @ normal
        for edge in edges[(side_first * side_second) <= 0]:
            vector = vertices[edge[1]] - vertices[edge[0]]
            perpendicular = vector - (vector @ normal) * normal
            length = float(cast(float, np.linalg.norm(perpendicular)))
            if length <= 1e-12 * max(1.0, float(cast(float, np.linalg.norm(vector)))):
                continue
            second = perpendicular / length
            axes = np.stack((normal, second, np.cross(normal, second)))
            projected = vertices @ axes.T
            extents = projected.max(axis=0) - projected.min(axis=0)
            volume = float(np.prod(extents))
            area = float(2 * (
                extents[0] * extents[1]
                + extents[1] * extents[2]
                + extents[2] * extents[0]
            ))
            candidates.append((volume, area))
    assert candidates
    best_volume = min(volume for volume, _ in candidates)
    best_area = min(
        area for volume, area in candidates if volume <= best_volume * (1 + 1e-12)
    )
    return best_volume, best_area


def _khachiyan_covariance(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the weighted covariance of float64 lifted Khachiyan ascent.

    This follows the homogeneous ``(d + 1) x (d + 1)`` formulation (Todd and Yildirim;
    MIRP ``Data3DMinimumEnvelopingEllipsoid``) rather than the implementation's Schur
    complement, so the two share only the algorithm, not the algebra. The MVEE shape
    matrix is ``inv(covariance) / d``.
    """
    data = np.asarray(points, dtype=np.float64)
    data = data - data.mean(axis=0)
    count, dimension = data.shape
    lifted = np.vstack((data.T, np.ones((1, count))))
    weights = np.full(count, 1.0 / count)
    for _ in range(200_000):
        scatter = (lifted * weights) @ lifted.T
        values = np.einsum("ji,jk,ki->i", lifted, np.linalg.inv(scatter), lifted)
        maximum = float(values.max())
        index = int(values.argmax())
        step = (maximum - dimension - 1) / ((dimension + 1) * (maximum - 1))
        updated = weights * (1 - step)
        updated[index] += step
        change = float(cast(float, np.linalg.norm(updated - weights)))
        weights = updated
        if change <= 1e-3:
            break
    else:
        raise AssertionError("float64 Khachiyan reference did not converge")
    center = data.T @ weights
    return (data.T * weights) @ data - np.outer(center, center)


def _mvee_reference(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return descending MVEE semi-axes by float64 lifted Khachiyan ascent."""
    covariance = _khachiyan_covariance(points)
    shape = np.linalg.inv(covariance) / covariance.shape[0]
    return np.sort(1 / np.sqrt(np.linalg.eigvalsh(shape)))[::-1]


def _exact_spectrum(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the eigenvalues of a float64 symmetric 3x3 positive-definite matrix.

    The characteristic polynomial is formed in exact rationals from the stored entries
    and each root is bisected from a factor-two bracket around the float64 estimate, so
    a strongly graded matrix keeps its small eigenvalues to full relative accuracy.
    """
    m = [[Fraction(float(value)) for value in row] for row in matrix]
    trace = m[0][0] + m[1][1] + m[2][2]
    minors = (m[0][0] * m[1][1] - m[0][1] * m[1][0]
              + m[0][0] * m[2][2] - m[0][2] * m[2][0]
              + m[1][1] * m[2][2] - m[1][2] * m[2][1])
    determinant = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                   - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    def characteristic(value: Fraction) -> Fraction:
        return value * (value * (value - trace) + minors) - determinant

    roots: list[float] = []
    for estimate in np.linalg.eigvalsh(matrix):
        low, high = Fraction(float(estimate) / 2), Fraction(float(estimate) * 2)
        assert characteristic(low) * characteristic(high) < 0
        for _ in range(80):
            middle = Fraction((float(low) + float(high)) / 2)
            if characteristic(low) * characteristic(middle) <= 0:
                high = middle
            else:
                low = middle
        roots.append(float((low + high) / 2))
    return np.asarray(roots, dtype=np.float64)


def _cloud(seed: int, size: int = 35) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(size, 3))
    return points[np.linalg.norm(points, axis=1) < 2.5]


def _ombb_checked(points: NDArray[np.float64]) -> tuple[float, float]:
    volume, area = oriented_bounding_box_measures(mx.array(points, dtype=mx.float32))
    return float(cast(float, volume.item())), float(cast(float, area.item()))


def _mvee_checked(
    points: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float64]:
    semi = minimum_volume_ellipsoid_axes(mx.array(points, dtype=mx.float32))
    return np.asarray([float(cast(float, value.item())) for value in semi])


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_ombb_matches_float64_face_flush_brute_force(seed: int) -> None:
    points = _cloud(seed)
    volume, area = _ombb_checked(points)
    expected_volume, expected_area = _ombb_reference(points)
    assert volume == pytest.approx(expected_volume, rel=2e-4)
    assert area == pytest.approx(expected_area, rel=2e-4)


def test_rotated_box_ombb_recovers_the_true_box() -> None:
    half = np.array([1.5, 0.8, 2.2])
    corners = np.asarray(list(product((-1.0, 1.0), repeat=3))) * half
    rotation, _ = np.linalg.qr(np.random.default_rng(13).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    points = corners @ rotation.T + np.array([4.0, -2.5, 7.0])
    volume, area = _ombb_checked(points)
    assert volume == pytest.approx(8 * prod(half), rel=2e-5)
    assert area == pytest.approx(
        8 * (half[0] * half[1] + half[1] * half[2] + half[2] * half[0]), rel=2e-5
    )


def _face_midpoints(shape: tuple[int, int, int], spacing: tuple[float, ...]) -> NDArray[
    np.float64
]:
    """Return the marching-cubes vertices of a full box mask: boundary face centers."""
    centers = np.argwhere(np.ones(shape, dtype=np.bool_)).astype(np.float64)
    points: list[NDArray[np.float64]] = []
    for axis in range(3):
        for side in (-0.5, 0.5):
            face = centers[centers[:, axis] == (0 if side < 0 else shape[axis] - 1)].copy()
            face[:, axis] += side
            points.append(face)
    return np.concatenate(points) * np.asarray(spacing)


@pytest.mark.parametrize(
    ("shape", "spacing", "extents"),
    [((6, 1, 2), (1.0, 2.0, 0.5), (6.0, 2.0, 1.0)),
     ((2, 1, 3), (1.0, 2.0, 0.5), (2.0, 2.0, 1.5))],
)
def test_ombb_exact_volume_tie_selects_smallest_area(
    shape: tuple[int, int, int], spacing: tuple[float, ...],
    extents: tuple[float, float, float],
) -> None:
    # The face-center hull of a full box is enclosed by the axis-aligned box with the
    # physical extents, and equally by a box tilted about the long axis: the chamfered
    # cross-section of a 2 x 1 rectangle has a tilted strip pair 2/sqrt(4.25) by
    # sqrt(4.25), product exactly 2. Volumes tie in exact arithmetic, so the smallest
    # area is the axis-aligned box's (hand values; the float64 face-flush brute force
    # agrees). Float32 rounding of the tied volumes must not decide which box is used.
    points = _face_midpoints(shape, spacing)
    x, y, z = extents
    expected_area = 2 * (x * y + y * z + z * x)
    reference_volume, reference_area = _ombb_reference(points)
    assert reference_volume == pytest.approx(x * y * z, rel=1e-12)
    assert reference_area == pytest.approx(expected_area, rel=1e-12)
    volume, area = _ombb_checked(points)
    assert volume == pytest.approx(x * y * z, rel=2e-6)
    assert area == pytest.approx(expected_area, rel=2e-6)


@pytest.mark.parametrize("seed", [8, 9, 10])
def test_mvee_matches_float64_khachiyan_on_random_hulls(seed: int) -> None:
    points = _cloud(seed)
    semi = _mvee_checked(points)
    hull = convex_hull(points)
    expected = _mvee_reference(np.asarray(hull.vertices, dtype=np.float64))
    np.testing.assert_allclose(semi, expected, rtol=1e-5)


def test_mvee_of_sampled_ellipsoid_surface_recovers_known_axes() -> None:
    axes = np.array([3.0, 2.0, 1.3])
    count = 400
    index = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * index / count)
    azimuth = pi * (1 + sqrt(5)) * index
    unit = np.stack((
        np.sin(polar) * np.cos(azimuth),
        np.sin(polar) * np.sin(azimuth),
        np.cos(polar),
    ), axis=1)
    rotation, _ = np.linalg.qr(np.random.default_rng(21).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    points = (unit * axes) @ rotation.T
    semi = _mvee_checked(points)
    np.testing.assert_allclose(semi, np.sort(axes)[::-1], rtol=5e-3)


def test_planar_hull_raises_for_enclosing_geometry() -> None:
    rng = np.random.default_rng(31)
    planar = np.column_stack((rng.normal(size=(20, 2)), np.zeros(20)))
    with pytest.raises(RuntimeError, match="degenerate"):
        oriented_bounding_box_measures(mx.array(planar, dtype=mx.float32))
    with pytest.raises(RuntimeError, match="degenerate"):
        minimum_volume_ellipsoid_axes(mx.array(planar, dtype=mx.float32))


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_public_densities_match_independent_mesh_and_hull_geometry(
    conventions: Conventions,
) -> None:
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    spacing = (2.0, 2.0, 2.0)
    names = (*GEOMETRY_NAMES, "MeshVolume", "SurfaceArea")
    computed = convention_shape_statistics(conventions, mask.astype(np.bool_), spacing, names)
    values = {
        name: float(np.asarray(computed[f"shape_{name}"], dtype=np.float64).item())
        for name in names
    }

    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1), 0.5, method="lorensen")
    physical = vertices.astype(np.float64) * np.asarray(spacing)
    triangles = physical[faces]
    mesh_area = float(
        np.linalg.norm(np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0],
        ), axis=1).sum() / 2
    )
    mesh_volume = float(abs(np.einsum(
        "ij,ij->", np.cross(triangles[:, 0], triangles[:, 1]), triangles[:, 2],
    )) / 6)
    ombb_volume, ombb_area = _ombb_reference(physical)
    hull = convex_hull(physical)
    hull_points = np.asarray(hull.vertices, dtype=np.float64)
    mvee = _mvee_reference(hull_points)
    # This mask's marching surface has octahedral symmetry, so its MVEE is the sphere
    # through the extreme hull vertices; the independent float64 MVEE confirms equal
    # semi-axes, and the sphere area is exact.
    np.testing.assert_allclose(mvee, mvee[0], rtol=2e-3)
    mvee_volume = 4 * pi / 3 * float(mvee[0] ** 3)
    mvee_area = 4 * pi * float(mvee[0] ** 2)
    expected = {
        "VolumeDensityOrientedMinimumBoundingBox": mesh_volume / ombb_volume,
        "AreaDensityOrientedMinimumBoundingBox": mesh_area / ombb_area,
        "VolumeDensityMinimumVolumeEnclosingEllipsoid": mesh_volume / mvee_volume,
        "AreaDensityMinimumVolumeEnclosingEllipsoid": mesh_area / mvee_area,
    }
    for name in GEOMETRY_NAMES:
        assert values[name] == pytest.approx(expected[name], rel=2e-4), name
    assert values["MeshVolume"] == pytest.approx(mesh_volume, rel=2e-5)
    assert values["SurfaceArea"] == pytest.approx(mesh_area, rel=2e-5)


def _diagonal_surface(count: int) -> NDArray[np.float32]:
    """Return the float32 half-grid hull of ``count`` voxels on the body diagonal."""
    centers = np.stack([np.arange(count, dtype=np.float64)] * 3, axis=1)
    corners = np.asarray(list(product((-0.5, 0.5), repeat=3)), dtype=np.float64)
    surface = (centers[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    hull = convex_hull(surface)
    return np.asarray(hull.vertices, dtype=np.float32)


def test_mvee_slender_diagonal_surface_matches_float64_reference() -> None:
    # 200 voxels on the body diagonal form an approximately 283-mm-long tube with
    # sub-voxel cross-section: a float32 covariance loses its smallest axis.
    points = _diagonal_surface(200)
    semi = _mvee_checked(points)
    expected = _mvee_reference(points.astype(np.float64))
    np.testing.assert_allclose(semi, expected, rtol=1e-4)


@pytest.mark.parametrize("offset", [0.0, 512.0])
def test_mvee_thin_rotated_cuboid_matches_float64_reference(offset: float) -> None:
    half = (-1.0, 1.0)
    corners = np.asarray(
        list(product(half, half, (-0.001, 0.001))), dtype=np.float64,
    )
    theta = pi / 4
    rotation = np.asarray([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ], dtype=np.float64)
    points = (corners @ rotation.T + offset).astype(np.float32)
    semi = _mvee_checked(points)
    expected = _mvee_reference(points.astype(np.float64))
    np.testing.assert_allclose(semi, expected, rtol=1e-4)


@pytest.mark.parametrize("seed", [8, 9, 10])
def test_mvee_thin_oblique_hull_matches_affine_preimage_khachiyan(seed: int) -> None:
    # A 1e4 : 1 : 1e-2 hull in a random orientation. The weighted covariance has
    # condition ~1e12, so neither a cofactor inverse nor float64 inv() on these points
    # resolves Khachiyan's lifted values (the cofactor path gave a 28% middle-axis error
    # on such a hull). Khachiyan's weights are invariant under invertible affine maps,
    # so the oracle runs float64 ascent on the exact affine preimage (rotation back,
    # axes divided by their scales) of the float32 input hull, which is well
    # conditioned. Its covariance C maps to D C D, whose exact rational spectrum gives
    # the semi-axes sqrt(3 * eig).
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    scales = np.asarray([1e4, 1.0, 1e-2])
    points = ((_cloud(seed) * scales) @ rotation.T).astype(np.float32)
    semi = _mvee_checked(points)
    vertices = np.asarray(convex_hull(points.astype(np.float64)).vertices, dtype=np.float64)
    preimage = (vertices @ rotation) / scales
    graded = _khachiyan_covariance(preimage) * np.outer(scales, scales)
    expected = np.sort(np.sqrt(3 * _exact_spectrum(graded)))[::-1]
    np.testing.assert_allclose(semi, expected, rtol=1e-5)


def test_mvee_axes_are_scale_equivariant_far_from_unit_scale() -> None:
    # Multiplying the points by 2**60 (or 2**-60) scales every semi-axis by the same
    # power of two. A unit-scale cofactor inverse forms sixth powers of the
    # coordinates, which overflow (or flush) float32 there.
    points = _cloud(8)
    expected = _mvee_reference(np.asarray(convex_hull(points).vertices, dtype=np.float64))
    for exponent in (60, -60):
        semi = _mvee_checked(points * 2.0**exponent)
        np.testing.assert_allclose(semi, expected * 2.0**exponent, rtol=1e-5)

