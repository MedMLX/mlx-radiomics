"""Mathematically defined shape meshes at the public extraction boundary.

These values come from the midpoint marching-cubes geometry, not from a PyRadiomics
fixture. The midpoint isosurface of one voxel is the octahedron through its six face
centers: its volume is one sixth of the voxel volume and its eight faces have total area
``sqrt((s0*s1)**2 + (s1*s2)**2 + (s2*s0)**2)``. A closed mesh's signed volume is
independent of the array origin, so translating an identical ROI must not change its
mesh features. Two diagonal voxels exercise the ambiguous face, which must produce two closed
octahedra whichever side of the face is considered.
"""

from math import pi, prod, sqrt

import numpy as np
import pytest
from conftest import Conventions, convention_shape_statistics, settings_for
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_resampled

from mlxradiomics import Settings, extract
from mlxradiomics.config.settings import SHAPE, SHAPE2D, EllipsoidArea
from mlxradiomics.features.geometry.ellipsoids import (
    ellipsoid_area_ibsi_series,
    ellipsoid_area_normalized,
)

MESH_FEATURES = (
    "MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio", "Sphericity",
    "Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterColumn",
    "Maximum2DDiameterRow",
)
ANALYTIC_FEATURES = ("MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio",
                     "Sphericity")
AXIS_FEATURES = ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "Elongation",
                 "Flatness")


@pytest.mark.parametrize("shape", [(3, 4, 5), (21, 23, 25)])
def test_cuboid_mesh_diameters_match_face_center_extrema(shape: tuple[int, int, int]) -> None:
    # A cuboid's midpoint mesh reaches a face center by half a voxel at each end.
    # Opposite extreme vertices span n voxels on one axis and n-1 on the others.
    # The small surface uses one pass; the larger surface has partial pair tiles.
    spacing = (0.75, 1.25, 2.5)
    mask = np.ones(shape, dtype=np.uint8)
    names = ("Maximum3DDiameter", "Maximum2DDiameterSlice",
             "Maximum2DDiameterColumn", "Maximum2DDiameterRow")
    values = _values(mask, spacing, names)

    def diameter(axes: tuple[int, ...]) -> float:
        return sqrt(max(sum(((shape[axis] - (axis != face)) * spacing[axis])**2
                            for axis in axes) for face in axes))

    assert values[names[0]] == pytest.approx(diameter((0, 1, 2)), rel=2e-6)
    for plane, name in enumerate(names[1:]):
        axes = tuple(axis for axis in range(3) if axis != plane)
        assert values[name] == pytest.approx(diameter(axes), rel=2e-6)


@pytest.mark.parametrize("separation", [128, 512, 1024])
def test_disconnected_2d_mesh_area_is_additive(separation: int) -> None:
    mask = np.zeros((separation + 1,) * 2, np.uint8)
    mask[0, 0] = mask[-1, -1] = 1
    result = extract(hcc_resampled(mask.shape), mask, spacing=(0.977, 0.977),
                     settings=Settings(features=("shape2D_MeshSurface",)))
    assert result.features["original_shape2D_MeshSurface"] == pytest.approx(0.977**2, rel=1e-6)


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_collinear_population_has_zero_transverse_axes(scale: float) -> None:
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[np.arange(4), np.arange(4), np.arange(4)] = 1
    values = _values(mask, (scale, 1.5 * scale, 2 * scale), AXIS_FEATURES)
    assert values["MajorAxisLength"] == pytest.approx(4 * sqrt(9.0625) * scale, rel=1e-6)
    for name in ("MinorAxisLength", "LeastAxisLength", "Elongation", "Flatness"):
        assert values[name] == pytest.approx(0, abs=1e-6 * scale)


def test_thin_full_rank_shape_keeps_positive_least_axis() -> None:
    mask = np.ones((2, 2, 2), np.uint8)
    values = _values(mask, (1e-5, 1, 2), AXIS_FEATURES)
    assert values["LeastAxisLength"] == pytest.approx(2e-5, rel=1e-6)
    assert values["Flatness"] == pytest.approx(5e-6, rel=1e-6)


@pytest.mark.parametrize("spacing", [0.1, 0.001])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_correlated_thin_shape_resolves_transverse_axes(spacing: float, axis: int) -> None:
    mask = np.zeros((512, 2, 2), np.uint8)
    mask[:, 0, 0] = mask[:, 1, 0] = mask[:, 1, 1] = 1
    physical = [spacing, spacing, spacing]
    physical[axis] = 1.0
    values = _values(np.moveaxis(mask, 0, axis),
                     (physical[0], physical[1], physical[2]), AXIS_FEATURES)
    # The transverse covariance is s^2 / 9 * [[2, 1], [1, 2]], independent of length.
    major = 4 * sqrt((512**2 - 1) / 12)
    least, minor = 4 * spacing / 3, 4 * spacing / sqrt(3)
    for name, expected in (("MajorAxisLength", major), ("MinorAxisLength", minor),
                           ("LeastAxisLength", least), ("Elongation", minor / major),
                           ("Flatness", least / major)):
        assert values[name] == pytest.approx(expected, rel=2e-6), name


@pytest.mark.parametrize("separation", [10000, 1000000])
def test_distant_3d_components_add_surface_area(separation: int) -> None:
    mask = np.zeros((separation + 1, 1, 1), np.uint8)
    mask[0, 0, 0] = mask[-1, 0, 0] = 1
    spacing = (0.977, 0.977, 0.977)
    values = _values(mask, spacing, ANALYTIC_FEATURES)
    area, volume = 2 * sqrt(3) * spacing[0]**2, prod(spacing) / 3
    assert values["SurfaceArea"] == pytest.approx(area, rel=1e-6)
    assert values["SurfaceVolumeRatio"] == pytest.approx(area / volume, rel=1e-6)
    assert values["Sphericity"] == pytest.approx((36 * pi * volume**2)**(1 / 3) / area,
                                               rel=1e-6)


def _values(
    mask: NDArray[np.uint8], spacing: tuple[float, float, float],
    names: tuple[str, ...] = SHAPE,
) -> dict[str, float]:
    image = hcc_resampled(mask.shape).astype(np.float32)
    settings = settings_for("pyradiomics", features=tuple(f"shape_{name}" for name in names))
    result = extract(image, mask, settings=settings, spacing=spacing)
    return {
        name: float(np.asarray(result.features[f"original_shape_{name}"], dtype=np.float64).item())
        for name in names
    }


def _octahedron_area(spacing: tuple[float, float, float]) -> float:
    """Area of the midpoint isosurface around one isolated voxel."""
    s0, s1, s2 = spacing
    return sqrt((s0 * s1)**2 + (s1 * s2)**2 + (s2 * s0)**2)


def test_isolated_voxel_matches_analytic_octahedron() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    spacing = (2.0, 0.5, 1.5)
    values = _values(mask, spacing, ANALYTIC_FEATURES)

    voxel_volume = prod(spacing)
    mesh_volume = voxel_volume / 6
    surface_area = _octahedron_area(spacing)
    assert values["MeshVolume"] == pytest.approx(mesh_volume, rel=1e-6)
    assert values["VoxelVolume"] == pytest.approx(voxel_volume, rel=1e-6)
    assert values["SurfaceArea"] == pytest.approx(surface_area, rel=1e-6)
    assert values["SurfaceVolumeRatio"] == pytest.approx(surface_area / mesh_volume, rel=1e-6)
    assert values["Sphericity"] == pytest.approx(
        (36 * pi * mesh_volume**2)**(1 / 3) / surface_area, rel=1e-6
    )


def test_diagonal_voxel_pair_is_two_closed_octahedra() -> None:
    # The two voxels share only an edge. Their shared face is ambiguous; the mesh
    # must keep them separate, independent of where the ROI starts in the array.
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 1] = 1
    spacing = (1.5, 0.75, 2.0)
    values = _values(mask, spacing)

    mesh_volume = 2 * prod(spacing) / 6
    surface_area = 2 * _octahedron_area(spacing)
    assert values["MeshVolume"] == pytest.approx(mesh_volume, rel=1e-6)
    assert values["VoxelVolume"] == pytest.approx(2 * prod(spacing), rel=1e-6)
    assert values["SurfaceArea"] == pytest.approx(surface_area, rel=1e-6)
    assert values["SurfaceVolumeRatio"] == pytest.approx(surface_area / mesh_volume, rel=1e-6)

    translated = np.zeros((10, 12, 9), dtype=np.uint8)
    translated[4:8, 5:9, 4:7] = mask
    moved = _values(translated, spacing)
    for name in MESH_FEATURES:
        assert moved[name] == pytest.approx(values[name], rel=1e-6, abs=1e-9), name
    for name in AXIS_FEATURES:
        assert moved[name] == pytest.approx(values[name], rel=1e-5, abs=1e-5), name


def test_translated_ambiguous_roi_preserves_all_shape_features() -> None:
    # The random boundary creates many diagonal-corner faces. The mesh is evaluated in
    # the cropped ROI frame, so its features must be identical; the axis lengths also remove
    # the integer origin before converting voxel positions to float32.
    rng = np.random.default_rng(1)
    mask = np.zeros((8, 11, 14), dtype=np.uint8)
    mask[2:6, 3:8, 4:10] = rng.random((4, 5, 6)) > 0.35
    # A large index shift along a non-integral spacing axis also exposes loss of
    # voxel differences when physical float32 coordinates precede centering.
    translated = np.zeros((8, 11, 10014), dtype=np.uint8)
    translated[:, :, 10000:10014] = mask
    spacing = (3.0, 0.977, 0.977)

    original = _values(mask, spacing)
    moved = _values(translated, spacing)
    for name in MESH_FEATURES:
        assert moved[name] == pytest.approx(original[name], rel=1e-6, abs=1e-9), name
    for name in AXIS_FEATURES:
        assert moved[name] == pytest.approx(original[name], rel=1e-5, abs=1e-5), name


def test_disconnected_ambiguous_components_add_volume() -> None:
    # Each pair is two octahedra; the combined mesh contains four. The two pairs sit at
    # different array origins, so this also checks that component volumes do not depend
    # on which component defines the combined bounding box.
    pair = np.zeros((4, 4, 3), dtype=np.uint8)
    pair[1, 1, 1] = 1
    pair[2, 2, 1] = 1
    combined = np.zeros((21, 21, 21), dtype=np.uint8)
    combined[1:5, 1:5, 1:4] = pair
    combined[17:21, 17:21, 17:20] = pair
    spacing = (3.0, 0.977, 0.977)

    single = _values(pair, spacing, ("MeshVolume",))["MeshVolume"]
    total = _values(combined, spacing, ("MeshVolume",))["MeshVolume"]
    assert total == pytest.approx(2 * single, rel=1e-6)


def test_enclosed_cavity_subtracts_octahedron() -> None:
    # The outer block surface is unchanged by removing an interior voxel. The cavity
    # contributes the same octahedron as an isolated voxel, with reversed orientation,
    # so signed volumes subtract and areas add.
    solid = np.ones((5, 5, 5), dtype=np.uint8)
    shell = solid.copy()
    shell[2, 2, 2] = 0
    spacing = (1.5, 0.8, 2.2)
    solid_values = _values(solid, spacing, ("MeshVolume", "SurfaceArea"))
    shell_values = _values(shell, spacing, ("MeshVolume", "SurfaceArea"))

    cavity_volume = prod(spacing) / 6
    cavity_area = _octahedron_area(spacing)
    assert shell_values["MeshVolume"] == pytest.approx(
        solid_values["MeshVolume"] - cavity_volume, rel=1e-6
    )
    assert shell_values["SurfaceArea"] == pytest.approx(
        solid_values["SurfaceArea"] + cavity_area, rel=1e-6
    )


def test_shape_family_scales_with_uniform_spacing() -> None:
    rng = np.random.default_rng(2)
    mask = np.zeros((6, 7, 5), dtype=np.uint8)
    mask[1:5, 1:6, 1:4] = rng.random((4, 5, 3)) > 0.4
    factors = {
        "MeshVolume": 8.0, "VoxelVolume": 8.0, "SurfaceArea": 4.0,
        "SurfaceVolumeRatio": 0.5, "Sphericity": 1.0,
        "Maximum3DDiameter": 2.0, "Maximum2DDiameterSlice": 2.0,
        "Maximum2DDiameterColumn": 2.0, "Maximum2DDiameterRow": 2.0,
        "MajorAxisLength": 2.0, "MinorAxisLength": 2.0, "LeastAxisLength": 2.0,
        "Elongation": 1.0, "Flatness": 1.0,
        "Compactness1": 1.0, "Compactness2": 1.0, "SphericalDisproportion": 1.0,
        "Asphericity": 1.0, "VolumeDensityAxisAlignedBoundingBox": 1.0,
        "AreaDensityAxisAlignedBoundingBox": 1.0,
        "VolumeDensityApproximateEnclosingEllipsoid": 1.0,
        "AreaDensityApproximateEnclosingEllipsoid": 1.0,
        "VolumeDensityConvexHull": 1.0, "AreaDensityConvexHull": 1.0,
    }
    geometry_names = tuple(factors)
    base = _values(mask, (1.0, 1.0, 1.0), geometry_names)
    scaled = _values(mask, (2.0, 2.0, 2.0), geometry_names)
    for name in geometry_names:
        assert scaled[name] == pytest.approx(base[name] * factors[name], rel=1e-5), name


def test_thin_full_rank_2d_covariance_keeps_positive_minor_axis() -> None:
    # Two voxel pairs separated by 4096 share a nearly collinear diagonal. The exact
    # population covariance is [[4194304, 4194304], [4194304, 4194304.25]]; the 0.25 is
    # not representable at float32's ulp near 4194304, so a float32 covariance loses the
    # minor axis entirely. The compensated covariance/spectrum must retain it.
    mask = np.zeros((4097, 4098), dtype=np.uint8)
    mask[0, 0] = mask[0, 1] = 1
    mask[4096, 4096] = mask[4096, 4097] = 1
    points = np.array(
        ((0.0, 0.0), (0.0, 1.0), (4096.0, 4096.0), (4096.0, 4097.0)),
        dtype=np.float64,
    )
    centered = points - np.mean(points, axis=0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(points))
    expected = {
        "MajorAxisLength": 4 * np.sqrt(eigen[-1]),
        "MinorAxisLength": 4 * np.sqrt(eigen[0]),
        "Elongation": np.sqrt(eigen[0] / eigen[-1]),
    }
    values = _values2d(mask, (1.0, 1.0), tuple(expected))
    for name, reference in expected.items():
        assert values[name] == pytest.approx(reference, rel=2e-6), name
    assert values["MinorAxisLength"] > 0



@pytest.mark.parametrize("scale", [1e-25, 1e25])
def test_shape_axes_preserve_units_when_squared_spacing_is_outside_float32(scale: float) -> None:
    # A 2x2 population has variance s^2/4 on both axes and axis lengths 2s.
    # Squaring these spacings directly underflows/overflows float32 although 2s fits.
    mask = np.ones((2, 2), dtype=np.uint8)
    values = _values2d(mask, (scale, scale),
                       ("MajorAxisLength", "MinorAxisLength", "Elongation"))
    assert values["MajorAxisLength"] == pytest.approx(2 * scale, rel=2e-6, abs=0)
    assert values["MinorAxisLength"] == pytest.approx(2 * scale, rel=2e-6, abs=0)
    assert values["Elongation"] == pytest.approx(1.0, rel=2e-6)


def test_major_axis_request_does_not_require_an_unresolved_minor_axis() -> None:
    mask = np.ones((2, 2), dtype=np.uint8)
    spacing = (1.0, 1e-25)
    values = _values2d(mask, spacing, ("MajorAxisLength",))
    assert values["MajorAxisLength"] == pytest.approx(2.0, rel=2e-6)
    # Exact integer rank is two: a lost positive minor axis must raise, never become 0.
    with pytest.raises(RuntimeError, match="resolve a requested axis"):
        _values2d(mask, spacing, ("MinorAxisLength",))


def test_selected_shape_features_match_the_full_request() -> None:
    rng = np.random.default_rng(11)
    mask = np.zeros((7, 8, 6), dtype=np.uint8)
    mask[1:6, 2:7, 1:5] = rng.random((5, 5, 4)) > 0.35
    spacing = (3.0, 0.977, 1.4)
    full = _values(mask, spacing)
    groups = (
        ("VoxelVolume",),
        ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "Elongation", "Flatness"),
        ("MeshVolume", "SurfaceArea", "Sphericity"),
        ("Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterRow"),
    )
    for group in groups:
        selected = _values(mask, spacing, group)
        assert set(selected) == set(group)
        for name in group:
            assert selected[name] == pytest.approx(full[name], rel=1e-6), name


def test_selected_shape2d_features_match_the_full_request() -> None:
    rng = np.random.default_rng(12)
    mask = np.zeros((9, 10), dtype=np.uint8)
    mask[1:8, 2:9] = rng.random((7, 7)) > 0.3
    spacing = (2.5, 0.8)
    full = _values2d(mask, spacing)
    groups = (
        ("PixelSurface",),
        ("MajorAxisLength", "MinorAxisLength", "Elongation"),
        ("MeshSurface", "Perimeter", "PerimeterSurfaceRatio", "Sphericity"),
        ("MaximumDiameter",),
    )
    for group in groups:
        selected = _values2d(mask, spacing, group)
        assert set(selected) == set(group)
        for name in group:
            assert selected[name] == pytest.approx(full[name], rel=1e-6), name


def _values2d(
    mask: NDArray[np.uint8], spacing: tuple[float, float],
    names: tuple[str, ...] = SHAPE2D,
) -> dict[str, float]:
    image = hcc_resampled(mask.shape).astype(np.float32)
    settings = settings_for("pyradiomics",
        features=tuple(f"shape2D_{name}" for name in names))
    result = extract(image, mask, settings=settings, spacing=spacing)
    return {
        name: float(np.asarray(
            result.features[f"original_shape2D_{name}"], dtype=np.float64,
        ).item())
        for name in names
    }


def _shape_values(
    mask: NDArray[np.uint8], spacing: tuple[float, float, float],
    names: tuple[str, ...],
    conventions: Conventions = "pyradiomics",
) -> dict[str, float]:
    """Evaluate shape statistics directly for the given convention set."""
    computed = convention_shape_statistics(conventions, mask.astype(np.bool_), spacing, names)
    return {
        name: float(np.asarray(computed[f"shape_{name}"], dtype=np.float64).item())
        for name in names
    }


@pytest.mark.parametrize("spacing", [1e-12, 1e10])
def test_isolated_voxel_extreme_spacing_restores_units(spacing: float) -> None:
    # The octahedron has V=s^3/6, A=sqrt(3)s^2, sphericity=pi^(1/3)/sqrt(3), and
    # its opposite face centers are one spacing apart. Squaring the physical edge
    # crosses float32's range at these spacings; the scaled mesh must not.
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    values = _shape_values(
        mask, (spacing, spacing, spacing),
        ("MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio",
         "Sphericity", "Maximum3DDiameter"),
    )
    assert values["MeshVolume"] == pytest.approx(spacing**3 / 6, rel=2e-6, abs=0)
    assert values["VoxelVolume"] == pytest.approx(spacing**3, rel=2e-6, abs=0)
    assert values["SurfaceArea"] == pytest.approx(sqrt(3) * spacing**2, rel=2e-6, abs=0)
    assert values["SurfaceVolumeRatio"] == pytest.approx(
        6 * sqrt(3) / spacing, rel=2e-6, abs=0
    )
    assert values["Sphericity"] == pytest.approx(pi ** (1 / 3) / sqrt(3), rel=2e-6)
    assert values["Maximum3DDiameter"] == pytest.approx(spacing, rel=2e-6, abs=0)


@pytest.mark.parametrize("spacing", [1e-12, 1e10])
def test_isolated_pixel_extreme_spacing_restores_units(spacing: float) -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    values = _values2d(
        mask, (spacing, spacing),
        ("MeshSurface", "PixelSurface", "Perimeter", "PerimeterSurfaceRatio",
         "Sphericity", "MaximumDiameter"),
    )
    assert values["MeshSurface"] == pytest.approx(spacing**2 / 2, rel=2e-6, abs=0)
    assert values["PixelSurface"] == pytest.approx(spacing**2, rel=2e-6, abs=0)
    assert values["Perimeter"] == pytest.approx(2 * sqrt(2) * spacing, rel=2e-6, abs=0)
    assert values["PerimeterSurfaceRatio"] == pytest.approx(4 * sqrt(2) / spacing, rel=2e-6, abs=0)
    assert values["Sphericity"] == pytest.approx(sqrt(pi) / 2, rel=2e-6)
    assert values["MaximumDiameter"] == pytest.approx(spacing, rel=2e-6, abs=0)


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_box_morphology_matches_analytic_extent_and_ellipsoid(
    conventions: Conventions,
) -> None:
    # The midpoint mesh clips the voxel box edges and corners. For n=3 its
    # grid volume is n^3-3n/2+2/3; its area consists of six flat faces,
    # twelve diagonal edge strips, and eight corner triangles. It is convex.
    # The AEE is the sphere whose semi-axes are 2 sqrt(lambda). Each axis has
    # sum of squared deviations 18 over 27 voxels, divided by N (pyradiomics) or N-1 (ibsi).
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    spacing = (2.0, 2.0, 2.0)
    names = (
        "MeshVolume", "SurfaceArea", "Sphericity", "Compactness1", "Compactness2",
        "SphericalDisproportion", "Asphericity",
        "VolumeDensityAxisAlignedBoundingBox", "AreaDensityAxisAlignedBoundingBox",
        "VolumeDensityApproximateEnclosingEllipsoid",
        "AreaDensityApproximateEnclosingEllipsoid",
        "VolumeDensityConvexHull", "AreaDensityConvexHull",
    )
    values = _shape_values(mask, spacing, names, conventions)
    sides = tuple(3 * value for value in spacing)
    volume = (27 - 4.5 + 2 / 3) * prod(spacing)
    area = (24 + 12 * sqrt(2) + sqrt(3)) * spacing[0] ** 2
    sphericity = (36 * pi * volume**2) ** (1 / 3) / area
    assert values["MeshVolume"] == pytest.approx(volume, rel=2e-6)
    assert values["SurfaceArea"] == pytest.approx(area, rel=2e-6)
    assert values["Sphericity"] == pytest.approx(sphericity, rel=2e-6)
    assert values["Compactness1"] == pytest.approx(
        volume / (sqrt(pi) * area**1.5), rel=2e-6
    )
    assert values["Compactness2"] == pytest.approx(
        36 * pi * volume**2 / area**3, rel=2e-6
    )
    assert values["SphericalDisproportion"] == pytest.approx(
        area / (36 * pi * volume**2) ** (1 / 3), rel=2e-6
    )
    assert values["Asphericity"] == pytest.approx(
        (area**3 / (36 * pi * volume**2)) ** (1 / 3) - 1, rel=2e-6, abs=1e-6
    )
    assert values["VolumeDensityAxisAlignedBoundingBox"] == pytest.approx(
        volume / prod(sides), rel=2e-6
    )
    assert values["AreaDensityAxisAlignedBoundingBox"] == pytest.approx(
        area / (6 * sides[0] ** 2), rel=2e-6
    )
    assert values["VolumeDensityConvexHull"] == pytest.approx(1, rel=2e-6)
    assert values["AreaDensityConvexHull"] == pytest.approx(1, rel=2e-6)

    variance = 18 / (27 if conventions == "pyradiomics" else 26)
    semi = tuple(2 * sqrt(variance * value**2) for value in spacing)
    ellipsoid_volume = 4 * pi / 3 * prod(semi)
    aee_area = 4 * pi * semi[0] ** 2
    assert values["VolumeDensityApproximateEnclosingEllipsoid"] == pytest.approx(
        volume / ellipsoid_volume, rel=2e-6
    )
    assert values["AreaDensityApproximateEnclosingEllipsoid"] == pytest.approx(
        area / aee_area, rel=2e-6
    )


@pytest.mark.parametrize("ulps", [1, 3, 64, 4096])
def test_near_spherical_oblate_area_stays_below_the_unit_sphere(ulps: int) -> None:
    import mlx.core as mx

    # Polar radius 1 - k*2**-24 below a unit equator. With q = e**2 computed exactly,
    # atanh(e)/e = sum q**n/(2n+1) converges quickly and keeps the area below 4*pi.
    c = 1 - ulps * 2.0**-24
    q = (1 - c) * (1 + c)
    expected = 2 * pi * (1 + c * c * sum(q**n / (2 * n + 1) for n in range(40)))
    actual = float(np.asarray(
        ellipsoid_area_normalized(mx.array(1.0), mx.array(1.0), mx.array(c)), dtype=np.float64,
    ).item())
    assert actual == pytest.approx(expected, rel=2e-7)
    assert actual <= float(np.float32(4 * pi))


@pytest.mark.parametrize(
    "axes",
    [(10.0, 5.0, 1.0), (20.0, 2.0, 1.0), (100.0, 1.0001, 1.0), (1000.0, 1.0, 0.5),
     (2.0, 1.5, 1.0), (1.0, 1 - 3 * 2.0**-24, 1 - 6 * 2.0**-24)],
)
def test_elongated_triaxial_ellipsoid_area_matches_carlson_rg(
    axes: tuple[float, float, float],
) -> None:
    import mlx.core as mx
    from scipy.special import elliprg

    # Exact surface 4*pi*a*b*c*R_G(a**-2, b**-2, c**-2) in float64 (DLMF 19.33.2).
    # Elongated axes make the Legendre remainder decay as (1-c**2/a**2)**n / n**2, so
    # 10:5:1 already needs far more than 512 terms for a 1e-7 bound.
    a, b, c = (float(np.float32(value)) for value in axes)
    expected = 4 * pi * a * b * c * float(elliprg(a**-2, b**-2, c**-2))
    actual = float(np.asarray(
        ellipsoid_area_normalized(mx.array(a), mx.array(b), mx.array(c)), dtype=np.float64,
    ).item())
    assert actual == pytest.approx(expected, rel=2e-6)


def _printed_series_area(a: float, b: float, c: float) -> float:
    """IBSI printed series (degree 20 inclusive) in float64 from NumPy's Legendre values."""
    from numpy.polynomial import legendre

    alpha, beta = sqrt(1 - (b / a) ** 2), sqrt(1 - (c / a) ** 2)
    argument = (alpha * alpha + beta * beta) / (2 * alpha * beta)
    return 4 * pi * a * b * sum(
        (alpha * beta) ** degree / (1 - 4 * degree * degree)
        * float(legendre.legval(argument, [0.0] * degree + [1.0]))
        for degree in range(21)
    )


def test_ibsi_series_reproduces_the_printed_expression() -> None:
    import mlx.core as mx

    def series(*axes: float) -> float:
        value = ellipsoid_area_ibsi_series(*(mx.array(axis) for axis in axes))
        return float(np.asarray(value, dtype=np.float64).item())

    # The printed eccentricities do not recover the (3, 2, 2) spheroid's exact
    # 67.6729 mm^2; a degree-20 evaluation gives 59.333825294 mm^2.
    assert series(3.0, 2.0, 2.0) == pytest.approx(59.333825294, rel=2e-6)
    assert series(3.0, 3.0, 3.0) == pytest.approx(36 * pi, rel=2e-6)
    for axes in ((5.0, 3.0, 1.0), (2.5, 1.7, 1.2), (10.0, 5.0, 1.0)):
        assert series(*axes) == pytest.approx(_printed_series_area(*axes), rel=2e-6)
    # alpha = 0 is the limit of the recurrence, not a division by zero.
    assert series(3.0, 3.0, 2.0) == pytest.approx(
        _printed_series_area(3.0, 3.0 - 1e-6, 2.0), rel=1e-5,
    )


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
def test_aee_area_density_follows_the_ellipsoid_area_convention(conventions: Conventions) -> None:
    import mlx.core as mx

    grid = np.indices((9, 7, 5)) - np.array([4, 3, 2])[:, None, None, None]
    mask = ((grid[0] / 4.2) ** 2 + (grid[1] / 3.1) ** 2 + (grid[2] / 2.3) ** 2 <= 1)
    names = ("SurfaceArea", "MajorAxisLength", "MinorAxisLength", "LeastAxisLength",
             "AreaDensityApproximateEnclosingEllipsoid")
    values = _shape_values(mask.astype(np.uint8), (1.0, 1.2, 2.0), names, conventions)
    semi = tuple(values[name] / 2 for name in names[1:4])
    if conventions == "ibsi":
        area = _printed_series_area(*semi)
    else:
        area = float(np.asarray(
            ellipsoid_area_normalized(*(mx.array(axis) for axis in semi)), dtype=np.float64,
        ).item())
    assert values["AreaDensityApproximateEnclosingEllipsoid"] == pytest.approx(
        values["SurfaceArea"] / area, rel=2e-5,
    )


@pytest.mark.parametrize("spacing", [1e-25, 1e25])
def test_dimensionless_shape_does_not_require_physical_measures(spacing: float) -> None:
    mask = np.ones((1, 1, 1), dtype=np.uint8)
    actual = _shape_values(mask, (spacing, spacing, spacing), ("Sphericity",))
    assert actual["Sphericity"] == pytest.approx(pi ** (1 / 3) / sqrt(3), rel=2e-6)


def test_anisotropic_voxel_area_survives_intermediate_scale_extremes() -> None:
    mask = np.ones((1, 1, 1), dtype=np.uint8)
    actual = _shape_values(mask, (1e25, 1e-25, 1e-25), ("SurfaceArea", "MeshVolume"))
    assert actual["SurfaceArea"] == pytest.approx(sqrt(2), rel=2e-6)
    assert actual["MeshVolume"] == pytest.approx(1e-25 / 6, rel=2e-6, abs=0)


def test_convention_sets_use_distinct_covariance_denominators_and_matching_aee() -> None:
    # On the 3x3x3 grid, each coordinate has sum of squared deviations 18.
    # Population covariance divides by 27; sample covariance divides by 26.
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    names = (
        "MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "Elongation", "Flatness",
        "VolumeDensityApproximateEnclosingEllipsoid",
        "AreaDensityApproximateEnclosingEllipsoid",
    )
    volume = 139 / 6
    area = 24 + 12 * sqrt(2) + sqrt(3)
    cases: tuple[tuple[Conventions, float], ...] = (
        ("pyradiomics", 18 / 27), ("ibsi", 18 / 26),
    )
    for conventions, variance in cases:
        result = extract(
            hcc_resampled(mask.shape), mask,
            settings=settings_for(
                conventions,
                features=tuple(f'shape_{name}' for name in names),
            ),
        )
        semi = 2 * sqrt(variance)
        expected = {
            "MajorAxisLength": 2 * semi, "MinorAxisLength": 2 * semi,
            "LeastAxisLength": 2 * semi, "Elongation": 1.0, "Flatness": 1.0,
            "VolumeDensityApproximateEnclosingEllipsoid": volume / (4 * pi * semi**3 / 3),
            "AreaDensityApproximateEnclosingEllipsoid": area / (4 * pi * semi**2),
        }
        for name, value in expected.items():
            assert result.features[f"original_shape_{name}"] == pytest.approx(value, rel=2e-6)


def test_covariance_divisor_and_ellipsoid_area_are_independent_settings() -> None:
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    names = ("MajorAxisLength", "AreaDensityApproximateEnclosingEllipsoid")
    volume_area = 24 + 12 * sqrt(2) + sqrt(3)
    features = tuple(f"shape_{name}" for name in names)
    sample_exact = extract(hcc_resampled(mask.shape), mask, settings=Settings(
        features=features, ellipsoid_area="exact"))
    population_series = extract(hcc_resampled(mask.shape), mask, settings=Settings(
        features=features, covariance_divisor="n"))
    # Spheres have the same exact and printed-series area; only the divisor differs.
    for result, variance in ((sample_exact, 18 / 26), (population_series, 18 / 27)):
        semi = 2 * sqrt(variance)
        assert result.features["original_shape_MajorAxisLength"] == pytest.approx(
            2 * semi, rel=2e-6)
        assert result.features[
            "original_shape_AreaDensityApproximateEnclosingEllipsoid"
        ] == pytest.approx(volume_area / (4 * pi * semi**2), rel=2e-6)


def test_aee_area_density_uses_the_printed_series_only_for_ibsi_series() -> None:
    # The published RDD2 values follow the manual's printed series; "exact" keeps the
    # exact surface. A 6x4x3 box has three distinct axes, so the two forms differ.
    mask = np.ones((6, 4, 3), dtype=np.uint8)
    names = ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "SurfaceArea",
             "AreaDensityApproximateEnclosingEllipsoid")
    from scipy.special import elliprg

    def values(area: EllipsoidArea) -> dict[str, float]:
        result = extract(
            hcc_resampled(mask.shape), mask,
            settings=Settings(ellipsoid_area=area,
                              features=tuple(f"shape_{name}" for name in names)),
        )
        return {name: float(result.features[f"original_shape_{name}"]) for name in names}

    for area in ("ibsi-series", "exact"):
        v = values(area)
        a, b, c = (v[name] / 2 for name in names[:3])
        if area == "ibsi-series":
            expected = _printed_series_area(a, b, c)
        else:
            expected = 4 * pi * a * b * c * float(elliprg(a**-2, b**-2, c**-2))
        assert v["AreaDensityApproximateEnclosingEllipsoid"] == pytest.approx(
            v["SurfaceArea"] / expected, rel=2e-6,
        )


def test_ibsi_shape2d_uses_sample_covariance() -> None:
    # Four corners have squared deviations totaling one on each axis, hence 1/3.
    mask = np.ones((2, 2), dtype=np.uint8)
    result = extract(
        hcc_resampled(mask.shape), mask,
        settings=Settings(features=("shape2D_MajorAxisLength",)),
    )
    assert result.features["original_shape2D_MajorAxisLength"] == pytest.approx(
        4 / sqrt(3), rel=2e-6,
    )


def test_ibsi_singleton_shape_has_zero_axes() -> None:
    mask = np.ones((1, 1, 1), dtype=np.uint8)
    names = ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength")
    result = extract(
        hcc_resampled(mask.shape), mask,
        settings=Settings(features=tuple(f"shape_{n}" for n in names)),
    )
    for name in names:
        assert result.features[f"original_shape_{name}"] == 0


def test_mesh_refuses_cropped_extents_beyond_exact_half_grid_vertices() -> None:
    # Two unit voxels 2**23 apart: float32 cannot hold their half-grid vertices, so
    # the extent is refused; a short separation keeps the exact 2*sqrt(3).
    def surface(separation: int) -> float:
        mask = np.zeros((1, 1, separation + 1), np.uint8)
        mask[0, 0, [0, separation]] = 1
        result = extract(hcc_resampled(mask.shape), mask,
                         settings=Settings(features=("shape_SurfaceArea",), bin_count=1))
        return float(result.features["original_shape_SurfaceArea"])

    assert surface(2**10) == pytest.approx(2 * sqrt(3), rel=1e-6)
    with pytest.raises(ValueError, match="cropped ROI extent"):
        surface(2**23)
