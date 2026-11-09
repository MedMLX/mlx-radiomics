"""IBSI shape features of 3-D masks (``shape``) and 2-D masks (``shape2D``)."""

from collections.abc import Collection
from math import pi, prod

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import SHAPE, SHAPE2D, CovarianceDivisor, EllipsoidArea
from mlxradiomics.features.geometry.arithmetic import (
    SpacingParts,
    checked_measure,
    scale_pow2,
    scaled_ratio,
    spacing_parts,
)
from mlxradiomics.features.geometry.boxes import ombb_measures
from mlxradiomics.features.geometry.diameters import diameter_maxima
from mlxradiomics.features.geometry.ellipsoids import (
    aee_densities,
    ellipsoid_area_normalized,
    enclosing_densities,
    mvee_semi_axes,
)
from mlxradiomics.features.geometry.hull import hull_measures, scipy_hull
from mlxradiomics.features.geometry.mesh import MeshGeometry, MeshGeometryContext
from mlxradiomics.features.geometry.moments import coordinate_rank, shape_eigenvalues
from mlxradiomics.numerics.runtime import array_from_bool, evaluate, mx

_SHAPE_MESH = frozenset({"MeshVolume", "SurfaceArea", "SurfaceVolumeRatio", "Sphericity"})
_SHAPE2D_MESH = frozenset({"MeshSurface", "Perimeter", "PerimeterSurfaceRatio", "Sphericity"})
_AXIS_FEATURES = frozenset({
    "MajorAxisLength", "MinorAxisLength", "LeastAxisLength", "Elongation", "Flatness",
})
_SHAPE_DIAMETERS = frozenset({
    "Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterColumn",
    "Maximum2DDiameterRow",
})
_SHAPE2D_DIAMETERS = frozenset({"MaximumDiameter"})
_MORPHOLOGY_3D = (
    "Compactness1", "Compactness2", "SphericalDisproportion", "Asphericity",
    "VolumeDensityAxisAlignedBoundingBox", "AreaDensityAxisAlignedBoundingBox",
    "VolumeDensityApproximateEnclosingEllipsoid",
    "AreaDensityApproximateEnclosingEllipsoid",
    "VolumeDensityOrientedMinimumBoundingBox", "AreaDensityOrientedMinimumBoundingBox",
    "VolumeDensityMinimumVolumeEnclosingEllipsoid",
    "AreaDensityMinimumVolumeEnclosingEllipsoid",
    "VolumeDensityConvexHull", "AreaDensityConvexHull",
)
_HULL_FEATURES = frozenset({"VolumeDensityConvexHull", "AreaDensityConvexHull"})
_OMBB_FEATURES = frozenset({
    "VolumeDensityOrientedMinimumBoundingBox",
    "AreaDensityOrientedMinimumBoundingBox",
})
_MVEE_FEATURES = frozenset({
    "VolumeDensityMinimumVolumeEnclosingEllipsoid",
    "AreaDensityMinimumVolumeEnclosingEllipsoid",
})
_HULL_GEOMETRY_FEATURES = _HULL_FEATURES | _OMBB_FEATURES | _MVEE_FEATURES


_AEE_FEATURES = frozenset({
    "VolumeDensityApproximateEnclosingEllipsoid",
    "AreaDensityApproximateEnclosingEllipsoid",
})
_SHAPE_MESH_ALL = _SHAPE_MESH | frozenset(_MORPHOLOGY_3D)


def _sphericity_3d(geometry: MeshGeometry) -> mx.array:
    """Dimensionless ``(36 pi V^2)^(1/3) / A`` from separately scaled measures."""
    exponent = 2 * geometry.enclosed_exponent / 3 - geometry.boundary_exponent
    whole = int(exponent // 1)
    fraction = exponent - whole
    base = ((36 * pi) ** (1 / 3) * mx.power(geometry.enclosed_norm, 2 / 3)
            / geometry.boundary_norm)
    return scale_pow2(base, whole) * (2.0 ** fraction)


def _sphericity_2d(geometry: MeshGeometry) -> mx.array:
    """Dimensionless ``2 sqrt(pi A) / P`` from separately scaled measures."""
    exponent = geometry.enclosed_exponent / 2 - geometry.boundary_exponent
    whole = int(exponent // 1)
    fraction = exponent - whole
    base = 2 * mx.sqrt(pi * geometry.enclosed_norm) / geometry.boundary_norm
    return scale_pow2(base, whole) * (2.0 ** fraction)


def _aabb_area_density(
    geometry: MeshGeometry, extents: tuple[float, ...], parts: SpacingParts,
) -> mx.array:
    """Return mesh area divided by the axis-aligned bounding-box surface area."""
    mantissa = mx.array(parts.mantissa, dtype=mx.float32)
    terms: list[mx.array] = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        term = extents[i] * extents[j] * mantissa[i] * mantissa[j]
        terms.append(scale_pow2(
            mx.array(term, dtype=mx.float32),
            parts.exponent[i] + parts.exponent[j] - parts.boundary_exponent,
        ))
    aabb_norm = 2 * (terms[0] + terms[1] + terms[2])
    return geometry.boundary_norm / aabb_norm


def shape_statistics(
    mask: NDArray[np.bool_], spacing: tuple[float, ...], scratch_mib: int,
    features: Collection[str] | None = None,
    *, covariance_divisor: CovarianceDivisor = "n-1", ellipsoid_area: EllipsoidArea = "ibsi-series",
    _mesh_context: MeshGeometryContext | None = None,
) -> dict[str, mx.array]:
    """Compute the requested ``shape`` (3-D mask) or ``shape2D`` (2-D mask) features.

    Only dependencies of the requested names are evaluated: count-only and axes-only
    requests skip the marching mesh, mesh-only and diameter-only requests skip the
    covariance spectrum, and the diameter kernel runs only for diameter features. With
    ``features=None`` every name in the mask dimension's family is requested.

    * Volume and area: MeshVolume / MeshSurface come from the marching mesh; VoxelVolume /
      PixelSurface are voxel count times voxel volume (area). SurfaceArea (3-D) and
      Perimeter (2-D) are the mesh boundary measure.
    * Ratios: SurfaceVolumeRatio and PerimeterSurfaceRatio are boundary over enclosed
      measure. Sphericity is ``(36 pi V^2)^(1/3) / A`` in 3-D and ``2 sqrt(pi A) / P`` in 2-D.
    * Compactness 1 is ``V / (sqrt(pi) A^(3/2))``; Compactness 2 is ``36 pi V^2 / A^3``;
      SphericalDisproportion is ``A / (36 pi V^2)^(1/3)``; Asphericity is
      ``(A^3 / (36 pi V^2))^(1/3) - 1`` (3-D only).
    * Densities: AABB volume/area densities use the mask's axis-aligned bounding box,
      which is also the mesh vertex bounding box. Convex-hull densities use SciPy only for
      hull connectivity and evaluate areas/volumes from the hull triangles on Metal. AEE
      volume and area use the PCA semi-axes; ``ellipsoid_area="ibsi-series"`` (default)
      evaluates the AEE area with the series printed in the IBSI manual, which reproduces
      the published area-density reference, and ``"exact"`` uses the corrected, converged
      Legendre series. OMBB volume and area use the minimum-volume box among orientations
      with one face
      flush with a hull face (an approximation of O'Rourke's exact optimum), evaluated
      in bounded Metal projection blocks; equal minimum
      volumes select the smallest area. MVEE uses Khachiyan barycentric coordinate ascent
      on the hull vertices with the IBSI tolerance 0.001, a closed-form 3x3 Schur solve,
      and the same ellipsoid-area routine; its volume and area densities use the resulting
      semi-axes. All three density pairs use the mesh volume and area.
    * Diameters: the largest distance between boundary vertices. In 3-D,
      Maximum2DDiameterSlice, Column and Row restrict to vertex pairs sharing their axis-0,
      axis-1 and axis-2 coordinate respectively. Every unordered pair is evaluated with
      the same scale-safe component norm; larger meshes only split the pair range into
      bounded Metal tiles whose maxima are then reduced.
    * Axis lengths: ``4 * sqrt(eigenvalue)`` of the ROI voxel-center covariance
      (spacing applied). ``covariance_divisor="n-1"`` (default, IBSI) uses the sample
      divisor N-1 and ``"n"`` the population divisor N. A singleton has zero axes.
      AEE semi-axes use the same divisor. Major
      and Minor use the largest and
      second-largest eigenvalues (Least uses the smallest, 3-D only); Elongation is
      ``sqrt(minor / major)`` and Flatness ``sqrt(least / major)``.

    Mesh boundary, enclosed measures and diameters are evaluated from binary spacing
    mantissas and exponents, so a representable result does not overflow through an
    intermediate squared or cubed spacing.

    Args:
        mask: Boolean ROI mask, 3-D for ``shape`` or 2-D for ``shape2D``.
        spacing: Voxel spacing per mask axis, in the unit of the requested lengths.
        scratch_mib: Budget that sizes mesh chunks and bounds diameter tile output.
        features: Bare feature names (without the ``shape_`` or ``shape2D_`` prefix) to
            compute; only those names are returned. Defaults to every feature in the mask
            dimension's family.
        covariance_divisor: ``"n-1"`` (default) or ``"n"``, as above.
        ellipsoid_area: ``"ibsi-series"`` (default) or ``"exact"``, as above.

    Returns:
        Requested feature name with its ``shape_`` or ``shape2D_`` prefix mapped to a
        float32 array of shape ``(1,)``. Lengths, areas and volumes use the spacing
        unit and its square/cube; boundary-to-enclosed-measure ratios use its inverse,
        and the other ratios are dimensionless.

    Raises:
        ValueError: If spacing does not have one value per mask axis, a name is not a
            shape feature for the mask dimension, the mask is empty, or the mesh has no
            boundary vertices.
        RuntimeError: If a requested dimensional measure or diameter is outside float32's
            representable range, the covariance spectrum is non-finite or below its PSD
            bound, a requested positive axis is too ill-conditioned, the convex hull is
            degenerate (including a planar or collinear point set for OMBB/MVEE), or MVEE
            loses positive definiteness or exceeds its iteration cap.
    """
    if covariance_divisor not in {"n-1", "n"}:
        raise ValueError("covariance_divisor must be n-1 or n")
    if ellipsoid_area not in {"ibsi-series", "exact"}:
        raise ValueError("ellipsoid_area must be ibsi-series or exact")
    ndim = mask.ndim
    if len(spacing) != ndim:
        raise ValueError("spacing must contain one value per mask axis")
    known = (frozenset(SHAPE) | frozenset(_MORPHOLOGY_3D)
             if ndim == 3 else frozenset(SHAPE2D))
    requested = known if features is None else frozenset(features)
    unsupported = requested - known
    if unsupported:
        raise ValueError(
            f"unsupported {ndim}D shape features: {', '.join(sorted(unsupported))}"
        )
    if not requested:
        return {}
    if not np.any(mask):
        raise ValueError("shape mask is empty")
    parts = spacing_parts(spacing)
    mesh_features = _SHAPE_MESH_ALL if ndim == 3 else _SHAPE2D_MESH
    diameter_features = _SHAPE_DIAMETERS if ndim == 3 else _SHAPE2D_DIAMETERS
    needs_mesh = bool(requested & mesh_features)
    needs_diameter = bool(requested & diameter_features)
    needs_axes = bool(requested & (_AXIS_FEATURES | _AEE_FEATURES))
    context = _mesh_context or MeshGeometryContext(mask, spacing, scratch_mib)
    geometry = (context.get(
        include_vertices=needs_diameter or bool(requested & _HULL_GEOMETRY_FEATURES),
    ) if needs_mesh or needs_diameter else None)
    values: dict[str, mx.array] = {}
    if "VoxelVolume" in requested or "PixelSurface" in requested:
        count = mx.sum(array_from_bool(mask).astype(mx.int32))
        physical = scale_pow2(
            count.astype(mx.float32) * prod(parts.mantissa), parts.enclosed_exponent,
        )
        name = "PixelSurface" if ndim == 2 else "VoxelVolume"
        values[name] = checked_measure(physical, name)
    if geometry is not None and needs_mesh:
        if ndim == 2:
            mesh_values: dict[str, mx.array] = {}
            if "MeshSurface" in requested:
                mesh_values["MeshSurface"] = checked_measure(
                    geometry.enclosed, "MeshSurface",
                )
            if "Perimeter" in requested:
                mesh_values["Perimeter"] = checked_measure(geometry.boundary, "Perimeter")
            if "PerimeterSurfaceRatio" in requested:
                mesh_values["PerimeterSurfaceRatio"] = checked_measure(
                    scaled_ratio(
                        geometry.boundary_norm, geometry.boundary_exponent,
                        geometry.enclosed_norm, geometry.enclosed_exponent,
                    ),
                    "PerimeterSurfaceRatio",
                )
            if "Sphericity" in requested:
                mesh_values["Sphericity"] = checked_measure(
                    _sphericity_2d(geometry), "Sphericity",
                )
        else:
            mesh_values = {}
            if "MeshVolume" in requested:
                mesh_values["MeshVolume"] = checked_measure(
                    geometry.enclosed, "MeshVolume",
                )
            if "SurfaceArea" in requested:
                mesh_values["SurfaceArea"] = checked_measure(
                    geometry.boundary, "SurfaceArea",
                )
            if "SurfaceVolumeRatio" in requested:
                mesh_values["SurfaceVolumeRatio"] = checked_measure(
                    scaled_ratio(
                        geometry.boundary_norm, geometry.boundary_exponent,
                        geometry.enclosed_norm, geometry.enclosed_exponent,
                    ),
                    "SurfaceVolumeRatio",
                )
            if requested & {
                "Sphericity", "Compactness1", "Compactness2",
                "SphericalDisproportion", "Asphericity",
            }:
                sphericity = _sphericity_3d(geometry)
                if "Sphericity" in requested:
                    mesh_values["Sphericity"] = checked_measure(sphericity, "Sphericity")
                if "Compactness1" in requested:
                    mesh_values["Compactness1"] = mx.power(sphericity, 1.5) / (6 * pi)
                if "Compactness2" in requested:
                    mesh_values["Compactness2"] = sphericity * sphericity * sphericity
                if "SphericalDisproportion" in requested:
                    mesh_values["SphericalDisproportion"] = 1 / sphericity
                if "Asphericity" in requested:
                    mesh_values["Asphericity"] = 1 / sphericity - 1
            if requested & {
                "VolumeDensityAxisAlignedBoundingBox",
                "AreaDensityAxisAlignedBoundingBox",
            }:
                extents = tuple(
                    float(value)
                    for value in np.ptp(context.coordinates(), axis=0).astype(np.float64) + 1
                )
                mesh_values["VolumeDensityAxisAlignedBoundingBox"] = (
                    geometry.grid_enclosed / (6 * prod(extents))
                )
                mesh_values["AreaDensityAxisAlignedBoundingBox"] = _aabb_area_density(
                    geometry, extents, parts,
                )
            if requested & _HULL_GEOMETRY_FEATURES:
                evaluate(geometry.vertices)
                grid = np.asarray(geometry.vertices, dtype=np.float64)
                if requested & _HULL_FEATURES:
                    hull = scipy_hull(grid)
                    hull_area, hull_volume = hull_measures(
                        hull, geometry.vertices, parts,
                    )
                    mesh_values["VolumeDensityConvexHull"] = (
                        geometry.grid_enclosed / (6 * hull_volume)
                    )
                    mesh_values["AreaDensityConvexHull"] = (
                        geometry.boundary_norm / hull_area
                    )
                if requested & (_OMBB_FEATURES | _MVEE_FEATURES):
                    # Convex-hull geometry is evaluated in physical millimetres normalized
                    # by the largest spacing, which keeps float32 projections near unit
                    # scale; the exact binary scale is restored in the density ratio.
                    physical_scale = max(spacing)
                    normalized = grid * (
                        np.asarray(spacing, dtype=np.float64) / physical_scale
                    )
                    hull = scipy_hull(normalized)
                    hull_points = normalized[hull.vertices]
                    centered = hull_points - hull_points.mean(axis=0)
                    if requested & _OMBB_FEATURES:
                        volume, area = ombb_measures(centered, hull)
                        volume_density, area_density = enclosing_densities(
                            geometry, volume, area, physical_scale,
                        )
                        mesh_values["VolumeDensityOrientedMinimumBoundingBox"] = (
                            volume_density
                        )
                        mesh_values["AreaDensityOrientedMinimumBoundingBox"] = area_density
                    if requested & _MVEE_FEATURES:
                        a, b, c = mvee_semi_axes(hull_points)
                        volume = (4 * pi / 3) * a * b * c
                        area = (
                            ellipsoid_area_normalized(a, b, c)
                            if "AreaDensityMinimumVolumeEnclosingEllipsoid" in requested
                            else mx.array(1.0)
                        )
                        volume_density, area_density = enclosing_densities(
                            geometry, volume, area, physical_scale,
                        )
                        mesh_values["VolumeDensityMinimumVolumeEnclosingEllipsoid"] = (
                            volume_density
                        )
                        mesh_values["AreaDensityMinimumVolumeEnclosingEllipsoid"] = (
                            area_density
                        )
        values.update({name: value for name, value in mesh_values.items() if name in requested})
    if needs_axes:
        voxel_coordinates = context.coordinates()
        # Translate before conversion: covariance is translation invariant, and a large
        # source origin would discard small voxel differences.
        voxel_coordinates = voxel_coordinates - voxel_coordinates.min(axis=0)
        integer_coordinates = mx.array(voxel_coordinates, dtype=mx.int64)
        rank = coordinate_rank(integer_coordinates)
        required_axes: set[int] = set()
        if requested & ({"MajorAxisLength", "Elongation", "Flatness"} | _AEE_FEATURES):
            required_axes.add(ndim - 1)
        if requested & ({"MinorAxisLength", "Elongation"} | _AEE_FEATURES):
            required_axes.add(ndim - 2)
        if requested & ({"LeastAxisLength", "Flatness"} | _AEE_FEATURES):
            required_axes.add(0)
        eigen, physical_scale = shape_eigenvalues(
            voxel_coordinates, spacing, rank, frozenset(required_axes),
        )
        count = voxel_coordinates.shape[0]
        if covariance_divisor == "n-1" and count > 1:
            # IBSI reference PCA uses sample covariance. Rescale the already
            # conditioned population spectrum so axes and AEE densities share
            # that denominator; exact null axes and axis ratios are not affected.
            eigen = eigen * (count / (count - 1))
        axis_values = {
            "MajorAxisLength": (4 * mx.sqrt(eigen[-1])) * physical_scale,
            "MinorAxisLength": (4 * mx.sqrt(eigen[-2])) * physical_scale,
            "Elongation": mx.sqrt(eigen[-2] / eigen[-1]),
        }
        if ndim == 3:
            axis_values.update(
                LeastAxisLength=(4 * mx.sqrt(eigen[0])) * physical_scale,
                Flatness=mx.sqrt(eigen[0] / eigen[-1]),
            )
        values.update({name: value for name, value in axis_values.items() if name in requested})
        if requested & _AEE_FEATURES:
            if geometry is None:
                raise RuntimeError("approximate enclosing ellipsoid requires a mesh")
            # ellipsoid_area="ibsi-series" evaluates the AEE area with the manual's printed
            # series, which the published RDD2 values follow; "exact" keeps the exact surface.
            volume_density, area_density = aee_densities(
                geometry, eigen, physical_scale,
                "AreaDensityApproximateEnclosingEllipsoid" in requested,
                printed_series=ellipsoid_area == "ibsi-series",
            )
            if "VolumeDensityApproximateEnclosingEllipsoid" in requested:
                values["VolumeDensityApproximateEnclosingEllipsoid"] = volume_density
            if "AreaDensityApproximateEnclosingEllipsoid" in requested:
                values["AreaDensityApproximateEnclosingEllipsoid"] = area_density
    if needs_diameter:
        if geometry is None:
            raise RuntimeError("diameter features require mesh boundary vertices")
        maximum = diameter_maxima(geometry.vertices, spacing, scratch_mib, ndim)
        evaluate(maximum)
        maximum_host = np.asarray(maximum, dtype=np.float32)
        if not np.all(np.isfinite(maximum_host)):
            raise RuntimeError("shape diameter exceeds the float32 range")
        if ndim == 2:
            values["MaximumDiameter"] = maximum[3]
        else:
            diameter_values = {
                "Maximum3DDiameter": maximum[3],
                "Maximum2DDiameterSlice": maximum[0],
                "Maximum2DDiameterColumn": maximum[1],
                "Maximum2DDiameterRow": maximum[2],
            }
            values.update(
                {name: value for name, value in diameter_values.items() if name in requested}
            )
        if maximum_host[3] <= 0:
            name = "MaximumDiameter" if ndim == 2 else "Maximum3DDiameter"
            raise RuntimeError(f"{name} underflows the float32 range")
    evaluate(*values.values())
    family = "shape2D" if ndim == 2 else "shape"
    return {f"{family}_{name}": value[None] for name, value in values.items()}
