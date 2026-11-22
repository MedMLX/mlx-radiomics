"""Bounded pairwise diameter reductions on mesh vertices."""

from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import spacing_parts
from mlxradiomics.features.geometry.hull import scipy_hull
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

_DISTANCE = """
using Distance = float;
template <int D>
inline float pair_distance(float3 delta, constant float* spacing) {
    float component[D];
    float scale = 0;
    for (int d = 0; d < D; d++) {
        component[d] = delta[d] * spacing[d];
        scale = max(scale, fabs(component[d]));
    }
    if (isinf(scale)) return INFINITY;
    if (scale == 0.0f) return 0.0f;
    float squares = 0;
    for (int d = 0; d < D; d++) {
        float normalized = component[d] / scale;
        squares += normalized * normalized;
    }
    return scale * sqrt(squares);
}
"""


_SCALED_DISTANCE = """
// Positive binary32 bit patterns preserve ordering, including subnormals that
// Metal could flush in a float maximum. Reinterpret after reducing the distances.
using Distance = uint;
template <int D>
inline uint pair_distance(float3 delta, constant float* spacing,
                          constant int* exponents) {
    int exponent = -2048;
    for (int d = 0; d < D; d++) {
        if (delta[d] != 0) exponent = max(exponent, exponents[d]);
    }
    if (exponent == -2048) return 0;
    float squares = 0;
    for (int d = 0; d < D; d++) {
        if (delta[d] == 0) continue;
        float component = metal::ldexp(delta[d] * spacing[d], exponents[d] - exponent);
        squares = metal::fma(component, component, squares);
    }
    float magnitude = metal::sqrt(squares);
    if (metal::ilogb(magnitude) + exponent < -126) {
        return uint(metal::rint(metal::ldexp(magnitude, exponent + 149)));
    }
    return as_type<uint>(metal::ldexp(magnitude, exponent));
}
"""


def _distance_deltas(ndim: int, second: str) -> str:
    """Read pair coordinates at the call site for either Metal address space."""
    components = [
        f"points[t * {ndim} + {axis}] - {second}[q * {ndim} + {axis}]"
        for axis in range(ndim)
    ]
    if ndim == 2:
        components.append("0.0f")
    return "float3(" + ", ".join(components) + ")"


def _distance_source(ndim: int, scaled: bool) -> tuple[str, list[str], str]:
    """Select the norm and output representation for the spacing range."""
    names = ["points", "spacing"]
    delta = _distance_deltas(ndim, "points")
    if scaled:
        return _SCALED_DISTANCE, [*names, "exponents"], (
            f"pair_distance<{ndim}>({delta}, spacing, exponents)"
        )
    return _DISTANCE, names, f"pair_distance<{ndim}>({delta}, spacing)"


@lru_cache(maxsize=8)
def _diameter_kernel(ndim: int, scaled: bool) -> MetalKernel:
    """Compile the single-pass maximum-diameter kernel.

    Thread ``t`` scans every ``q < t`` and records the overall maximum distance plus,
    in 3-D, the per-plane maxima for pairs sharing a coordinate on each axis.
    Distances are computed with a max-component scale so they cannot overflow.
    """
    header, names, distance = _distance_source(ndim, scaled)
    planes = "\n".join(
        f"if (points[t * {ndim} + {d}] == points[q * {ndim} + {d}]) "
        f"maximum[{d}] = max(maximum[{d}], distance);"
        for d in range(ndim)
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_diameters",
            input_names=names,
            output_names=["diameters"],
            header=header,
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= points_shape[0]) return;
        Distance maximum[4] = {{0, 0, 0, 0}};
        for (uint q = 0; q < t; q++) {{
            Distance distance = {distance};
            maximum[3] = max(maximum[3], distance);
            {planes if ndim == 3 else ""}
        }}
        for (int d = 0; d < 4; d++) diameters[t * 4 + d] = maximum[d];
        """,
        ),
    )


_DIAMETER_MAXIMA = 4


# Cap the number of q layers; the scratch budget can lower this further.
_DIAMETER_MAX_TILES = 16


# Below one full 256-thread layer a single dispatch already bounds the pair work, so
# tiny meshes keep the smaller output and avoid tile scheduling.
_DIAMETER_TILED_MIN_VERTICES = 256


def _diameter_tiling(vertex_count: int, scratch_mib: int) -> tuple[int, int] | None:
    """Return the bounded q-range tiling for exhaustive diameter pairs, or ``None``.

    The single-pass kernel gives vertex ``t`` a serial ``q < t`` loop. Splitting that
    range across grid layers multiplies the resident threadgroups and shortens each
    thread's dependent distance chain while every unordered pair is still visited
    exactly once. A tile writes four float32 maxima per vertex, so ``scratch_mib``
    caps the tile count; small meshes keep the single pass to avoid expanding the
    reduction input.

    Returns:
        ``(tiles, tile_width)`` with ``tiles >= 2``, or ``None`` for one pass.
    """
    if vertex_count < _DIAMETER_TILED_MIN_VERTICES:
        return None
    tile_bytes = vertex_count * _DIAMETER_MAXIMA * 4
    budget = scratch_mib * 1024**2
    tiles = min(_DIAMETER_MAX_TILES, vertex_count, max(1, budget // tile_bytes))
    if tiles < 2:
        return None
    return tiles, (vertex_count + tiles - 1) // tiles


@lru_cache(maxsize=8)
def _tiled_diameter_kernel(ndim: int, scaled: bool) -> MetalKernel:
    """Compile the diameter kernel that splits each vertex's ``q`` range across grid layers.

    Every layer owns one ``q`` range; the condition and norm are identical to the
    single-pass kernel, so tiling partitions the work without changing any distance.
    """
    header, names, distance = _distance_source(ndim, scaled)
    planes = "\n".join(
        f"if (points[t * {ndim} + {d}] == points[q * {ndim} + {d}]) "
        f"maximum[{d}] = max(maximum[{d}], distance);"
        for d in range(ndim)
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_tiled_diameters",
            input_names=[*names, "limits"],
            output_names=["diameters"],
            header=header,
            source=f"""
        uint t = thread_position_in_grid.x;
        uint tile = thread_position_in_grid.y;
        uint count = (uint)points_shape[0];
        if (t >= count) return;
        uint width = (uint)limits[0];
        uint begin = tile * width;
        uint end = min(begin + width, t);
        Distance maximum[4] = {{0, 0, 0, 0}};
        for (uint q = begin; q < end; q++) {{
            Distance distance = {distance};
            maximum[3] = max(maximum[3], distance);
            {planes if ndim == 3 else ""}
        }}
        uint offset = (tile * count + t) * 4;
        for (int d = 0; d < 4; d++) diameters[offset + d] = maximum[d];
        """,
        ),
    )


def _all_pairs(
    vertices: mx.array,
    spacing: list[mx.array],
    scratch_mib: int,
    ndim: int,
) -> mx.array:
    """Return the overall and per-plane maxima over every unordered vertex pair."""
    scaled = len(spacing) > 1
    dtype = mx.uint32 if scaled else mx.float32
    vertex_count = vertices.shape[0]
    tiling = _diameter_tiling(vertex_count, scratch_mib)
    if tiling is None:
        diameters = _diameter_kernel(ndim, scaled)(
            inputs=[vertices, *spacing],
            grid=(vertex_count, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(vertex_count, _DIAMETER_MAXIMA)],
            output_dtypes=[dtype],
        )[0]
        return mx.max(diameters, axis=0)
    tiles, width = tiling
    diameters = _tiled_diameter_kernel(ndim, scaled)(
        inputs=[vertices, *spacing, mx.array([width], dtype=mx.int32)],
        grid=(vertex_count, tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tiles, vertex_count, _DIAMETER_MAXIMA)],
        output_dtypes=[dtype],
    )[0]
    return mx.max(mx.reshape(diameters, (-1, _DIAMETER_MAXIMA)), axis=0)


@lru_cache(maxsize=2)
def _plane_diameter_kernel(scaled: bool) -> MetalKernel:
    """Compile the maximum distance search within each ordered vertex group."""
    header, names, distance = _distance_source(3, scaled)
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_plane_diameters",
            input_names=[*names, "starts"],
            output_names=["maxima"],
            header=header,
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= uint(points_shape[0])) return;
        Distance maximum = 0;
        for (uint q = starts[t]; q < t; q++) {{
            maximum = max(maximum, {distance});
        }}
        maxima[t] = maximum;
        """,
        ),
    )


def _plane_maximum(
    vertices: mx.array, coordinates: NDArray[np.float64], spacing: list[mx.array], axis: int,
) -> mx.array:
    """Compare only vertices that lie on the same half-index plane."""
    order = np.argsort(coordinates[:, axis], kind="stable")
    planes = coordinates[order, axis]
    changes = np.flatnonzero(planes[1:] != planes[:-1]) + 1
    starts = np.zeros(len(order), dtype=np.uint32)
    starts[changes] = changes
    np.maximum.accumulate(starts, out=starts)
    points = mx.take(vertices, mx.array(order.astype(np.int32)), axis=0)
    scaled = len(spacing) > 1
    dtype = mx.uint32 if scaled else mx.float32
    [maxima] = _plane_diameter_kernel(scaled)(
        inputs=[points, *spacing, mx.array(starts)],
        grid=(len(order), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(len(order),)],
        output_dtypes=[dtype],
    )
    return mx.max(maxima)


@lru_cache(maxsize=1)
def _farthest_hull_kernel(ndim: int) -> MetalKernel:
    """Return each vertex's maximum ordinary-precision distance to the hull."""
    delta = _distance_deltas(ndim, "hull")
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_farthest_hull_vertex",
        input_names=["points", "hull", "spacing"],
        output_names=["farthest"],
        header=_DISTANCE,
        source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= points_shape[0]) return;
        float maximum = 0;
        for (uint q = 0; q < hull_shape[0]; q++) {{
            maximum = max(maximum, pair_distance<{ndim}>({delta}, spacing));
        }}
        farthest[t] = maximum;
        """,
    ))


def _search_maxima(
    vertices: mx.array, spacing: list[mx.array], scratch_mib: int, ndim: int,
    hull_indices: NDArray[np.int32] | None,
) -> mx.array:
    """Return the overall and per-plane maximum boundary-vertex distances on Metal.

    Distance from a fixed point is convex, so an overall diameter is attained
    by hull vertices. Plane diameters need their own vertices and group by
    equal grid coordinates. Small or scratch-limited meshes use all pairs.
    """
    count = vertices.shape[0]
    # Bound host hull and plane-ordering work. Small meshes already fit in a
    # short all-pairs dispatch, avoiding a host topology pass.
    if count < 8192 or count * 128 > scratch_mib * 1024**2:
        return _all_pairs(vertices, spacing, scratch_mib, ndim)

    coordinates = np.asarray(vertices, dtype=np.float64)
    if hull_indices is None:
        hull_indices = scipy_hull(coordinates).vertices
    hull_vertices = mx.take(vertices, mx.array(hull_indices), axis=0)
    overall = _all_pairs(hull_vertices, spacing, scratch_mib, ndim)[3]
    if len(spacing) == 1:
        # Exact distances are maximized on the hull. Recheck near-maximal vertices
        # so rounding cannot remove a pair that wins in the exhaustive float32 search.
        [farthest] = _farthest_hull_kernel(ndim)(
            inputs=[vertices, hull_vertices, spacing[0]],
            grid=(count, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(count,)], output_dtypes=[mx.float32],
        )
        keep = farthest >= overall * (1.0 - 2.0**-16)
        evaluate(keep)
        indices = np.flatnonzero(np.asarray(keep, dtype=np.bool_)).astype(np.int32)
        candidates = mx.take(vertices, mx.array(indices), axis=0)
        overall = _all_pairs(candidates, spacing, scratch_mib, ndim)[3]
    if ndim == 2:
        return mx.stack([mx.zeros_like(overall)] * 3 + [overall])
    planes: list[mx.array] = []
    for axis in range(3):
        maximum = _plane_maximum(vertices, coordinates, spacing, axis)
        evaluate(maximum)
        planes.append(maximum)
    return mx.stack([*planes, overall])


def diameter_maxima(
    vertices: mx.array, spacing: tuple[float, ...], scratch_mib: int, ndim: int,
    *, hull_indices: NDArray[np.int32] | None = None,
) -> mx.array:
    """Find diameter maxima, reusing grid-hull vertex indices when already available."""
    # The smallest nonzero grid displacement is half a voxel. Keep it
    # and its contribution to the planar norm even at the float32 normal boundary.
    if min(spacing) >= 2 * np.finfo(np.float32).tiny and max(spacing) <= np.finfo(np.float32).max:
        return _search_maxima(
            vertices, [mx.array(spacing, dtype=mx.float32)], scratch_mib, ndim, hull_indices,
        )
    parts = spacing_parts(spacing)
    scales = [mx.array(parts.mantissa, dtype=mx.float32), mx.array(parts.exponent, dtype=mx.int32)]
    return _search_maxima(vertices, scales, scratch_mib, ndim, hull_indices).view(mx.float32)
