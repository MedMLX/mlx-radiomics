"""Bounded pairwise diameter reductions on mesh vertices."""

from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx


@lru_cache(maxsize=4)
def _diameter_kernel(ndim: int) -> MetalKernel:
    """Compile the single-pass maximum-diameter kernel.

    Thread ``t`` scans every ``q < t`` and records the overall maximum distance plus,
    in 3-D, the per-plane maxima for pairs sharing a coordinate on each axis.
    Distances are computed with a max-component scale so they cannot overflow.
    """
    planes = "\n".join(
        f"if (points[t * {ndim} + {d}] == points[q * {ndim} + {d}]) "
        f"maximum[{d}] = max(maximum[{d}], distance);"
        for d in range(ndim)
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_diameters",
            input_names=["points", "spacing"],
            output_names=["diameters"],
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= points_shape[0]) return;
        float maximum[4] = {{0, 0, 0, 0}};
        for (uint q = 0; q < t; q++) {{
            float component[{ndim}];
            float scale = 0;
            for (int d = 0; d < {ndim}; d++) {{
                component[d] = (points[t * {ndim} + d] - points[q * {ndim} + d])
                               * spacing[d];
                scale = max(scale, fabs(component[d]));
            }}
            float distance;
            if (isinf(scale)) {{
                distance = INFINITY;
            }} else if (scale == 0.0f) {{
                distance = 0.0f;
            }} else {{
                float squares = 0;
                for (int d = 0; d < {ndim}; d++) {{
                    float normalized = component[d] / scale;
                    squares += normalized * normalized;
                }}
                distance = scale * sqrt(squares);
            }}
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


@lru_cache(maxsize=4)
def _tiled_diameter_kernel(ndim: int) -> MetalKernel:
    """Compile the diameter kernel that splits each vertex's ``q`` range across grid layers.

    Every layer owns one ``q`` range; the condition and norm are identical to the
    single-pass kernel, so tiling partitions the work without changing any distance.
    """
    planes = "\n".join(
        f"if (points[t * {ndim} + {d}] == points[q * {ndim} + {d}]) "
        f"maximum[{d}] = max(maximum[{d}], distance);"
        for d in range(ndim)
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_tiled_diameters",
            input_names=["points", "spacing", "limits"],
            output_names=["diameters"],
            source=f"""
        uint t = thread_position_in_grid.x;
        uint tile = thread_position_in_grid.y;
        uint count = (uint)points_shape[0];
        if (t >= count) return;
        uint width = (uint)limits[0];
        uint begin = tile * width;
        uint end = min(begin + width, t);
        float maximum[4] = {{0, 0, 0, 0}};
        for (uint q = begin; q < end; q++) {{
            float component[{ndim}];
            float scale = 0;
            for (int d = 0; d < {ndim}; d++) {{
                component[d] = (points[t * {ndim} + d] - points[q * {ndim} + d])
                               * spacing[d];
                scale = max(scale, fabs(component[d]));
            }}
            float distance;
            if (isinf(scale)) {{
                distance = INFINITY;
            }} else if (scale == 0.0f) {{
                distance = 0.0f;
            }} else {{
                float squares = 0;
                for (int d = 0; d < {ndim}; d++) {{
                    float normalized = component[d] / scale;
                    squares += normalized * normalized;
                }}
                distance = scale * sqrt(squares);
            }}
            maximum[3] = max(maximum[3], distance);
            {planes if ndim == 3 else ""}
        }}
        uint offset = (tile * count + t) * 4;
        for (int d = 0; d < 4; d++) diameters[offset + d] = maximum[d];
        """,
        ),
    )


def diameter_maxima(
    vertices: mx.array,
    spacing: tuple[float, ...],
    scratch_mib: int,
    ndim: int,
    hull_vertices: NDArray[np.int32] | None = None,
) -> mx.array:
    """Return the overall and per-axis maximum boundary-vertex distances on Metal.

    Every unordered vertex pair is evaluated exactly once with the same scale-safe
    float32 component norm. A large mesh splits the ``q < t`` range into bounded
    tiles (``_diameter_tiling``) and reduces the per-tile maxima with a Metal max;
    the maximum of maxima selects the same float32 distance values as one pass.

    With ``hull_vertices`` (3-D only, the convex-hull vertex indices of ``vertices``)
    the same maxima are found without visiting every pair; see
    ``_pruned_diameter_maxima``.
    """
    if hull_vertices is not None:
        return _pruned_diameter_maxima(vertices, spacing, scratch_mib, hull_vertices)
    vertex_count = vertices.shape[0]
    spacing_array = mx.array(spacing, dtype=mx.float32)
    tiling = _diameter_tiling(vertex_count, scratch_mib)
    if tiling is None:
        diameters = _diameter_kernel(ndim)(
            inputs=[vertices, spacing_array],
            grid=(vertex_count, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(vertex_count, _DIAMETER_MAXIMA)],
            output_dtypes=[mx.float32],
        )[0]
        return mx.max(diameters, axis=0)
    tiles, width = tiling
    diameters = _tiled_diameter_kernel(ndim)(
        inputs=[vertices, spacing_array, mx.array([width], dtype=mx.int32)],
        grid=(vertex_count, tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tiles, vertex_count, _DIAMETER_MAXIMA)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.max(mx.reshape(diameters, (-1, _DIAMETER_MAXIMA)), axis=0)


# Below this vertex count one exhaustive pass is cheaper than the hull, the grouped plane
# pass and the pruning bound together, so shape features keep the exhaustive kernel.
PRUNED_DIAMETER_MIN_VERTICES = 32768


# Relative slack of the pruning bound. The float32 component norm is within about ten
# ulps (fast-math division and square root included) of the exact distance between the
# same float32 grid points, far below 2**-20; the slack is wider still.
_PRUNE_SLACK = 2.0**-16


def _distance_source(first: str, second: str) -> str:
    """Metal statements setting ``distance`` from ``first`` and ``second`` 3-vectors.

    The statements are those of ``_diameter_kernel``: the same scale-safe norm, so a
    pair evaluated here rounds exactly as it does in the exhaustive kernels.
    """
    return f"""
            float component[3];
            float scale = 0;
            for (int d = 0; d < 3; d++) {{
                component[d] = ({first}[d] - {second}[d])
                               * spacing[d];
                scale = max(scale, fabs(component[d]));
            }}
            float distance;
            if (isinf(scale)) {{
                distance = INFINITY;
            }} else if (scale == 0.0f) {{
                distance = 0.0f;
            }} else {{
                float squares = 0;
                for (int d = 0; d < 3; d++) {{
                    float normalized = component[d] / scale;
                    squares += normalized * normalized;
                }}
                distance = scale * sqrt(squares);
            }}"""


@lru_cache(maxsize=1)
def _grouped_diameter_kernel() -> MetalKernel:
    """Compile the kernel for the largest distance within coordinate groups.

    ``points`` are sorted so that each group of equal coordinates is contiguous, and
    ``starts[t]`` is the first row of row ``t``'s group. Row ``t`` scans the earlier rows
    of its own group, so every unordered pair sharing the grouped coordinate is visited
    exactly once and no other pair is.
    """
    distance = _distance_source("(points + t * 3)", "(points + q * 3)")
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_grouped_diameters",
            input_names=["points", "spacing", "starts"],
            output_names=["diameters"],
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= points_shape[0]) return;
        float maximum = 0;
        for (uint q = uint(starts[t]); q < t; q++) {{
            {distance}
            maximum = max(maximum, distance);
        }}
        diameters[t] = maximum;
        """,
        ),
    )


@lru_cache(maxsize=1)
def _farthest_kernel() -> MetalKernel:
    """Compile the kernel for each point's largest distance to a set of hull vertices."""
    distance = _distance_source("(points + t * 3)", "(hull + h * 3)")
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="mlxradiomics_farthest_hull_vertex",
            input_names=["points", "hull", "spacing"],
            output_names=["farthest"],
            source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= points_shape[0]) return;
        float maximum = 0;
        for (uint h = 0; h < hull_shape[0]; h++) {{
            {distance}
            maximum = max(maximum, distance);
        }}
        farthest[t] = maximum;
        """,
        ),
    )


def _plane_maxima(vertices: mx.array, spacing_array: mx.array) -> mx.array:
    """Return the three per-plane maxima over pairs sharing an axis-0, 1 or 2 coordinate.

    Each axis sorts the vertices by that coordinate on Metal and marks every row with the
    first row of its run of equal float32 values; ``_grouped_diameter_kernel`` then visits
    exactly the pairs that the exhaustive kernel counts for that plane.
    """
    count = vertices.shape[0]
    index = mx.arange(count, dtype=mx.int32)
    blocks: list[mx.array] = []
    starts: list[mx.array] = []
    for axis in range(3):
        ordered = mx.take(vertices, mx.argsort(vertices[:, axis]), axis=0)
        values = ordered[:, axis]
        first = mx.concatenate([mx.array([True]), values[1:] != values[:-1]])
        blocks.append(ordered)
        starts.append(mx.cummax(mx.where(first, index, 0)) + axis * count)
    grouped = _grouped_diameter_kernel()(
        inputs=[mx.concatenate(blocks), spacing_array, mx.concatenate(starts)],
        grid=(3 * count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(3 * count,)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.max(mx.reshape(grouped, (3, count)), axis=1)


def _pruned_diameter_maxima(
    vertices: mx.array,
    spacing: tuple[float, ...],
    scratch_mib: int,
    hull_vertices: NDArray[np.int32],
) -> mx.array:
    """Return the ``diameter_maxima`` values of a 3-D mesh without the all-pairs pass.

    Plane maxima visit only the pairs that share the plane coordinate
    (``_plane_maxima``), which is exactly the pair set the exhaustive kernel counts.

    The overall maximum ``F`` is the largest float32 distance over all vertex pairs. Its
    exact value is attained between convex-hull vertices, but the rounded maximum need
    not be, so the hull pairs only give a lower bound ``L <= F``. For each vertex the
    largest distance to any vertex is its largest distance to a hull vertex (distance to
    a fixed point is convex). A vertex whose farthest hull vertex is closer than
    ``L * (1 - 2**-16)`` therefore belongs to no pair whose float32 distance can reach
    ``F``; the remaining vertices contain the pair attaining ``F``, and the exhaustive
    kernel over them returns ``F`` itself. Only vertices nearly as eccentric as the
    diameter endpoints survive, so this pass is short.
    """
    spacing_array = mx.array(spacing, dtype=mx.float32)
    hull = mx.take(vertices, mx.array(hull_vertices, dtype=mx.int32), axis=0)
    lower = diameter_maxima(hull, spacing, scratch_mib, 3)[3]
    farthest = _farthest_kernel()(
        inputs=[vertices, hull, spacing_array],
        grid=(vertices.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(vertices.shape[0],)],
        output_dtypes=[mx.float32],
    )[0]
    keep = farthest >= lower * (1 - _PRUNE_SLACK)
    planes = _plane_maxima(vertices, spacing_array)
    evaluate(keep, planes)
    kept = np.flatnonzero(np.asarray(keep, dtype=np.bool_)).astype(np.int32)
    candidates = mx.take(vertices, mx.array(kept), axis=0)
    overall = diameter_maxima(candidates, spacing, scratch_mib, 3)[3]
    return mx.concatenate([planes, overall[None]])
