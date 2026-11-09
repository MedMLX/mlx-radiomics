"""Bounded pairwise diameter reductions on mesh vertices."""

from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx


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
) -> mx.array:
    """Return the overall and per-axis maximum boundary-vertex distances on Metal.

    Every unordered vertex pair is evaluated exactly once with the same scale-safe
    float32 component norm. A large mesh splits the ``q < t`` range into bounded
    tiles (``_diameter_tiling``) and reduces the per-tile maxima with a Metal max;
    the maximum of maxima selects the same float32 distance values as one pass.
    """
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
