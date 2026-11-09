"""Integer pair histograms with MLX indexed addition or a shared Metal kernel."""

from functools import lru_cache
from math import prod
from typing import cast

from mlxradiomics.numerics.protocols import IndexedAddition, MetalKernel
from mlxradiomics.numerics.runtime import mx

# Each group scans a bounded tile. Four integer histograms distribute hot-bin
# contention; their combined storage is at most 16 KiB at the admitted 32 x 32
# cells.
_TILE_PAIRS = 4096
_HISTOGRAM_SHARDS = 4
_MAX_TILED_CELLS = 32 * 32


@lru_cache(maxsize=32)
def _tiled_metal_histogram(width: int, ng: int, columns: int) -> MetalKernel:
    """Accumulate a tile locally, then flush each occupied bin once to device memory.

    Every pair belongs to exactly one tile of one row. The local counters contain
    at most ``_TILE_PAIRS`` observations in total, so merging their uint32 counts
    is exact. The final uint32 histogram has the same count range as the direct
    kernel. Barriers include every lane, including lanes past the last valid pair.
    """
    chunks = (width + _TILE_PAIRS - 1) // _TILE_PAIRS
    cells = ng * columns
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_glcm_histogram_tiled",
        input_names=["left", "right", "valid"], output_names=["counts"],
        atomic_outputs=True,
        source=f"""
            uint lane = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.x;
            uint row = group / {chunks};
            uint chunk = group % {chunks};
            threadgroup atomic_uint local_counts[{_HISTOGRAM_SHARDS * cells}];
            for (uint cell = lane; cell < {_HISTOGRAM_SHARDS * cells}; cell += 256) {{
                atomic_store_explicit(&local_counts[cell], 0u, memory_order_relaxed);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shard = lane % {_HISTOGRAM_SHARDS};
            ulong begin = ulong(chunk) * {_TILE_PAIRS};
            for (uint i = lane; i < {_TILE_PAIRS}; i += 256) {{
                ulong position = begin + i;
                if (position < {width}) {{
                    ulong t = ulong(row) * {width} + position;
                    if (valid[t]) {{
                        uint cell = (uint(left[t]) - 1u) * {columns} + uint(right[t]) - 1u;
                        atomic_fetch_add_explicit(
                            &local_counts[shard * {cells} + cell],
                            1u, memory_order_relaxed);
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint cell = lane; cell < {cells}; cell += 256) {{
                uint total = 0u;
                for (uint shard = 0; shard < {_HISTOGRAM_SHARDS}; ++shard) {{
                    total += atomic_load_explicit(
                        &local_counts[shard * {cells} + cell], memory_order_relaxed);
                }}
                if (total != 0u) {{
                    atomic_fetch_add_explicit(
                        &counts[ulong(row) * {cells} + cell], total, memory_order_relaxed);
                }}
            }}
        """,
    ))


@lru_cache(maxsize=64)
def _roi_histogram_kernel(
    shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], ng: int,
) -> MetalKernel:
    """Count every admitted ROI direction directly from resident bins and mask."""
    strides = tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    coordinates = "\n".join(
        f"int c{axis} = int(position / {stride}) % {size};"
        for axis, (stride, size) in enumerate(zip(strides, shape, strict=True))
    )
    updates: list[str] = []
    for direction, offset in enumerate(offsets):
        inside = " && ".join(
            f"c{axis} + ({delta}) >= 0 && c{axis} + ({delta}) < {size}"
            for axis, (delta, size) in enumerate(zip(offset, shape, strict=True)) if delta
        )
        step = sum(stride * delta for stride, delta in zip(strides, offset, strict=True))
        updates.append(f"""
            if ({inside or 'true'}) {{
                uint q = uint(int(position) + ({step}));
                if (mask[q]) {{
                    uint index = ({direction}u * {ng}u + uint(gray)) * {ng}u
                                 + uint(bins[q] - 1);
                    atomic_fetch_add_explicit(&counts[index], 1u, memory_order_relaxed);
                }}
            }}
        """)
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_roi_glcm_counts", input_names=["bins", "mask"],
        output_names=["counts"], atomic_outputs=True,
        source=f"""
            uint position = thread_position_in_grid.x;
            if (position >= {prod(shape)}u || !mask[position]) return;
            int gray = bins[position] - 1;
            {coordinates}
            {''.join(updates)}
        """,
    ))


def roi_histograms(
    bins: mx.array, mask: mx.array, offsets: tuple[tuple[int, ...], ...], ng: int,
) -> mx.array:
    """Exact uint32 directional counts without materializing voxel-pair arrays.

    Inputs are same-shaped contiguous integer bins and bool masks on a 2-D or 3-D
    grid. Valid bins are 1-based and at most ``ng``. Each masked source voxel visits
    each in-grid offset once; no direction has more than the grid's voxel count.
    The caller admits the count storage and keeps grids below 2**30 voxels.
    """
    return _roi_histogram_kernel(tuple(bins.shape), offsets, ng)(
        inputs=[mx.reshape(bins, (-1,)), mx.reshape(mask, (-1,))],
        grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(len(offsets), ng, ng)], output_dtypes=[mx.uint32], init_value=0,
    )[0]


@lru_cache(maxsize=32)
def _metal_histogram(batch: int, width: int, ng: int, columns: int) -> MetalKernel:
    """Compile the atomic kernel counting valid ``(left, right)`` level pairs per row."""
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_pair_histogram",
        input_names=["left", "right", "valid"], output_names=["counts"],
        atomic_outputs=True,
        source=f"""
            size_t t = thread_position_in_grid.x;
            if (t < {batch * width} && valid[t]) {{
                size_t row = t / {width};
                size_t index = (row * {ng} + left[t] - 1) * {columns} + right[t] - 1;
                atomic_fetch_add_explicit(&counts[index], 1u, memory_order_relaxed);
            }}
        """,
    ))


def histogram(
    left: mx.array, right: mx.array, valid: mx.array, ng: int, *, method: str,
    columns: int | None = None,
) -> mx.array:
    """Count gray-level co-occurrences for a batch of voxel-pair lists.

    Args:
        left: Integer array ``(batch, pairs)`` of 1-based gray levels of the first voxel
            of each pair. Valid-pair levels must be in ``1..ng``; this is not checked.
        right: Same shape; valid indices on the second axis lie in ``1..columns``.
        valid: Boolean array of the same shape; pairs with False are not counted (their
            gray levels may be 0).
        ng: Number of gray levels.
        columns: Number of second-axis bins, defaulting to ``ng``. Texture size
            matrices use this for their run/dependence/zone-size or distance axis.
        method: ``"mlx"`` accumulates with MLX indexed addition; ``"metal"`` uses
            integer atomic adds. Long rows with at most 1,024 cells first count
            in bounded threadgroup memory to reduce device-atomic contention.

    Returns:
        uint32 array ``(batch, ng, columns)``; entry ``[b, i - 1, j - 1]`` is the number of
        valid pairs with ``left == i`` and ``right == j``. The matrix is not symmetrized.

    Raises:
        ValueError: If ``method`` is neither ``"mlx"`` nor ``"metal"``.
        RuntimeError: If ``method`` is ``"metal"`` and Metal is unavailable.
    """
    batch, width = left.shape
    columns = ng if columns is None else columns
    if method not in ("mlx", "metal"):
        raise ValueError("histogram method must be mlx or metal")
    if batch == 0 or width == 0:
        # No pairs: a zero-length scatter is rejected and a Metal dispatch faults.
        return mx.zeros((batch, ng, columns), dtype=mx.uint32)
    if method == "metal":
        if not mx.metal.is_available():
            raise RuntimeError("Metal histogram requires Apple Metal")
        if 0 < ng * columns <= _MAX_TILED_CELLS and width >= _TILE_PAIRS:
            chunks = (width + _TILE_PAIRS - 1) // _TILE_PAIRS
            return _tiled_metal_histogram(width, ng, columns)(
                inputs=[left, right, valid], grid=(batch * chunks * 256, 1, 1),
                threadgroup=(256, 1, 1), output_shapes=[(batch, ng, columns)],
                output_dtypes=[mx.uint32], init_value=0,
            )[0]
        return _metal_histogram(batch, width, ng, columns)(
            inputs=[left, right, valid], grid=(batch * width, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(batch, ng, columns)], output_dtypes=[mx.uint32], init_value=0,
        )[0]
    rows = mx.arange(batch, dtype=mx.int32)[:, None]
    # Invalid pairs may carry gray level 0; clamp their index and add 0 for them below.
    index = rows * ng * columns + mx.maximum(left - 1, 0) * columns + mx.maximum(right - 1, 0)
    result = mx.zeros((batch * ng * columns,), dtype=mx.uint32)
    counts = cast(IndexedAddition, result.at[mx.reshape(index, (-1,))]).add(
        mx.reshape(valid.astype(mx.uint32), (-1,))
    )
    return mx.reshape(counts, (batch, ng, columns))
