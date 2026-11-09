"""GLDM, NGTDM, GLRLM and GLSZM statistics built on Metal.

Each function works on a batch of rows, where a row is the whole ROI (ROI mode) or one local
neighborhood (voxel mode). ``bins`` holds 1-based gray levels, ``mask`` marks voxels that
belong to the row, and ``shape`` is the row's spatial shape. GLDM, GLRLM and GLSZM use
integer counts; NGTDM also accumulates float32 neighborhood differences. Neighborhood
geometry comes from ``preprocessing.directions``.
"""

from dataclasses import replace
from functools import lru_cache
from math import prod
from typing import cast

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.distributions import ngtdm_statistics, size_statistics
from mlxradiomics.features.sparse_size import sparse_size_statistics, use_sparse_size
from mlxradiomics.imaging.preprocessing import directions
from mlxradiomics.numerics.histogram import histogram
from mlxradiomics.numerics.protocols import IndexedAddition, MetalKernel
from mlxradiomics.numerics.reduction import directional_mean
from mlxradiomics.numerics.runtime import evaluate, mx, trim_cache
from mlxradiomics.numerics.sparse_counts import coordinate_counts


def neighbor_code(shape: tuple[int, ...], offset: tuple[int, ...], name: str) -> str:
    """Return Metal source declaring ``int name`` as the flat index of ``position + offset``.

    The index is -1 when that neighbor falls outside ``shape``.
    """
    strides = [prod(shape[i + 1:]) for i in range(len(shape))]
    checks = [f"int((position / {stride}) % {size}) + ({delta}) >= 0 && "
              f"int((position / {stride}) % {size}) + ({delta}) < {size}"
              for size, stride, delta in zip(shape, strides, offset, strict=True)]
    delta = sum(s * d for s, d in zip(strides, offset, strict=True))
    return f"int {name} = ({' && '.join(checks)}) ? int(position) + ({delta}) : -1;"


@lru_cache(maxsize=128)
def neighborhood_kernel(
    shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], alpha: int,
    families: tuple[str, ...] = ("gldm", "ngtdm"),
) -> MetalKernel:
    """Compile the per-voxel GLDM/NGTDM neighborhood kernel.

    For each voxel, over the masked neighbors at ``offsets``, it computes GLDM
    dependence (1 + neighbors within ``alpha`` gray levels), the NGTDM
    ``|center - mean neighbor gray level|`` and whether the voxel has at least one
    masked neighbor. Only the outputs of ``families`` are accumulated.
    """
    accumulation = ("sum += bins[row + q]; n++; " if "ngtdm" in families else "")
    accumulation += (f"dependence += abs(bins[t] - bins[row + q]) <= {alpha}; "
                     if "gldm" in families else "")
    neighbors = "\n".join(
        f"{{ {neighbor_code(shape, offset, 'q')} if (q >= 0 && mask[row + q]) "
        f"{{ {accumulation} }} }}" for offset in offsets)
    outputs = (["dependences"] if "gldm" in families else [])
    outputs += ["difference", "eligible"] if "ngtdm" in families else []
    stores = "dependences[t] = dependence;" if "gldm" in families else ""
    if "ngtdm" in families:
        stores += """
        // Subtract exact integer sums before division; rounded means lose small
        // differences at high gray levels. The admitted index/label range fits int64.
        difference[t] = n > 0 ? float(abs(long(n) * long(bins[t]) - sum)) / float(n) : 0;
        eligible[t] = mask[t] && n > 0;
        """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_neighborhood_" + "_".join(families), input_names=["bins", "mask"],
        output_names=outputs,
        source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= bins_shape[0] * bins_shape[1]) return;
        uint position = t % {prod(shape)}, row = t - position;
        int dependence = 1, n = 0; long sum = 0;
        if (mask[t]) {{ {neighbors} }}
        {stores}
        """,
    ))


def neighborhood_arrays(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...],
    offsets: tuple[tuple[int, ...], ...], alpha: int, families: tuple[str, ...],
) -> dict[str, mx.array]:
    """Calculate the neighborhood arrays of the requested families."""
    names = ["dependences"] if "gldm" in families else []
    dtypes = [mx.int32] if "gldm" in families else []
    if "ngtdm" in families:
        names += ["difference", "eligible"]
        dtypes += [mx.float32, mx.bool_]
    values = neighborhood_kernel(shape, offsets, alpha, families)(
        inputs=[bins, mask], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[bins.shape] * len(names), output_dtypes=dtypes,
    )
    return dict(zip(names, values, strict=True))


def count_matrix(bins: mx.array, sizes: mx.array, valid: mx.array, ng: int, ns: int) -> mx.array:
    """Count valid 1-based (gray level, size) pairs as uint32 ``(batch, ng, ns)``.

    Use the shared integer Metal histogram without materializing scatter indices
    or updates. Invalid entries are skipped before reading their pair indices.
    """
    return histogram(bins, sizes, valid, ng, method="metal", columns=ns)


@lru_cache(maxsize=64)
def _ngtdm_reduction_kernel(width: int, ng: int, chunks: int, threads: int) -> MetalKernel:
    """Reduce one row/level/tile per group, without floating-point atomics."""
    tile = (width + chunks - 1) // chunks
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_ngtdm_levels",
        input_names=["bins", "eligible", "difference"], output_names=["counts", "sums"],
        source=f"""
        uint lane = thread_position_in_threadgroup.x;
        uint chunk = threadgroup_position_in_grid.x % {chunks};
        uint level = threadgroup_position_in_grid.x / {chunks} + 1;
        uint row = threadgroup_position_in_grid.y;
        uint count = 0; float total = 0;
        for (size_t p = size_t(chunk) * {tile} + lane;
             p < min(size_t(chunk + 1) * {tile}, size_t({width})); p += {threads}) {{
            size_t index = size_t(row) * {width} + p;
            if (eligible[index] && bins[index] == int(level)) {{
                ++count; total += difference[index];
            }}
        }}
        threadgroup uint nc[{threads}];
        threadgroup float sc[{threads}];
        nc[lane] = count; sc[lane] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = {threads // 2}; step > 0; step /= 2) {{
            if (lane < step) {{ nc[lane] += nc[lane + step]; sc[lane] += sc[lane + step]; }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (lane == 0) {{
            size_t out = (size_t(row) * {ng} + level - 1) * {chunks} + chunk;
            counts[out] = nc[0]; sums[out] = sc[0];
        }}
        """,
    ))


def ngtdm_level_reductions(
    bins: mx.array, eligible: mx.array, difference: mx.array, ng: int,
) -> tuple[mx.array, mx.array]:
    """Return integer occurrences and float32 difference sums per row/gray level.

    Inputs share ``(batch, width)``; eligible bins lie in ``1..ng``. Fixed group
    trees preserve fractional differences without floating-point atomic updates
    or one host wait per level. Partial tables contain at most
    ``batch * max(width, ng)`` entries each, avoiding a dense Ng-by-voxel mask.
    """
    batch, width = bins.shape
    if batch == 0 or width == 0:
        return mx.zeros((batch, ng), dtype=mx.int64), mx.zeros((batch, ng), dtype=mx.float32)
    chunks = max(1, min((width + 2047) // 2048, width // ng))
    threads = min(256, 1 << (width - 1).bit_length())
    counts, sums = _ngtdm_reduction_kernel(width, ng, chunks, threads)(
        inputs=[bins, eligible, difference], grid=(ng * chunks * threads, batch, 1),
        threadgroup=(threads, 1, 1), output_shapes=[(batch, ng, chunks)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )
    return mx.sum(counts.astype(mx.int64), axis=2), mx.sum(sums, axis=2)


@lru_cache(maxsize=64)
def _roi_neighborhood_totals_kernel(width: int) -> MetalKernel:
    """Compile the NGTDM per-level count and difference-sum kernel.

    It matches MLX's ``row_reduce_looped`` float32 sum: four adjacent inputs
    per lane, increasing blocks, ``simd_sum``, then a ``simd_sum`` of SIMD totals.
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_roi_ngtdm_totals",
        input_names=["bins", "eligible", "difference"], output_names=["n", "s"],
        source=f"""
            uint lane = thread_position_in_threadgroup.x;
            uint level = threadgroup_position_in_grid.x + 1;
            uint threads = threads_per_threadgroup.x;
            int count = 0;
            float total = 0.0f;
            for (uint block = 0; block < {width} / (threads * 4); ++block) {{
                uint base = block * threads * 4 + lane * 4;
                for (uint i = 0; i < 4; ++i) {{
                    uint index = base + i;
                    bool selected = eligible[index] && bins[index] == int(level);
                    count += int(selected);
                    total = (selected ? difference[index] : 0.0f) + total;
                }}
            }}
            uint tail = ({width} / (threads * 4)) * threads * 4 + lane * 4;
            for (uint i = 0; i < 4 && tail + i < {width}; ++i) {{
                uint index = tail + i;
                bool selected = eligible[index] && bins[index] == int(level);
                count += int(selected);
                total = (selected ? difference[index] : 0.0f) + total;
            }}
            total = simd_sum(0.0f + total);
            count = simd_sum(count);
            threadgroup float partial_s[32];
            threadgroup int partial_n[32];
            uint simd = lane / 32, within = lane % 32;
            if (threads > 32) {{
                if (within == 0) {{ partial_s[simd] = total; partial_n[simd] = count; }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                total = simd_sum(lane < threads / 32 ? partial_s[lane] : 0.0f);
                count = simd_sum(lane < threads / 32 ? partial_n[lane] : 0);
            }}
            if (lane == 0) {{ n[level - 1] = count; s[level - 1] = total; }}
        """,
    ))


def neighborhood_totals(
    bins: mx.array, eligible: mx.array, difference: mx.array, ng: int,
    *, queued: bool = False,
) -> tuple[mx.array, mx.array]:
    """Use the native ROI addition order or fixed tiled reductions for other rows.

    The queued single-row ROI path reproduces MLX's native row-reduction tree.
    Smaller or batched rows use bounded partial tables without per-level host
    waits. Neither path uses floating-point atomic additions.
    """
    if (queued and bins.shape[0] == 1 and bins.shape[1] > 64
            and difference.dtype == mx.float32):
        width = bins.shape[1]
        threads = (32 if width <= 512 else 128 if width <= 1024
                   else min(1024, ((width + 3) // 4 + 31) // 32 * 32))
        n, s = _roi_neighborhood_totals_kernel(width)(
            inputs=[bins, eligible, difference], grid=(ng * threads, 1, 1),
            threadgroup=(threads, 1, 1), output_shapes=[(1, ng), (1, ng)],
            output_dtypes=[mx.int32, mx.float32],
        )
        return n, s
    counts, sums = ngtdm_level_reductions(bins, eligible, difference, ng)
    return counts.astype(mx.int32), sums


def neighborhood_statistics(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...], ng: int, settings: Settings,
    *, queue_levels: bool = False,
) -> dict[str, mx.array]:
    """Compute GLDM and NGTDM features from each voxel's neighborhood.

    Neighbors are the voxels at every offset in ``directions(shape, settings)`` and its
    negation, so ``settings.distances`` applies (Chebyshev distance, in voxels). Only masked
    neighbors count. GLDM dependence is 1 plus the number of neighbors within
    ``settings.gldm_alpha`` gray levels of the center, so sizes run from 1 to
    ``len(offsets) + 1``; every masked center contributes, including an isolated center
    with dependence 1. NGTDM compares each eligible center, one with at least one masked
    neighbor, with the mean gray level of its masked neighbors. Only eligible centers
    contribute to NGTDM occurrence counts and differences; isolated centers are excluded.

    Returns:
        ``gldm_*`` and/or ``ngtdm_*`` arrays ``(batch,)``, depending on ``settings.features``.
    """
    offsets = directions(shape, settings)
    offsets = offsets + tuple(tuple(-v for v in d) for d in offsets)
    families = tuple(family for family in ("gldm", "ngtdm")
                     if any(name.startswith(family + "_") for name in settings.features))
    arrays = neighborhood_arrays(bins, mask, shape, offsets, settings.gldm_alpha, families)
    output: dict[str, mx.array] = {}
    if any(n.startswith("gldm_") for n in settings.features):
        counts = count_matrix(bins, arrays["dependences"], mask, ng, len(offsets) + 1)
        selected = tuple(name.removeprefix("gldm_") for name in settings.features
                         if name.startswith("gldm_"))
        output.update({f"gldm_{k}": v for k, v in size_statistics(
            counts, "gldm", selected,
        ).items()})
    if any(n.startswith("ngtdm_") for n in settings.features):
        occurrences, differences = neighborhood_totals(
            bins, arrays["eligible"], arrays["difference"], ng, queued=queue_levels,
        )
        selected = tuple(name.removeprefix("ngtdm_") for name in settings.features
                         if name.startswith("ngtdm_"))
        output.update({f"ngtdm_{k}": v for k, v in ngtdm_statistics(
            occurrences, differences, selected,
        ).items()})
    return output


@lru_cache(maxsize=128)
def run_kernel(shape: tuple[int, ...], offset: tuple[int, ...]) -> MetalKernel:
    """Compile the per-voxel run-length kernel for one direction.

    For each voxel along ``offset``, ``lengths`` holds the length of the
    same-gray-level run that starts there and ``starts`` flags run starts. A run of
    length one is valid.
    """
    before = neighbor_code(shape, tuple(-d for d in offset), "before")
    after = neighbor_code(shape, offset, "after")
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_runs", input_names=["bins", "mask"],
        output_names=["lengths", "starts"], source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= bins_shape[0] * bins_shape[1]) return;
        uint position = t % {prod(shape)}, row = t - position;
        {before}
        bool start = mask[t] && (before < 0 || !mask[row + before] ||
                                bins[row + before] != bins[t]);
        int length = 1;
        if (start) {{
            while (true) {{ {after}
                if (after < 0 || !mask[row + after] || bins[row + after] != bins[t]) break;
                position = after; length++;
            }}
        }}
        lengths[t] = length; starts[t] = start;
        """,
    ))


def run_statistics(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...], ng: int, settings: Settings,
    *, batch_directions: bool = False,
) -> dict[str, mx.array]:
    """Compute selected GLRLM features, averaged over directions.

    A run is a maximal sequence of consecutive masked voxels with one gray level along a
    direction; a length-one run is a valid run. Only distance-1 directions are used (at most
    13 in 3-D, 4 in 2-D or with ``force2d_axis``), independent of ``settings.distances``.
    Length-one axes still define singleton runs along the corresponding directions.
    A direction contributes to a
    row whenever at least one run starts in it, so an isolated masked voxel is not
    discarded. Rows with no run in any direction have undefined features and return NaN.

    Returns:
        Selected ``glrlm_*`` arrays ``(batch,)``.
    """
    selected = {
        name.removeprefix("glrlm_") for name in settings.features if name.startswith("glrlm_")
    }
    # Pair-based direction pruning would remove valid singleton-only directions when an
    # image axis has length one. Enumerate the dimensional directions, then let the run
    # kernel apply the actual image bounds. Explicit force-2D restrictions still apply.
    offsets = directions(tuple(max(size, 2) for size in shape), replace(settings, distances=(1,)))
    if batch_directions and bins.shape[0] == 1:
        return _roi_run_statistics(bins, mask, shape, ng, settings, offsets, selected)
    totals: dict[str, mx.array] = {}
    active_count = mx.zeros((bins.shape[0],), dtype=mx.int32)
    for offset in offsets:
        lengths, starts = run_kernel(shape, offset)(
            inputs=[bins, mask], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[bins.shape] * 2, output_dtypes=[mx.int32, mx.bool_])
        counts = count_matrix(bins, lengths, starts, ng, max(shape))
        active = mx.any(starts, axis=1)
        active_count = active_count + active.astype(mx.int32)
        # Selected names skip constructing unrequested reductions; only requested
        # outputs cross this per-direction evaluation boundary.
        values = size_statistics(counts, "glrlm", selected)
        totals = {k: totals.get(k, mx.zeros_like(v)) + mx.where(active, v, 0)
                  for k, v in values.items()}
        evaluate(active_count, *totals.values())
    return {f"glrlm_{k}": mx.where(active_count > 0, v / active_count, float("nan"))
            for k, v in totals.items()}


@lru_cache(maxsize=64)
def _roi_run_counts_kernel(
    shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], ng: int,
) -> MetalKernel:
    """Compile the kernel that accumulates GLRLM counts for all ``offsets``.

    Each run start walks its run and atomically increments
    ``counts[direction, gray - 1, length - 1]``.
    """
    updates: list[str] = []
    for direction, offset in enumerate(offsets):
        before = neighbor_code(shape, tuple(-d for d in offset), "before")
        after = neighbor_code(shape, offset, "after")
        updates.append(f"""
            {{
                position = t;
                {before}
                if (before < 0 || !mask[before] || bins[before] != gray) {{
                    int length = 1;
                    while (true) {{
                        {after}
                        if (after < 0 || !mask[after] || bins[after] != gray) break;
                        position = after;
                        ++length;
                    }}
                    uint index = ({direction}u * {ng}u + uint(gray - 1))
                                 * {max(shape)}u + uint(length - 1);
                    atomic_fetch_add_explicit(&counts[index], 1u, memory_order_relaxed);
                }}
            }}
        """)
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_roi_run_counts", input_names=["bins", "mask"],
        output_names=["counts"], atomic_outputs=True,
        source=f"""
            uint t = thread_position_in_grid.x;
            if (t >= {prod(shape)}u || !mask[t]) return;
            int gray = bins[t];
            uint position;
            {''.join(updates)}
        """,
    ))


def _roi_run_statistics(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...], ng: int, settings: Settings,
    offsets: tuple[tuple[int, ...], ...], selected: set[str],
) -> dict[str, mx.array]:
    """Evaluate directional matrices together and average them in direction order."""
    columns = max(shape)
    group = max(1, min(4, len(offsets), int(settings.scratch_mib * 1024**2)
                       // max(1, ng * columns * 4)))
    count_bytes = len(offsets) * ng * columns * 4
    direct_counts: mx.array | None = None
    if count_bytes <= min(64 * 1024**2, settings.scratch_mib * 1024**2 // 2):
        direct_counts = _roi_run_counts_kernel(shape, offsets, ng)(
            inputs=[bins, mask], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(len(offsets), ng, columns)], output_dtypes=[mx.uint32],
            init_value=0,
        )[0]
        # The directional matrices fit together, so their statistics are batched too,
        # reserving space for the int64/float32 intermediates. The final directional
        # reduction below is ordered by direction.
        group = max(1, min(len(offsets), (settings.scratch_mib * 1024**2 - count_bytes)
                           // max(1, ng * columns * 256)))
    blocks: list[mx.array] = []
    all_active: list[mx.array] = []
    names: tuple[str, ...] = ()
    for start in range(0, len(offsets), group):
        matrices: list[mx.array] = []
        active_rows: list[mx.array] = []
        if direct_counts is not None:
            block_counts = direct_counts[start:start + group]
            active = mx.any(block_counts > 0, axis=(1, 2))
            active_rows.extend(active[index:index + 1] for index in range(block_counts.shape[0]))
            matrices.append(block_counts)
        else:
            for offset in offsets[start:start + group]:
                lengths, starts = run_kernel(shape, offset)(
                    inputs=[bins, mask], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[bins.shape] * 2, output_dtypes=[mx.int32, mx.bool_],
                )
                counts = count_matrix(bins, lengths, starts, ng, columns)
                active = mx.any(starts, axis=1)
                evaluate(counts, active)
                matrices.append(counts)
                active_rows.append(active)
                del lengths, starts
        trim_cache()
        values = size_statistics(mx.concatenate(matrices, axis=0), "glrlm", selected)
        names = tuple(values)
        block = mx.stack(list(values.values()))
        evaluate(block)
        blocks.append(block)
        all_active.extend(active_rows)
    means = directional_mean(
        mx.concatenate(blocks, axis=1), mx.concatenate(all_active), compensated=False,
    )
    return {f"glrlm_{name}": means[index:index + 1] for index, name in enumerate(names)}


_ZONE_HEADER = r"""
// Union-find over one row's flat positions. ``parent[row + x]`` is ``root`` (the row
// width) while x is a set root, otherwise a smaller position of the same zone. Pointers
// only decrease (atomic minimum), so every chain ends at the set's smallest position.
inline int zone_root(device atomic_int* parent, uint row, int x, int root) {
    int p = atomic_load_explicit(&parent[row + x], memory_order_relaxed);
    while (p != root) {
        int g = atomic_load_explicit(&parent[row + p], memory_order_relaxed);
        if (g == root) return p;
        // Path halving: g is a smaller ancestor of x in the same set.
        atomic_fetch_min_explicit(&parent[row + x], g, memory_order_relaxed);
        x = g;
        p = atomic_load_explicit(&parent[row + x], memory_order_relaxed);
    }
    return x;
}

// Link the larger root under the smaller. If another thread linked b first, ``old`` is
// b's former parent; b now points at min(old, a), so a must still be joined with old.
inline void zone_unite(device atomic_int* parent, uint row, int a, int b, int root) {
    while (true) {
        a = zone_root(parent, row, a, root);
        b = zone_root(parent, row, b, root);
        if (a == b) return;
        if (a > b) { int s = a; a = b; b = s; }
        int old = atomic_fetch_min_explicit(&parent[row + b], a, memory_order_relaxed);
        if (old == root) return;
        b = old;
    }
}
"""


@lru_cache(maxsize=64)
def _zone_merge_kernel(
    shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...],
) -> MetalKernel:
    """Compile the union-find kernel that links same-level masked neighbors into zones."""
    width = prod(shape)
    neighbors = "\n".join(
        f"{{ {neighbor_code(shape, offset, 'q')} if (q >= 0 && mask[row + q] && "
        f"bins[row + q] == level) zone_unite(parent, row, int(position), q, {width}); }}"
        for offset in offsets)
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_zone_merge", input_names=["bins", "mask"],
        output_names=["parent"], header=_ZONE_HEADER, atomic_outputs=True, source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= bins_shape[0] * bins_shape[1]) return;
        uint position = t % {width}, row = t - position;
        if (!mask[t]) return;
        int level = bins[t];
        {neighbors}
        """,
    ))


@lru_cache(maxsize=64)
def _zone_root_kernel(width: int) -> MetalKernel:
    """Compile the kernel that labels each voxel with its zone's root position."""
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_zone_roots", input_names=["parent"], output_names=["labels"],
        source=f"""
        uint t = thread_position_in_grid.x;
        if (t >= parent_shape[0] * parent_shape[1]) return;
        uint position = t % {width}, row = t - position;
        int x = int(position);
        for (int p = parent[t]; p != {width}; p = parent[row + x]) x = p;
        labels[t] = x;
        """,
    ))


def zone_statistics(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...], ng: int, settings: Settings,
    *, fixed_columns: bool = False,
) -> dict[str, mx.array]:
    """Compute GLSZM features.

    A zone is a connected set of masked voxels with the same gray level, using the full
    unit-distance neighborhood (26-connected in 3-D, 8-connected in 2-D or in-plane with
    ``force2d_axis``); ``settings.distances`` is ignored, as in PyRadiomics. Each zone is
    labelled with its smallest flat voxel index (:func:`zone_labels`).

    With ``fixed_columns`` the size axis spans every possible zone size, ``prod(shape)``,
    instead of the largest zone in the batch. Voxel batches use it so the fixed pairwise
    reduction tree, and hence each center's float32 result, does not depend on which
    centers share its batch.

    Returns:
        ``glszm_*`` arrays ``(batch,)``.
    """
    width = prod(shape)
    ids = mx.broadcast_to(mx.arange(width, dtype=mx.int32)[None, :], bins.shape)
    labels = zone_labels(bins, mask, shape, directions(shape, replace(settings, distances=(1,))))
    rows = mx.arange(bins.shape[0], dtype=mx.int32)[:, None]
    indexes = mx.reshape(rows * width + labels, (-1,))
    storage = mx.zeros((bins.size,), dtype=mx.int32)
    sizes = cast(IndexedAddition, storage.at[indexes]).add(mx.reshape(mask.astype(mx.int32), (-1,)))
    sizes = mx.reshape(sizes, bins.shape)
    maximum_size = width if fixed_columns else max(1, int(cast(int, mx.max(sizes).item())))
    selected = tuple(name.removeprefix("glszm_") for name in settings.features
                     if name.startswith("glszm_"))
    starts = mask & (ids == labels)
    if not fixed_columns and use_sparse_size(ng, maximum_size):
        keys, counts = coordinate_counts(bins, sizes, starts, maximum_size)
        return {f"glszm_{k}": value for k, value in sparse_size_statistics(
            keys, counts, maximum_size, "glszm", selected,
        ).items()}
    counts = count_matrix(bins, sizes, starts, ng, maximum_size)
    return {f"glszm_{k}": v for k, v in size_statistics(counts, "glszm", selected).items()}


def zone_labels(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...],
    offsets: tuple[tuple[int, ...], ...],
) -> mx.array:
    """Label each masked voxel with the smallest flat row position of its zone.

    ``offsets`` holds one offset of each opposite pair (``preprocessing.directions``);
    union is symmetric, so each voxel joins only those neighbors with the same gray
    level. One union-find pass links roots by atomic minimum and a second resolves every
    position to its root. A set's root is never linked, and linking only points a root at
    a smaller one, so each zone's root is its smallest position. Unmasked voxels keep their own
    position. Rows (ROIs or voxel neighborhoods) are labelled independently.
    """
    width = prod(shape)
    parent = _zone_merge_kernel(shape, offsets)(
        inputs=[bins, mask], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[bins.shape], output_dtypes=[mx.int32], init_value=width,
    )[0]
    return _zone_root_kernel(width)(
        inputs=[parent], grid=(bins.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[bins.shape], output_dtypes=[mx.int32],
    )[0]


def texture_statistics(
    bins: mx.array, mask: mx.array, shape: tuple[int, ...], ng: int, settings: Settings,
    *, fixed_columns: bool = False,
) -> dict[str, mx.array]:
    """Compute texture features for the GLDM, NGTDM, GLRLM or GLSZM families requested.

    ``fixed_columns`` is forwarded to :func:`zone_statistics` for voxel batches.
    """
    output: dict[str, mx.array] = {}
    if any(n.startswith(("gldm_", "ngtdm_")) for n in settings.features):
        output.update(neighborhood_statistics(
            bins, mask, shape, ng, settings, queue_levels=not fixed_columns,
        ))
    if any(n.startswith("glrlm_") for n in settings.features):
        output.update(run_statistics(
            bins, mask, shape, ng, settings, batch_directions=not fixed_columns,
        ))
    if any(n.startswith("glszm_") for n in settings.features):
        output.update(zone_statistics(bins, mask, shape, ng, settings,
                                      fixed_columns=fixed_columns))
    return output
