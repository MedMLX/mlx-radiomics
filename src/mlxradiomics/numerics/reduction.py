"""Fixed adjacent-input addition trees in bounded Metal threadgroup memory."""

from functools import lru_cache
from typing import cast

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx


@lru_cache(maxsize=64)
def _pairwise_kernel(width: int, block: int, dtype: mx.Dtype) -> MetalKernel:
    """Compile the fixed adjacent-pair tree that sums ``block``-wide chunks of each row."""
    if dtype not in (mx.float32, mx.int32):
        raise ValueError("pairwise reduction requires float32 or int32")
    scalar = "float" if dtype == mx.float32 else "int"
    chunks = (width + block - 1) // block
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_pairwise_sum", input_names=["values"], output_names=["sums"],
        source=f"""
            uint lane = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.x;
            uint row = group / {chunks};
            uint chunk = group % {chunks};
            threadgroup {scalar} work[{block}];
            for (uint i = lane; i < {block}; i += 256) {{
                ulong index = ulong(chunk) * {block} + i;
                work[i] = index < {width} ? values[ulong(row) * {width} + index] : 0;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint step = 1; step < {block}; step *= 2) {{
                for (uint i = lane; i < {block} / (2 * step); i += 256) {{
                    uint left = i * 2 * step;
                    work[left] = work[left] + work[left + step];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (lane == 0) sums[group] = work[0];
        """,
    ))


def sum_rows(values: mx.array) -> mx.array:
    """Sum each row with a fixed pairwise (adjacent-pair) addition tree.

    The tree for a row depends only on its width, not on the batch size, so one center gets
    the same float32 sum however voxels are batched. Odd tails are zero-padded.

    Args:
        values: Float32 or int32 array ``(batch, ...)``; trailing axes are flattened.

    Returns:
        Array ``(batch,)`` of the same dtype.

    Raises:
        ValueError: If a row has more than one element and the dtype is not float32 or int32.
            Singleton rows are returned without invoking the dtype-checked reduction kernel.
    """
    values = mx.reshape(values, (values.shape[0], -1))
    while values.shape[1] > 1:
        width = values.shape[1]
        # Power-of-two contiguous blocks preserve the original adjacent-pair tree.
        # Zero padding replaces odd tails; threadgroup storage never exceeds 4 KiB.
        block = min(1024, 1 << (width - 1).bit_length())
        chunks = (width + block - 1) // block
        values = _pairwise_kernel(width, block, values.dtype)(
            inputs=[values], grid=(values.shape[0] * chunks * 256, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[(values.shape[0], chunks)],
            output_dtypes=[values.dtype],
        )[0]
    return values[:, 0]


@lru_cache(maxsize=32)
def _compensated_tree_kernel(block: int, masked_pairs: bool) -> MetalKernel:
    """Compile the two-sum tree that reduces one ``(row, block)`` chunk per threadgroup.

    Widths are read from input shapes at run time, so one compiled kernel serves
    every row width; only the power-of-two block and the leaf form select a variant.
    """
    levels = block.bit_length() - 1
    if masked_pairs:
        # Leaves are where(valid, high / scale, 0) followed by where(valid, low / scale, 0).
        inputs = ["high", "low", "valid", "scale"]
        width = "2 * ulong(high_shape[1])"
        load = """
                float leaf = 0.0f;
                if (index < width) {
                    ulong half_width = width / 2;
                    ulong column = index < half_width ? index : index - half_width;
                    ulong at = ulong(row) * half_width + column;
                    float part = index < half_width ? high[at] : low[at];
                    leaf = valid[at] ? part / scale[row] : 0.0f;
                }
                work[i] = leaf;"""
    else:
        inputs = ["values"]
        width = "ulong(values_shape[1])"
        load = "work[i] = index < width ? values[ulong(row) * width + index] : 0.0f;"
    writes = "".join(
        f"""
                    if (level == {level}) {{
                        // Level inputs have ceil(width / 2**(level - 1)) entries; only
                        // their floor(entries / 2) real pairs are stored.
                        ulong pairs = ((width + {(1 << (level - 1)) - 1}) >> {level - 1}) / 2;
                        ulong pair = ulong(chunk) * {block >> level} + i;
                        if (pair < pairs) error{level - 1}[ulong(row) * pairs + pair] = error;
                    }}"""
        for level in range(1, levels + 1)
    )
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_compensated_tree",
        input_names=inputs,
        output_names=["totals", *(f"error{level}" for level in range(levels))],
        source=f"""
            uint lane = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.x;
            ulong width = {width};
            uint chunks = uint((width + {block - 1}) / {block});
            uint row = group / chunks;
            uint chunk = group % chunks;
            threadgroup float work[{block}];
            for (uint i = lane; i < {block}; i += 256) {{
                ulong index = ulong(chunk) * {block} + i;
                {load}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint level = 0;
            for (uint step = 1; step < {block}; step *= 2) {{
                level += 1;
                for (uint i = lane; i < {block} / (2 * step); i += 256) {{
                    uint left_index = i * 2 * step;
                    float left = work[left_index];
                    float right = work[left_index + step];
                    float total = left + right;
                    float back = total - left;
                    float error = (left - (total - back)) + (right - back);
                    work[left_index] = total;{writes}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (lane == 0) totals[group] = work[0];
        """,
    ))


_ERROR_CHAIN = 16


@lru_cache(maxsize=8)
def _error_chain_kernel(count: int, carry: bool, final: bool) -> MetalKernel:
    """Compile the sequential fold of per-level two-sum errors.

    It adds ``((carry + s0) + s1) + ...`` and, on the final link, ``total + error``:
    the same float32 additions, in the same order, as an initial zero plus each
    level sum.
    """
    inputs = [*(["carry"] if carry else []), *(f"s{i}" for i in range(count)),
              *(["total"] if final else [])]
    additions = "".join(f"error = error + s{i}[row];\n" for i in range(count))
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_compensated_error_chain",
        input_names=inputs, output_names=["out"],
        source=f"""
            uint row = thread_position_in_grid.x;
            if (row >= s0_shape[0]) return;
            float error = {"carry[row]" if carry else "0.0f"};
            {additions}
            out[row] = {"total[row] + error" if final else "error"};
        """,
    ))


def _compensated_tree(
    inputs: list[mx.array], rows: int, width: int, masked_pairs: bool,
) -> mx.array:
    """Evaluate the two-sum tree of :func:`compensated_sum_rows` over ``width`` leaves.

    Power-of-two blocks zero-pad odd tails. A padded pair returns its real operand
    unchanged with an exactly zero error, so the partial sums equal the carried-tail tree
    up to the sign of a zero total, which the final ``total + error`` with a
    positive-zero error removes. Each level's errors of real pairs are written to their
    own ``(rows, floor(entries / 2))`` array and reduced by the same ``mx.sum`` as the
    level-by-level tree, then accumulated level by level from zero.
    """
    level_sums: list[mx.array] = []
    current = inputs
    while width > 1:
        block = min(1024, 1 << (width - 1).bit_length())
        levels = block.bit_length() - 1
        chunks = (width + block - 1) // block
        shapes = [(rows, ((width + (1 << level) - 1) >> level) // 2) for level in range(levels)]
        outputs = _compensated_tree_kernel(block, masked_pairs)(
            inputs=current, grid=(rows * chunks * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, chunks), *shapes],
            output_dtypes=[mx.float32] * (1 + levels),
        )
        level_sums.extend(mx.sum(error, axis=1) for error in outputs[1:])
        current = [outputs[0]]
        masked_pairs = False
        width = chunks
    total = current[0][:, 0]
    if not level_sums:
        return total + mx.zeros((rows,), dtype=mx.float32)
    error: mx.array | None = None
    for start in range(0, len(level_sums), _ERROR_CHAIN):
        link = level_sums[start : start + _ERROR_CHAIN]
        final = start + _ERROR_CHAIN >= len(level_sums)
        error = _error_chain_kernel(len(link), error is not None, final)(
            inputs=[*([error] if error is not None else []), *link, *([total] if final else [])],
            grid=(rows, 1, 1), threadgroup=(min(256, rows), 1, 1),
            output_shapes=[(rows,)], output_dtypes=[mx.float32],
        )[0]
    assert error is not None
    return error


def compensated_sum_rows(values: mx.array) -> mx.array:
    """Sum rows with a two-sum pairwise tree, folding each addition's rounding error back.

    :func:`mlxradiomics.numerics.reduction.sum_rows` rounds every partial sum, so a cancelling
    sequence such as ``[1e8, 1, -1e8]`` loses the small exact total: the plain tree sums
    ``1e8 + 1`` to ``1e8`` and subtracts ``1e8`` to zero. The two-sum identity records the
    exact rounding error of every addition; reducing those errors alongside the partial
    sums recovers the exact total to first order. Operands must already be scaled so no
    partial sum can overflow. Reduction remains float32 rather than exactly rounded
    float64, matching the documented first-order precision contract.

    The tree adds adjacent pairs and carries an odd tail to the next level. Each level's
    error vector is reduced with ``mx.sum`` and the level sums are added in tree order to
    an initial zero, so results match evaluating every level as separate MLX operations.
    One Metal pass reduces up to ten levels per 1024-wide block in threadgroup memory.

    Raises:
        ValueError: If ``values`` is not two-dimensional float32.
    """
    if values.ndim != 2 or values.dtype != mx.float32:
        raise ValueError("compensated reduction requires a two-dimensional float32 array")
    rows, width = values.shape
    if rows == 0:
        return mx.zeros((0,), dtype=mx.float32)
    return _compensated_tree([values], rows, width, masked_pairs=False)


def compensated_masked_pair_sum_rows(
    high: mx.array, low: mx.array, valid: mx.array, scale: mx.array,
) -> mx.array:
    """Compensated row sum of a float-float pair's masked, scaled high and low parts.

    Equals :func:`compensated_sum_rows` applied to
    ``concatenate([where(valid, high / scale[:, None], 0),
    where(valid, low / scale[:, None], 0)], axis=1)``: high parts precede low parts in
    one tree, so cancelling residuals cannot round a surviving small residual away in a
    separate reduction. The mask, division and concatenation are formed while loading
    the tree's leaves instead of as full-width intermediate arrays.

    Args:
        high: Float32 ``(rows, n)`` high parts.
        low: Float32 ``(rows, n)`` low parts.
        valid: Boolean ``(rows, n)``; excluded entries contribute positive zero.
        scale: Float32 ``(rows,)`` divisor applied to both parts of each row.

    Raises:
        ValueError: If the shapes or dtypes differ from those above.
    """
    if (
        high.ndim != 2 or high.shape != low.shape or high.shape != valid.shape
        or scale.shape != (high.shape[0],) or high.dtype != mx.float32
        or low.dtype != mx.float32 or scale.dtype != mx.float32 or valid.dtype != mx.bool_
    ):
        raise ValueError("masked pair reduction requires matching float32 rows and a bool mask")
    rows, columns = high.shape
    if rows == 0:
        return mx.zeros((0,), dtype=mx.float32)
    return _compensated_tree([high, low, valid, scale], rows, 2 * columns, masked_pairs=True)


class CompensatedTotal:
    """Running float32 sum of row vectors that keeps each addition's rounding error.

    Averages over slices or directions add one feature vector at a time. A plain running
    sum is order-dependent under cancellation: ``-11718750 + -105/256`` rounds the small
    term away, and a later ``+11718750`` then leaves a false zero. Each addition here
    records its exact two-sum error, and :meth:`value` folds the accumulated errors back
    in, so the result is first-order independent of the addition order. Memory stays two
    vectors however many matrices are added. A non-finite total is returned unchanged:
    its two-sum error is NaN and carries no information.
    """

    def __init__(self, rows: int, dtype: mx.Dtype = mx.float32) -> None:
        """Start zero totals and errors for ``rows`` features."""
        self.total = mx.zeros((rows,), dtype=dtype)
        self.error = mx.zeros((rows,), dtype=dtype)

    def add(self, value: mx.array) -> None:
        """Add ``value`` and accumulate the addition's exact two-sum error."""
        total = self.total + value
        back = total - self.total
        self.error = self.error + ((self.total - (total - back)) + (value - back))
        self.total = total

    def arrays(self) -> tuple[mx.array, mx.array]:
        """Return the running arrays, for evaluating the lazy graph between additions."""
        return self.total, self.error

    def value(self) -> mx.array:
        """Return the total with its accumulated error folded in."""
        return mx.where(mx.isfinite(self.total), self.total + self.error, self.total)


@lru_cache(maxsize=2)
def _directional_mean_kernel(compensated: bool) -> MetalKernel:
    """Compile the one-thread-per-feature mean over active directions."""
    correction = """
        float next = total + value;
        float back = next - total;
        error = error + ((total - (next - back)) + (value - back));
        total = next;
    """ if compensated else "total = total + value;"
    result = "(isfinite(total) ? total + error : total)" if compensated else "total"
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_directional_mean",
        input_names=["values", "active"], output_names=["means"],
        source=f"""
            uint feature = thread_position_in_grid.x;
            if (feature >= values_shape[0]) return;
            uint directions = values_shape[1];
            int count = 0;
            float total = 0.0f, error = 0.0f;
            for (uint direction = 0; direction < directions; ++direction) {{
                bool present = active[direction];
                count += int(present);
                float value = present ? values[feature * directions + direction] : 0.0f;
                {correction}
            }}
            means[feature] = count > 0 ? {result} / float(count) : NAN;
        """,
    ))


def directional_mean(
    values: mx.array, active: mx.array, *, compensated: bool,
) -> mx.array:
    """Average ROI directions in their original order, one thread per feature.

    ``values`` is contiguous float32 ``(features, directions)`` and ``active`` is
    a boolean vector. The compensated path retains CompensatedTotal's two-sum
    operations and final nonfinite handling. The ordinary path retains sequential
    additions, including the initial zero. Only the host's scalar graph is removed.
    """
    return _directional_mean_kernel(compensated)(
        inputs=[values, active], grid=(values.shape[0], 1, 1),
        threadgroup=(32, 1, 1), output_shapes=[(values.shape[0],)],
        output_dtypes=[mx.float32],
    )[0]
