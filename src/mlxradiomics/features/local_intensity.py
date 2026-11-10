"""Physical 1 cm³ spherical local/global intensity peaks (IBSI VJGA/0F91).

Sphere means are needed only at candidate centers: every ROI voxel for the global peak
and the maximum-intensity ROI voxels for the local peak. :func:`_sphere_means` evaluates
them with one Metal kernel, one thread per center, and the exact in-image voxel count.
Bounded integral images use exact sums and row prefixes for the global peak; other
images use compensated float32 values with source residuals when present. Local-only
requests upload the maximum candidates' support. Inputs whose magnitudes could overflow
the compensated sums or flush their residuals, and non-finite inputs, use the tiled
convolution path (:func:`_convolution_peaks`) instead.
"""

from collections.abc import Collection
from functools import lru_cache
from itertools import product
from math import floor, pi, prod
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import array_from_bool, evaluate, mx

# Sphere sums stay below 2**126 when every |value| times the sphere's voxel count does,
# and a nonzero float64 value at or above 2**-60 keeps its nonzero float32 residual at
# or above 2**-113, inside the normal range, so Metal cannot flush it.
_MAX_SUM_EXPONENT = 126
_MIN_EXPONENT = -60
_THREADS = 256
type _SphereSum = Literal["compensated", "float32", "integer", "prefix"]


def _sphere(
    shape: tuple[int, ...], spacing: tuple[float, ...],
) -> tuple[tuple[int, ...], NDArray[np.bool_]]:
    """Kernel half-widths and the physical 1 cm³ sphere's voxel-center membership."""
    radius = (3000 / (4 * pi)) ** (1 / 3)
    half = tuple(
        min(size - 1, floor(radius / step)) for size, step in zip(shape, spacing, strict=True)
    )
    coordinates = np.ogrid[tuple(slice(-value, value + 1) for value in half)]
    squared = sum(
        (coordinate * step) ** 2 for coordinate, step in zip(coordinates, spacing, strict=True)
    )
    membership: NDArray[np.bool_] = np.asarray(squared <= radius**2, dtype=np.bool_)
    return half, membership


def _sphere_rows(half: tuple[int, ...], sphere: NDArray[np.bool_]) -> NDArray[np.int32]:
    """``(dz, dy, w)`` per sphere row: the row holds offsets ``dx`` in ``[-w, w]``."""
    rows: list[tuple[int, int, int]] = []
    for dz in range(sphere.shape[0]):
        for dy in range(sphere.shape[1]):
            inside = np.flatnonzero(sphere[dz, dy])
            if inside.size == 0:
                continue
            width = int(inside[-1]) - half[2]
            if inside.size != 2 * width + 1 or int(inside[0]) != half[2] - width:
                raise AssertionError("physical sphere rows must be symmetric and contiguous")
            rows.append((dz - half[0], dy - half[1], width))
    return np.asarray(rows, dtype=np.int32).reshape(-1)


@lru_cache(maxsize=4)
def _geometry(
    shape: tuple[int, ...], spacing: tuple[float, ...],
) -> tuple[tuple[int, ...], NDArray[np.int32], int, int]:
    """Reuse immutable physical support and row descriptions across extractions."""
    half, sphere = _sphere(shape, spacing)
    rows = _sphere_rows(half, sphere)
    rows.flags.writeable = False
    return half, rows, sphere.size, int(np.count_nonzero(sphere))


@lru_cache(maxsize=4)
def _sphere_kernel(mode: _SphereSum = "compensated") -> MetalKernel:
    """Mean over in-image sphere voxels with exact or compensated accumulation.

    Thread ``t`` owns center ``t``. Each sphere row is clipped once to the box, so
    the in-image count is exact. General values accumulate with error-free two-sum
    and source residuals in a separate register. Bounded integral sums need no carry;
    their prefixes also stay inside int32. Contraction and reassociation are disabled
    so compensation is retained. Every mode corrects the quotient with a remainder term.
    """
    # Bounded integral sums are exactly representable at every step, so their
    # compensated carry is identically zero. Prefix rows retain that exact sum
    # while replacing the innermost voxel loop with two integer reads.
    accumulate = (
        """
            int total = high[base + uint(last)];
            if (first > 0) total -= high[base + uint(first - 1)];
            sum += float(total);
        """
        if mode == "prefix" else
        """
            for (int xx = first; xx <= last; ++xx) {
                sum += high[base + uint(xx)];
            }
        """
        if mode == "integer" else
        """
            for (int xx = first; xx <= last; ++xx) {
                two_sum_add(sum, carry, high[base + uint(xx)]);
            }
        """
        if mode == "float32" else
        """
            for (int xx = first; xx <= last; ++xx) {
                two_sum_add(sum, carry, high[base + uint(xx)]);
                residual += low[base + uint(xx)];
            }
        """
    )
    return cast(MetalKernel, mx.fast.metal_kernel(
        name=f"mlxradiomics_sphere_means_{mode}",
        input_names=["high", *(["low"] if mode == "compensated" else []),
                     "centers", "rows", "params"],
        output_names=["means"],
        header="""
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            inline void two_sum_add(thread float &s, thread float &c, float x) {
                float t = s + x;
                float v = t - s;
                c += (s - (t - v)) + (x - v);
                s = t;
            }
        """,
        source="""
            uint t = thread_position_in_grid.x;
            uint count = params[0];
            if (t >= count) return;
            uint row_count = params[1];
            int depth = int(params[2]), height = int(params[3]), width = int(params[4]);
            int z = centers[3 * t], y = centers[3 * t + 1], x = centers[3 * t + 2];
            float sum = 0, carry = 0, residual = 0;
            uint used = 0;
            for (uint r = 0; r < row_count; ++r) {
                int zz = z + rows[3 * r], yy = y + rows[3 * r + 1], reach = rows[3 * r + 2];
                if (zz < 0 || zz >= depth || yy < 0 || yy >= height) continue;
                int first = max(x - reach, 0), last = min(x + reach, width - 1);
                uint base = (uint(zz) * uint(height) + uint(yy)) * uint(width);
                ACCUMULATE
                used += uint(last - first + 1);
            }
            float n = float(used);
            float quotient = metal::precise::divide(sum, n);
            float remainder = metal::fma(-quotient, n, sum);
            means[t] = quotient + metal::precise::divide(remainder + (carry + residual), n);
        """.replace("ACCUMULATE", accumulate),
    ))


def _sphere_means(
    high: mx.array, low: mx.array | None, shape: tuple[int, ...], centers: NDArray[np.int64],
    rows: NDArray[np.int32], mode: _SphereSum = "compensated",
) -> mx.array:
    """Float32 sphere means at ``centers`` (box coordinates) over a box of ``shape``."""
    count = int(centers.shape[0])
    params = mx.array([count, rows.size // 3, *shape], dtype=mx.uint32)
    threads = ((count + _THREADS - 1) // _THREADS) * _THREADS
    return _sphere_kernel(mode)(
        inputs=[high, *([low] if low is not None else []),
                mx.array(centers.astype(np.int32).reshape(-1), dtype=mx.int32),
                mx.array(rows, dtype=mx.int32), params],
        grid=(threads, 1, 1), threadgroup=(_THREADS, 1, 1), output_shapes=[(count,)],
        output_dtypes=[mx.float32],
    )[0]


def _kernel_admits(values: NDArray[np.float64], sphere_voxels: int) -> bool:
    """Whether float-float sphere sums over ``values`` are finite and unflushed."""
    if not np.all(np.isfinite(values)):
        return False
    magnitude = np.abs(values)
    largest = float(magnitude.max(initial=0.0))
    if largest * sphere_voxels >= 2.0**_MAX_SUM_EXPONENT:
        return False
    nonzero = magnitude[magnitude > 0]
    return nonzero.size == 0 or float(nonzero.min()) >= 2.0**_MIN_EXPONENT


def _integer_sums_exact(values: NDArray[np.float64], sphere_voxels: int) -> bool:
    """Every sphere partial sum and row prefix must be an exactly bounded integer.

    The float32 sum remains exact through 2**24, independent of signs/order. Row
    prefixes use signed int32, and the row-width bound prevents their overflow.
    """
    largest = max(abs(float(values.min())), abs(float(values.max())))
    if largest * sphere_voxels > 2**24 or largest * values.shape[-1] > 2**31 - 1:
        return False
    return bool(np.all(values == np.trunc(values)))


def local_intensity_statistics(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    scratch_mib: int,
    selected: Collection[str],
) -> dict[str, mx.array]:
    """Return ROI peaks using all available image voxels inside each physical sphere.

    Spacing is in millimetres. Image boundaries truncate the sphere and its denominator;
    ROI boundaries do not. Maximum-intensity candidate ties use the source float64 values,
    with signed zeros equal. Sphere means are evaluated only at candidate centers by
    :func:`_sphere_means` when :func:`_kernel_admits` the ROI box plus sphere reach;
    otherwise by :func:`_convolution_peaks`.
    """
    if image.ndim != 3:
        raise ValueError("local intensity features require a 3D image")
    half, rows, sphere_size, sphere_voxels = _geometry(image.shape, spacing)
    budget = scratch_mib * 1024**2
    if sphere_size * 4 > budget // 2:
        raise ValueError("scratch_mib cannot hold the physical local-intensity sphere")
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return _convolution_peaks(image, mask, spacing, scratch_mib, selected)
    lower = np.maximum(occupied.min(axis=0) - np.asarray(half), 0)
    upper = np.minimum(occupied.max(axis=0) + 1 + np.asarray(half), image.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    # Admission only inspects magnitudes and finiteness, so it needs no contiguous
    # copy of a support that a local-only selection may immediately narrow.
    values = image[box]
    if not _kernel_admits(values, sphere_voxels):
        return _convolution_peaks(image, mask, spacing, scratch_mib, selected)
    is_maximum: NDArray[np.bool_] | None = None
    if "LocalIntensityPeak" in selected:
        roi_values = image[mask]
        is_maximum = roi_values == np.max(roi_values)
        if "GlobalIntensityPeak" not in selected and not np.all(is_maximum):
            # Admission above deliberately uses the original ROI support: moving a
            # remote exceptional value must not change which arithmetic path runs.
            # Once admitted, only tied maximum centers can affect the local peak.
            occupied = occupied[is_maximum]
            lower = np.maximum(occupied.min(axis=0) - np.asarray(half), 0)
            upper = np.minimum(occupied.max(axis=0) + 1 + np.asarray(half), image.shape)
            box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
            values = image[box]
    values = np.ascontiguousarray(values, dtype=np.float64)
    mode: _SphereSum
    low: mx.array | None = None
    if _integer_sums_exact(values, sphere_voxels):
        if "GlobalIntensityPeak" in selected:
            mode = "prefix"
            high = mx.reshape(mx.cumsum(
                mx.array(values, dtype=mx.int32), axis=2, dtype=mx.int32,
            ), (-1,))
        else:
            mode = "integer"
            high = mx.array(values.reshape(-1), dtype=mx.float32)
    else:
        high32 = values.astype(np.float32)
        low32 = (values - high32.astype(np.float64)).astype(np.float32)
        high = mx.array(high32.reshape(-1), dtype=mx.float32)
        mode = "float32"
        if np.any(low32):
            mode = "compensated"
            low = mx.array(low32.reshape(-1), dtype=mx.float32)
    shape = tuple(int(size) for size in values.shape)
    relative = occupied - lower
    peaks: dict[str, mx.array] = {}
    if "GlobalIntensityPeak" in selected:
        means = _sphere_means(high, low, shape, relative, rows, mode)
        peaks["GlobalIntensityPeak"] = mx.max(means)
        if is_maximum is not None:
            peaks["LocalIntensityPeak"] = mx.max(
                mx.where(array_from_bool(is_maximum), means, float("-inf")),
            )
    elif "LocalIntensityPeak" in selected:
        peaks["LocalIntensityPeak"] = mx.max(
            _sphere_means(high, low, shape, relative, rows, mode),
        )
    evaluate(*peaks.values())
    return {f"localintensity_{name}": mx.reshape(peaks[name], (1,)) for name in selected}


def _convolution_peaks(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    scratch_mib: int,
    selected: Collection[str],
) -> dict[str, mx.array]:
    """Tiled-convolution peaks for inputs the sphere kernel does not admit.

    Convolution is performed with the installed MLX 3-D primitive in bounded tiles.
    Values are power-of-two scaled before the sum so common large intensities cannot
    overflow it; values that the scaling would flush to zero are summed unscaled, so an
    unrelated extreme elsewhere in the tile cannot change a sphere's representable mean.
    """
    half, sphere_mask = _sphere(image.shape, spacing)
    kernel_shape = tuple(2 * value + 1 for value in half)
    budget = scratch_mib * 1024**2
    sphere = np.asarray(sphere_mask, dtype=np.float32)
    weight = mx.array(sphere)[None, ..., None]
    sign = mx.array(1, dtype=mx.uint64) << 63

    def source_keys(samples: NDArray[np.float64]) -> mx.array:
        bits = mx.array(np.ascontiguousarray(samples).view(np.uint64))
        keys = mx.where((bits & sign) != 0, ~bits, bits ^ sign)
        # Signed source zeros denote the same intensity and candidate set.
        return mx.where((bits & ~sign) == 0, sign, keys)

    maximum_key = mx.array(0, dtype=mx.uint64)
    if "LocalIntensityPeak" in selected:
        # Stream the exact source maximum instead of uploading volume-sized key,
        # value and mask arrays outside the convolution scratch budget.
        flat_image, flat_mask = image.reshape(-1), mask.reshape(-1)
        chunk = max(1, budget // 64)
        for begin in range(0, image.size, chunk):
            end = min(image.size, begin + chunk)
            keys = source_keys(flat_image[begin:end])
            valid = array_from_bool(flat_mask[begin:end])
            maximum_key = mx.maximum(
                maximum_key,
                mx.max(mx.where(valid, keys, mx.array(0, dtype=mx.uint64))),
            )
            evaluate(maximum_key)
    tile = list(image.shape)
    while (
        prod(
            min(size, block + 2 * halo)
            for size, block, halo in zip(image.shape, tile, half, strict=True)
        )
        * 64
        > budget - prod(kernel_shape) * 4
    ):
        axis = max(range(3), key=lambda k: tile[k])
        if tile[axis] == 1:
            raise ValueError("scratch_mib cannot hold one local-intensity convolution tile")
        tile[axis] = (tile[axis] + 1) // 2
    peaks = {name: mx.array(float("-inf")) for name in selected}
    padding = (half[0], half[1], half[2])
    for start in product(
        *(range(0, size, block) for size, block in zip(image.shape, tile, strict=True))
    ):
        stop = tuple(
            min(size, offset + block)
            for size, offset, block in zip(image.shape, start, tile, strict=True)
        )
        source_start = tuple(
            max(0, offset - halo) for offset, halo in zip(start, half, strict=True)
        )
        source_stop = tuple(
            min(size, end + halo) for size, end, halo in zip(image.shape, stop, half, strict=True)
        )
        source = tuple(slice(a, b) for a, b in zip(source_start, source_stop, strict=True))
        interior = tuple(
            slice(a - lo, b - lo) for a, b, lo in zip(start, stop, source_start, strict=True)
        )
        target = tuple(slice(a, b) for a, b in zip(start, stop, strict=True))
        if not np.any(mask[target]):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            uploaded = image[source].astype(np.float32)
            source_residual = (image[source] - uploaded.astype(np.float64)).astype(np.float32)
        block = mx.array(uploaded)
        has_residual = bool(np.any(source_residual))
        all_finite = bool(np.all(np.isfinite(uploaded)))
        roi = array_from_bool(mask[target])
        local_centers = (
            roi & (source_keys(image[target]) == maximum_key)
            if "LocalIntensityPeak" in selected else roi
        )
        finite = mx.isfinite(block)
        amplitude = mx.maximum(mx.max(mx.where(finite, mx.abs(block), 0)), 1)
        # Power-of-two scaling avoids a new division-rounding error before sums
        # cancel. Only values below the scaled float32 range are flushed; recover their
        # sums unscaled below so an unrelated extreme elsewhere in the tile cannot erase
        # a representable local mean (their total is bounded by the tile count times
        # scale * 2**-126, which cannot overflow).
        scale = mx.power(2.0, mx.minimum(mx.floor(mx.log2(amplitude)), 127))
        scaled = mx.where(finite, block / scale, 0)
        numerator = mx.conv3d(
            scaled[None, ..., None], weight, padding=padding
        )[0, ..., 0][interior]
        if has_residual:
            residual = mx.array(source_residual)
            scaled_low = mx.where(finite, residual / scale, 0)
            low_numerator = mx.conv3d(
                scaled_low[None, ..., None], weight, padding=padding,
            )[0, ..., 0][interior]
            flushed_residual = finite & (residual != 0) & (scaled_low == 0)
        else:
            residual = None
            low_numerator = mx.zeros_like(numerator)
            flushed_residual = None
        # A residual is far smaller than its value, so it can flush while the value
        # stays normal (1e8 + 1 next to a remote 3e38). Each part is recovered on its
        # own flush condition; testing only the value lost residuals whose values cancel.
        flushed = finite & (block != 0) & (scaled == 0)
        flush_flags = np.asarray(mx.stack([
            mx.any(flushed),
            mx.any(flushed_residual) if flushed_residual is not None
            else mx.array(False, dtype=mx.bool_),
        ]), dtype=np.bool_)
        if flush_flags[0]:
            flushed_numerator = mx.conv3d(
                mx.where(flushed, block, 0)[None, ..., None], weight, padding=padding
            )[0, ..., 0][interior]
        else:
            flushed_numerator = mx.zeros_like(numerator)
        if flush_flags[1] and residual is not None and flushed_residual is not None:
            flushed_low = mx.conv3d(
                mx.where(flushed_residual, residual, 0)[None, ..., None],
                weight,
                padding=padding,
            )[0, ..., 0][interior]
        else:
            flushed_low = mx.zeros_like(low_numerator)
        denominator = mx.conv3d(mx.ones_like(block)[None, ..., None], weight, padding=padding)[
            0, ..., 0
        ][interior]
        if not all_finite:
            invalid = mx.conv3d(
                (~finite).astype(mx.float32)[None, ..., None], weight, padding=padding,
            )[0, ..., 0][interior]
            candidates = roi if "GlobalIntensityPeak" in selected else local_centers
            if bool(mx.any(candidates & (invalid > 0)).item()):
                raise ValueError("local-intensity spheres require finite float32-range samples")
        mean = (
            (numerator / denominator + low_numerator / denominator) * scale
            + flushed_numerator / denominator
            + flushed_low / denominator
        )
        for name in selected:
            centers = local_centers if name == "LocalIntensityPeak" else roi
            peaks[name] = mx.maximum(peaks[name], mx.max(mx.where(centers, mean, float("-inf"))))
        evaluate(*peaks.values())
    return {f"localintensity_{name}": mx.reshape(value, (1,)) for name, value in peaks.items()}
