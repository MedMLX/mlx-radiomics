"""Intensity-dependent IBSI morphological features on Metal.

The morphological family uses two masks. The intensity mask defines the voxels whose
intensities enter an intensity-weighted quantity; the morphological mask defines the
shape whose volume and geometric centre are used. The IBSI manual distinguishes these
masks because resegmentation can remove high- or low-intensity voxels from the intensity
set while leaving the segmented shape intact.

All feature arithmetic runs on MLX arrays. Host NumPy work is limited to selecting source
voxels, translating integer coordinates, and centring and scaling source intensities
before they are uploaded as float32. Moran's I and Geary's C evaluate every unordered
pair once in a tiled Metal kernel without storing a pair matrix. Row outputs are bounded by
``scratch_mib``; the quadratic pass runs only when selected and is refused above
:data:`MAX_PAIRWISE_VOXELS` intensity-mask voxels.
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from functools import lru_cache
from math import frexp, isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.features.geometry.arithmetic import scale_pow2
from mlxradiomics.features.geometry.mesh import MeshGeometryContext, mesh_geometry
from mlxradiomics.numerics.pair_arithmetic import subtract_pairs
from mlxradiomics.numerics.precision import prepare_source_pair
from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.reduction import CompensatedTotal, compensated_sum_rows
from mlxradiomics.numerics.runtime import evaluate, mx

INTENSITY_MORPHOLOGY = (
    "CentreOfMassShift",
    "IntegratedIntensity",
    "MoransI",
    "GearysC",
)
_SUPPORTED = frozenset(INTENSITY_MORPHOLOGY)
# Moran's I and Geary's C sum over every voxel pair. Blocking bounds memory but
# not time: 2**20 voxels are about 5.5e11 distinct pairs. Larger ROIs are
# refused before any feature work starts.
MAX_PAIRWISE_VOXELS = 2**20


def require_pairwise_admission(features: Collection[str], voxels: int) -> None:
    """Refuse Moran's I or Geary's C when the intensity mask exceeds the pair budget.

    ``features`` are bare shape names; ``voxels`` counts the intensity mask after
    resegmentation. Raises ``ValueError`` naming the count and the limit.
    """
    selected = sorted(set(features) & {"MoransI", "GearysC"})
    if selected and voxels > MAX_PAIRWISE_VOXELS:
        raise ValueError(
            f"{' and '.join(selected)} require quadratic work for {voxels:,} ROI voxels, above the "
            f"{MAX_PAIRWISE_VOXELS:,}-voxel limit for quadratic spatial autocorrelation; "
            "deselect them or use a smaller or coarser ROI"
        )


@dataclass
class _IntensitySource:
    """Reuse sampled intensities and coordinates within one morphology invocation."""

    values: NDArray[np.float64]
    mask: NDArray[np.bool_]
    _coordinates: NDArray[np.int64] | None = field(default=None, init=False, repr=False)

    def coordinates(self) -> NDArray[np.int64]:
        """Return the ``(n, ndim)`` mask coordinates, computed once."""
        if self._coordinates is None:
            self._coordinates = np.argwhere(self.mask).astype(np.int64, copy=False)
        return self._coordinates


def _source_parts(scaled: NDArray[np.float64]) -> tuple[mx.array, mx.array, mx.array]:
    """Upload both source words and their complete representation-error bound.

    Coincident high words are harmless when their residuals retain the distinction.
    Feature-specific checks below compare the remaining source error with the quantity
    being calculated, rather than requiring every high word to be unique.
    """
    if not np.all(np.isfinite(scaled)):
        raise RuntimeError("intensity morphology source scaling is too ill-conditioned")
    high, low, error = prepare_source_pair(scaled)
    # A positive error bound must not disappear when Metal flushes subnormal words.
    error = np.where(error > 0, np.maximum(error, np.finfo(np.float32).tiny), error)
    return mx.array(high), mx.array(low), mx.array(error)


def _sum_parts(high: mx.array, low: mx.array) -> mx.array:
    """Fold both source words through one compensated reduction tree."""
    return compensated_sum_rows(mx.reshape(mx.concatenate([high, low]), (1, -1)))[0]


def _resolved_total(
    high: mx.array, low: mx.array, error: mx.array, *, allow_zero: bool = False,
) -> mx.array:
    """Compensate signed cancellation and qualify the remaining representation error.

    An integral may be zero; a centroid denominator may not. The sum has the same
    compensated float32 arithmetic contract as first-order Mean. Source-pair error
    must be below 2**-20 of a nonzero total, or exactly zero for a zero integral.
    """
    total = _sum_parts(high, low)
    bound = 2 * mx.sum(error)
    evaluate(total, bound)
    value = float(cast(float, total.item()))
    source_error = float(cast(float, bound.item()))
    if source_error > abs(value) * 2**-20 or (value == 0 and not allow_zero):
        raise RuntimeError("intensity morphology total intensity is too ill-conditioned to resolve")
    return total


def _centroid_shift(
    source: _IntensitySource,
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    mesh_context: MeshGeometryContext | None = None,
) -> mx.array:
    """Return the distance between geometric and intensity-weighted centroids.

    The geometric centroid is the exact rational ``S / n`` of int64 coordinate sums, so
    each intensity voxel's offset from it is the int64 numerator ``n * x - S`` (rounded
    once to float32), and constant weights on matching masks give exactly zero shift.
    Weights are scaled by an exact power of two and uploaded as a float32 value plus
    float32 residual, so neither the scaling nor a single float32 word distorts small
    weight differences; the weighted numerator is a compensated sum. Multiplicative
    scaling preserves a weighted centroid because numerator and denominator share the
    weights; an additive intensity offset would not, so weights keep their sign.
    """
    morphology_coordinates = (
        mesh_context.coordinates() if mesh_context is not None else
        source.coordinates() if source.mask is morphology_mask else
        np.argwhere(morphology_mask).astype(np.int64, copy=False)
    )
    intensity_coordinates = source.coordinates()
    reference = morphology_coordinates[0]
    count = morphology_coordinates.shape[0]
    # Coordinates relative to one morphological voxel keep every int64 numerator far
    # below 2**62 (count and extent are each below 2**31).
    morphology = mx.array(morphology_coordinates - reference)
    intensity = mx.array(intensity_coordinates - reference)
    numerators = (intensity * count - mx.sum(morphology, axis=0)).astype(mx.float32)
    values = source.values
    weight_scale = float(np.max(np.abs(values)))
    if weight_scale == 0:
        raise RuntimeError("intensity-weighted centroid is undefined for zero intensity")
    # Constant nonzero weights cancel from the centroid ratio exactly, including
    # constants whose binary64 value cannot be represented by two float32 words.
    scaled = (np.ones_like(values) if np.all(values == values[0]) else
              np.ldexp(values, -frexp(weight_scale)[1]))
    weights, residual_weights, source_error = _source_parts(scaled)
    denominator = _resolved_total(weights, residual_weights, source_error)
    products = mx.concatenate(
        [numerators * weights[:, None], numerators * residual_weights[:, None]], axis=0,
    )
    numerator = compensated_sum_rows(mx.transpose(products))
    error_bound = 2 * mx.sum(mx.abs(numerators) * source_error[:, None], axis=0)
    evaluate(numerator, error_bound)
    maximum_error = float(cast(float, mx.max(error_bound).item()))
    maximum_numerator = float(cast(float, mx.max(mx.abs(numerator)).item()))
    if maximum_error > maximum_numerator * 2**-20:
        raise RuntimeError("intensity-weighted centroid is too ill-conditioned to resolve")
    shift = numerator / denominator / float(count)
    axis_scale = mx.array(spacing, dtype=mx.float32)
    components = mx.where(shift != 0, shift * axis_scale, 0)
    largest = mx.max(mx.abs(components))
    displaced = mx.any(shift != 0)
    evaluate(largest, displaced)
    largest_value = float(cast(float, largest.item()))
    if not isfinite(largest_value):
        raise RuntimeError("CentreOfMassShift exceeds the float32 range")
    if largest_value == 0:
        # A nonzero grid-unit shift whose physical components all leave the normal
        # float32 range (Metal flushes subnormals) is an underflow, not a zero shift.
        if bool(displaced.item()):
            raise RuntimeError("CentreOfMassShift underflows the float32 range")
        return mx.array(0, dtype=mx.float32)
    result = largest * mx.sqrt(mx.sum((components / largest) * (components / largest)))
    evaluate(result)
    if not isfinite(float(cast(float, result.item()))):
        raise RuntimeError("CentreOfMassShift exceeds the float32 range")
    return result


def _integrated_intensity(
    source: _IntensitySource,
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    scratch_mib: int,
    mesh_context: MeshGeometryContext | None = None,
) -> mx.array:
    """Return mesh volume times mean intensity over the intensity mask."""
    geometry = (mesh_context.get() if mesh_context is not None else
                mesh_geometry(morphology_mask, spacing, scratch_mib, include_vertices=False))
    values = source.values
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return mx.array(0, dtype=mx.float32)
    _, exponent = frexp(scale)
    high, low, error = _source_parts(np.ldexp(values, -exponent))
    mean = _resolved_total(high, low, error, allow_zero=True) / float(values.size)
    result = scale_pow2(
        geometry.enclosed_norm * mean, geometry.enclosed_exponent + exponent,
    )
    evaluate(result)
    result_value = float(cast(float, result.item()))
    if not isfinite(result_value):
        raise RuntimeError("IntegratedIntensity exceeds the float32 range")
    if result_value == 0 and float(cast(float, mean.item())) != 0:
        raise RuntimeError("IntegratedIntensity underflows the float32 range")
    return result



# Pair tiles staged in threadgroup memory; one thread per row voxel in a threadgroup.
_PAIR_TILE = 256
# Rows per dispatch keep each command buffer short on large ROIs.
_PAIR_ROWS = 1 << 16


# Below this normalized spacing a one-step squared component would leave the float32
# normal range, which Metal flushes; such grids keep the per-pair scaled distance.
_UNSCALED_UNIT_FLOOR = 2.0**-63


@lru_cache(maxsize=2)
def _pair_kernel(scaled: bool) -> MetalKernel:
    """Tiled unordered-pair kernel for the inverse-distance weight and both numerators.

    Each thread owns row voxel ``i`` and visits only column voxels ``j > i`` through
    ``_PAIR_TILE``-wide threadgroup tiles, starting at its own threadgroup's tile, so
    the pairwise matrix is never stored. All three terms are symmetric in ``i`` and
    ``j`` (``-dx`` gives the same squared distance), so the row totals are doubled to
    give the ordered-pair sums exactly. Mask coordinates are distinct, so no visited pair
    is a self-pair.

    With ``scaled`` the distance divides each component by the largest before squaring,
    which keeps squares normal when a normalized spacing is below
    :data:`_UNSCALED_UNIT_FLOOR`. Otherwise every nonzero squared component is a normal
    float32 and the plain ``sqrt(qx*qx + qy*qy + qz*qz)`` is used, which drops the
    three divisions and the rescaling product along with their roundings. Weights,
    Moran and Geary terms are accumulated with error-free two-sum in float-float
    registers; contraction and reassociation are disabled so the compensation is not
    optimized away. Output row ``t`` holds
    (weight, moran, geary) as (high, low) pairs for row voxel ``start + t``.
    """
    distance = (
        """float largest = max(max(metal::abs(qx), metal::abs(qy)), metal::abs(qz));
                        float safe = largest > 0 ? largest : 1.0f;
                        qx = metal::precise::divide(qx, safe);
                        qy = metal::precise::divide(qy, safe);
                        qz = metal::precise::divide(qz, safe);
                        float distance = safe * metal::precise::sqrt(qx*qx + qy*qy + qz*qz);"""
        if scaled else
        "float distance = metal::precise::sqrt(qx*qx + qy*qy + qz*qz);"
    )
    return cast(MetalKernel, mx.fast.metal_kernel(
        name=f"mlxradiomics_spatial_pairs_{'scaled' if scaled else 'direct'}",
        input_names=["coords", "values", "unit", "params"],
        output_names=["sums"],
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
        source=f"""
            uint lane = thread_position_in_threadgroup.x;
            uint row = thread_position_in_grid.x;
            uint n = params[0], start = params[1], rows = params[2];
            bool active = row < rows;
            uint i = start + (active ? row : 0);
            // First row voxel of this threadgroup; row voxel i is first + lane.
            uint first = start + threadgroup_position_in_grid.x * {_PAIR_TILE};
            int ix = coords[3 * i], iy = coords[3 * i + 1], iz = coords[3 * i + 2];
            float vi = values[i];
            float ux = unit[0], uy = unit[1], uz = unit[2];
            float ws = 0, wc = 0, ms = 0, mc = 0, gs = 0, gc = 0;
            threadgroup int tx[{_PAIR_TILE}];
            threadgroup int ty[{_PAIR_TILE}];
            threadgroup int tz[{_PAIR_TILE}];
            threadgroup float tv[{_PAIR_TILE}];
            for (uint base = first; base < n; base += {_PAIR_TILE}) {{
                uint j = base + lane;
                if (j < n) {{
                    tx[lane] = coords[3 * j]; ty[lane] = coords[3 * j + 1];
                    tz[lane] = coords[3 * j + 2]; tv[lane] = values[j];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint count = min(uint({_PAIR_TILE}), n - base);
                // In the diagonal tile, column k is voxel first + k; keep only j > i.
                uint k0 = base == first ? lane + 1 : 0;
                if (active) {{
                    for (uint k = k0; k < count; ++k) {{
                        int dx = tx[k] - ix, dy = ty[k] - iy, dz = tz[k] - iz;
                        float qx = float(dx) * ux, qy = float(dy) * uy, qz = float(dz) * uz;
                        {distance}
                        float w = metal::precise::divide(1.0f, distance);
                        float vj = tv[k];
                        float diff = vi - vj;
                        two_sum_add(ws, wc, w);
                        two_sum_add(ms, mc, w * vi * vj);
                        two_sum_add(gs, gc, w * diff * diff);
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (active) {{
                // Doubling is exact and turns unordered-pair totals into ordered ones.
                uint o = 6 * row;
                sums[o] = 2 * ws; sums[o + 1] = 2 * wc; sums[o + 2] = 2 * ms;
                sums[o + 3] = 2 * mc; sums[o + 4] = 2 * gs; sums[o + 5] = 2 * gc;
            }}
        """,
    ))


def _pair_sums(
    coordinates: mx.array, centered: mx.array, unit: mx.array, count: int, scratch_mib: int,
    scaled: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    """Sum ordered pairs with bounded row outputs and compensated chunk totals."""
    rows_per_dispatch = min(_PAIR_ROWS, max(1, scratch_mib * 1024**2 // 64))
    accumulator = CompensatedTotal(6)
    for start in range(0, count, rows_per_dispatch):
        rows = min(rows_per_dispatch, count - start)
        params = mx.array([count, start, rows], dtype=mx.uint32)
        threads = ((rows + _PAIR_TILE - 1) // _PAIR_TILE) * _PAIR_TILE
        partial = _pair_kernel(scaled)(
            inputs=[coordinates, centered, unit, params], grid=(threads, 1, 1),
            threadgroup=(_PAIR_TILE, 1, 1), output_shapes=[(rows * 6,)],
            output_dtypes=[mx.float32],
        )[0]
        totals = compensated_sum_rows(mx.transpose(mx.reshape(partial, (rows, 6))))
        accumulator.add(totals)
        evaluate(*accumulator.arrays())
    totals = accumulator.value()
    return totals[0] + totals[1], totals[2] + totals[3], totals[4] + totals[5]


def _spatial_autocorrelation(
    source: _IntensitySource,
    spacing: tuple[float, ...],
    scratch_mib: int,
    names: frozenset[str],
) -> dict[str, mx.array]:
    """Return Moran's I and/or Geary's C with inverse-Euclidean-distance weights.

    A tiled Metal kernel evaluates each unordered pair once, retaining only row totals.
    ``scratch_mib`` bounds those outputs. The largest active spacing axis supplies a
    common distance scale that cancels from both ratios.
    """
    coordinates = source.coordinates()
    reference = coordinates[0]
    relative = (coordinates - reference).astype(np.int32)
    count = int(relative.shape[0])
    if count < 2:
        return {name: mx.array(float("nan")) for name in names}
    values = source.values
    reference_value = float(values[0])
    offsets = values - reference_value
    scale = float(np.max(np.abs(offsets)))
    if scale == 0:
        return {name: mx.array(float("nan")) for name in names}
    high, low, error = _source_parts(offsets / scale)
    mean = _sum_parts(high, low) / float(count)
    centered_high, centered_low = subtract_pairs(high, low, mean, mx.zeros_like(mean))
    centered = centered_high + centered_low
    denominator = compensated_sum_rows(mx.reshape(centered * centered, (1, -1)))[0]
    deviation = mx.sqrt(denominator / float(count))
    evaluate(deviation, error)
    spread = float(cast(float, deviation.item()))
    if spread == 0 or float(cast(float, mx.max(error).item())) > spread * 2**-20:
        raise RuntimeError(
            "spatial autocorrelation source spread is too ill-conditioned to resolve"
        )
    # An axis on which all coordinates agree contributes no pair distance. Do not
    # let its spacing consume the precision of the axes actually present in the ROI.
    active = tuple(bool(np.any(relative[:, axis] != 0)) for axis in range(len(spacing)))
    largest_spacing = max(value for value, used in zip(spacing, active, strict=True) if used)
    normalized_spacing = tuple(value / largest_spacing if used else 0.0
                               for value, used in zip(spacing, active, strict=True))
    units32 = np.asarray(normalized_spacing, dtype=np.float32)
    if any(float(units32[axis]) == 0 and used for axis, used in enumerate(active)):
        raise RuntimeError("spatial autocorrelation distances are too ill-conditioned")
    unit = mx.array(normalized_spacing, dtype=mx.float32)
    coordinates_mx = mx.array(relative)
    scaled = any(used and float(units32[axis]) < _UNSCALED_UNIT_FLOOR
                 for axis, used in enumerate(active))
    total_weight, moran_numerator, geary_numerator = _pair_sums(
        mx.reshape(coordinates_mx, (-1,)), centered, unit, count, scratch_mib, scaled,
    )
    evaluate(total_weight, moran_numerator, geary_numerator)
    # Every valid weight is positive, so any non-finite weight, including from a distance
    # flushed to zero, makes the total non-finite. One deferred check replaces a host
    # synchronization per block.
    if not isfinite(float(cast(float, total_weight.item()))):
        raise RuntimeError("spatial autocorrelation weights are too ill-conditioned")
    results: dict[str, mx.array] = {}
    if "MoransI" in names:
        results["MoransI"] = mx.where(
            total_weight > 0,
            (float(count) / total_weight) * moran_numerator / denominator,
            float("nan"),
        )
    if "GearysC" in names:
        results["GearysC"] = mx.where(
            total_weight > 0,
            ((count - 1) / (2 * total_weight)) * geary_numerator / denominator,
            float("nan"),
        )
    return results


def morphology_statistics(
    image: NDArray[np.generic],
    intensity_mask: NDArray[np.bool_],
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    scratch_mib: int,
    features: Collection[str],
    *, _mesh_context: MeshGeometryContext | None = None,
) -> dict[str, mx.array]:
    """Compute selected intensity-dependent morphological features.

    Args:
        image: Intensity image on the same grid as both masks.
        intensity_mask: Voxels whose intensities enter weighted quantities and spatial
            autocorrelation.
        morphology_mask: Segmented shape used for mesh volume and the geometric centroid.
        spacing: Voxel spacing per array axis in the requested length unit.
        scratch_mib: Working-memory budget for mesh chunks and pair blocks.
        features: Bare names from :data:`INTENSITY_MORPHOLOGY`.

    Returns:
        ``shape_<Name>`` keys mapped to float32 arrays of shape ``(1,)``. Moran's I and
        Geary's C are NaN for a constant intensity ROI or fewer than two intensity-mask
        voxels, where their weighted variance denominator is zero.

    Raises:
        ValueError: If shapes, dimensions, spacing, or feature names are invalid, a
            required mask is empty, or Moran's I or Geary's C is requested for more than
            :data:`MAX_PAIRWISE_VOXELS` intensity-mask voxels.
        RuntimeError: If the intensity-weighted centroid is undefined or a computed
            centre-of-mass shift exceeds or underflows float32.
    """
    requested = frozenset(features)
    unsupported = requested - _SUPPORTED
    if unsupported:
        raise ValueError(
            f"unsupported intensity morphology features: {', '.join(sorted(unsupported))}"
        )
    if not requested:
        return {}
    require_pairwise_admission(requested, int(np.count_nonzero(intensity_mask)))
    image_array = np.asarray(image)
    if image_array.shape != intensity_mask.shape or image_array.shape != morphology_mask.shape:
        raise ValueError("image and both masks must have the same shape")
    if morphology_mask.ndim != 3:
        raise ValueError("intensity morphology requires three-dimensional masks")
    if len(spacing) != 3:
        raise ValueError("spacing must contain one value per mask axis")
    if any(not isfinite(float(value)) or value <= 0 for value in spacing):
        raise ValueError("spacing values must be finite and positive")
    if not np.any(morphology_mask):
        raise ValueError("morphology mask is empty")
    if not np.any(intensity_mask):
        raise ValueError("intensity mask is empty")
    samples = image_array[intensity_mask]
    if not np.all(np.isfinite(samples)):
        raise ValueError("intensity morphology requires finite source intensities")
    source = _IntensitySource(samples.astype(np.float64), intensity_mask)
    values: dict[str, mx.array] = {}
    if "CentreOfMassShift" in requested:
        values["CentreOfMassShift"] = _centroid_shift(
            source, morphology_mask, spacing, _mesh_context,
        )
    if "IntegratedIntensity" in requested:
        values["IntegratedIntensity"] = _integrated_intensity(
            source, morphology_mask, spacing, scratch_mib, _mesh_context,
        )
    pair_names = requested & {"MoransI", "GearysC"}
    if pair_names:
        values.update(_spatial_autocorrelation(
            source, spacing, scratch_mib, pair_names,
        ))
    evaluate(*values.values())
    return {f"shape_{name}": value[None] for name, value in values.items()}
