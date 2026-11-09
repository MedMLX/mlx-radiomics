"""Voxel-map GLRLM, GLSZM, GLDM and NGTDM features with fused per-center Metal passes.

As in :mod:`mlxradiomics.voxel.glcm`, each thread reads one center's small
neighborhood once and evaluates the requested family from its occupied
(level, size) cells instead of a dense ``Ng x Ns`` matrix. Definitions follow
:func:`mlxradiomics.features.texture.texture_statistics` with native settings:

* GLRLM runs follow each unit direction of the neighborhood; directions with a run
  are averaged with an ordinary sum, as ``run_statistics`` does.
* GLSZM zones are same-level components under the full unit neighborhood.
* GLDM dependence counts masked neighbors at the selected distances within
  ``gldm_alpha`` gray levels; NGTDM compares eligible centers with the mean level of
  their masked neighbors.

Integer numerators (emphases with integer weights, nonuniformities, variances,
percentages, NGTDM contrast and strength) are exact int64 before one division;
other sums use two-sum compensation. Variances are formed as
``sum m (N i - S)**2 / N**3`` from exact integers, which retains the integer
centering of ``size_statistics``. Entropies use the ``count_entropy`` summand.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import cast

import numpy as np

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.voxel.glcm import KERNEL_HEADER, MAX_WINDOW, window_code

_SPECS = {
    "gldm": ("Dependence", "SmallDependence", "LargeDependence", "LowGrayLevel",
             "HighGrayLevel", "DependenceEntropy", "DependenceVariance", "DependencePercentage"),
    "glrlm": ("RunLength", "ShortRun", "LongRun", "LowGrayLevelRun", "HighGrayLevelRun",
              "RunEntropy", "RunVariance", "RunPercentage"),
    "glszm": ("SizeZone", "SmallArea", "LargeArea", "LowGrayLevelZone", "HighGrayLevelZone",
              "ZoneEntropy", "ZoneVariance", "ZonePercentage"),
}
NGTDM = ("Coarseness", "Contrast", "Busyness", "Complexity", "Strength")


def _size_names(family: str) -> dict[str, int]:
    """Map each size-family feature name of ``family`` to its slot in the kernel output."""
    size, small, large, low, high, entropy, variance, percentage = _SPECS[family]
    names = {
        small + "Emphasis": 0, large + "Emphasis": 1, "GrayLevelNonUniformity": 2,
        "GrayLevelNonUniformityNormalized": 3, size + "NonUniformity": 4,
        size + "NonUniformityNormalized": 5, "GrayLevelVariance": 6, variance: 7,
        entropy: 8, low + "Emphasis": 9, high + "Emphasis": 10,
        small + "LowGrayLevelEmphasis": 11, small + "HighGrayLevelEmphasis": 12,
        large + "LowGrayLevelEmphasis": 13, large + "HighGrayLevelEmphasis": 14,
        percentage: 15,
    }
    if family == "gldm":
        names["DependenceCountEnergy"] = 16
    return names


_SIZE_NAMES = {family: _size_names(family) for family in _SPECS}

_SIZE_HEADER = r"""
inline void add_size_cell(thread int* li, thread int* lj, thread int* lc, thread int& n,
                          int i, int j) {
    for (int k = 0; k < n; k++) {
        if (li[k] == i && lj[k] == j) { lc[k] += 1; return; }
    }
    li[n] = i; lj[n] = j; lc[n] = 1; n++;
}

inline void add_marginal(thread int* keys, thread long* counts, thread int& n, int key, long c) {
    for (int k = 0; k < n; k++) {
        if (keys[k] == key) { counts[k] += c; return; }
    }
    keys[n] = key; counts[n] = c; n++;
}

// sum m (N x - S)**2 / N**3 for marginal counts m at integer positions x.
inline float integer_variance(thread const int* keys, thread const long* counts, int n,
                              long total) {
    long s = 0;
    for (int k = 0; k < n; k++) s += counts[k] * long(keys[k]);
    long numerator = 0;
    for (int k = 0; k < n; k++) {
        long offset = total * long(keys[k]) - s;
        numerator += counts[k] * offset * offset;
    }
    return q_div(numerator, total * total * total);
}

// Size-family features of cells (gray level i, size j, count c); voxels is the
// denominator of the percentage (sum c*j for runs and zones, N for dependence).
template <int CAP>
inline void size_features(thread const int* li, thread const int* lj, thread const int* lc,
                          int nk, long voxels, thread float* out) {
    long n = 0;
    for (int k = 0; k < nk; k++) n += lc[k];
    float nf = float(n);
    Acc small = {0, 0}, low = {0, 0}, small_low = {0, 0}, small_high = {0, 0};
    Acc large_low = {0, 0}, entropy = {0, 0};
    long large = 0, high = 0, large_high = 0, energy = 0;
    int gi[CAP], sj[CAP];
    long gm[CAP], sm[CAP];
    int ng = 0, ns = 0;
    for (int k = 0; k < nk; k++) {
        long i = li[k], j = lj[k], c = lc[k];
        float cf = float(c);
        acc_add(small, cf / float(j * j));
        large += c * j * j;
        high += c * i * i;
        large_high += c * i * i * j * j;
        energy += c * c;
        acc_add(low, cf / float(i * i));
        acc_add(small_low, cf / (float(i * i) * float(j * j)));
        acc_add(small_high, float(c * i * i) / float(j * j));
        acc_add(large_low, float(c * j * j) / float(i * i));
        acc_add(entropy, entropy_term(c, n));
        add_marginal(gi, gm, ng, li[k], c);
        add_marginal(sj, sm, ns, lj[k], c);
    }
    long gln = 0, snu = 0;
    for (int k = 0; k < ng; k++) gln += gm[k] * gm[k];
    for (int k = 0; k < ns; k++) snu += sm[k] * sm[k];
    out[0] = acc_value(small) / nf;
    out[1] = q_div(large, n);
    out[2] = q_div(gln, n);
    out[3] = q_div(gln, n * n);
    out[4] = q_div(snu, n);
    out[5] = q_div(snu, n * n);
    out[6] = integer_variance(gi, gm, ng, n);
    out[7] = integer_variance(sj, sm, ns, n);
    out[8] = -acc_value(entropy) / nf;
    out[9] = acc_value(low) / nf;
    out[10] = q_div(high, n);
    out[11] = acc_value(small_low) / nf;
    out[12] = acc_value(small_high) / nf;
    out[13] = acc_value(large_low) / nf;
    out[14] = q_div(large_high, n);
    out[15] = float(n) / float(voxels);
    out[16] = q_div(energy, n * n);
}
"""


def _size_stores(slots: tuple[int, ...], target: str) -> str:
    """Return Metal stores of the requested size-family ``slots``, in order.

    ``target`` addresses ``{column}``. Outputs that are not stored are removed by
    the compiler from the inlined ``size_features``.
    """
    return "\n".join(f"{target.format(column=k)} = out[{slot}];" for k, slot in enumerate(slots))


def _local_index_code(ndim: int, local_shape: tuple[int, ...]) -> str:
    """Return Metal constants and ``local_slot`` mapping window coordinates to slots."""
    strides = [prod(local_shape[a + 1:]) for a in range(ndim)]
    halves = [s // 2 for s in local_shape]
    flat = " + ".join(f"x{a} * {strides[a]}" for a in range(ndim))
    inside = " && ".join(f"x{a} >= 0 && x{a} < {local_shape[a]}" for a in range(ndim))
    args = ", ".join(f"int x{a}" for a in range(ndim))
    return f"""
constant int LOCAL_HALF[{ndim}] = {{{", ".join(map(str, halves))}}};
inline int local_slot({args}) {{ return ({inside}) ? {flat} : -1; }}
"""


def _offsets_table(name: str, offsets: tuple[tuple[int, ...], ...]) -> str:
    """Return a flattened ``constant int`` table of ``offsets``."""
    values = [v for offset in offsets for v in offset] or [0]
    return f"constant int {name}[{len(values)}] = {{{', '.join(map(str, values))}}};"


def _window_table(local_shape: tuple[int, ...]) -> str:
    """Return the ``WINDOW`` table of offsets covering ``local_shape``, centered at zero."""
    axes = [range(-(s // 2), s // 2 + 1) if s > 1 else [0] for s in local_shape]
    window = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(local_shape), -1).T
    return f"constant int WINDOW[{window.size}] = {{{', '.join(map(str, window.reshape(-1)))}}};"


def _neighbor_slot(ndim: int, table: str, index: str) -> str:
    """Return Metal source for window slot ``q`` shifted by ``table[index]``, or -1 outside."""
    args = ", ".join(
        f"WINDOW[q * {ndim} + {a}] + LOCAL_HALF[{a}] + {table}[{index} * {ndim} + {a}]"
        for a in range(ndim)
    )
    return f"local_slot({args})"


@lru_cache(maxsize=64)
def _run_kernel(
    local_shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], slots: tuple[int, ...],
) -> MetalKernel:
    """Compile the per-center local GLRLM kernel for every direction in ``offsets``."""
    ndim = len(local_shape)
    window = prod(local_shape)
    directions = len(offsets)
    header = "\n".join((
        KERNEL_HEADER, _SIZE_HEADER, _window_table(local_shape),
        _local_index_code(ndim, local_shape), _offsets_table("RUN", offsets),
        _offsets_table("BACK", tuple(tuple(-v for v in o) for o in offsets)),
    ))
    previous = _neighbor_slot(ndim, "BACK", "d")
    coordinates = ", ".join(f"x{a}" for a in range(ndim))
    position = "\n".join(
        f"int x{a} = WINDOW[q * {ndim} + {a}] + LOCAL_HALF[{a}];" for a in range(ndim)
    )
    advance = "\n".join(f"x{a} += RUN[d * {ndim} + {a}];" for a in range(ndim))
    source = f"""
        uint t = thread_position_in_grid.x;
        uint total_centers = centers_shape[0];
        if (t >= total_centers * {directions}) return;
        uint center = t / {directions};
        int d = int(t % {directions});
        {window_code(ndim, window)}
        int li[{window}], lj[{window}], lc[{window}];
        int nk = 0;
        long voxels = 0;
        for (int q = 0; q < {window}; q++) {{
            int level = levels[q];
            if (level == 0) continue;
            voxels++;
            int before = {previous};
            if (before >= 0 && levels[before] == level) continue;
            {position}
            int length = 1;
            while (true) {{
                {advance}
                int next = local_slot({coordinates});
                if (next < 0 || levels[next] != level) break;
                length++;
            }}
            add_size_cell(li, lj, lc, nk, level, length);
        }}
        active[t] = nk > 0;
        if (nk == 0) return;
        float out[17];
        size_features<{window}>(li, lj, lc, nk, voxels, out);
        {_size_stores(slots, "values[ulong(t) * " + str(len(slots)) + " + {column}]")}
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_glrlm",
        input_names=["bins", "mask", "centers", "dims"],
        output_names=["values", "active"],
        header=header,
        source=source,
    ))


@lru_cache(maxsize=64)
def _run_average_kernel(directions: int, features: int) -> MetalKernel:
    """Average per-direction GLRLM features with ordinary running sums.

    Directions are added in order, as ``run_statistics`` accumulates them.
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_glrlm_average",
        input_names=["values", "active"],
        output_names=["averaged"],
        source=f"""
        uint center = thread_position_in_grid.x;
        uint total_centers = active_shape[0] / {directions};
        if (center >= total_centers) return;
        int count = 0;
        for (int d = 0; d < {directions}; d++) count += active[center * {directions} + d];
        for (int f = 0; f < {features}; f++) {{
            float total = 0.0f;
            for (int d = 0; d < {directions}; d++) {{
                uint t = center * {directions} + d;
                if (active[t]) total = total + values[ulong(t) * {features} + f];
            }}
            averaged[ulong(f) * total_centers + center] =
                count > 0 ? total / float(count) : NAN;
        }}
        """,
    ))


@lru_cache(maxsize=64)
def _zone_kernel(
    local_shape: tuple[int, ...], neighbors: tuple[tuple[int, ...], ...], slots: tuple[int, ...],
) -> MetalKernel:
    """Compile the per-center local GLSZM kernel under ``neighbors`` connectivity."""
    ndim = len(local_shape)
    window = prod(local_shape)
    header = "\n".join((
        KERNEL_HEADER, _SIZE_HEADER, _window_table(local_shape),
        _local_index_code(ndim, local_shape), _offsets_table("NEIGHBOR", neighbors),
    ))
    neighbor = _neighbor_slot(ndim, "NEIGHBOR", "m").replace("q * ", "slot * ")
    source = f"""
        uint center = thread_position_in_grid.x;
        uint total_centers = centers_shape[0];
        if (center >= total_centers) return;
        {window_code(ndim, window)}
        int li[{window}], lj[{window}], lc[{window}];
        int nk = 0;
        long voxels = 0;
        bool seen[{window}];
        for (int q = 0; q < {window}; q++) seen[q] = false;
        int stack[{window}];
        for (int q = 0; q < {window}; q++) {{
            int level = levels[q];
            if (level == 0 || seen[q]) continue;
            int top = 0, size = 0;
            stack[top++] = q;
            seen[q] = true;
            while (top > 0) {{
                int slot = stack[--top];
                size++;
                for (int m = 0; m < {len(neighbors)}; m++) {{
                    int other = {neighbor};
                    if (other >= 0 && !seen[other] && levels[other] == level) {{
                        seen[other] = true;
                        stack[top++] = other;
                    }}
                }}
            }}
            voxels += size;
            add_size_cell(li, lj, lc, nk, level, size);
        }}
        float out[17];
        if (nk == 0) {{
            for (int f = 0; f < 17; f++) out[f] = NAN;
        }} else {{
            size_features<{window}>(li, lj, lc, nk, voxels, out);
        }}
        {_size_stores(slots, "values[ulong({column}) * total_centers + center]")}
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_glszm",
        input_names=["bins", "mask", "centers", "dims"],
        output_names=["values"],
        header=header,
        source=source,
    ))


# Per-voxel NGTDM accumulation: eligible voxels' |level - mean neighbor level| by level.
_NGTDM_UPDATE = r"""
            if (count > 0) {
                // Exact integer difference before division, as the dense kernel forms it.
                long numerator = long(count) * long(level) - sum;
                float difference = float(numerator < 0 ? -numerator : numerator) / float(count);
                int k = 0;
                while (k < ng && gl[k] != level) k++;
                if (k == ng) { gl[k] = level; gn[k] = 0; gs[k] = Acc{0, 0}; ng++; }
                gn[k] += 1;
                acc_add(gs[k], difference);
            }
"""

# NGTDM features from the per-level eligible counts and difference sums.
_NGTDM_FEATURES = r"""
        // NGTDM from eligible centers.
        long total = 0;
        float s[WINDOW_SIZE];
        for (int k = 0; k < ng; k++) { total += gn[k]; s[k] = acc_value(gs[k]); }
        float coarseness = NAN, contrast = NAN, busyness = NAN, complexity = NAN,
              strength = NAN;
        if (total > 0) {
            float tf = float(total);
            Acc ps = {0, 0}, sum_s = {0, 0}, weighted = {0, 0}, complexity_sum = {0, 0};
            long contrast_num = 0, strength_num = 0, busy = 0;
            for (int a = 0; a < ng; a++) {
                acc_add(ps, float(gn[a]) * s[a]);
                acc_add(sum_s, s[a]);
                acc_add(weighted, float(gn[a]) * s[a]);
                for (int b = 0; b < ng; b++) {
                    long delta = long(gl[a]) - long(gl[b]);
                    contrast_num += gn[a] * gn[b] * delta * delta;
                    strength_num += (gn[a] + gn[b]) * delta * delta;
                    long product = gn[a] * long(gl[a]) - gn[b] * long(gl[b]);
                    busy += product < 0 ? -product : product;
                    if (a != b) {
                        float pa = float(gn[a]) / tf, pb = float(gn[b]) / tf;
                        acc_add(complexity_sum, float(delta < 0 ? -delta : delta)
                                * (pa * s[a] + pb * s[b]) / (pa + pb));
                    }
                }
            }
            float psf = acc_value(ps) / tf;
            float sum_sf = acc_value(sum_s);
            coarseness = psf > 0 ? 1.0f / psf : INFINITY;
            contrast = ng > 1 ? q_div(contrast_num, total * total) * sum_sf / tf
                                / float(ng * (ng - 1)) : 0.0f;
            busyness = busy > 0 ? acc_value(weighted) / float(busy)
                                : (psf > 0 ? INFINITY : NAN);
            complexity = acc_value(complexity_sum) / tf;
            float strength_value = q_div(strength_num, total);
            strength = sum_sf > 0 ? strength_value / sum_sf
                                  : (strength_value > 0 ? INFINITY : NAN);
        }
        ngtdm_values[0 * total_centers + center] = coarseness;
        ngtdm_values[1 * total_centers + center] = contrast;
        ngtdm_values[2 * total_centers + center] = busyness;
        ngtdm_values[3 * total_centers + center] = complexity;
        ngtdm_values[4 * total_centers + center] = strength;
"""


def _ngtdm_feature_code(slots: tuple[int, ...]) -> str:
    """Emit only the requested NGTDM reductions, retaining their arithmetic order.

    The complete selection keeps the original source. A partial selection omits
    stores and their unused reductions; Coarseness needs no gray-level pair loop.
    """
    if slots == tuple(range(len(NGTDM))):
        return _NGTDM_FEATURES
    selected = set(slots)
    setup = """
        long total = 0;
        float s[WINDOW_SIZE];
        for (int k = 0; k < ng; k++) { total += gn[k]; s[k] = acc_value(gs[k]); }
        float coarseness = NAN, contrast = NAN, busyness = NAN, complexity = NAN,
              strength = NAN;
        if (total > 0) {
            float tf = float(total);
    """
    declarations: list[str] = []
    level_updates: list[str] = []
    pair_updates: list[str] = []
    finish: list[str] = []
    if selected & {0, 2}:
        declarations.append("Acc ps = {0, 0};")
        level_updates.append("acc_add(ps, float(gn[a]) * s[a]);")
        finish.append("float psf = acc_value(ps) / tf;")
    if selected & {1, 4}:
        declarations.append("Acc sum_s = {0, 0};")
        level_updates.append("acc_add(sum_s, s[a]);")
        finish.append("float sum_sf = acc_value(sum_s);")
    if 1 in selected:
        declarations.append("long contrast_num = 0;")
        pair_updates.append("contrast_num += gn[a] * gn[b] * delta * delta;")
    if 4 in selected:
        declarations.append("long strength_num = 0;")
        pair_updates.append("strength_num += (gn[a] + gn[b]) * delta * delta;")
    if 2 in selected:
        declarations.extend(("Acc weighted = {0, 0};", "long busy = 0;"))
        level_updates.append("acc_add(weighted, float(gn[a]) * s[a]);")
        pair_updates.append("""
            long product = gn[a] * long(gl[a]) - gn[b] * long(gl[b]);
            busy += product < 0 ? -product : product;
        """)
    if 3 in selected:
        declarations.append("Acc complexity_sum = {0, 0};")
        pair_updates.append("""
            if (a != b) {
                float pa = float(gn[a]) / tf, pb = float(gn[b]) / tf;
                acc_add(complexity_sum, float(delta < 0 ? -delta : delta)
                        * (pa * s[a] + pb * s[b]) / (pa + pb));
            }
        """)
    if pair_updates:
        delta = "long delta = long(gl[a]) - long(gl[b]);" if selected & {1, 3, 4} else ""
        level_updates.append(
            "for (int b = 0; b < ng; b++) {\n" + delta + "\n"
            + "\n".join(pair_updates) + "\n}"
        )
    formulas = (
        "coarseness = psf > 0 ? 1.0f / psf : INFINITY;",
        """contrast = ng > 1 ? q_div(contrast_num, total * total) * sum_sf / tf
                            / float(ng * (ng - 1)) : 0.0f;""",
        """busyness = busy > 0 ? acc_value(weighted) / float(busy)
                            : (psf > 0 ? INFINITY : NAN);""",
        "complexity = acc_value(complexity_sum) / tf;",
        """float strength_value = q_div(strength_num, total);
            strength = sum_sf > 0 ? strength_value / sum_sf
                                  : (strength_value > 0 ? INFINITY : NAN);""",
    )
    stores = "\n".join(
        f"ngtdm_values[{column} * total_centers + center] = {NGTDM[slot].lower()};"
        for column, slot in enumerate(slots)
    )
    return (
        setup + "\n".join(declarations)
        + "\nfor (int a = 0; a < ng; a++) {\n" + "\n".join(level_updates) + "\n}\n"
        + "\n".join(finish) + "\n" + "\n".join(formulas[slot] for slot in slots)
        + "\n}\n" + stores
    )


@lru_cache(maxsize=64)
def _neighborhood_kernel(
    local_shape: tuple[int, ...], neighbors: tuple[tuple[int, ...], ...], alpha: int,
    dependence_slots: tuple[int, ...], ngtdm_slots: tuple[int, ...],
) -> MetalKernel:
    """GLDM and NGTDM features at their requested output slots.

    The two families share one neighbor loop. A family that is not requested is left
    out of the source; the other family's arithmetic is unchanged.
    """
    ndim = len(local_shape)
    window = prod(local_shape)
    ngtdm = bool(ngtdm_slots)
    header = "\n".join((
        KERNEL_HEADER, _SIZE_HEADER, _window_table(local_shape),
        _local_index_code(ndim, local_shape), _offsets_table("NEIGHBOR", neighbors),
        f"constant int WINDOW_SIZE = {window};",
    ))
    neighbor = _neighbor_slot(ndim, "NEIGHBOR", "m")
    ngtdm_state = """
        int gl[WINDOW_SIZE];
        long gn[WINDOW_SIZE];
        Acc gs[WINDOW_SIZE];
        int ng = 0;
    """
    dependence_stores = _size_stores(
        dependence_slots, "dependence_values[ulong({column}) * total_centers + center]",
    )
    dependence_features = f"""
        float out[17];
        size_features<{window}>(li, lj, lc, nk, voxels, out);
        {dependence_stores}
    """
    source = f"""
        uint center = thread_position_in_grid.x;
        uint total_centers = centers_shape[0];
        if (center >= total_centers) return;
        {window_code(ndim, window)}
        int li[{window}], lj[{window}], lc[{window}];
        int nk = 0;
        long voxels = 0;
        {ngtdm_state if ngtdm else ""}
        for (int q = 0; q < {window}; q++) {{
            int level = levels[q];
            if (level == 0) continue;
            voxels++;
            int dependence = 1, count = 0;
            long sum = 0;
            for (int m = 0; m < {len(neighbors)}; m++) {{
                int other = {neighbor};
                if (other >= 0 && levels[other] > 0) {{
                    sum += levels[other];
                    count++;
                    dependence += abs(levels[other] - level) <= {alpha};
                }}
            }}
            {"add_size_cell(li, lj, lc, nk, level, dependence);" if dependence_slots else ""}
            {_NGTDM_UPDATE if ngtdm else ""}
        }}
        {dependence_features if dependence_slots else ""}
        {_ngtdm_feature_code(ngtdm_slots) if ngtdm else ""}
    """
    outputs = ["dependence_values"] if dependence_slots else []
    outputs += ["ngtdm_values"] if ngtdm else []
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_neighborhood",
        input_names=["bins", "mask", "centers", "dims"],
        output_names=outputs,
        header=header,
        source=source,
    ))


def eligible(window: int, ng: int) -> bool:
    """Neighborhoods of at most 27 voxels; ``Ng`` bounds the int64 emphasis numerators."""
    return window <= MAX_WINDOW and window**3 * ng**4 < 2**62


def local_texture(
    bins: mx.array,
    mask: mx.array,
    image_shape: tuple[int, ...],
    centers: mx.array,
    local_shape: tuple[int, ...],
    run_offsets: tuple[tuple[int, ...], ...],
    zone_neighbors: tuple[tuple[int, ...], ...],
    dependence_neighbors: tuple[tuple[int, ...], ...],
    alpha: int,
    names: tuple[str, ...],
) -> dict[str, mx.array]:
    """Return the requested GLRLM, GLSZM, GLDM and NGTDM maps for a batch of centers.

    Args:
        bins: Flat int32 gray levels, 0 outside the sampling mask.
        mask: Flat boolean sampling mask.
        image_shape: Image shape in array-axis order.
        centers: Int32 ``(batch, ndim)`` center coordinates.
        local_shape: Neighborhood shape.
        run_offsets: Unit run directions (one of each opposite pair).
        zone_neighbors: Unit connectivity offsets, both signs.
        dependence_neighbors: GLDM/NGTDM neighbor offsets at the selected distances,
            both signs.
        alpha: GLDM dependence tolerance in gray levels.
        names: Requested feature names with family prefixes.

    Returns:
        Name to float32 ``(batch,)`` array.
    """
    batch = centers.shape[0]
    dims = mx.array(image_shape, dtype=mx.int32)
    inputs = [bins, mask, centers, dims]
    result: dict[str, mx.array] = {}
    run_slots, zone_slots, dependence_slots = (
        _slots(family, names) for family in ("glrlm", "glszm", "gldm")
    )
    if run_slots:
        directions = len(run_offsets)
        values, active = _run_kernel(local_shape, run_offsets, run_slots)(
            inputs=inputs, grid=(batch * directions, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(batch * directions, len(run_slots)), (batch * directions,)],
            output_dtypes=[mx.float32, mx.bool_],
        )
        averaged = _run_average_kernel(directions, len(run_slots))(
            inputs=[values, active], grid=(batch, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(len(run_slots), batch)], output_dtypes=[mx.float32],
        )[0]
        result.update(_select("glrlm", averaged, run_slots, names))
    if zone_slots:
        values = _zone_kernel(local_shape, zone_neighbors, zone_slots)(
            inputs=inputs, grid=(batch, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(len(zone_slots), batch)], output_dtypes=[mx.float32],
        )[0]
        result.update(_select("glszm", values, zone_slots, names))
    ngtdm_slots = tuple(index for index, name in enumerate(NGTDM) if f"ngtdm_{name}" in names)
    ngtdm = bool(ngtdm_slots)
    if dependence_slots or ngtdm:
        outputs = _neighborhood_kernel(
            local_shape, dependence_neighbors, alpha, dependence_slots, ngtdm_slots,
        )(
            inputs=inputs, grid=(batch, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(len(dependence_slots), batch)] * bool(dependence_slots)
            + [(len(ngtdm_slots), batch)] * ngtdm,
            output_dtypes=[mx.float32] * (bool(dependence_slots) + ngtdm),
        )
        if dependence_slots:
            result.update(_select("gldm", outputs[0], dependence_slots, names))
        if ngtdm:
            result.update({
                f"ngtdm_{NGTDM[slot]}": outputs[-1][index]
                for index, slot in enumerate(ngtdm_slots)
            })
    return result


def _slots(family: str, names: tuple[str, ...]) -> tuple[int, ...]:
    """Sorted size-family output slots requested in ``names``."""
    return tuple(sorted({
        _SIZE_NAMES[family][name.removeprefix(family + "_")]
        for name in names if name.startswith(family + "_")
    }))


def _select(
    family: str, values: mx.array, slots: tuple[int, ...], names: tuple[str, ...],
) -> dict[str, mx.array]:
    """Map requested names to the rows of ``values``, which hold ``slots`` in order."""
    return {
        name: values[slots.index(_SIZE_NAMES[family][name.removeprefix(family + "_")])]
        for name in names if name.startswith(family + "_")
    }
