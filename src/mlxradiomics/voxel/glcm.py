"""Voxel-map GLCM features with one fused Metal pass per center and direction.

A voxel map's local neighborhood is small: a radius-1 cube has 27 voxels, so one
direction has at most 18 voxel pairs and 36 symmetric counts. The dense path builds
an ``Ng x Ng`` matrix per center and direction and evaluates every feature with
generic array operations over all ``Ng**2`` cells. Here one thread builds the
occupied cells of one center's matrix for one direction and evaluates the requested
ordinary features from them; a second thread per center averages the directions.

Arithmetic contracts of the dense path are retained or tightened:

* Counts, marginals, ``Sx``, ``Sy`` and every polynomial numerator that the dense path
  forms from exact integers are int64 here. Autocorrelation, JointAverage, Contrast,
  ClusterTendency, ClusterShade, Correlation, DifferenceAverage, DifferenceVariance,
  JointEnergy, MaximumProbability, SumAverage and SumSquares are one exact numerator
  divided by an exact power of ``N``, rounded close to once.
* ClusterShade keeps the exact ``Q = N**2 A3 - 3 f N A2 + 2 f**3`` numerator, so
  cancelling odd moments are exactly zero.
* Entropies keep the integer-deficit ``log1p`` form of ``count_entropy``, and mutual
  information keeps the Pearson-series/direct split of ``_mutual_information_bits``.
* Float sums use two-sum compensation; direction averages use the same two-sum order
  as ``CompensatedTotal``.
* MCC uses the same centered standardized residual and the same cyclic Jacobi
  rotations and convergence rule as :mod:`mlxradiomics.numerics.eigen`, on the occupied levels
  only. Unoccupied levels have zero rows and columns, which the rotation rule never
  touches. A symmetric GLCM has a symmetric residual, whose largest absolute
  eigenvalue is the largest singular value; the Gram square is formed only for
  asymmetric matrices.

Values are not bit-identical to the dense path, whose reduction trees run over the
zero cells as well. Eligibility is limited to neighborhoods of at most 27 voxels and
the native texture definitions; other settings use the dense path.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import cast

import numpy as np

from mlxradiomics.numerics.protocols import MetalKernel
from mlxradiomics.numerics.runtime import evaluate, mx

# Ordinary per-direction outputs, in kernel output order.
_ORDINARY = (
    "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
    "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage", "DifferenceEntropy",
    "DifferenceVariance", "JointEnergy", "JointEntropy", "Imc1", "Imc2", "Idm", "Idmn",
    "Id", "Idn", "InverseVariance", "MaximumProbability", "SumAverage", "SumEntropy",
    "SumSquares",
)
_ALIASES = {"Dissimilarity": "DifferenceAverage", "SumVariance": "ClusterTendency"}
MAX_WINDOW = 27
_INT64_LIMIT = 2**62


def eligible(window: int, pairs: tuple[int, ...], ng: int, symmetric: bool) -> bool:
    """Return whether the fused kernels support this neighborhood and gray-level count.

    ``pairs`` holds the number of local voxel pairs per direction. The int64 bounds
    cover the largest numerators: ``N**3 (2 Ng)**3`` for ClusterShade and
    ``N**3 (2 Ng)**2`` for the quadratic moments, where ``N`` is a direction's count.
    """
    if not pairs or window > MAX_WINDOW:
        return False
    total = max(pairs) * (2 if symmetric else 1)
    return total**3 * (2 * ng) ** 3 * 4 < _INT64_LIMIT and total**3 < 2**24


KERNEL_HEADER = r"""
constant float LN2 = 0.6931471805599453f;

struct Acc { float s; float e; };

inline void acc_add(thread Acc& a, float v) {
    float t = a.s + v;
    float back = t - a.s;
    a.e += (a.s - (t - back)) + (v - back);
    a.s = t;
}

inline float acc_value(Acc a) { return metal::isfinite(a.s) ? a.s + a.e : a.s; }

// Exact int64 numerator over an exact positive denominator (< 2**24), with the
// float conversion remainder and the division residual folded back in.
inline float q_div(long num, long den) {
    if (num == 0) return 0.0f;
    float d = float(den);
    float h = float(num);
    long rem = num - long(h);
    float q = h / d;
    float e = metal::fma(-q, d, h);
    return q + (e + float(rem)) / d;
}

// count * log2(count / total), the count_entropy summand, with the integer deficit
// in log1p when the state holds more than half of the mass.
inline float entropy_term(long count, long total) {
    float tf = float(total);
    float lp = count * 2 > total ? log1p(float(count - total) / tf) / LN2
                                 : metal::log2(float(count) / tf);
    return float(count) * lp;
}

// expm1 for x <= 0. The compiler may fold log(exp(x)) to x, which defeats the usual
// exp/log correction, so small arguments use the Taylor series directly.
inline float expm1_nonpositive(float x) {
    if (x < -0.5f) return metal::exp(x) - 1.0f;
    float p = 1.0f / 3628800.0f;
    p = p * x + 1.0f / 362880.0f;
    p = p * x + 1.0f / 40320.0f;
    p = p * x + 1.0f / 5040.0f;
    p = p * x + 1.0f / 720.0f;
    p = p * x + 1.0f / 120.0f;
    p = p * x + 1.0f / 24.0f;
    p = p * x + 1.0f / 6.0f;
    p = p * x + 0.5f;
    return x + x * x * p;
}

inline void add_cell(thread int* ci, thread int* cj, thread int* cc, thread int& n,
                     int i, int j) {
    for (int k = 0; k < n; k++) {
        if (ci[k] == i && cj[k] == j) { cc[k] += 1; return; }
    }
    ci[n] = i; cj[n] = j; cc[n] = 1; n++;
}

// Insert ``count`` at ``level`` into an ascending level list, merging an existing level.
inline void add_level(thread int* levels, thread long* counts, thread int& n,
                      int level, long count) {
    int k = 0;
    while (k < n && levels[k] < level) k++;
    if (k < n && levels[k] == level) { counts[k] += count; return; }
    for (int m = n; m > k; m--) { levels[m] = levels[m - 1]; counts[m] = counts[m - 1]; }
    levels[k] = level; counts[k] = count; n++;
}

inline int find_level(thread const int* levels, int n, int level) {
    for (int k = 0; k < n; k++) if (levels[k] == level) return k;
    return -1;
}

inline void add_key(thread long* keys, thread long* counts, thread int& n, long key, long c) {
    for (int k = 0; k < n; k++) {
        if (keys[k] == key) { counts[k] += c; return; }
    }
    keys[n] = key; counts[n] = c; n++;
}
"""


def _tables(
    local_shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...],
) -> tuple[str, list[int]]:
    """Metal constant tables: window offsets and per-direction local pair indexes."""
    axes = [range(-(s // 2), s // 2 + 1) if s > 1 else [0] for s in local_shape]
    window = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(local_shape), -1).T
    indexes = np.arange(prod(local_shape), dtype=np.int32).reshape(local_shape)
    left: list[int] = []
    right: list[int] = []
    starts = [0]
    counts: list[int] = []
    for delta in offsets:
        pairs = tuple(zip(local_shape, delta, strict=True))
        lo = tuple(slice(max(0, -v), min(s, s - v)) for s, v in pairs)
        hi = tuple(slice(max(0, v), min(s, s + v)) for s, v in pairs)
        a, b = indexes[lo].reshape(-1), indexes[hi].reshape(-1)
        left.extend(int(v) for v in a)
        right.extend(int(v) for v in b)
        starts.append(len(left))
        counts.append(len(a))

    def table(name: str, kind: str, values: list[int]) -> str:
        body = ", ".join(map(str, values or [0]))
        return f"constant {kind} {name}[{max(1, len(values))}] = {{{body}}};"

    source = "\n".join((
        table("WINDOW", "int", [int(v) for v in window.reshape(-1)]),
        table("PAIR_LEFT", "ushort", left),
        table("PAIR_RIGHT", "ushort", right),
        table("PAIR_START", "ushort", starts),
    ))
    return source, counts


def window_code(ndim: int, window: int) -> str:
    """Return Metal source loading a center's neighborhood gray levels into ``levels``.

    Level 0 marks a position outside the image or mask.
    """
    coordinate = " && ".join(
        f"(z{a} = center_at[{a}] + WINDOW[q * {ndim} + {a}]) >= 0 && z{a} < dims[{a}]"
        for a in range(ndim)
    )
    flat = " + ".join(
        f"long(z{a})" + "".join(f" * long(dims[{b}])" for b in range(a + 1, ndim))
        for a in range(ndim)
    )
    declarations = ", ".join(f"z{a}" for a in range(ndim))
    return f"""
        int center_at[{ndim}];
        for (int a = 0; a < {ndim}; a++) center_at[a] = centers[center * {ndim} + a];
        int levels[{window}];
        for (int q = 0; q < {window}; q++) {{
            int {declarations};
            bool inside = {coordinate};
            long flat = inside ? {flat} : 0;
            levels[q] = inside && mask[flat] ? bins[flat] : 0;
        }}
    """


def _cells_code(window: int, max_cells: int, symmetric: bool) -> str:
    """Return Metal source collecting one direction's occupied GLCM cells and marginals."""
    second = "add_cell(ci, cj, cc, nc, b, a);" if symmetric else ""
    return f"""
        int ci[{max_cells}], cj[{max_cells}], cc[{max_cells}];
        int nc = 0;
        for (int p = PAIR_START[d]; p < PAIR_START[d + 1]; p++) {{
            int a = levels[PAIR_LEFT[p]], b = levels[PAIR_RIGHT[p]];
            if (a > 0 && b > 0) {{ add_cell(ci, cj, cc, nc, a, b); {second} }}
        }}
        long n = 0;
        for (int k = 0; k < nc; k++) n += cc[k];
        int rl[{window}], cl[{window}];
        long rn[{window}], cn[{window}];
        int nr = 0, ncol = 0;
        for (int k = 0; k < nc; k++) {{
            add_level(rl, rn, nr, ci[k], cc[k]);
            add_level(cl, cn, ncol, cj[k], cc[k]);
        }}
    """


# Per-direction value of each _ORDINARY feature, by slot. Each expression reads only
# accumulators of its own feature, so dropping unrequested stores leaves the others'
# arithmetic unchanged.
_STORES = (
    "q_div(auto_num, n)",
    "q_div(sx, n)",
    "acc_value(prominence) / nf",
    "q_div(shade, n3)",
    "q_div(tendency_num, n3)",
    "q_div(contrast_num, n)",
    "(nr == 1 || ncol == 1) ? 1.0f\n"
    "                 : float(covariance) / metal::sqrt(float(vx) * float(vy))",
    "q_div(absolute_num, n)",
    "-acc_value(difference_entropy) / nf",
    "q_div(variance_num, n3)",
    "q_div(energy_num, n * n)",
    "-acc_value(joint_entropy) / nf",
    "denominator > 0 ? -information / denominator : 0.0f",
    "metal::sqrt(max(-expm1_nonpositive(-2.0f * information), 0.0f))",
    "acc_value(idm) / nf",
    "acc_value(idmn) / nf",
    "acc_value(id) / nf",
    "acc_value(idn) / nf",
    "acc_value(inverse) / nf",
    "float(maximum) / nf",
    "q_div(s, n)",
    "-acc_value(sum_entropy) / nf",
    "q_div(vx, n3)",
)
_SUM_SLOTS = {21}  # SumEntropy reads the i + j table.
_DIFFERENCE_SLOTS = {8, 9}  # DifferenceEntropy and DifferenceVariance read the |i - j| table.
_INFORMATION_SLOTS = {12, 13}  # Imc1 and Imc2 read the mutual information.


# Optional per-direction blocks, emitted only when a requested slot reads them.
_SUM_ENTROPY = r"""
Acc sum_entropy = {0, 0};
for (int k = 0; k < ns; k++) acc_add(sum_entropy, entropy_term(sum_counts[k], n));
"""
_DIFFERENCE_MOMENTS = r"""
Acc difference_entropy = {0, 0};
long variance_num = 0;
for (int k = 0; k < nd; k++) {
    acc_add(difference_entropy, entropy_term(diff_counts[k], n));
    long centered = n * diff_keys[k] - absolute_num;
    variance_num += diff_counts[k] * centered * centered;
}
"""
_MUTUAL_INFORMATION = r"""
// Mutual information, as in _mutual_information_bits.
bool full = nc == nr * ncol;
float largest = 0.0f;
if (full) {
    for (int k = 0; k < nc; k++) {
        long r = rn[find_level(rl, nr, ci[k])], q = cn[find_level(cl, ncol, cj[k])];
        float ratio = float(long(cc[k]) * n - r * q) / (float(r) * float(q));
        largest = max(largest, metal::abs(ratio));
    }
}
float information;
if (full && largest <= 0.5f) {
    Acc linear = {0, 0}, correction = {0, 0};
    for (int k = 0; k < nc; k++) {
        long r = rn[find_level(rl, nr, ci[k])], q = cn[find_level(cl, ncol, cj[k])];
        float residual = float(long(cc[k]) * n - r * q);
        acc_add(linear, (residual / float(r)) * (residual / float(q)));
        float ratio = residual / (float(r) * float(q));
        float polynomial = 0.0f;
        for (int m = 24; m > 1; m--) {
            float coefficient = (m % 2) ? 1.0f / float(m) : -1.0f / float(m);
            polynomial = ratio * polynomial + coefficient;
        }
        acc_add(correction, float(cc[k]) * (ratio * ratio * polynomial));
    }
    information = acc_value(linear) / (nf * nf) + acc_value(correction) / nf;
} else {
    Acc direct = {0, 0};
    for (int k = 0; k < nc; k++) {
        long r = rn[find_level(rl, nr, ci[k])], q = cn[find_level(cl, ncol, cj[k])];
        float denominator = float(r) * float(q);
        float ratio = float(long(cc[k]) * n - r * q) / denominator;
        float quotient = float(cc[k]) * nf / denominator;
        float log_ratio = quotient > 0.5f ? log1p(ratio)
                                          : metal::log(max(quotient, 1e-38f));
        acc_add(direct, float(cc[k]) * log_ratio);
    }
    information = acc_value(direct) / nf;
}
information = information / LN2;
"""


@lru_cache(maxsize=64)
def _direction_kernel(
    local_shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], symmetric: bool,
    slots: tuple[int, ...],
) -> MetalKernel:
    """Per-direction kernel writing only the _ORDINARY features at ``slots``, in order.

    The sum/difference tables and the mutual-information block are emitted only for
    slots that read them; the compiler removes the other unused scalar accumulators.
    """
    tables, counts = _tables(local_shape, offsets)
    window = prod(local_shape)
    ndim = len(local_shape)
    directions = len(offsets)
    max_cells = max(counts) * (2 if symmetric else 1)
    features = len(slots)
    sums = bool(_SUM_SLOTS & set(slots))
    differences = bool(_DIFFERENCE_SLOTS & set(slots))
    information = bool(_INFORMATION_SLOTS & set(slots))
    sum_tables = (f"long sum_keys[{max_cells}], sum_counts[{max_cells}];\n        int ns = 0;"
                  if sums else "")
    difference_tables = (
        f"long diff_keys[{max_cells}], diff_counts[{max_cells}];\n        int nd = 0;"
        if differences else ""
    )
    sum_insert = "add_key(sum_keys, sum_counts, ns, i + j, c);" if sums else ""
    difference_insert = "add_key(diff_keys, diff_counts, nd, delta, c);" if differences else ""
    stores = "\n".join(
        f"        out[{column}] = {_STORES[slot]};" for column, slot in enumerate(slots)
    )
    source = f"""
        uint t = thread_position_in_grid.x;
        uint total_centers = centers_shape[0];
        if (t >= total_centers * {directions}) return;
        uint center = t / {directions};
        int d = int(t % {directions});
        {window_code(ndim, window)}
        {_cells_code(window, max_cells, symmetric)}
        active[t] = int(n);
        if (n == 0) return;
        long ng = long(gray_levels[0]);
        float nf = float(n);
        long sx = 0, sy = 0;
        for (int k = 0; k < nr; k++) sx += rn[k] * long(rl[k]);
        for (int k = 0; k < ncol; k++) sy += cn[k] * long(cl[k]);
        long s = sx + sy;
        long floor_mean = s / n;
        long f = s - n * floor_mean;
        long auto_num = 0, contrast_num = 0, absolute_num = 0, energy_num = 0, maximum = 0;
        long tendency_num = 0, covariance = 0, a2 = 0, a3 = 0;
        Acc prominence = {{0, 0}}, id = {{0, 0}}, idm = {{0, 0}}, idn = {{0, 0}};
        Acc idmn = {{0, 0}}, inverse = {{0, 0}}, joint_entropy = {{0, 0}};
        float ngf = float(ng), ng2f = float(ng * ng);
        {sum_tables}
        {difference_tables}
        for (int k = 0; k < nc; k++) {{
            long i = ci[k], j = cj[k], c = cc[k];
            long delta = i > j ? i - j : j - i;
            auto_num += c * i * j;
            contrast_num += c * delta * delta;
            absolute_num += c * delta;
            energy_num += c * c;
            maximum = max(maximum, c);
            long cluster = n * (i + j) - s;
            tendency_num += c * cluster * cluster;
            covariance += c * (n * i - sx) * (n * j - sy);
            long a = i + j - floor_mean;
            a2 += c * a * a;
            a3 += c * a * a * a;
            float cluster_f = q_div(cluster, n);
            float square = cluster_f * cluster_f;
            float cf = float(c);
            acc_add(prominence, cf * (square * square));
            acc_add(id, cf / float(1 + delta));
            acc_add(idm, cf / float(1 + delta * delta));
            acc_add(idn, cf * (ngf / float(ng + delta)));
            acc_add(idmn, cf * (ng2f / float(ng * ng + delta * delta)));
            if (delta > 0) acc_add(inverse, cf / float(delta * delta));
            acc_add(joint_entropy, entropy_term(c, n));
            {sum_insert}
            {difference_insert}
        }}
        long vx = 0, vy = 0;
        Acc hx = {{0, 0}}, hy = {{0, 0}};
        for (int k = 0; k < nr; k++) {{
            long offset = n * long(rl[k]) - sx;
            vx += rn[k] * offset * offset;
            acc_add(hx, entropy_term(rn[k], n));
        }}
        for (int k = 0; k < ncol; k++) {{
            long offset = n * long(cl[k]) - sy;
            vy += cn[k] * offset * offset;
            acc_add(hy, entropy_term(cn[k], n));
        }}
        {_SUM_ENTROPY if sums else ''}
        {_DIFFERENCE_MOMENTS if differences else ''}
        {_MUTUAL_INFORMATION if information else ''}
        float hxf = -acc_value(hx) / nf, hyf = -acc_value(hy) / nf;
        float denominator = max(hxf, hyf);
        long n3 = n * n * n;
        long shade = n * n * a3 - 3 * f * n * a2 + 2 * f * f * f;
        device float* out = values + ulong(t) * {features};
{stores}
    """
    kernel = mx.fast.metal_kernel(
        name="mlxradiomics_local_glcm",
        input_names=["bins", "mask", "centers", "dims", "gray_levels"],
        output_names=["values", "active"],
        header=KERNEL_HEADER + tables,
        source=source,
    )
    return cast(MetalKernel, kernel)


@lru_cache(maxsize=16)
def _mcc_kernel(
    local_shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...], symmetric: bool,
) -> MetalKernel:
    """Compile the per-center, per-direction GLCM MCC kernel.

    Symmetric counts give a symmetric residual whose largest absolute eigenvalue is
    MCC; asymmetric counts use its Gram matrix, as in ``eigen.maximal_correlation``.
    """
    tables, counts = _tables(local_shape, offsets)
    window = prod(local_shape)
    ndim = len(local_shape)
    directions = len(offsets)
    max_cells = max(counts) * (2 if symmetric else 1)
    # Symmetric counts give a symmetric residual R; MCC is its largest |eigenvalue|.
    # Asymmetric counts use the Gram matrix R R^T, as in eigen.maximal_correlation.
    matrix = "" if symmetric else f"float gram[{window * window}];"
    if symmetric:
        target = "residual"
        prepare = ""
    else:
        target = "gram"
        prepare = f"""
            for (int a = 0; a < nr; a++) for (int b = a; b < nr; b++) {{
                float total = 0.0f;
                for (int k = 0; k < ncol; k++)
                    total += residual[a * {window} + k] * residual[b * {window} + k];
                gram[a * {window} + b] = gram[b * {window} + a] = total;
            }}
        """
    source = f"""
        uint t = thread_position_in_grid.x;
        uint total_centers = centers_shape[0];
        if (t >= total_centers * {directions}) return;
        uint center = t / {directions};
        int d = int(t % {directions});
        {window_code(ndim, window)}
        {_cells_code(window, max_cells, symmetric)}
        converged[t] = true;
        if (n == 0 || nr < 2 || ncol < 2) {{ mcc[t] = NAN; return; }}
        float residual[{window * window}];
        {matrix}
        float nf = float(n);
        for (int a = 0; a < nr; a++) for (int b = 0; b < ncol; b++) {{
            long count = 0;
            for (int k = 0; k < nc; k++) if (ci[k] == rl[a] && cj[k] == cl[b]) count = cc[k];
            float numerator = float(count * n - rn[a] * cn[b]);
            residual[a * {window} + b] =
                numerator / (nf * metal::sqrt(float(rn[a]) * float(cn[b])));
        }}
        {prepare}
        thread float* work = {target};
        int size = nr;
        bool done = false;
        for (int sweep = 0; sweep < 100; sweep++) {{
            done = true;
            for (int p = 0; p < size; p++) {{
                for (int q = p + 1; q < size; q++) {{
                    float apq = work[p * {window} + q];
                    float app = work[p * {window} + p];
                    float aqq = work[q * {window} + q];
                    float threshold = (1e-7f * metal::sqrt(metal::abs(app)))
                                      * metal::sqrt(metal::abs(aqq));
                    if (metal::abs(apq) <= threshold) continue;
                    done = false;
                    float delta = 0.5f * aqq - 0.5f * app;
                    float scale = max(metal::abs(delta), metal::abs(apq));
                    float dd = delta / scale, off = apq / scale;
                    float tt = metal::copysign(1.0f, delta) * off /
                               (metal::abs(dd) + metal::sqrt(dd * dd + off * off));
                    float c = metal::rsqrt(1 + tt * tt), s = tt * c;
                    for (int k = 0; k < size; k++) {{
                        if (k == p || k == q) continue;
                        float akp = work[k * {window} + p];
                        float akq = work[k * {window} + q];
                        work[k * {window} + p] = work[p * {window} + k] = c * akp - s * akq;
                        work[k * {window} + q] = work[q * {window} + k] = s * akp + c * akq;
                    }}
                    work[p * {window} + p] = app - tt * apq;
                    work[q * {window} + q] = aqq + tt * apq;
                    work[p * {window} + q] = work[q * {window} + p] = 0;
                }}
            }}
            if (done) break;
        }}
        float largest = 0.0f;
        for (int p = 0; p < size; p++) {{
            float value = work[p * {window} + p];
            largest = max(largest, {"metal::abs(value)" if symmetric else "value"});
        }}
        mcc[t] = {"largest" if symmetric else "metal::sqrt(max(largest, 0.0f))"};
        converged[t] = done;
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_mcc",
        input_names=["bins", "mask", "centers", "dims"],
        output_names=["mcc", "converged"],
        header=KERNEL_HEADER + tables,
        source=source,
    ))


@lru_cache(maxsize=16)
def _average_kernel(directions: int, features: int) -> MetalKernel:
    """Average per-direction features with two-sum running totals.

    Directions are added in order, as ``CompensatedTotal`` does.
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_glcm_average",
        input_names=["values", "active"],
        output_names=["averaged"],
        header=KERNEL_HEADER,
        source=f"""
        uint center = thread_position_in_grid.x;
        uint total_centers = active_shape[0] / {directions};
        if (center >= total_centers) return;
        int angles = 0;
        for (int d = 0; d < {directions}; d++) angles += active[center * {directions} + d] > 0;
        for (int f = 0; f < {features}; f++) {{
            Acc total = {{0, 0}};
            for (int d = 0; d < {directions}; d++) {{
                uint t = center * {directions} + d;
                if (active[t] > 0) acc_add(total, values[ulong(t) * {features} + f]);
            }}
            averaged[ulong(f) * total_centers + center] =
                angles > 0 ? acc_value(total) / float(angles) : NAN;
        }}
        """,
    ))


@lru_cache(maxsize=16)
def _mcc_average_kernel(directions: int) -> MetalKernel:
    """Average MCC over directions.

    A direction without a valid pair, or with one occupied level, is NaN and skipped.
    """
    return cast(MetalKernel, mx.fast.metal_kernel(
        name="mlxradiomics_local_mcc_average",
        input_names=["mcc"],
        output_names=["averaged"],
        source=f"""
        uint center = thread_position_in_grid.x;
        uint total_centers = mcc_shape[0] / {directions};
        if (center >= total_centers) return;
        float total = 0.0f;
        int defined = 0;
        for (int d = 0; d < {directions}; d++) {{
            float value = mcc[center * {directions} + d];
            if (!metal::isnan(value)) {{ total += value; defined++; }}
        }}
        averaged[center] = defined > 0 ? total / float(defined) : NAN;
        """,
    ))


def local_glcm(
    bins: mx.array,
    mask: mx.array,
    image_shape: tuple[int, ...],
    centers: mx.array,
    local_shape: tuple[int, ...],
    offsets: tuple[tuple[int, ...], ...],
    gray_levels: int,
    symmetric: bool,
    names: tuple[str, ...],
) -> dict[str, mx.array]:
    """Return direction-averaged GLCM features for each center's local neighborhood.

    Args:
        bins: Flat int32 gray levels, 0 outside the sampling mask.
        mask: Flat boolean sampling mask.
        image_shape: Image shape in array-axis order.
        centers: Int32 ``(batch, ndim)`` center coordinates.
        local_shape: Neighborhood shape; odd sizes, or 1 along a ``force2d_axis``.
        offsets: Pair directions within ``local_shape`` (``preprocessing.directions``).
        gray_levels: Global gray-level count ``Ng``, used by Idn and Idmn.
        symmetric: Add each direction's counts to their transpose.
        names: GLCM feature names without the ``glcm_`` prefix.

    Returns:
        ``glcm_<name>`` float32 arrays ``(batch,)``. A center with no valid pair in any
        direction is NaN; MCC averages only directions where it is defined.

    Raises:
        RuntimeError: If an MCC Jacobi solve has not converged after 100 sweeps.
    """
    batch = centers.shape[0]
    directions = len(offsets)
    dims = mx.array(image_shape, dtype=mx.int32)
    result: dict[str, mx.array] = {}
    ordinary = tuple(name for name in names if name != "MCC")
    if ordinary:
        slots = tuple(
            sorted({_ORDINARY.index(_ALIASES.get(name, name)) for name in ordinary})
        )
        kernel = _direction_kernel(local_shape, offsets, symmetric, slots)
        values, active = kernel(
            inputs=[bins, mask, centers, dims, mx.array([gray_levels], dtype=mx.int32)],
            grid=(batch * directions, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(batch * directions, len(slots)), (batch * directions,)],
            output_dtypes=[mx.float32, mx.int32],
        )
        averaged = _average_kernel(directions, len(slots))(
            inputs=[values, active], grid=(batch, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(len(slots), batch)], output_dtypes=[mx.float32],
        )[0]
        for name in ordinary:
            slot = _ORDINARY.index(_ALIASES.get(name, name))
            result[f"glcm_{name}"] = averaged[slots.index(slot)]
    if "MCC" in names:
        mcc, converged = _mcc_kernel(local_shape, offsets, symmetric)(
            inputs=[bins, mask, centers, dims],
            grid=(batch * directions, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(batch * directions,), (batch * directions,)],
            output_dtypes=[mx.float32, mx.bool_],
        )
        result["glcm_MCC"] = _mcc_average_kernel(directions)(
            inputs=[mcc], grid=(batch, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(batch,)], output_dtypes=[mx.float32],
        )[0]
        all_converged = mx.all(converged)
        evaluate(result["glcm_MCC"], all_converged)
        if not bool(all_converged.item()):
            raise RuntimeError("Metal Jacobi eigenvalue calculation did not converge")
    return result
