"""Small-array acceptance oracles independent of the Metal implementations.

Counts are enumerated with Python integers. Scalar formulas use binary64 and
``fsum``; geometry uses analytic polygons or scikit-image's classic marching
cubes. These are test references, never runtime feature backends.
"""

from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from itertools import product
from math import expm1, fsum, log2, nan, pi, prod, sqrt

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from radiomics_reference.mathematics import glrlm_features, ngtdm_features

type Index = tuple[int, ...]


def entropy(probabilities: Iterable[float]) -> float:
    """Evaluate Shannon entropy on positive support, in bits."""
    return -fsum(p * log2(p) for p in probabilities if p > 0)


def _quantile_fraction(
    values: list[float], numerator: int, denominator: int,
) -> Fraction:
    """Return the exactly interpolated order statistic at ``(n-1) * p / q``."""
    ordered = sorted(values)
    position = Fraction((len(ordered) - 1) * numerator, denominator)
    lower = position.numerator // position.denominator
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    left = Fraction.from_float(ordered[lower])
    right = Fraction.from_float(ordered[upper])
    return left + (right - left) * fraction


def _ieee_ratio(numerator: float, denominator: float) -> float:
    """Divide with IEEE-754 zero semantics instead of raising ``ZeroDivisionError``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def firstorder(
    image: NDArray[np.float64], bins: NDArray[np.int32], mask: NDArray[np.bool_],
    shift: float, voxel_volume: float, kurtosis: str = "raw", zero_variance_moments: str = "nan",
) -> dict[str, float]:
    """Evaluate source-intensity statistics with exact rational centering.

    The mean offset is formed relative to one observed source value.  Moments and
    quantiles then use :class:`fractions.Fraction`, so a rounded absolute mean cannot
    corrupt the centered moments of two distinct values that differ by one float64 ulp.
    """
    if kurtosis not in {"raw", "excess"}:
        raise ValueError("kurtosis must be raw or excess")
    if zero_variance_moments not in {"nan", "zero"}:
        raise ValueError("zero_variance_moments must be nan or zero")
    values = [float(x) for x in image[mask]]
    n = len(values)
    fractions = [Fraction.from_float(x) for x in values]
    origin = fractions[0]
    offsets = [x - origin for x in fractions]
    mean_offset = sum(offsets, Fraction()) / n
    mean_fraction = origin + mean_offset
    centered = [x - mean_offset for x in offsets]
    variance_fraction = sum((x * x for x in centered), Fraction()) / n
    third_fraction = sum((x**3 for x in centered), Fraction()) / n
    fourth_fraction = sum((x**4 for x in centered), Fraction()) / n
    variance = float(variance_fraction)
    q10 = _quantile_fraction(values, 1, 10)
    q25 = _quantile_fraction(values, 1, 4)
    q50 = _quantile_fraction(values, 1, 2)
    q75 = _quantile_fraction(values, 3, 4)
    q90 = _quantile_fraction(values, 9, 10)
    robust = [x for x in fractions if q10 <= x <= q90]
    if robust:
        robust_mean = sum(robust, Fraction()) / len(robust)
        robust_deviation = (
            sum((abs(x - robust_mean) for x in robust), Fraction()) / len(robust)
        )
        robust_value = float(robust_deviation)
    else:
        robust_value = nan
    mean_deviation = sum((abs(x - mean_fraction) for x in fractions), Fraction()) / n
    median_deviation = sum((abs(x - q50) for x in fractions), Fraction()) / n
    if variance_fraction == 0:
        skewness = nan if zero_variance_moments == "nan" else 0.0
        kurtosis_value = nan if zero_variance_moments == "nan" else 0.0
    else:
        # Form dimensionless ratios before binary64 conversion: the dimensional
        # variance can underflow even when standardized moments are representable.
        skewness = sqrt(float(third_fraction**2 / variance_fraction**3))
        if third_fraction < 0:
            skewness = -skewness
        kurtosis_value = float(fourth_fraction / variance_fraction**2)
        if kurtosis == "excess":
            kurtosis_value -= 3.0
    energy = fsum((x + shift)**2 for x in values)
    probabilities = [count / n for count in Counter(int(x) for x in bins[mask]).values()]
    return {
        "Mean": float(mean_fraction), "Minimum": float(min(fractions)),
        "Maximum": float(max(fractions)),
        "Range": float(max(fractions) - min(fractions)), "Variance": variance,
        "Skewness": skewness, "Kurtosis": kurtosis_value,
        "Energy": energy, "TotalEnergy": energy * voxel_volume,
        "RootMeanSquared": sqrt(energy / n), "Entropy": entropy(probabilities),
        "Uniformity": fsum(p * p for p in probabilities),
        "10Percentile": float(q10), "90Percentile": float(q90),
        "Median": float(q50),
        "InterquartileRange": float(q75 - q25),
        "MeanAbsoluteDeviation": float(mean_deviation),
        "MedianAbsoluteDeviation": float(median_deviation),
        "CoefficientOfVariation": _ieee_ratio(sqrt(variance), float(mean_fraction)),
        "QuartileCoefficientOfDispersion": _ieee_ratio(
            float(q75 - q25), float(q75 + q25)
        ),
        "RobustMeanAbsoluteDeviation": robust_value,
    }


def offsets(ndim: int, distances: tuple[int, ...], axis: int | None) -> list[Index]:
    """Enumerate the full Chebyshev shells without importing production directions."""
    radius = max(distances)
    return [d for d in product(range(-radius, radius + 1), repeat=ndim)
            if max(map(abs, d)) in distances and (axis is None or d[axis] == 0)]


def neighbors(index: Index, shape: tuple[int, ...], directions: list[Index]) -> Iterable[Index]:
    """Yield only in-array neighbors; masks are applied by each counting definition."""
    for delta in directions:
        other = tuple(i + d for i, d in zip(index, delta, strict=True))
        if all(0 <= i < size for i, size in zip(other, shape, strict=True)):
            yield other


def glcm_matrix(counts: dict[tuple[int, int], int], ng: int) -> dict[str, float]:
    """Evaluate all 24 definitions from one independently counted contingency table."""
    total = sum(counts.values())
    p = {key: count / total for key, count in counts.items()}
    px: Counter[int] = Counter()
    py: Counter[int] = Counter()
    for (i, j), count in counts.items():
        px[i] += count
        py[j] += count
    ux = fsum(i * count for i, count in px.items()) / total
    uy = fsum(j * count for j, count in py.items()) / total
    vx = fsum((i - ux)**2 * count for i, count in px.items()) / total
    vy = fsum((j - uy)**2 * count for j, count in py.items()) / total
    diff: dict[int, float] = {}
    sums: dict[int, float] = {}
    for (i, j), probability in p.items():
        diff[abs(i - j)] = diff.get(abs(i - j), 0) + probability
        sums[i + j] = sums.get(i + j, 0) + probability
    da = fsum(k * probability for k, probability in diff.items())
    cluster_tendency = fsum(v * (i + j - ux - uy)**2 for (i, j), v in p.items())
    hx, hy = entropy(c / total for c in px.values()), entropy(c / total for c in py.values())
    information = fsum(probability * log2(counts[i, j] * total / (px[i] * py[j]))
                       for (i, j), probability in p.items())
    mcc = nan
    if len(px) >= 2 and len(py) >= 2:
        # Singular values evaluate the definition independently of the production
        # Gram eigenproblem. Zero marginal rows/columns are excluded explicitly.
        normalized = np.array([[counts.get((i, j), 0) / sqrt(px[i] * py[j])
                                for j in sorted(py)] for i in sorted(px)])
        mcc = float(np.linalg.svd(normalized, compute_uv=False)[1])
    return {
        "Autocorrelation": fsum(v * i * j for (i, j), v in p.items()),
        "Contrast": fsum(v * (i - j)**2 for (i, j), v in p.items()),
        "Correlation": (1.0 if len(px) == 1 or len(py) == 1 else
                        fsum(v * (i - ux) * (j - uy) for (i, j), v in p.items()) / sqrt(vx * vy)),
        "DifferenceAverage": da, "Dissimilarity": da,
        "JointEnergy": fsum(v * v for v in p.values()),
        "JointEntropy": entropy(p.values()),
        "Id": fsum(v / (1 + abs(i - j)) for (i, j), v in p.items()),
        "Idm": fsum(v / (1 + (i - j)**2) for (i, j), v in p.items()),
        "JointAverage": ux,
        "ClusterProminence": fsum(v * (i + j - ux - uy)**4 for (i, j), v in p.items()),
        "ClusterShade": fsum(v * (i + j - ux - uy)**3 for (i, j), v in p.items()),
        "ClusterTendency": cluster_tendency, "SumVariance": cluster_tendency,
        "DifferenceEntropy": entropy(diff.values()),
        "DifferenceVariance": fsum(v * (k - da)**2 for k, v in diff.items()),
        "Imc1": -information / max(hx, hy) if max(hx, hy) else 0,
        "Imc2": sqrt(-expm1(-2 * max(information, 0))),
        "Idmn": fsum(v / (1 + (i - j)**2 / ng**2) for (i, j), v in p.items()),
        "Idn": fsum(v / (1 + abs(i - j) / ng) for (i, j), v in p.items()),
        "InverseVariance": fsum(v / (i - j)**2 for (i, j), v in p.items() if i != j),
        "MaximumProbability": max(p.values()),
        "SumAverage": fsum(k * v for k, v in sums.items()), "SumEntropy": entropy(sums.values()),
        "SumSquares": vx, "MCC": mcc,
    }


def glcm(
    bins: NDArray[np.int32], mask: NDArray[np.bool_], settings: Settings, ng: int,
) -> dict[str, float]:
    """Average directional features, excluding empty pairs and undefined MCC values."""
    by_name: dict[str, list[float]] = {}
    for delta in offsets(bins.ndim, settings.distances, settings.force2d_axis):
        if next(x for x in delta if x) < 0:
            continue
        counts: Counter[tuple[int, int]] = Counter()
        for index in np.ndindex(bins.shape):
            if mask[index]:
                for other in neighbors(index, bins.shape, [delta]):
                    if mask[other]:
                        i, j = int(bins[index]), int(bins[other])
                        counts[i, j] += 1
                        if settings.symmetrical_glcm:
                            counts[j, i] += 1
        if counts:
            for name, value in glcm_matrix(counts, ng).items():
                by_name.setdefault(name, []).append(value)
    if not by_name:
        return dict.fromkeys(glcm_matrix({(1, 1): 1}, ng), nan)
    return {name: fsum(defined) / len(defined) if defined else nan
            for name, values in by_name.items()
            for defined in [[v for v in values if not np.isnan(v)]]}


def size_features(
    counts: Counter[tuple[int, int]], family: str, *, voxel_count: int | None = None,
) -> dict[str, float]:
    """Evaluate GLDM, GLSZM or GLDZM definitions from explicit matrix counts.

    ``family="gldzm"`` treats the second key as a zone distance and therefore requires
    an explicit ``voxel_count`` for ``ZonePercentage``; ``distance * zones`` is not the
    number of voxels.
    """
    if family not in {"gldm", "glszm", "gldzm"}:
        raise ValueError(f"unsupported size family: {family}")
    total = sum(counts.values())
    p = {key: count / total for key, count in counts.items()}
    gray: Counter[int] = Counter()
    size: Counter[int] = Counter()
    for (i, j), count in counts.items():
        gray[i] += count
        size[j] += count
    ui = fsum(i * c for i, c in gray.items()) / total
    uj = fsum(j * c for j, c in size.items()) / total
    dependence = family == "gldm"
    distance_zone = family == "gldzm"
    small, large = (("SmallDependence", "LargeDependence") if dependence
                    else ("SmallDistance", "LargeDistance") if distance_zone
                    else ("SmallArea", "LargeArea"))
    group = "Dependence" if dependence else "ZoneDistance" if distance_zone else "SizeZone"
    low, high = (("LowGrayLevel", "HighGrayLevel") if dependence
                 else ("LowGrayLevelZone", "HighGrayLevelZone"))
    variance = ("DependenceVariance" if dependence
                else "ZoneDistanceVariance" if distance_zone else "ZoneVariance")
    entropy_name = ("DependenceEntropy" if dependence
                    else "ZoneDistanceEntropy" if distance_zone else "ZoneEntropy")
    result = {
        small + "Emphasis": fsum(v / j**2 for (_, j), v in p.items()),
        large + "Emphasis": fsum(v * j**2 for (_, j), v in p.items()),
        "GrayLevelNonUniformity": fsum(c * c for c in gray.values()) / total,
        "GrayLevelNonUniformityNormalized":
            fsum(c * c for c in gray.values()) / total**2,
        group + "NonUniformity": fsum(c * c for c in size.values()) / total,
        group + "NonUniformityNormalized": fsum(c * c for c in size.values()) / total**2,
        "GrayLevelVariance": fsum(v * (i - ui)**2 for (i, _), v in p.items()),
        variance: fsum(v * (j - uj)**2 for (_, j), v in p.items()),
        entropy_name: entropy(p.values()),
        low + "Emphasis": fsum(v / i**2 for (i, _), v in p.items()),
        high + "Emphasis": fsum(v * i**2 for (i, _), v in p.items()),
        small + "LowGrayLevelEmphasis": fsum(v / (i * j)**2 for (i, j), v in p.items()),
        small + "HighGrayLevelEmphasis": fsum(v * i**2 / j**2 for (i, j), v in p.items()),
        large + "LowGrayLevelEmphasis": fsum(v * j**2 / i**2 for (i, j), v in p.items()),
        large + "HighGrayLevelEmphasis": fsum(v * (i * j)**2 for (i, j), v in p.items()),
    }
    if dependence:
        # The GLDM matrix contains one center per intensity-mask voxel, so N_s == N_v.
        result["DependenceCountEnergy"] = fsum(v * v for v in p.values())
        result["DependencePercentage"] = 1.0
    elif distance_zone:
        if voxel_count is None:
            raise ValueError("gldzm requires voxel_count")
        result["ZonePercentage"] = total / voxel_count
    else:
        result["ZonePercentage"] = total / sum(j * c for (_, j), c in counts.items())
    return result


def _histogram_percentile(
    counts: Counter[int], ng: int, numerator: int, denominator: int,
) -> float:
    """Interpolate an order statistic of a discrete histogram by exact rank."""
    total = sum(counts.values())
    position = Fraction((total - 1) * numerator, denominator)
    lower = position.numerator // position.denominator
    fraction = float(position - lower)
    upper = min(lower + 1, total - 1)

    def value_at(index: int) -> int:
        cumulative = 0
        for level in range(1, ng + 1):
            cumulative += counts.get(level, 0)
            if cumulative > index:
                return level
        return ng

    left, right = value_at(lower), value_at(upper)
    return left + (right - left) * fraction


def _histogram_gradients(counts: Counter[int], ng: int) -> list[Fraction]:
    """Return H' with exact halves; endpoints use one-sided differences."""
    if ng == 1:
        return [Fraction(0)]
    values = [counts.get(level, 0) for level in range(1, ng + 1)]
    doubled = [2 * (values[1] - values[0])]
    doubled.extend(values[i + 1] - values[i - 1] for i in range(1, ng - 1))
    doubled.append(2 * (values[-1] - values[-2]))
    return [Fraction(value, 2) for value in doubled]


def histogram_features(
    bins: NDArray[np.int32], mask: NDArray[np.bool_], ng: int,
) -> dict[str, float]:
    """Evaluate the 23 IBSI intensity-histogram definitions independently."""
    counts: Counter[int] = Counter(int(x) for x in bins[mask])
    total = sum(counts.values())
    p = {level: count / total for level, count in counts.items()}
    mean = fsum(level * probability for level, probability in p.items())
    variance = fsum((level - mean)**2 * probability for level, probability in p.items())
    third = fsum((level - mean)**3 * probability for level, probability in p.items())
    fourth = fsum((level - mean)**4 * probability for level, probability in p.items())
    q10 = _histogram_percentile(counts, ng, 1, 10)
    q25 = _histogram_percentile(counts, ng, 1, 4)
    q50 = _histogram_percentile(counts, ng, 1, 2)
    q75 = _histogram_percentile(counts, ng, 3, 4)
    q90 = _histogram_percentile(counts, ng, 9, 10)
    retained = {
        level: count for level, count in counts.items() if q10 <= level <= q90
    }
    retained_total = sum(retained.values())
    if retained_total:
        retained_mean = fsum(level * count for level, count in retained.items()) / retained_total
        robust = fsum(
            abs(level - retained_mean) * count for level, count in retained.items()
        ) / retained_total
    else:
        robust = nan
    max_count = max(counts.values())
    candidates = [level for level in range(1, ng + 1) if counts.get(level, 0) == max_count]
    mode = min(candidates, key=lambda level: (abs(level - mean), level))
    gradient = _histogram_gradients(counts, ng)
    maximum_gradient = max(gradient)
    minimum_gradient = min(gradient)
    return {
        "Mean": mean,
        "Variance": variance,
        "Skewness": 0.0 if variance == 0 else third / variance**1.5,
        "Kurtosis": 0.0 if variance == 0 else fourth / variance**2 - 3,
        "Median": q50,
        "Minimum": min(counts),
        "10Percentile": q10,
        "90Percentile": q90,
        "Maximum": max(counts),
        "Mode": float(mode),
        "InterquartileRange": q75 - q25,
        "Range": float(max(counts) - min(counts)),
        "MeanAbsoluteDeviation": fsum(
            abs(level - mean) * probability for level, probability in p.items()
        ),
        "RobustMeanAbsoluteDeviation": robust,
        "MedianAbsoluteDeviation": fsum(
            abs(level - q50) * probability for level, probability in p.items()
        ),
        "CoefficientOfVariation": _ieee_ratio(sqrt(variance), mean),
        "QuartileCoefficientOfDispersion": _ieee_ratio(q75 - q25, q75 + q25),
        "Entropy": entropy(p.values()),
        "Uniformity": fsum(probability * probability for probability in p.values()),
        "MaximumGradient": float(maximum_gradient),
        "MaximumGradientGrayLevel": float(1 + gradient.index(maximum_gradient)),
        "MinimumGradient": float(minimum_gradient),
        "MinimumGradientGrayLevel": float(1 + gradient.index(minimum_gradient)),
    }


def dependence_zones(
    bins: NDArray[np.int32], mask: NDArray[np.bool_], settings: Settings,
) -> dict[str, float]:
    """Count all dependence centers and flood-fill equal-level unit-distance zones."""
    dependence: Counter[tuple[int, int]] = Counter()
    adjacent = offsets(bins.ndim, settings.distances, settings.force2d_axis)
    remaining = {index for index in np.ndindex(bins.shape) if mask[index]}
    for index in remaining:
        level = int(bins[index])
        size = 1 + sum(mask[other] and abs(int(bins[other]) - level) <= settings.gldm_alpha
                       for other in neighbors(index, bins.shape, adjacent))
        dependence[level, int(size)] += 1
    zones: Counter[tuple[int, int]] = Counter()
    connected = offsets(bins.ndim, (1,), settings.force2d_axis)
    while remaining:
        start = remaining.pop()
        level, size, frontier = int(bins[start]), 1, [start]
        while frontier:
            for other in neighbors(frontier.pop(), bins.shape, connected):
                if other in remaining and int(bins[other]) == level:
                    remaining.remove(other)
                    frontier.append(other)
                    size += 1
        zones[level, size] += 1
    return {**{f"gldm_{k}": v for k, v in size_features(dependence, "gldm").items()},
            **{f"glszm_{k}": v for k, v in size_features(zones, "glszm").items()}}


def intensity_features(
    image: NDArray[np.float64], bins: NDArray[np.int32], mask: NDArray[np.bool_],
    settings: Settings, spacing: tuple[float, ...], ng: int,
) -> dict[str, float]:
    """Return supported intensity definitions for a sampling mask and global bins."""
    families = {
        "firstorder": firstorder(
            image, bins, mask, settings.voxel_array_shift, prod(spacing),
            settings.kurtosis, settings.zero_variance_moments,
        ),
        "glcm": glcm(bins, mask, settings, ng),
        "glrlm": glrlm_features(bins, mask, settings.force2d_axis),
        "ngtdm": ngtdm_features(
            bins, mask, settings.distances, settings.force2d_axis,
            ibsi_caps=settings.ngtdm_singular_cases == "ibsi-caps",
        ),
        "intensityhistogram": histogram_features(bins, mask, ng),
    }
    return {**{f"{family}_{name}": value for family, values in families.items()
               for name, value in values.items()}, **dependence_zones(bins, mask, settings)}


def shape_features(
    mask: NDArray[np.bool_], spacing: tuple[float, ...], covariance_divisor: str = "n",
) -> dict[str, float]:
    """Evaluate all shape outputs on a rectangular 2D ROI or any small 3D mask.

    ``covariance_divisor`` is ``"n"`` (population) or ``"n-1"`` (sample) for the axis covariance.
    """
    points = np.argwhere(mask).astype(np.float64) * np.asarray(spacing)
    centered = points - points.mean(axis=0)
    samples = len(points) - 1 if covariance_divisor == "n-1" and len(points) > 1 else len(points)
    eigen = np.linalg.svd(centered, compute_uv=False)**2 / samples
    result = {"MajorAxisLength": 4 * sqrt(float(eigen[0])),
              "MinorAxisLength": 4 * sqrt(float(eigen[1])),
              "Elongation": sqrt(float(eigen[1] / eigen[0]))}
    if mask.ndim == 2:
        # Half-index contour of a filled rectangle: four half-voxel corner cuts.
        occupied = np.argwhere(mask)
        lo, hi = occupied.min(axis=0), occupied.max(axis=0)
        a, b = (int(x) for x in hi - lo + 1)
        if int(mask.sum()) != a * b:
            raise ValueError("the analytic 2D oracle requires a filled rectangle")
        vertices = np.array([[-.5, 0], [0, -.5], [a - 1, -.5], [a - .5, 0],
                             [a - .5, b - 1], [a - 1, b - .5], [0, b - .5], [-.5, b - 1]])
        vertices *= np.asarray(spacing)
        following = np.roll(vertices, -1, axis=0)
        measure = float(abs(np.sum(vertices[:, 0] * following[:, 1]
                                   - vertices[:, 1] * following[:, 0])) / 2)
        boundary = float(np.linalg.norm(vertices - following, axis=1).sum())
        diameter = float(np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2).max())
        result.update(MeshSurface=measure, PixelSurface=int(mask.sum()) * prod(spacing),
                      Perimeter=boundary, PerimeterSurfaceRatio=boundary / measure,
                      Sphericity=2 * sqrt(pi * measure) / boundary, MaximumDiameter=diameter)
    else:
        from skimage.measure import marching_cubes

        vertices_raw, faces, _, _ = marching_cubes(np.pad(mask, 1), .5, method="lorensen")
        vertices = vertices_raw.astype(np.float64) * np.asarray(spacing)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        boundary = float(np.linalg.norm(cross, axis=1).sum() / 2)
        measure = float(abs(np.einsum("ij,ij->", np.cross(triangles[:, 0], triangles[:, 1]),
                                     triangles[:, 2])) / 6)
        distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
        planar = [float(distances[vertices[:, None, axis] == vertices[None, :, axis]].max())
                  for axis in range(3)]
        result.update(MeshVolume=measure, VoxelVolume=int(mask.sum()) * prod(spacing),
                      SurfaceArea=boundary, SurfaceVolumeRatio=boundary / measure,
                      Sphericity=(36 * pi * measure**2)**(1 / 3) / boundary,
                      Maximum3DDiameter=float(distances.max()), Maximum2DDiameterSlice=planar[0],
                      Maximum2DDiameterColumn=planar[1], Maximum2DDiameterRow=planar[2],
                      LeastAxisLength=4 * sqrt(float(eigen[2])),
                      Flatness=sqrt(float(eigen[2] / eigen[0])))
    family = "shape2D" if mask.ndim == 2 else "shape"
    return {f"{family}_{name}": value for name, value in result.items()}
