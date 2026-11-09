"""Independent scalar references for corrected texture and moment mathematics.

These helpers enumerate small arrays in Python loops and evaluate the definitions
directly. They do not import the production feature arithmetic, so sampling, direction
aggregation and degenerate-case defects remain observable.
"""

from collections.abc import Iterator
from itertools import product
from math import inf, log2, nan

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings

GLRLM_NAMES = (
    "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance", "RunVariance",
    "RunEntropy", "LowGrayLevelRunEmphasis", "HighGrayLevelRunEmphasis",
    "ShortRunLowGrayLevelEmphasis", "ShortRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis", "LongRunHighGrayLevelEmphasis",
)
NGTDM_NAMES = ("Coarseness", "Contrast", "Busyness", "Complexity", "Strength")
_EPS = 2.220446049250313e-16

type _Index = tuple[int, ...]


def _indexes(shape: tuple[int, ...]) -> Iterator[_Index]:
    yield from product(*(range(size) for size in shape))


def _inside(index: _Index, shape: tuple[int, ...]) -> bool:
    return all(0 <= value < size for value, size in zip(index, shape, strict=True))


def _shift(index: _Index, offset: _Index) -> _Index:
    return tuple(value + delta for value, delta in zip(index, offset, strict=True))


def _run_directions(shape: tuple[int, ...], force2d_axis: int | None = None) -> tuple[_Index, ...]:
    """Return the dimensional distance-1 directions, including singleton-only lines."""
    return tuple(
        offset for offset in product((-1, 0, 1), repeat=len(shape))
        if max(map(abs, offset)) == 1
        and next((value for value in offset if value), 0) > 0
        and (force2d_axis is None or offset[force2d_axis] == 0)
    )


def _neighbor_offsets(
    distances: tuple[int, ...], ndim: int, force2d_axis: int | None = None,
) -> tuple[_Index, ...]:
    radius = max(distances)
    zero = (0,) * ndim
    return tuple(
        offset for offset in product(range(-radius, radius + 1), repeat=ndim)
        if offset != zero and max(map(abs, offset)) in distances
        and (force2d_axis is None or offset[force2d_axis] == 0)
    )


def _glrlm_direction(
    counts: dict[tuple[int, int], int], runs: int, voxels: int,
) -> dict[str, float]:
    if runs == 0:
        return dict.fromkeys(GLRLM_NAMES, nan)
    levels = sorted({level for level, _ in counts})
    lengths = sorted({length for _, length in counts})
    probability = {(level, length): count / runs for (level, length), count in counts.items()}
    per_length = {
        length: sum(count for (_level, run_length), count in counts.items()
                    if run_length == length)
        for length in lengths
    }
    per_level = {
        level: sum(count for (run_level, _length), count in counts.items()
                   if run_level == level)
        for level in levels
    }
    mean_level = sum(p * level for (level, _), p in probability.items())
    mean_length = sum(p * length for (_, length), p in probability.items())
    return {
        "ShortRunEmphasis": sum(p / length**2 for (_, length), p in probability.items()),
        "LongRunEmphasis": sum(p * length**2 for (_, length), p in probability.items()),
        "GrayLevelNonUniformity": sum(per_level[i]**2 for i in levels) / runs,
        "GrayLevelNonUniformityNormalized": sum(per_level[i]**2 for i in levels) / runs**2,
        "RunLengthNonUniformity": sum(per_length[j]**2 for j in lengths) / runs,
        "RunLengthNonUniformityNormalized": sum(per_length[j]**2 for j in lengths) / runs**2,
        "RunPercentage": runs / voxels,
        "GrayLevelVariance": sum(
            p * (level - mean_level)**2 for (level, _), p in probability.items()
        ),
        "RunVariance": sum(
            p * (length - mean_length)**2 for (_, length), p in probability.items()
        ),
        "RunEntropy": -sum(p * log2(p + _EPS) for p in probability.values()),
        "LowGrayLevelRunEmphasis": sum(
            p / level**2 for (level, _), p in probability.items()
        ),
        "HighGrayLevelRunEmphasis": sum(
            p * level**2 for (level, _), p in probability.items()
        ),
        "ShortRunLowGrayLevelEmphasis": sum(
            p / (level**2 * length**2) for (level, length), p in probability.items()
        ),
        "ShortRunHighGrayLevelEmphasis": sum(
            p * level**2 / length**2 for (level, length), p in probability.items()
        ),
        "LongRunLowGrayLevelEmphasis": sum(
            p * length**2 / level**2 for (level, length), p in probability.items()
        ),
        "LongRunHighGrayLevelEmphasis": sum(
            p * level**2 * length**2 for (level, length), p in probability.items()
        ),
    }


def glrlm_features(
    bins: NDArray[np.int32], mask: NDArray[np.bool_], force2d_axis: int | None = None,
) -> dict[str, float]:
    """Evaluate all GLRLM statistics by enumerating runs along every declared direction."""
    if bins.shape != mask.shape:
        raise ValueError("bins and mask must share a shape")
    shape = bins.shape
    directions = _run_directions(shape, force2d_axis)
    voxels = int(mask.sum())
    if not directions or voxels == 0:
        return dict.fromkeys(GLRLM_NAMES, nan)
    indexes = tuple(_indexes(shape))
    per_direction: list[dict[str, float]] = []
    for offset in directions:
        counts: dict[tuple[int, int], int] = {}
        runs = 0
        reverse = tuple(-value for value in offset)
        for index in indexes:
            if not bool(mask[index]):
                continue
            before = _shift(index, reverse)
            if (_inside(before, shape) and bool(mask[before])
                    and int(bins[before]) == int(bins[index])):
                continue
            level = int(bins[index])
            length = 1
            position = index
            while True:
                following = _shift(position, offset)
                if (not _inside(following, shape) or not bool(mask[following])
                        or int(bins[following]) != level):
                    break
                position = following
                length += 1
            counts[(level, length)] = counts.get((level, length), 0) + 1
            runs += 1
        per_direction.append(_glrlm_direction(counts, runs, voxels))
    return {
        name: sum(values[name] for values in per_direction) / len(per_direction)
        for name in GLRLM_NAMES
    }


def ngtdm_features(
    bins: NDArray[np.int32],
    mask: NDArray[np.bool_],
    distances: tuple[int, ...] = (1,),
    force2d_axis: int | None = None,
) -> dict[str, float]:
    """Evaluate NGTDM statistics from eligible centers and their scalar neighbor means."""
    if bins.shape != mask.shape:
        raise ValueError("bins and mask must share a shape")
    shape = bins.shape
    offsets = _neighbor_offsets(distances, len(shape), force2d_axis)
    occurrence: dict[int, int] = {}
    difference: dict[int, float] = {}
    for index in _indexes(shape):
        if not bool(mask[index]):
            continue
        neighbors: list[int] = []
        for offset in offsets:
            neighbor = _shift(index, offset)
            if _inside(neighbor, shape) and bool(mask[neighbor]):
                neighbors.append(int(bins[neighbor]))
        if not neighbors:
            continue
        level = int(bins[index])
        occurrence[level] = occurrence.get(level, 0) + 1
        difference[level] = (difference.get(level, 0.0)
                             + abs(level - sum(neighbors) / len(neighbors)))
    total = sum(occurrence.values())
    if total == 0:
        return dict.fromkeys(NGTDM_NAMES, nan)
    levels = sorted(occurrence)
    probability = {level: occurrence[level] / total for level in levels}
    raw = sum(probability[level] * difference[level] for level in levels)
    gray_levels = len(levels)
    contrast_sum = sum(
        probability[i] * probability[j] * (i - j)**2 for i in levels for j in levels
    )
    busyness_denominator = sum(
        abs(i * probability[i] - j * probability[j]) for i in levels for j in levels
    )
    strength_numerator = sum(
        (probability[i] + probability[j]) * (i - j)**2 for i in levels for j in levels
    )
    complexity_numerator = sum(
        abs(i - j) * (probability[i] * difference[i] + probability[j] * difference[j])
        / (probability[i] + probability[j])
        for i in levels for j in levels
    )
    total_difference = sum(difference[level] for level in levels)
    return {
        "Coarseness": inf if raw == 0 else 1 / raw,
        "Contrast": (0.0 if gray_levels <= 1
                     else contrast_sum * total_difference / total
                     / (gray_levels * (gray_levels - 1))),
        "Busyness": (nan if busyness_denominator == 0 and raw == 0
                     else inf if busyness_denominator == 0
                     else raw / busyness_denominator),
        "Complexity": complexity_numerator / total,
        "Strength": (nan if total_difference == 0 and strength_numerator == 0
                     else inf if total_difference == 0
                     else strength_numerator / total_difference),
    }


def glcm_features(
    bins: NDArray[np.int32], mask: NDArray[np.bool_], settings: Settings,
) -> dict[str, float]:
    """Average JointAverage over nonempty directions and MCC over defined directions.

    Each local direction is evaluated independently. MCC requires at least two occupied
    row and column marginal states; a rank-one matrix with those states has valid MCC 0.
    """
    shape = bins.shape
    offsets = _neighbor_offsets(settings.distances, bins.ndim, settings.force2d_axis)
    joint: list[float] = []
    mcc: list[float] = []
    for offset in offsets:
        if next((value for value in offset if value), 0) < 0:
            continue
        if any(abs(value) >= size for value, size in zip(offset, shape, strict=True)):
            continue
        # Retain actual occupied levels, including gaps in the global bin numbering.
        pairs: list[tuple[int, int]] = []
        for index in _indexes(shape):
            neighbor = _shift(index, offset)
            if bool(mask[index]) and _inside(neighbor, shape) and bool(mask[neighbor]):
                pairs.append((int(bins[index]), int(bins[neighbor])))
        if not pairs:
            continue
        if settings.symmetrical_glcm:
            pairs = pairs + [(right, left) for left, right in pairs]
        levels = sorted({level for pair in pairs for level in pair})
        indexes = {level: index for index, level in enumerate(levels)}
        counts = np.zeros((len(levels), len(levels)), dtype=np.float64)
        for left, right in pairs:
            counts[indexes[left], indexes[right]] += 1
        p = counts / counts.sum()
        px, py = p.sum(axis=1), p.sum(axis=0)
        joint.append(float(np.dot(levels, px)))
        rows, columns = px > 0, py > 0
        if np.count_nonzero(rows) < 2 or np.count_nonzero(columns) < 2:
            continue
        occupied = p[np.ix_(rows, columns)]
        normalized = occupied / np.sqrt(px[rows, None] * py[None, columns])
        # Singular values independently evaluate the squared-eigenvalue definition.
        mcc.append(float(np.linalg.svd(normalized, compute_uv=False)[1]))
    return {
        "JointAverage": sum(joint) / len(joint) if joint else nan,
        "MCC": sum(mcc) / len(mcc) if mcc else nan,
    }


def masked_neighborhood_mean(
    values: NDArray[np.float64], mask: NDArray[np.bool_], radius: int = 1,
) -> NDArray[np.float64]:
    """Mean of each mask voxel's in-mask cubic neighborhood; background stays zero."""
    result = np.zeros(values.shape, dtype=np.float64)
    for index in np.argwhere(mask):
        center = tuple(int(value) for value in index)
        window = tuple(
            slice(max(position - radius, 0), min(position + radius + 1, size))
            for position, size in zip(center, values.shape, strict=True)
        )
        result[center] = np.mean(values[window][mask[window]])
    return result


def gradient_magnitude(
    values: NDArray[np.generic], spacing: tuple[float, ...],
) -> NDArray[np.float64]:
    """Central-difference gradient magnitude with edge voxels replicated."""
    data = np.asarray(values, dtype=np.float64)
    squared = np.zeros_like(data)
    for axis, size in enumerate(data.shape):
        indexes = np.arange(size, dtype=np.int64)
        left = np.take(data, np.maximum(indexes - 1, 0), axis=axis)
        right = np.take(data, np.minimum(indexes + 1, size - 1), axis=axis)
        squared += ((right - left) / (2 * spacing[axis])) ** 2
    return np.sqrt(squared)
