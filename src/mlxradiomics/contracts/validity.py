"""Classification of evaluated feature values."""

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.contracts.types import (
    FeatureValidity,
)


def empty_validity(excluded: int) -> FeatureValidity:
    """Initialize counts without treating uncomputed map background as valid zero."""
    return {
        "finite": 0,
        "valid_zero": 0,
        "nan": 0,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "excluded": excluded,
    }


def accumulate_validity(
    counts: FeatureValidity,
    values: NDArray[np.float32] | NDArray[np.float64],
) -> None:
    """Classify evaluated host samples; this never recomputes feature arithmetic."""
    finite = int(np.count_nonzero(np.isfinite(values)))
    counts["finite"] += finite
    counts["valid_zero"] += int(np.count_nonzero(values == 0))
    if finite == values.size:
        # Completed finite results cannot add to any exceptional-value category.
        return
    counts["nan"] += int(np.count_nonzero(np.isnan(values)))
    counts["positive_infinity"] += int(np.count_nonzero(np.isposinf(values)))
    counts["negative_infinity"] += int(np.count_nonzero(np.isneginf(values)))
