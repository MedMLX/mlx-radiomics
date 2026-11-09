"""Scikit-image LBP contract used for independent transform comparisons."""

import numpy as np
from numpy.typing import NDArray

def local_binary_pattern(
    image: NDArray[np.float64], P: int, R: float, method: str = "default",
) -> NDArray[np.float64]: ...
