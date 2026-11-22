from typing import Literal

import numpy as np
from numpy.typing import NDArray

def map_coordinates(
    input: NDArray[np.generic],
    coordinates: NDArray[np.float64],
    *,
    order: int = 3,
    mode: Literal["mirror", "constant", "nearest"] = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> NDArray[np.generic]: ...

def spline_filter(
    input: NDArray[np.generic],
    order: int = 3,
    *,
    output: type[np.float64],
    mode: Literal["mirror", "constant", "nearest"] = "mirror",
) -> NDArray[np.float64]: ...

def maximum_filter(
    input: NDArray[np.bool_],
    *,
    size: tuple[int, ...],
    mode: Literal["constant"],
    cval: float,
) -> NDArray[np.bool_]: ...

def zoom(
    input: NDArray[np.generic],
    zoom: tuple[float, ...],
    *,
    order: int = 3,
    prefilter: bool = True,
) -> NDArray[np.generic]: ...
