from typing import Literal

import numpy as np
from numpy.typing import NDArray

def swtn(
    data: NDArray[np.float64],
    wavelet: str,
    level: int,
    start_level: int = 0,
    axes: tuple[int, ...] | None = None,
    trim_approx: Literal[False] = False,
    norm: bool = False,
) -> list[dict[str, NDArray[np.float64]]]: ...
