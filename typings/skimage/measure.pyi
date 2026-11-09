from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def marching_cubes(
    volume: NDArray[np.generic],
    level: float | None = None,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    gradient_direction: str = "descent",
    step_size: int = 1,
    allow_degenerate: bool = True,
    method: str = "lewiner",
    mask: NDArray[np.bool_] | None = None,
) -> tuple[
    NDArray[np.float32] | NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
