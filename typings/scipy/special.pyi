"""Derivative-free spherical harmonics and Carlson's symmetric elliptic integral."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

def sph_harm_y(
    n: int,
    m: int,
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    *,
    diff_n: Literal[0] = 0,
) -> NDArray[np.complex128]: ...
def elliprg(x: float, y: float, z: float) -> np.float64: ...
