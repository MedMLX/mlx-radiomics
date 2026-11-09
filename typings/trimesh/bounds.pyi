"""Local typing for trimesh 4.9.0 oriented bounds."""

import numpy as np
from numpy.typing import NDArray
from trimesh.convex import Trimesh

def oriented_bounds(
    obj: Trimesh,
    angle_digits: int = 1,
    ordered: bool = True,
    normal: NDArray[np.float64] | None = None,
    coplanar_tol: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
