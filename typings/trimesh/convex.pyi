"""Local typing for trimesh 4.9.0 convex hull construction."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

class Trimesh(Protocol):
    @property
    def vertices(self) -> NDArray[np.float64]: ...

    @property
    def face_normals(self) -> NDArray[np.float64]: ...

    @property
    def face_adjacency(self) -> NDArray[np.int64]: ...

    @property
    def face_adjacency_edges(self) -> NDArray[np.int64]: ...


def convex_hull(mesh: NDArray[np.float64], repair: bool = True) -> Trimesh: ...
