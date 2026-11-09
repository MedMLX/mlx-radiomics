from typing import Protocol

import numpy as np
from numpy.typing import NDArray

class _MeshVertices(Protocol):
    @property
    def vertices(self) -> NDArray[np.float64]: ...

def icosphere(subdivisions: int = 3, radius: float = 1.0) -> _MeshVertices: ...
