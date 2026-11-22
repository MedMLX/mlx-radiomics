"""The package executes only on Apple Silicon Metal through the shared runtime."""

from collections.abc import Callable
from functools import wraps
from platform import machine, system
from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core import MissingDependencyError, import_mlx
from numpy.typing import NDArray

from mlxradiomics.contracts.types import MapDtype
from mlxradiomics.numerics.protocols import EvaluationRuntime

if TYPE_CHECKING:
    import mlx.core as mx
else:
    if system() != "Darwin" or machine() != "arm64":
        raise MissingDependencyError(
            "mlxradiomics requires macOS arm64",
            hint="Run on Apple Silicon; host preprocessing needs no Metal.",
        )
    mx = import_mlx()


__all__ = [
    "array_from_bool",
    "evaluate",
    "mx",
    "releases_cache",
    "require_retained_capacity",
    "trim_cache",
    "validate_map_dtype",
]

# MLX accepts NumPy bool arrays; its DLPack constructor annotation omits them.
array_from_bool = cast(Callable[[NDArray[np.bool_]], mx.array], mx.array)
evaluate = cast(EvaluationRuntime, mx).eval


def trim_cache() -> None:
    """Retain at most 64 MiB of reusable freed Metal buffers at a cleanup boundary.

    Small ROI calls reuse their allocations; larger workloads release the cache
    instead of retaining every buffer size seen across successive images. This
    bounds cached buffers only, not live arrays or an extraction's working memory.
    """
    if mx.get_cache_memory() > 64 * 1024**2:
        mx.clear_cache()


def releases_cache[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Bound MLX's freed-buffer cache when a public extraction ends.

    MLX keeps freed Metal buffers for reuse, and by default the cache may grow toward
    the device memory limit. Successive extractions use differently sized buffers, so
    without this a long batch of cases keeps every size it has seen: on 512x512 CT
    with wavelets the cache would otherwise grow to tens of gigabytes. Clearing
    releases all unreferenced buffers when the cache exceeds the 64 MiB reuse
    allowance and also runs on error. Buffers held by the propagating traceback are
    cached when it is dropped and subject to the bound at the next extraction boundary.
    """

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        finally:
            trim_cache()

    return wrapper


def validate_map_dtype(map_dtype: MapDtype) -> None:
    """Reject unsupported retained-map storage before extraction or file loading."""
    if map_dtype not in ("float64", "float32"):
        raise ValueError("map_dtype must be 'float64' or 'float32'")


def require_retained_capacity(voxels: int, maps: int, map_dtype: MapDtype = "float64") -> None:
    """Admit dense maps against one quarter of physical unified memory.

    Count the complete selection, including every derived image, before allocating
    any map. The remaining memory serves inputs, host preprocessing and GPU scratch;
    this payload admission is not a reservation or a bound on total process memory.
    """
    required = voxels * maps * (4 if map_dtype == "float32" else 8)
    budget = int(mx.device_info(mx.gpu)["memory_size"]) // 4
    if required > budget:
        raise MemoryError(
            f"retaining {maps:,} {map_dtype} voxel maps requires {required:,} bytes, above "
            f"the {budget:,}-byte retained-map budget (one quarter of unified memory); "
            "use sink or export/export_files to stream the maps"
        )
