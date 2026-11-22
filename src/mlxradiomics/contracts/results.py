"""Typed results shared by array, file and map-export workflows."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from mlxradiomics.contracts.types import ExecutionMetadata, VoxelArray

Value = TypeVar("Value", covariant=True, default=float | VoxelArray)


@dataclass(frozen=True)
class Result(Generic[Value]):
    """Extracted features and their execution metadata.

    Keys are ``<image>_<family>_<name>``, such as ``original_firstorder_Mean``.
    ROI extraction returns ``Result[float]``; voxel extraction returns
    ``Result[NDArray[np.float64]]`` by default, or float32 maps when requested,
    with zero outside the ROI. A streaming sink
    receives evaluated float32 batches and leaves ``features`` empty.

    NaN denotes an undefined feature; infinity denotes an unbounded float32
    result, including overflow. ``metadata["feature_validity"]`` records those
    cases separately from finite values and uncomputed background. Effective
    settings, transforms and processed file geometry are recorded in metadata.
    """

    features: Mapping[str, Value]
    metadata: ExecutionMetadata


@dataclass(frozen=True)
class MapExport:
    """Published voxel maps, the JSON report and their execution metadata.

    ``report`` and every value in ``maps`` are absolute paths. Each map is a
    float32 NIfTI on the processed grid. The report contains the serialized
    metadata and file paths; ``metadata`` retains the typed Python values.
    """

    report: Path
    maps: Mapping[str, Path]
    metadata: ExecutionMetadata
