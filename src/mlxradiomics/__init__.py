"""Radiomics on Apple Silicon: extract features or export voxel maps.

``extract`` and ``export`` accept aligned NumPy arrays. Their ``*_files``
counterparts read NIfTI/NRRD image and mask files and preserve physical geometry.
Both extraction routes return ``Result``; both exporters return ``MapExport``.

Importing or listing this namespace does not initialize MLX or SimpleITK.
Low-level image/filter operations live in their respective modules; MedMLX and
multi-case workflows are available through ``mlxradiomics.medmlx`` and
``mlxradiomics.workflows.study``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, cast

from mlxradiomics.config.settings import FEATURES, Settings, list_features

if TYPE_CHECKING:
    from mlxradiomics.config.spatial import IbsiInterpolation as IbsiInterpolation
    from mlxradiomics.config.spatial import ImageSettings as ImageSettings
    from mlxradiomics.config.spatial import Interpolator as Interpolator
    from mlxradiomics.config.transforms import ImageType as ImageType
    from mlxradiomics.config.transforms import TransformSettings as TransformSettings
    from mlxradiomics.contracts.results import MapExport as MapExport
    from mlxradiomics.contracts.results import Result as Result
    from mlxradiomics.contracts.types import VoxelSink as VoxelSink
    from mlxradiomics.contracts.types import VoxelSinkFactory as VoxelSinkFactory
    from mlxradiomics.workflows.exporting import export as export
    from mlxradiomics.workflows.exporting import export_files as export_files
    from mlxradiomics.workflows.extraction import extract as extract
    from mlxradiomics.workflows.extraction import extract_files as extract_files

__version__ = version("mlxradiomics")

__all__ = [
    "FEATURES",
    "IbsiInterpolation",
    "ImageSettings",
    "ImageType",
    "Interpolator",
    "MapExport",
    "Result",
    "Settings",
    "TransformSettings",
    "VoxelSink",
    "VoxelSinkFactory",
    "__version__",
    "export",
    "export_files",
    "extract",
    "extract_files",
    "list_features",
]

_EXPORT_MODULES = {
    "ImageSettings": "config.spatial",
    "TransformSettings": "config.transforms",
    "ImageType": "config.transforms",
    "Interpolator": "config.spatial",
    "IbsiInterpolation": "config.spatial",
    "Result": "contracts.results",
    "MapExport": "contracts.results",
    "VoxelSink": "contracts.types",
    "VoxelSinkFactory": "contracts.types",
    "extract": "workflows.extraction",
    "extract_files": "workflows.extraction",
    "export": "workflows.exporting",
    "export_files": "workflows.exporting",
}


def __getattr__(name: str) -> object:
    """Resolve a public API name on first access."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = cast(object, getattr(import_module(f"mlxradiomics.{module}"), name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public API without loading the modules behind lazy names."""
    return sorted(__all__)
