"""Bounded disk-backed export of voxel feature maps for original and derived images.

``export`` and ``export_files`` reuse the extraction pipeline and the same
bounded spool as the MedMLX runner (``mlxradiomics.medmlx.run``): evaluated batches go
to one compact scratch file, then one disk-backed float32 map at a time is written
through private staging and published atomically. No full map is retained after it is written.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path

import numpy as np
from medmlx_core.errors import InvalidInputError

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings
from mlxradiomics.config.transforms import TransformSettings
from mlxradiomics.contracts.results import MapExport
from mlxradiomics.contracts.types import HostArray
from mlxradiomics.imaging.io import Volume
from mlxradiomics.imaging.spatial import load_image_pair
from mlxradiomics.workflows.reporting import publish_report


def export(
    image: HostArray,
    mask: HostArray,
    output_dir: str | os.PathLike[str],
    *,
    settings: Settings | None = None,
    transforms: TransformSettings | None = None,
    spacing: tuple[float, ...] | None = None,
    allow_synthetic_geometry: bool = False,
    overwrite: bool = False,
    temporary_dir: str | os.PathLike[str] | None = None,
    max_temporary_mib: int | None = None,
) -> MapExport:
    """Stream derived-image voxel maps to NIfTI files with bounded host storage.

    The input arrays are aligned and use NumPy axis order. Every selected image type is
    extracted in turn; evaluated batches are spooled to one compact scratch file and one
    float32 map at a time is written under ``<output_dir>/maps``. Each map carries the
    image-type prefix, for example ``wavelet-LLH_glcm_Contrast``. ``features.json`` is a
    strict schema-2 report containing the processed geometry, effective settings and map
    paths. Publication uses private staging and refuses existing outputs unless
    ``overwrite`` is true.

    Shape features are unavailable in voxel mode and are rejected before extraction.

    Args:
        image: Aligned 2D or 3D integer or floating-point NumPy array.
        mask: Boolean or integer label array with the same shape.
        output_dir: Directory for ``features.json`` and ``maps/``; created if missing.
        settings: Feature selection and preprocessing; defaults to ``Settings()``.
        transforms: Derived images to export; defaults to the original image.
        spacing: Positive millimeters per NumPy axis; defaults to unit spacing.
        allow_synthetic_geometry: Acknowledge zero origin and identity direction in the
            saved maps. False preserves the export but warns; use ``export_files`` to
            retain physical geometry from a source image.
        overwrite: Replace existing outputs instead of refusing them.
        temporary_dir: Existing directory for voxel scratch; ``None`` uses the system
            temporary directory.
        max_temporary_mib: Positive cap in MiB for logical voxel scratch (compact spool
            plus one dense float32 grid); not an RSS cap.

    Returns:
        The published report and map paths with the extraction metadata.

    Raises:
        InvalidInputError: Invalid inputs or settings, an unsafe or occupied output
            target, or insufficient scratch or output staging space.
        ModelExecutionError: Feature extraction or map writing failed.
    """
    if type(allow_synthetic_geometry) is not bool:
        raise InvalidInputError("allow_synthetic_geometry must be a boolean")
    if not allow_synthetic_geometry:
        warnings.warn(
            "Array export writes synthetic geometry (zero origin, identity direction; "
            "1 mm spacing if omitted). Use export_files to retain source geometry, or "
            "set allow_synthetic_geometry=True to acknowledge this coordinate system.",
            UserWarning, stacklevel=2,
        )
    settings = settings or Settings()
    transforms = transforms or TransformSettings()
    array = np.asarray(image)
    ndim = array.ndim
    if len(spacing or ()) not in (0, ndim):
        raise InvalidInputError("spacing must contain one value per image axis")
    used_spacing = spacing or (1.0,) * ndim
    try:
        reference = Volume(array, used_spacing, (0.0,) * ndim, tuple(np.eye(ndim).ravel().tolist()))
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    saved = publish_report(
        image, mask, output_dir=output_dir, settings=settings, transforms=transforms,
        reference=reference, inputs=(),
        source_geometry=None, image_settings=None, overwrite=overwrite,
        temporary_dir=temporary_dir, max_temporary_mib=max_temporary_mib,
    )
    return MapExport(saved.report, saved.maps, saved.metadata)


def export_files(
    image_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    settings: Settings | None = None,
    image_settings: ImageSettings | None = None,
    transforms: TransformSettings | None = None,
    overwrite: bool = False,
    temporary_dir: str | os.PathLike[str] | None = None,
    max_temporary_mib: int | None = None,
) -> MapExport:
    """Read one image/mask pair and export derived-image voxel maps to disk.

    Combines :func:`load_image_pair` with :func:`export`. Normalization happens
    while loading, before resampling and filtering. Map geometry follows the processed
    grid; source geometry and effective spatial options are recorded in the report.

    Args:
        image_path: NIfTI or NRRD image.
        mask_path: NIfTI or NRRD integer label mask.
        output_dir: Directory for ``features.json`` and ``maps/``; created if missing.
        settings: Feature selection and preprocessing; defaults to ``Settings()``.
        image_settings: Resampling, IBSI interpolation and mask-correction options.
        transforms: Derived images to export; defaults to the original image.
        overwrite: Replace existing outputs instead of refusing them.
        temporary_dir: Existing directory for voxel scratch; ``None`` uses the system
            temporary directory.
        max_temporary_mib: Positive cap in MiB for logical voxel scratch (compact spool
            plus one dense float32 grid); not an RSS cap.

    Returns:
        The published report and map paths with the extraction metadata.

    Raises:
        InvalidInputError: Invalid inputs or settings, an unsafe or occupied output
            target, or insufficient scratch or output staging space.
        ModelExecutionError: Feature extraction or map writing failed.
    """
    options = settings or Settings()
    source_image, source_mask = Path(image_path).expanduser(), Path(mask_path).expanduser()
    try:
        pair = load_image_pair(
            source_image, source_mask, settings=options, image_settings=image_settings,
        )
    # load_image_pair raises ValueError (geometry, label) and RuntimeError (unreadable
    # file); truncated or unreadable compressed files surface as OSError or EOFError.
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise InvalidInputError(str(error)) from error
    extraction = dataclasses.replace(options, normalize=False, remove_outliers=None)
    saved = publish_report(
        pair.image.array, pair.mask.array, output_dir=output_dir, settings=extraction,
        transforms=transforms or TransformSettings(),
        reference=pair.image, inputs=(source_image, source_mask),
        source_geometry=pair.source_geometry,
        image_settings=image_settings or ImageSettings(), overwrite=overwrite,
        temporary_dir=temporary_dir, max_temporary_mib=max_temporary_mib,
        metadata_settings=options,
        input_volumes=(source_image, source_mask),
    )
    return MapExport(saved.report, saved.maps, saved.metadata)
