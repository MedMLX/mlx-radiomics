"""Shared extraction, bounded map writing and atomic report publication."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings
from mlxradiomics.config.transforms import TransformSettings
from mlxradiomics.contracts.json import JsonValue, json_safe
from mlxradiomics.contracts.metadata import file_metadata, report_details
from mlxradiomics.contracts.types import (
    ExecutionMetadata,
    HostArray,
    ImageGeometryPayload,
    RunDetails,
    VoxelSink,
    VoxelSinkFactory,
)
from mlxradiomics.imaging.io import Volume, write_feature_map
from mlxradiomics.imaging.transforms import feature_image_names
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.extraction import extract
from mlxradiomics.workflows.publication import (
    Publication,
    check_output_replacement,
    output_lock,
    prepare_output_targets,
    reclaim_stale_staging,
    reject_input_aliases,
    reject_output_aliases,
)


@dataclass(frozen=True)
class PublishedReport:
    """Published paths and the metadata used by Python and MedMLX result adapters."""

    report: Path
    table: Path | None
    maps: Mapping[str, Path]
    metadata: ExecutionMetadata
    details: RunDetails


def _map_layout(
    reference: Volume,
    settings: Settings,
    transforms: TransformSettings,
) -> dict[str, tuple[str, ...]]:
    """Return the prefixed map names of each derived image, in output order.

    Shape and local-intensity peak features are refused because they have no voxel
    maps; names must be nonempty and unique.
    """
    if any(name.startswith("shape") for name in settings.features):
        raise ValueError("shape features are unavailable in voxel mode")
    if any(name.startswith("localintensity_") for name in settings.features):
        raise ValueError("physical local-intensity peak features are unavailable in voxel mode")
    image_names = feature_image_names(
        reference.array.shape,
        reference.spacing,
        transforms,
        force2d_axis=settings.force2d_axis,
    )
    layout = {
        image: tuple(f"{image}_{feature}" for feature in settings.features) for image in image_names
    }
    names = tuple(name for group in layout.values() for name in group)
    if not names or len(set(names)) != len(names):
        raise ValueError("derived image names must be nonempty and unique")
    return layout


def _prepare_map_writer(
    reference: Volume,
    layout: Mapping[str, tuple[str, ...]],
    paths: Mapping[str, Path],
    publication: Publication,
    directory: Path,
    temporary_dir: str | os.PathLike[str] | None,
    max_temporary_mib: int | None,
) -> tuple[VoxelSinkFactory, dict[str, str]]:
    """Admit voxel scratch and output space, then return a per-image map writer.

    The returned sink factory stores one derived image's maps at a time and writes
    each as a staged float32 NIfTI; the dictionary fills with the written map paths.
    Placement and space failures raise ``InvalidInputError`` before extraction.
    """
    scratch = (
        Path(temporary_dir).expanduser()
        if temporary_dir is not None
        else Path(tempfile.gettempdir())
    )
    # Cover uncompressed maps, compression overhead and report headroom.
    estimate = len(paths) * (2 * prod(reference.array.shape) * 4 + 65536)
    limit = max_temporary_mib * 1024**2 if max_temporary_mib is not None else None
    try:
        if estimate > shutil.disk_usage(directory).free:
            raise ValueError("insufficient output filesystem space for voxel staging")
        reserve = estimate if scratch.stat().st_dev == directory.stat().st_dev else 0
        # Admit placement and the per-image budget before any extraction or filtering.
        with voxel_map_store(
            reference.array.shape,
            next(iter(layout.values())),
            temporary_dir=scratch,
            max_temporary_bytes=limit,
            reserve_free_bytes=reserve,
        ):
            pass
    except (ValueError, OSError) as error:
        raise InvalidInputError(str(error)) from error
    written: dict[str, str] = {}

    @contextmanager
    def write_image(image: str) -> Generator[VoxelSink]:
        with voxel_map_store(
            reference.array.shape,
            layout[image],
            temporary_dir=scratch,
            max_temporary_bytes=limit,
            reserve_free_bytes=reserve,
        ) as store:
            yield store.append
            for name, values in store.maps():
                write_feature_map(publication.stage(paths[name]), values, reference=reference)
                written[name] = paths[name].as_posix()

    return write_image, written


def _write_csv(path: Path, values: Mapping[str, float]) -> None:
    """Write one header row of feature names and one row of values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(values.keys())
        writer.writerow(values.values())


def publish_report(
    image: HostArray,
    mask: HostArray,
    *,
    output_dir: str | os.PathLike[str],
    settings: Settings,
    transforms: TransformSettings,
    reference: Volume,
    inputs: tuple[Path, ...],
    source_geometry: ImageGeometryPayload | None,
    image_settings: ImageSettings | None,
    overwrite: bool,
    temporary_dir: str | os.PathLike[str] | None,
    max_temporary_mib: int | None,
    metadata_settings: Settings | None = None,
    voxel: bool = True,
    input_volumes: tuple[Path, ...] = (),
) -> PublishedReport:
    """Extract once and publish a complete schema-2 report, CSV or streamed map set."""
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be a boolean")
    if max_temporary_mib is not None and (
        type(max_temporary_mib) is not int or max_temporary_mib <= 0
    ):
        raise InvalidInputError("max_temporary_mib must be a positive integer")
    try:
        settings.validate(image.ndim)
        layout = _map_layout(reference, settings, transforms) if voxel else {}
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {directory}")
    report = directory / "features.json"
    table = None if voxel else directory / "features.csv"
    paths = {
        name: directory / "maps" / f"{name}.nii.gz" for names in layout.values() for name in names
    }
    outputs = (report, *(() if table is None else (table,)), *paths.values())
    with output_lock(directory):
        reclaim_stale_staging(directory)
        prepare_output_targets(directory, outputs)
        reject_input_aliases(outputs, inputs)
        reject_output_aliases(outputs)
        existing = [path.as_posix() for path in outputs if path.exists()]
        check_output_replacement(existing, overwrite=overwrite, inputs=input_volumes)
        try:
            publication = Publication.create(directory)
        except OSError as error:
            raise InvalidInputError(f"cannot stage outputs in: {directory}") from error
        try:
            maps: dict[str, str] = {}
            sink = None
            if voxel:
                sink, maps = _prepare_map_writer(
                    reference,
                    layout,
                    paths,
                    publication,
                    directory,
                    temporary_dir,
                    max_temporary_mib,
                )
            try:
                result = extract(
                    image,
                    mask,
                    settings=settings,
                    transforms=transforms,
                    spacing=reference.spacing,
                    mode="voxel" if voxel else "roi",
                    sink_factory=sink,
                )
            except (InvalidInputError, OSError):
                raise
            except ValueError as error:
                raise InvalidInputError(str(error)) from error
            except (RuntimeError, ImportError) as error:
                raise ModelExecutionError(f"mlxradiomics extraction failed: {error}") from error
            metadata = file_metadata(
                result.metadata,
                reference,
                settings=metadata_settings,
                source_geometry=source_geometry,
                image_settings=image_settings,
            )
            details = report_details(metadata, count=len(paths) if voxel else len(result.features))
            payload: dict[str, JsonValue] = {
                "schema_version": 2,
                "mode": "voxel" if voxel else "roi",
                "details": cast(JsonValue, details),
            }
            if voxel:
                if set(maps) != set(paths):
                    raise ModelExecutionError("voxel extraction did not produce every selected map")
                payload["maps"] = cast(JsonValue, maps)
            else:
                values: dict[str, float] = {}
                for name, value in result.features.items():
                    if isinstance(value, np.ndarray):
                        raise ModelExecutionError(f"ROI feature '{name}' did not return a scalar")
                    values[name] = value
                assert table is not None
                _write_csv(publication.stage(table), values)
                payload["features"] = cast(JsonValue, values)
            publication.stage(report).write_text(
                json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            publication.publish(overwrite=overwrite)
        finally:
            publication.cleanup()
    return PublishedReport(report, table, paths, metadata, details)
