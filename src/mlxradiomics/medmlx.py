"""MedMLX declaration and end-to-end runner for radiomics extraction.

Reads one NIfTI/NRRD image and label mask, optionally resamples or corrects the mask,
computes the requested original and derived images on Metal, and writes a features JSON
(plus CSV in ROI mode, or one NIfTI map per feature in voxel mode).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from medmlx_core.errors import InvalidInputError
from medmlx_core.outputs import Artifact, artifact

from mlxradiomics.config.parameters import ExtractionParameters, load_params, parse_params
from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings
from mlxradiomics.config.transforms import TransformSettings
from mlxradiomics.contracts.types import (
    ImageGeometryPayload,
    RunDetails,
)
from mlxradiomics.imaging.io import Volume
from mlxradiomics.imaging.spatial import LoadedCase, prepare_image_pair
from mlxradiomics.workflows.reporting import publish_report

# Registered through the ``medmlx.models`` entry point in pyproject.toml; MedMLX
# calls ``run`` with an image path, mask path and output directory.
MODEL = {
    "id": "mlxradiomics",
    "requires_weights": False,
    "label": "Radiomics feature extraction on Metal",
    "task": "feature-extraction",
    "run": "mlxradiomics.medmlx:run",
    # Approximate GPU memory in GB for the default ROI features of one organ.
    "peak_memory_gb": 0.1,
    "summary": (
        "ROI and voxel radiomics features for one image and label mask on "
        "Apple Silicon using native MLX."
    ),
}


class RunResult(TypedDict):
    """Paths by role, their MedMLX descriptions and metadata from :func:`run`.

    ``outputs`` names each file by role: ``features_json``, ``features_csv`` when
    written, and one role per map named by its feature. ``artifacts`` describes every
    ``outputs`` role.
    """

    outputs: dict[str, str]
    artifacts: dict[str, Artifact]
    details: RunDetails


def _load_parameters(
    params: Mapping[str, object] | None,
    params_path: str | os.PathLike[str] | None,
    label: int | None,
    ndim: int,
) -> ExtractionParameters:
    """Resolve ``params`` or ``params_path`` into validated extraction parameters.

    With neither, native defaults are used. ``label`` overrides the parameter
    file's label. Invalid input raises ``InvalidInputError``.
    """
    if params is not None and params_path is not None:
        raise InvalidInputError("pass only one of params or params_path")
    try:
        loaded = load_params(Path(params_path).expanduser()) if params_path else None
        source = params if params is not None else loaded
        result = (
            parse_params(source, ndim=ndim) if source is not None
            else ExtractionParameters(Settings(), ImageSettings(), TransformSettings())
        )
        if label is not None:
            settings = dataclasses.replace(result.settings, label=label)
            settings.validate(ndim)
            result = dataclasses.replace(result, settings=settings)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    return result


def _scalar_field_artifact(
    *, processed: Volume, source: ImageGeometryPayload | None,
) -> Artifact:
    """Describe a voxel map on the input grid only when every geometry field matches.

    A cropped or resampled map is still physically positioned by its NIfTI affine, but it
    is no longer an array over the input volume's voxel grid. Such a map uses
    ``image_physical`` and omits ``grid_of``, which must name an input argument sharing the
    output grid. The run metadata keeps ``source_geometry`` for input provenance.
    """
    matches_input = source is not None and (
        tuple(processed.array.shape) == tuple(source["shape"])
        and tuple(processed.spacing) == tuple(source["spacing"])
        and tuple(processed.origin) == tuple(source["origin"])
        and tuple(processed.direction) == tuple(source["direction"])
    )
    if matches_input:
        return artifact(kind="scalar_field", coordinate_space="image_voxel", grid_of="image_path")
    return artifact(kind="scalar_field", coordinate_space="image_physical")


def run(
    *,
    image_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    params: Mapping[str, object] | None = None,
    params_path: str | os.PathLike[str] | None = None,
    label: int | None = None,
    voxel: bool = False,
    overwrite: bool = False,
    temporary_dir: str | os.PathLike[str] | None = None,
    max_temporary_mib: int | None = None,
) -> RunResult:
    """Extract radiomics features from one NIfTI/NRRD image and label mask.

    File I/O dependencies are included. No model weights or checkpoint directory are used.

    Args:
        image_path: 2D/3D ``.nii``, ``.nii.gz``, or ``.nrrd``/``.nhdr`` image.
        mask_path: An integer label mask. A geometry mismatch is refused unless the params
            enable ``correctMask`` or ``resampledPixelSpacing``.
        output_dir: Directory for the outputs; created if missing.
        params: A PyRadiomics-style params mapping (``setting``, ``featureClass`` and
            ``imageType`` sections); see :func:`mlxradiomics.config.parameters.parse_params`.
            Mutually exclusive with ``params_path``.
        params_path: A YAML file holding the same mapping.
        label: Optional mask-label override. ``None`` keeps the label from ``params`` or
            ``params_path``, or uses ``1`` if neither supplies a label.
        voxel: Boolean; extract one feature map per voxel instead of one value per feature.
            Evaluated ROI batches are spooled to one temporary file; one disk-backed
            float32 map at a time is expanded and written. Temporary files are removed
            on failure or success. Disk use grows with ROI size, feature count and image
            types.
        temporary_dir: Existing directory for voxel scratch; None uses the system default.
            Publication staging remains on the output filesystem for atomic renames.
        max_temporary_mib: Positive integer cap for logical voxel scratch (compact spool
            plus one dense float32 grid), in MiB. Checked before extraction and each batch.
            This excludes staged/final outputs and is not an RSS cap. Output filesystem
            capacity is admitted separately with a conservative staging estimate.
        overwrite: Boolean permission to replace existing outputs instead of refusing;
            outputs that alias the supplied image, mask or params paths are refused.
            Outputs that alias one another or are symbolic links are also refused.
            One run owns the output directory through extraction and publication;
            overlapping calls fail before extraction. Private staging files are
            published only after all writes finish. A publication failure restores
            replaced files and removes new outputs. An operating-system lock releases
            when the process exits; abrupt process termination can leave staging files.
            Readers may observe individual replacements during publication.
            Existing outputs are never replaced for detached NRRD inputs, because they
            may alias a referenced data file. Paired NIfTI/Analyze inputs are rejected.

    Returns:
        ``outputs`` maps roles to paths and ``artifacts`` describes them.
        ``outputs["features_json"]`` points to a
        metadata-plus-values (ROI) or metadata-plus-map-paths (voxel) JSON file. ROI mode
        also writes ``features_csv``; voxel mode also writes one float32 NIfTI per feature
        and image type at ``<output_dir>/maps/<prefix>_<feature>.nii.gz``, reusing the
        processed image geometry (zero outside the ROI, with nonfinite values possible
        inside). A map's artifact uses ``coordinate_space="image_voxel"`` and
        ``grid_of="image_path"`` only when the processed grid exactly matches the input's
        shape, spacing, origin and direction. After a crop or resampling it uses
        ``image_physical`` with no ``grid_of``, and ``source_geometry`` in the execution
        metadata records the input grid. Feature keys carry the image-type prefix, for
        example ``original_<family>_<Name>`` or ``wavelet-LLH_<family>_<Name>``. Spacing is
        in mm. Undefined or overflowed ROI scalars can be NaN or infinite. Report schema 2
        encodes them as the strings ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``;
        finite values remain JSON numbers. The same encoding applies to metadata.

    Raises:
        InvalidInputError: An input file is missing, unreadable or invalid, image/mask
            geometry disagrees without correction, params are invalid, an output aliases
            a supplied input, ``voxel`` is not a boolean, or outputs exist and ``overwrite``
            is false or is not a boolean.
        ModelExecutionError: Extraction itself failed.
    """
    return run_loaded_case(
        LoadedCase(Path(image_path).expanduser(), Path(mask_path).expanduser()),
        output_dir=output_dir, params=params, params_path=params_path, label=label,
        voxel=voxel, overwrite=overwrite, temporary_dir=temporary_dir,
        max_temporary_mib=max_temporary_mib,
    )


def run_loaded_case(
    case: LoadedCase,
    *,
    output_dir: str | os.PathLike[str],
    params: Mapping[str, object] | None = None,
    params_path: str | os.PathLike[str] | None = None,
    label: int | None = None,
    voxel: bool = False,
    overwrite: bool = False,
    temporary_dir: str | os.PathLike[str] | None = None,
    max_temporary_mib: int | None = None,
) -> RunResult:
    """Run one label with case-scoped source I/O and the ordinary publication contract."""
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be a boolean")
    if type(voxel) is not bool:  # JSON callers can send "false", which is truthy
        raise InvalidInputError("voxel must be a boolean")
    if max_temporary_mib is not None and (
        type(max_temporary_mib) is not int or max_temporary_mib <= 0
    ):
        raise InvalidInputError("max_temporary_mib must be a positive integer")
    if not voxel and (temporary_dir is not None or max_temporary_mib is not None):
        raise InvalidInputError("temporary storage options require voxel=True")

    source_image, source_mask = case.image_path, case.mask_path
    try:
        # Parse dimension-dependent parameters before the one full validated image read.
        dimension = case.dimension()
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        # SimpleITK raises RuntimeError, and gzip OSError or EOFError, for unreadable files.
        raise InvalidInputError(str(error)) from error
    parameters = _load_parameters(params, params_path, label, dimension)
    settings = parameters.settings
    transforms = parameters.transforms
    try:
        pair = prepare_image_pair(
            *case.volumes(), settings=settings,
            image_settings=parameters.image_settings,
        )
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise InvalidInputError(str(error)) from error
    processed = pair.image
    inputs = (
        source_image,
        source_mask,
        *((Path(params_path).expanduser(),) if params_path else ()),
    )
    extraction = dataclasses.replace(settings, normalize=False, remove_outliers=None)
    saved = publish_report(
        processed.array, pair.mask.array, output_dir=output_dir, settings=extraction,
        transforms=transforms, reference=processed, inputs=inputs,
        source_geometry=pair.source_geometry, image_settings=parameters.image_settings,
        overwrite=overwrite, temporary_dir=temporary_dir, max_temporary_mib=max_temporary_mib,
        metadata_settings=settings, voxel=voxel,
        input_volumes=(source_image, source_mask),
    )
    outputs = {"features_json": saved.report.as_posix()}
    artifacts = {"features_json": artifact(kind="report", coordinate_space="not_applicable")}
    if saved.table is not None:
        outputs["features_csv"] = saved.table.as_posix()
        artifacts["features_csv"] = artifact(
            kind="measurement", coordinate_space="not_applicable",
        )
    # Feature names have an image-type prefix, so they cannot collide with the report roles.
    for name, path in saved.maps.items():
        outputs[name] = path.as_posix()
        artifacts[name] = _scalar_field_artifact(processed=processed, source=pair.source_geometry)
    return {
        "outputs": outputs,
        "artifacts": artifacts,
        "details": saved.details,
    }
