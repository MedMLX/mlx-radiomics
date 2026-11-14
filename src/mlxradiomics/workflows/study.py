"""Multi-case and multi-label study extraction through :func:`mlxradiomics.medmlx.run`.

Each requested case and label gets its own output directory, so filenames cannot collide
and every combination has an explicit status. Failures are reported per combination and
do not prevent later combinations from running; cancellation still propagates.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from medmlx_core.errors import InvalidInputError, ModelExecutionError
from medmlx_core.outputs import Artifact

from mlxradiomics.imaging.spatial import LoadedCase
from mlxradiomics.medmlx import run_loaded_case
from mlxradiomics.workflows.publication import reject_input_aliases

_CASE_KEYS: frozenset[str] = frozenset(("case_id", "image_path", "mask_path"))


class StudyCase(TypedDict):
    """One image/mask pair in a study."""

    case_id: str
    image_path: str
    mask_path: str


class CaseOutcome(TypedDict):
    """Status for one ``(case_id, label)`` combination.

    ``status`` is ``completed`` or ``failed``. Completed entries carry the runner's
    ``outputs`` and ``artifacts`` from :func:`mlxradiomics.medmlx.run`; failed entries
    carry the exception type and message.
    """

    case_id: str
    label: int
    status: Literal["completed", "failed"]
    output_dir: str
    outputs: dict[str, str] | None
    artifacts: dict[str, Artifact] | None
    error_type: str | None
    error_message: str | None


def _case(value: object) -> StudyCase:
    """Validate one study case mapping; ``case_id`` must be a single safe path component."""
    if not isinstance(value, Mapping):
        raise InvalidInputError("each study case must be a mapping")
    mapping = cast(Mapping[str, object], value)
    keys = set(mapping)
    if len(keys) != 3 or not all(key in _CASE_KEYS for key in keys):
        missing = sorted(_CASE_KEYS - keys)
        unexpected = sorted(keys - _CASE_KEYS)
        raise InvalidInputError(
            f"study case keys must be exactly {sorted(_CASE_KEYS)}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    case_id, image, mask = mapping["case_id"], mapping["image_path"], mapping["mask_path"]
    if not isinstance(case_id, str) or not case_id:
        raise InvalidInputError("study case_id must be a nonempty string")
    if case_id in {".", ".."} or "/" in case_id or "\\" in case_id or "\0" in case_id:
        raise InvalidInputError("study case_id must be a single safe path component")
    if not isinstance(image, str) or not isinstance(mask, str):
        raise InvalidInputError("study image_path and mask_path must be strings")
    return {"case_id": case_id, "image_path": image, "mask_path": mask}


def _labels(values: Sequence[int]) -> tuple[int, ...]:
    """Validate study labels as a nonempty sequence of unique positive integers."""
    labels = tuple(values)
    if not labels or any(type(label) is not int or label <= 0 for label in labels):
        raise InvalidInputError("study labels must be a nonempty sequence of positive integers")
    if len(set(labels)) != len(labels):
        raise InvalidInputError("study labels must be unique")
    return labels


def run_study(
    cases: Sequence[StudyCase | Mapping[str, object]],
    output_dir: str | os.PathLike[str],
    *,
    labels: Sequence[int] = (1,),
    params: Mapping[str, object] | None = None,
    params_path: str | os.PathLike[str] | None = None,
    voxel: bool = False,
    overwrite: bool = False,
    temporary_dir: str | os.PathLike[str] | None = None,
    max_temporary_mib: int | None = None,
) -> list[CaseOutcome]:
    """Run one image/mask pair per case and label, reporting every outcome.

    Outputs go to ``<output_dir>/<case_id>/label-<label>``. Case identifiers must be
    safe single path components, unique after case folding and Unicode normalization
    for case-insensitive filesystems. Labels must be unique positive integers. All case
    and label arguments and destination aliases are checked before extraction starts.
    Study image, mask and parameter files must live outside every case output directory,
    including through symbolic links; existing files there cannot be hard links to
    study inputs. Input, I/O and extraction failures are recorded and the remaining
    combinations still run. Unexpected programming errors propagate immediately.
    Malformed shared ``params`` or ``params_path`` content is an input error for every
    combination: each outcome records ``InvalidInputError`` and feature extraction does
    not start.

    Args:
        cases: Mappings with exactly ``case_id``, ``image_path`` and ``mask_path``.
        output_dir: Study root; each combination writes ``<case_id>/label-<label>``.
        labels: Mask labels to extract for every case.
        params: PyRadiomics-style parameter mapping shared by every combination.
            Mutually exclusive with ``params_path``.
        params_path: YAML file holding the same mapping.
        voxel: Export voxel maps instead of ROI features.
        overwrite: Replace existing outputs instead of refusing them.
        temporary_dir: Existing directory for voxel scratch.
        max_temporary_mib: Positive cap in MiB for logical voxel scratch.

    Returns:
        One outcome per case and label, in case-major order.

    Raises:
        InvalidInputError: If cases, identifiers, labels, boolean flags or destination
            aliases fail preflight. Other runner argument, parameter and output-directory
            errors are reported in each affected combination's outcome.
    """
    if type(voxel) is not bool:
        raise InvalidInputError("voxel must be a boolean")
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be a boolean")
    if not cases:
        raise InvalidInputError("study requires at least one case")
    parsed = tuple(_case(value) for value in cases)
    identifiers = tuple(unicodedata.normalize("NFD", case["case_id"]).casefold() for case in parsed)
    if len(set(identifiers)) != len(identifiers):
        raise InvalidInputError(
            "study case_id values must be unique after case folding and Unicode normalization"
        )
    selected_labels = _labels(labels)
    root = Path(output_dir).expanduser().resolve()
    destinations = tuple(
        (root / case["case_id"] / f"label-{label}").resolve()
        for case in parsed for label in selected_labels
    )
    if len(set(destinations)) != len(destinations):
        raise InvalidInputError("study output destinations must be unique")
    inputs = tuple(
        Path(value).expanduser().resolve()
        for case in parsed for value in (case["image_path"], case["mask_path"])
    )
    if params_path is not None:
        inputs += (Path(params_path).expanduser().resolve(),)
    if any(source.is_relative_to(destination) for source in inputs for destination in destinations):
        raise InvalidInputError("study inputs must be outside all case output directories")
    existing_inputs = tuple(source for source in inputs if source.is_file())
    existing_outputs = tuple(
        path for destination in destinations for path in destination.rglob("*") if path.is_file()
    )
    reject_input_aliases(existing_outputs, existing_inputs)
    outcomes: list[CaseOutcome] = []
    for case in parsed:
        loaded = LoadedCase(
            Path(case["image_path"]).expanduser(), Path(case["mask_path"]).expanduser(),
        )
        for label in selected_labels:
            destination = root / case["case_id"] / f"label-{label}"
            try:
                result = run_loaded_case(
                    loaded,
                    output_dir=destination, params=params, params_path=params_path,
                    label=label, voxel=voxel, overwrite=overwrite,
                    temporary_dir=temporary_dir, max_temporary_mib=max_temporary_mib,
                )
            except (InvalidInputError, ModelExecutionError, OSError) as error:
                outcomes.append({
                    "case_id": case["case_id"], "label": label, "status": "failed",
                    "output_dir": destination.as_posix(), "outputs": None, "artifacts": None,
                    "error_type": type(error).__name__, "error_message": str(error),
                })
            else:
                outcomes.append({
                    "case_id": case["case_id"], "label": label, "status": "completed",
                    "output_dir": destination.as_posix(), "outputs": result["outputs"],
                    "artifacts": result["artifacts"],
                    "error_type": None, "error_message": None,
                })
        del loaded
    return outcomes
