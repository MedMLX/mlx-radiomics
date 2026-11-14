"""Multi-case and multi-label extraction keeps identities and reports every outcome."""

import json
from pathlib import Path
from typing import Never

import numpy as np
import pytest
import SimpleITK as sitk
from medmlx_core.errors import InvalidInputError
from radiomics_reference.real_images import hcc_crop

from mlxradiomics.workflows.study import run_study

MEAN = {"featureClass": {"firstorder": ["Mean"]}}


def _case(root: Path, case_id: str, offset: int) -> dict[str, str]:
    directory = root / case_id
    directory.mkdir(parents=True)
    values, liver, tumor = hcc_crop(np.s_[105:111, 215:222, 62:70])
    # A controlled HU offset distinguishes case identities; anatomical labels
    # remain the source-aligned, mutually exclusive liver and tumor annotations.
    labels = np.where(tumor != 0, 2, liver).astype(np.uint8)
    assert np.any(labels == 1) and np.any(labels == 2)
    image = sitk.GetImageFromArray((values + offset).astype(np.int16))
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    image_path, mask_path = directory / "image.nii.gz", directory / "mask.nii.gz"
    sitk.WriteImage(image, image_path.as_posix())
    sitk.WriteImage(mask, mask_path.as_posix())
    return {
        "case_id": case_id, "image_path": image_path.as_posix(),
        "mask_path": mask_path.as_posix(),
    }


def test_study_runs_every_case_and_label_independently(tmp_path: Path) -> None:
    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    outcomes = run_study(cases, tmp_path / "out", labels=(1, 2), params=MEAN)
    assert [(outcome["case_id"], outcome["label"], outcome["status"]) for outcome in outcomes] == [
        ("case-a", 1, "completed"), ("case-a", 2, "completed"),
        ("case-b", 1, "completed"), ("case-b", 2, "completed"),
    ]
    values: dict[tuple[str, int], float] = {}
    for outcome in outcomes:
        outputs = outcome["outputs"]
        assert outputs is not None
        payload = json.loads(Path(outputs["features_json"]).read_text())
        assert payload["details"]["label"] == outcome["label"]
        assert Path(outputs["features_json"]).parent == (
            tmp_path / "out" / outcome["case_id"] / f"label-{outcome['label']}"
        )
        values[(outcome["case_id"], outcome["label"])] = payload["features"][
            "original_firstorder_Mean"
        ]
    assert len(set(values.values())) == len(values)


@pytest.mark.parametrize("spatial", [False, True])
def test_study_reads_each_source_once_and_matches_independent_label_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spatial: bool,
) -> None:
    """Reuse source pixels while keeping normalization/crop/resegmentation label-local."""
    from mlxradiomics.imaging import spatial as spatial_io
    from mlxradiomics.medmlx import run

    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    params: dict[str, object] = {
        **MEAN,
        **({"setting": {"normalize": True, "resampledPixelSpacing": [.7, 1.1, 1.3],
                        "padDistance": 1, "resegmentRange": [-1.5, 1.5]}} if spatial else {}),
    }
    reads: list[tuple[str, Path]] = []
    read_image, read_mask = spatial_io.read_image, spatial_io.read_mask

    def image(path: Path) -> spatial_io.Volume:
        reads.append(("image", path))
        return read_image(path)

    def mask(path: Path) -> spatial_io.Volume:
        reads.append(("mask", path))
        return read_mask(path)

    monkeypatch.setattr(spatial_io, "read_image", image)
    monkeypatch.setattr(spatial_io, "read_mask", mask)
    outcomes = run_study(cases, tmp_path / "study", labels=(5, 1, 2), params=params)
    assert reads == [
        (kind, Path(case[f"{kind}_path"])) for case in cases for kind in ("image", "mask")
    ]
    assert [outcome["status"] for outcome in outcomes] == [
        "failed", "completed", "completed", "failed", "completed", "completed",
    ]
    for case in cases:
        for label in (1, 2):
            baseline = run(
                image_path=case["image_path"], mask_path=case["mask_path"],
                output_dir=tmp_path / "independent" / case["case_id"] / f"label-{label}",
                params=params, label=label,
            )
            actual = tmp_path / "study" / case["case_id"] / f"label-{label}" / "features.json"
            assert json.loads(actual.read_text()) == json.loads(
                Path(baseline["outputs"]["features_json"]).read_text(),
            )


def test_study_records_missing_label_and_continues(tmp_path: Path) -> None:
    case = _case(tmp_path / "inputs", "case-a", 0)
    outcomes = run_study([case], tmp_path / "out", labels=(1, 5), params=MEAN)
    assert [outcome["status"] for outcome in outcomes] == ["completed", "failed"]
    assert outcomes[1]["error_type"] == "InvalidInputError"
    assert outcomes[1]["outputs"] is None
    assert outcomes[1]["artifacts"] is None
    assert "absent" in (outcomes[1]["error_message"] or "")
    assert (tmp_path / "out" / "case-a" / "label-1" / "features.json").is_file()


def test_study_records_source_precision_failure_and_continues(tmp_path: Path) -> None:
    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    source = sitk.ReadImage(cases[0]["image_path"])
    values = sitk.GetArrayFromImage(source).astype(np.float64)
    values[1, 1, 1] = 2.0**-130
    image = sitk.GetImageFromArray(values)
    image.CopyInformation(source)
    sitk.WriteImage(image, cases[0]["image_path"])

    outcomes = run_study(cases, tmp_path / "out", params=MEAN, voxel=True)

    assert [outcome["status"] for outcome in outcomes] == ["failed", "completed"]
    assert outcomes[0]["error_type"] == "ModelExecutionError"
    assert "nonzero float64 sources below 2**-126" in (outcomes[0]["error_message"] or "")
    assert not (tmp_path / "out" / "case-a" / "label-1" / "features.json").exists()
    assert (tmp_path / "out" / "case-b" / "label-1" / "features.json").is_file()


def test_study_propagates_programming_errors_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    output = tmp_path / "out"
    report = output / "case-a" / "label-1" / "features.json"
    report.parent.mkdir(parents=True)
    report.write_text("previous report\n")
    failure = TypeError("injected extraction bug")

    def fail_extraction(*args: object, **kwargs: object) -> Never:
        raise failure

    monkeypatch.setattr("mlxradiomics.workflows.reporting.extract", fail_extraction)
    with pytest.raises(TypeError, match="injected extraction bug") as caught:
        run_study(cases, output, params=MEAN, overwrite=True)

    assert caught.value is failure
    assert report.read_text() == "previous report\n"
    assert not list(output.rglob(".mlxradiomics-*"))
    assert not (output / "case-b").exists()


def test_study_records_invalid_transform_dimension_and_continues(tmp_path: Path) -> None:
    cases = [
        _case(tmp_path / "inputs", "case-2d", 0),
        _case(tmp_path / "inputs", "case-3d", 20),
    ]
    for key in ("image_path", "mask_path"):
        source = cases[0][key]
        values = sitk.GetArrayFromImage(sitk.ReadImage(source))[2]
        sitk.WriteImage(sitk.GetImageFromArray(values), source)
    outcomes = run_study(
        cases, tmp_path / "out", voxel=True,
        params={
            "setting": {"lbp3DLevels": 1, "lbp3DIcosphereSubdivision": 0},
            "imageType": {"LBP3D": {}},
            "featureClass": {"firstorder": ["Mean"]},
        },
    )
    assert [outcome["status"] for outcome in outcomes] == ["failed", "completed"]
    assert outcomes[0]["error_type"] == "InvalidInputError"
    assert "requires a 3D image" in (outcomes[0]["error_message"] or "")
    outputs, artifacts = outcomes[1]["outputs"], outcomes[1]["artifacts"]
    assert outputs is not None
    assert artifacts is not None
    maps = {
        role: path for role, path in outputs.items()
        if artifacts[role]["kind"] == "scalar_field"
    }
    assert set(maps) == {"lbp-3D-m1_firstorder_Mean", "lbp-3D-k_firstorder_Mean"}
    assert all(Path(path).is_file() for path in maps.values())


def test_study_rejects_duplicate_or_unsafe_identifiers(tmp_path: Path) -> None:
    case = _case(tmp_path / "inputs", "case-a", 0)
    with pytest.raises(InvalidInputError, match="unique"):
        run_study([case, case], tmp_path / "out", params=MEAN)
    unsafe = dict(case)
    unsafe["case_id"] = "../escape"
    with pytest.raises(InvalidInputError, match="path component"):
        run_study([unsafe], tmp_path / "out", params=MEAN)
    with pytest.raises(InvalidInputError, match="unique"):
        run_study([case], tmp_path / "out", labels=(1, 1), params=MEAN)


@pytest.mark.parametrize("identifiers", [("case-a", "CASE-A"), ("caf\u00e9", "cafe\u0301")])
def test_study_rejects_filesystem_name_collisions_before_replacement(
    tmp_path: Path, identifiers: tuple[str, str],
) -> None:
    source = _case(tmp_path / "inputs", "source", 0)
    cases = [dict(source, case_id=identifier) for identifier in identifiers]
    report = tmp_path / "out" / identifiers[0] / "label-1" / "features.json"
    report.parent.mkdir(parents=True)
    report.write_text("previous result\n")
    with pytest.raises(InvalidInputError, match="unique"):
        run_study(cases, tmp_path / "out", params=MEAN, overwrite=True)
    assert report.read_text() == "previous result\n"


def test_study_rejects_aliased_destinations_before_replacement(tmp_path: Path) -> None:
    source = _case(tmp_path / "inputs", "source", 0)
    cases = [dict(source, case_id=identifier) for identifier in ("case-a", "case-b")]
    report = tmp_path / "out" / "case-a" / "label-1" / "features.json"
    report.parent.mkdir(parents=True)
    report.write_text("previous result\n")
    (tmp_path / "out" / "case-b").symlink_to(report.parent.parent, target_is_directory=True)
    with pytest.raises(InvalidInputError, match="destinations must be unique"):
        run_study(cases, tmp_path / "out", params=MEAN, overwrite=True)
    assert report.read_text() == "previous result\n"


def test_study_classifies_shared_malformed_params_yaml(tmp_path: Path) -> None:
    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    params = tmp_path / "params.yaml"
    params.write_text("setting: [", encoding="utf-8")
    outcomes = run_study(cases, tmp_path / "out", labels=(1, 2), params_path=params)
    assert [(outcome["status"], outcome["error_type"]) for outcome in outcomes] == [
        ("failed", "InvalidInputError"),
        ("failed", "InvalidInputError"),
        ("failed", "InvalidInputError"),
        ("failed", "InvalidInputError"),
    ]
    assert all("invalid params YAML" in (outcome["error_message"] or "")
               for outcome in outcomes)


@pytest.mark.parametrize("alias", ["path", "hardlink"])
def test_study_preserves_future_case_input_in_earlier_output_directory(
    tmp_path: Path, alias: str,
) -> None:
    cases = [_case(tmp_path / "inputs", "case-a", 0), _case(tmp_path / "inputs", "case-b", 20)]
    future_input = (
        tmp_path / "out" / "case-a" / "label-1" / "maps" / "original_firstorder_Mean.nii.gz"
    )
    future_input.parent.mkdir(parents=True)
    original = Path(cases[1]["image_path"]).read_bytes()
    if alias == "hardlink":
        future_input.hardlink_to(cases[1]["image_path"])
        message = "output aliases an input"
    else:
        future_input.write_bytes(original)
        cases[1]["image_path"] = future_input.as_posix()
        message = "outside all case output directories"
    with pytest.raises(InvalidInputError, match=message):
        run_study(cases, tmp_path / "out", params=MEAN, voxel=True, overwrite=True)
    assert future_input.read_bytes() == original
    assert Path(cases[1]["image_path"]).read_bytes() == original
