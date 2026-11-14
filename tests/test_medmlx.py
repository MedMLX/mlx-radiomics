"""The medmlx runner end to end, and its argument rejection."""

import csv
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from threading import Event
from typing import Never, cast

import numpy as np
import pytest
import SimpleITK as sitk
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from medmlx_core.outputs import validate_outputs
from radiomics_reference.mathematics import masked_neighborhood_mean
from radiomics_reference.real_images import hcc_crop

import mlxradiomics
from mlxradiomics.contracts.json import JsonValue
from mlxradiomics.imaging.io import read_image
from mlxradiomics.medmlx import run

MEAN = {"featureClass": {"firstorder": ["Mean"]}}
# Untyped view of the runner for deserialized invalid argument values.
run_untyped = cast(Callable[..., object], run)


def _write_pair(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    values, labels, _ = hcc_crop(np.s_[105:109, 215:220, 62:68])
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(labels)
    image.SetSpacing((1.5, 1.0, 0.8))
    image.SetOrigin((3.0, -2.0, 5.0))
    mask.CopyInformation(image)
    image_path, mask_path = directory / "image.nii.gz", directory / "mask.nii.gz"
    sitk.WriteImage(image, image_path.as_posix())
    sitk.WriteImage(mask, mask_path.as_posix())
    return image_path, mask_path


def test_run_roi_writes_independent_first_order_values(tmp_path: Path) -> None:
    image_path, mask_path = _write_pair(tmp_path / "inputs")
    params = tmp_path / "params.yaml"
    params.write_text("featureClass:\n  firstorder: [Mean, Variance]\n")
    output = tmp_path / "out"
    output.mkdir()
    (output / "features.json").write_bytes(b"previous metadata")
    (output / "features.csv").write_bytes(b"previous features")
    unrelated = tmp_path / "unrelated.csv"
    unrelated.hardlink_to(output / "features.csv")
    result = run(image_path=image_path, mask_path=mask_path, output_dir=output,
                 params_path=params, overwrite=True)
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float64)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    selected = array[labels == 1]
    expected = {"original_firstorder_Mean": float(np.mean(selected)),
                "original_firstorder_Variance": float(np.var(selected))}
    paths, artifacts = validate_outputs(result)
    assert paths == {
        "features_json": str(output / "features.json"),
        "features_csv": str(output / "features.csv"),
    }
    assert {role: entry["kind"] for role, entry in artifacts.items()} == {
        "features_json": "report", "features_csv": "measurement",
    }
    payload = json.loads(Path(result["files"]["features_json"]).read_text())
    assert payload["features"] == pytest.approx(expected)
    with Path(cast(str, result["files"]["features_csv"])).open(newline="") as handle:
        header, values = csv.reader(handle)
    assert dict(zip(header, map(float, values), strict=True)) == pytest.approx(expected)
    assert unrelated.read_bytes() == b"previous features"


def test_run_voxel_writes_local_means_on_the_input_grid(tmp_path: Path) -> None:
    image_path, mask_path = _write_pair(tmp_path / "inputs")
    result = run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
                 params=MEAN, voxel=True)
    source, mask = read_image(image_path), read_image(mask_path)
    expected = masked_neighborhood_mean(source.array.astype(np.float64), mask.array == 1)
    name = "original_firstorder_Mean"
    paths, artifacts = validate_outputs(result)
    assert set(paths) == set(artifacts) == {"features_json", name}
    assert paths[name] == result["files"]["maps"][name]
    assert artifacts[name]["kind"] == "scalar_field"
    assert artifacts[name].get("grid_of") == "image_path"
    assert artifacts[name]["coordinate_space"] == "image_voxel"
    output = read_image(result["files"]["maps"][name])
    np.testing.assert_allclose(output.array.astype(np.float64), expected, rtol=1e-4, atol=1e-4)
    assert output.spacing == source.spacing
    assert output.origin == source.origin
    assert output.direction == source.direction
    assert result["details"]["feature_count"] == 1
    payload = json.loads(Path(result["files"]["features_json"]).read_text())
    assert payload["maps"] == result["files"]["maps"]


def _reject_json_constant(value: str) -> Never:
    """Make Python's JSON decoder reject non-standard numeric tokens like strict clients."""
    raise ValueError(f"invalid JSON numeric token: {value}")


@pytest.mark.parametrize(
    "bin_width,encoded", [(float("nan"), "NaN"), (float("inf"), "Infinity"),
                          (-float("inf"), "-Infinity")],
)
def test_run_roi_reports_nonfinite_values_as_json_strings(
    tmp_path: Path, bin_width: float, encoded: str,
) -> None:
    image_path, mask_path = _write_pair(tmp_path / "inputs")
    # Constant intensities give mathematically undefined skewness; a separate
    # energy shift produces overflow without relying on unstable moment arithmetic.
    background, _, _ = hcc_crop(np.s_[0:4, 0:5, 60:66])
    assert np.all(background == -2048)
    # Rebase constant source background to zero so JSON also exercises valid-zero counts.
    image = background - background.flat[0]
    node = sitk.GetImageFromArray(image)
    node.CopyInformation(sitk.ReadImage(mask_path))
    sitk.WriteImage(node, image_path)
    # Fixed-count binning ignores binWidth; its non-finite metadata must still be valid JSON.
    params = {"setting": {"binCount": 1, "binWidth": bin_width, "voxelArrayShift": 1e30,
                          "zeroVarianceMoments": "nan"},
              "featureClass": {"firstorder": ["Mean", "Energy", "Skewness"]}}
    result = run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
                 params=params)
    payload = cast(dict[str, JsonValue], json.loads(
        Path(result["files"]["features_json"]).read_text(), parse_constant=_reject_json_constant,
    ))
    assert payload["schema_version"] == 2
    features = payload["features"]
    assert isinstance(features, dict)
    assert features == {"original_firstorder_Mean": 0.0,
                        "original_firstorder_Energy": "Infinity",
                        "original_firstorder_Skewness": "NaN"}
    assert payload["details"] == result["details"]
    settings = result["details"]["execution"]["settings"]
    assert isinstance(settings, dict)
    assert settings["bin_width"] == encoded
    execution = result["details"]["execution"]
    counts = execution["feature_validity"]
    assert isinstance(counts, dict)
    assert counts == {
        "original_firstorder_Mean": {"finite": 1, "valid_zero": 1, "nan": 0,
                                     "positive_infinity": 0, "negative_infinity": 0,
                                     "excluded": 0},
        "original_firstorder_Energy": {"finite": 0, "valid_zero": 0, "nan": 0,
                                       "positive_infinity": 1, "negative_infinity": 0,
                                       "excluded": 0},
        "original_firstorder_Skewness": {"finite": 0, "valid_zero": 0, "nan": 1,
                                         "positive_infinity": 0, "negative_infinity": 0,
                                         "excluded": 0},
    }
    runtime = execution["runtime"]
    assert isinstance(runtime, dict)
    versions = runtime["versions"]
    assert isinstance(versions, dict)
    assert versions["mlx"] == version("mlx")
    assert versions["SimpleITK"] == version("SimpleITK")
    assert execution["source_geometry"] == execution["image_geometry"]


@pytest.mark.parametrize(
    "problem", ["missing_image", "bad_suffix", "label_out_of_range", "non_bool_voxel",
                "existing_output"],
)
def test_run_rejects_bad_arguments(tmp_path: Path, problem: str) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    arguments: dict[str, object] = {"image_path": image, "mask_path": mask, "output_dir": output,
                                    "params": MEAN}
    if problem == "missing_image":
        arguments["image_path"] = tmp_path / "absent.nii.gz"
    elif problem == "bad_suffix":
        arguments["image_path"] = image.rename(tmp_path / "image.txt")
    elif problem == "label_out_of_range":
        arguments["label"] = 0
    elif problem == "non_bool_voxel":
        arguments["voxel"] = "false"
    else:
        output.mkdir()
        (output / "features.json").write_bytes(b"previous")
    with pytest.raises(InvalidInputError):
        run_untyped(**arguments)


@pytest.mark.parametrize(
    "problem", ["metadata_directory", "map_parent_file", "dangling_symlink",
                "input_symlink", "input_hardlink", "output_hardlink", "params_alias"],
)
def test_run_rejects_unsafe_targets_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    output.mkdir()
    metadata, csv_path = output / "features.json", output / "features.csv"
    if problem == "metadata_directory":
        metadata.mkdir()
    elif problem == "map_parent_file":
        (output / "maps").write_bytes(b"unrelated")
    elif problem == "dangling_symlink":
        metadata.symlink_to(output / "absent")
    elif problem == "input_symlink":
        metadata.symlink_to(image)
    elif problem == "input_hardlink":
        metadata.hardlink_to(image)
    elif problem == "output_hardlink":
        metadata.write_bytes(b"previous")
        csv_path.hardlink_to(metadata)
    else:
        metadata.write_text("featureClass:\n  firstorder: [Mean]\n")
    image_bytes, mask_bytes = image.read_bytes(), mask.read_bytes()

    def forbidden_extraction(*args: object, **kwargs: object) -> Never:
        raise AssertionError("unsafe targets must be rejected before extraction")

    monkeypatch.setattr("mlxradiomics.workflows.engine.extract", forbidden_extraction)
    with pytest.raises(InvalidInputError, match="output|aliases"):
        run(image_path=image, mask_path=mask, output_dir=output,
            params=None if problem == "params_alias" else MEAN,
            params_path=metadata if problem == "params_alias" else None,
            voxel=problem == "map_parent_file", overwrite=True)
    assert image.read_bytes() == image_bytes
    assert mask.read_bytes() == mask_bytes


@pytest.mark.parametrize(
    "voxel,existing,rollback_failure",
    [(False, False, False), (True, False, False), (False, True, False), (True, True, False),
     (False, True, True), (True, True, True)],
)
def test_run_rolls_back_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    voxel: bool, existing: bool, rollback_failure: bool,
) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    data = (output / "maps" / "original_firstorder_Mean.nii.gz"
            if voxel else output / "features.csv")
    metadata = output / "features.json"
    data.parent.mkdir(parents=True)
    previous = (
        {data: b"previous feature output", metadata: b"previous metadata"} if existing else {}
    )
    for path, value in previous.items():
        path.write_bytes(value)
    real_replace, real_link = os.replace, os.link
    failed = False

    def fail_metadata_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str],
    ) -> None:
        nonlocal failed
        if Path(destination) == metadata and not failed:
            failed = True
            raise OSError("injected publication failure")
        if Path(destination) == data and failed and rollback_failure:
            raise OSError("injected rollback failure")
        real_replace(source, destination)

    def fail_metadata_link(
        source: str | os.PathLike[str], destination: str | os.PathLike[str],
        *, follow_symlinks: bool = True,
    ) -> None:
        nonlocal failed
        if Path(destination) == metadata and not failed:
            failed = True
            raise OSError("injected publication failure")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "replace", fail_metadata_replace)
    monkeypatch.setattr(os, "link", fail_metadata_link)
    expected_error = ModelExecutionError if rollback_failure else OSError
    message = "cannot restore outputs" if rollback_failure else "injected publication failure"
    with pytest.raises(expected_error, match=message):
        run(image_path=image, mask_path=mask, output_dir=output, params=MEAN,
            voxel=voxel, overwrite=existing)
    assert failed
    if rollback_failure:
        recovery, = output.glob(".mlxradiomics-*")
        assert (recovery / f"backup-{data.name}").read_bytes() == previous[data]
        assert metadata.read_bytes() == previous[metadata]
    else:
        assert {path: path.read_bytes() for path in (data, metadata) if path.exists()} == previous
        assert not list(output.glob(".mlxradiomics-*"))


_PUBLICATION_SIGNAL_SCRIPT = """
import json
import os
from pathlib import Path
import signal
import sys

from mlxradiomics.medmlx import run

image, mask, output = map(Path, sys.argv[1:4])
signum = getattr(signal, sys.argv[4])
existing, repeat = sys.argv[5] == "1", sys.argv[6] == "1"
policy = sys.argv[7]
handled = []

def custom_handler(received, frame):
    handled.append(received)

if policy == "custom":
    signal.signal(signal.SIGTERM, custom_handler)
elif policy == "ignored":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
previous_handler = signal.getsignal(signal.SIGTERM)
targets = {output / "features.csv", output / "features.json"}
sent = []
real_replace, real_link = os.replace, os.link

def send_signal():
    sent.append(signum)
    (output.parent / "publication-signals.json").write_text(json.dumps(sent))
    os.kill(os.getpid(), signum)

def interrupt_after_mutation(source, destination):
    destination = Path(destination)
    if destination not in targets:
        return
    if not sent:
        send_signal()
    elif repeat and len(sent) == 1 and Path(source).name.startswith("backup-"):
        send_signal()

def replace(source, destination):
    real_replace(source, destination)
    interrupt_after_mutation(source, destination)

def link(source, destination, *, follow_symlinks=True):
    real_link(source, destination, follow_symlinks=follow_symlinks)
    interrupt_after_mutation(source, destination)

os.replace, os.link = replace, link
try:
    run(image_path=image, mask_path=mask, output_dir=output,
        params={"featureClass": {"firstorder": ["Mean"]}}, overwrite=existing)
except KeyboardInterrupt:
    if signum != signal.SIGINT:
        raise
    assert signal.getsignal(signal.SIGTERM) is previous_handler
else:
    assert policy in ("custom", "ignored"), "default signal did not interrupt publication"
    assert signal.getsignal(signal.SIGTERM) is previous_handler
    assert handled == ([signal.SIGTERM] if policy == "custom" else [])
"""


@pytest.mark.parametrize(
    "signum,existing,repeat,policy",
    [
        (signal.SIGTERM, False, False, "default"),
        (signal.SIGTERM, True, False, "default"),
        (signal.SIGTERM, True, True, "default"),
        (signal.SIGINT, False, False, "default"),
        (signal.SIGINT, True, False, "default"),
        (signal.SIGTERM, False, False, "custom"),
        (signal.SIGTERM, False, False, "ignored"),
    ],
)
def test_run_publication_handles_process_signals_without_mixing_outputs(
    tmp_path: Path, signum: signal.Signals, existing: bool, repeat: bool, policy: str,
) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    output.mkdir()
    metadata, data = output / "features.json", output / "features.csv"
    previous = (
        {metadata: b"previous metadata", data: b"previous feature output"} if existing else {}
    )
    for path, value in previous.items():
        path.write_bytes(value)
    environment = dict(os.environ)
    # Keep child processes on the parent's source or installed-wheel package.
    environment["PYTHONPATH"] = str(Path(mlxradiomics.__file__).resolve().parents[1])

    # Signal after the real syscall succeeds, before publication records its return.
    result = subprocess.run(
        [sys.executable, "-c", _PUBLICATION_SIGNAL_SCRIPT, str(image), str(mask), str(output),
         signum.name, "1" if existing else "0", "1" if repeat else "0", policy],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )

    expected_code = -signal.SIGTERM if signum == signal.SIGTERM and policy == "default" else 0
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert json.loads((tmp_path / "publication-signals.json").read_text()) == [int(signum)] * (
        2 if repeat else 1
    )
    if policy == "default":
        assert {path: path.read_bytes() for path in (metadata, data) if path.exists()} == previous
    else:
        assert metadata.is_file() and data.is_file()
    assert not list(output.glob(".mlxradiomics-*"))


def test_run_refuses_a_concurrent_call_to_the_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    publishing, resume = Event(), Event()
    real_replace = os.replace
    blocked = False

    def pause_publication(
        source: str | os.PathLike[str], destination: str | os.PathLike[str],
    ) -> None:
        nonlocal blocked
        if Path(destination) == output / "features.csv" and not blocked:
            blocked = True
            publishing.set()
            if not resume.wait(timeout=20):
                raise TimeoutError("publication was not resumed")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", pause_publication)
    # overwrite=True exercises replacement while the first call owns the directory.
    with ThreadPoolExecutor(max_workers=1) as workers:
        first = workers.submit(run, image_path=image, mask_path=mask, output_dir=output,
                               params=MEAN, overwrite=True)
        try:
            assert publishing.wait(timeout=20)
            with pytest.raises(InvalidInputError, match="another run"):
                run(image_path=image, mask_path=mask, output_dir=output, params=MEAN,
                    overwrite=True)
        finally:
            resume.set()
        result = first.result(timeout=20)
    assert Path(result["files"]["features_json"]).is_file()
    assert {path.name for path in output.iterdir()} == {"features.json", "features.csv"}


def test_run_preserves_an_output_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, mask = _write_pair(tmp_path / "inputs")
    output = tmp_path / "out"
    metadata = output / "features.json"
    real_replace, real_link = os.replace, os.link

    def create_late_output(destination: str | os.PathLike[str]) -> None:
        if Path(destination) == metadata:
            metadata.write_bytes(b"created by another writer")

    def replace_after_creation(
        source: str | os.PathLike[str], destination: str | os.PathLike[str],
    ) -> None:
        create_late_output(destination)
        real_replace(source, destination)

    def link_after_creation(
        source: str | os.PathLike[str], destination: str | os.PathLike[str],
        *, follow_symlinks: bool = True,
    ) -> None:
        create_late_output(destination)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "replace", replace_after_creation)
    monkeypatch.setattr(os, "link", link_after_creation)
    with pytest.raises(InvalidInputError, match="overwrite=True"):
        run(image_path=image, mask_path=mask, output_dir=output, params=MEAN)
    assert metadata.read_bytes() == b"created by another writer"
    assert {path.name for path in output.iterdir()} == {"features.json"}


def test_invalid_filter_dimension_is_an_input_error(tmp_path: Path) -> None:
    """A planar filter on a 2D image is a rejected request, not an execution failure."""
    values, labels, _ = hcc_crop(np.s_[103:111, 214:222, 64])
    image = sitk.GetImageFromArray(values.astype(np.float32))
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    sitk.WriteImage(image, tmp_path / "image.nii.gz")
    sitk.WriteImage(mask, tmp_path / "mask.nii.gz")
    with pytest.raises(InvalidInputError, match="3-D image"):
        run(
            image_path=tmp_path / "image.nii.gz", mask_path=tmp_path / "mask.nii.gz",
            output_dir=tmp_path / "out", voxel=True,
            params={"imageType": {"Mean": {"support": 3, "planar": True}},
                    "featureClass": {"firstorder": ["Mean"]}},
        )
    assert not (tmp_path / "out" / "features.json").exists()
