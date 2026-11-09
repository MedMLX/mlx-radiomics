"""Registered export resource admission and interruption preserve existing results."""

from pathlib import Path
from typing import Never

import numpy as np
import pytest
import SimpleITK as sitk
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray

from mlxradiomics.imaging.io import Volume
from mlxradiomics.medmlx import run

MEAN = {"featureClass": {"firstorder": ["Mean"]}}


def _inputs(root: Path, shape: tuple[int, ...]) -> tuple[Path, Path]:
    image = sitk.GetImageFromArray(np.full(shape, 17, dtype=np.int16))
    mask = sitk.GetImageFromArray(np.ones(shape, dtype=np.uint8))
    paths = root / "image.nii.gz", root / "mask.nii.gz"
    sitk.WriteImage(image, paths[0])
    sitk.WriteImage(mask, paths[1])
    return paths


@pytest.mark.parametrize("problem", ["missing_directory", "dense_budget", "invalid_budget"])
def test_runner_rejects_invalid_scratch_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    paths = _inputs(tmp_path, (65, 65, 65) if problem == "dense_budget" else (4, 5))
    scratch = tmp_path / "scratch"
    if problem != "missing_directory":
        scratch.mkdir()

    def forbidden_extraction(*args: object, **kwargs: object) -> Never:
        raise AssertionError("resource admission must precede extraction")

    monkeypatch.setattr("mlxradiomics.workflows.engine.extract", forbidden_extraction)
    with pytest.raises(InvalidInputError):
        run(image_path=paths[0], mask_path=paths[1], output_dir=tmp_path / "out",
            params=MEAN, voxel=True, temporary_dir=scratch,
            max_temporary_mib=0 if problem == "invalid_budget" else 1)
    assert not list((tmp_path / "out").glob(".mlxradiomics-*"))
    if scratch.exists():
        assert not list(scratch.iterdir())


@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt])
def test_runner_removes_owned_scratch_and_staging_after_writer_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException],
) -> None:
    paths = _inputs(tmp_path, (4, 5))
    scratch, output = tmp_path / "scratch", tmp_path / "out"
    scratch.mkdir()
    marker = scratch / "unrelated"
    marker.write_bytes(b"preserved")
    (output / "maps").mkdir(parents=True)
    previous = {output / "features.json": b"previous report",
                output / "maps" / "original_firstorder_Mean.nii.gz": b"previous map"}
    for path, value in previous.items():
        path.write_bytes(value)

    def fail_write(
        path: str | Path, array: NDArray[np.float32] | NDArray[np.float64], *, reference: Volume,
    ) -> Never:
        raise interruption("injected writer interruption")

    monkeypatch.setattr("mlxradiomics.workflows.reporting.write_feature_map", fail_write)
    with pytest.raises(interruption, match="injected writer interruption"):
        run(image_path=paths[0], mask_path=paths[1], output_dir=output,
            params=MEAN, voxel=True, overwrite=True, temporary_dir=scratch, max_temporary_mib=1)
    assert {path: path.read_bytes() for path in previous} == previous
    assert list(scratch.iterdir()) == [marker]
    assert not list(output.glob(".mlxradiomics-*"))


def test_runner_reclaims_abandoned_staging_and_keeps_recovery_copies(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, (4, 5))
    output = tmp_path / "out"
    output.mkdir()
    abandoned = output / ".mlxradiomics-abcdefgh"
    abandoned.mkdir()
    (abandoned / ".mlxradiomics-staging").write_text("1\n")
    (abandoned / "features.json").write_bytes(b"staged but unpublished")
    empty = output / ".mlxradiomics-12345678"
    empty.mkdir()
    recovery = output / ".mlxradiomics-zyxwvuts"
    recovery.mkdir()
    (recovery / ".mlxradiomics-staging").write_text("1\n")
    backup = recovery / "backup-features.json"
    backup.write_bytes(b"recovery copy")
    unrelated = output / ".mlxradiomics-notours"
    unrelated.mkdir()
    (unrelated / "keep").write_bytes(b"not a private staging name")

    run(image_path=paths[0], mask_path=paths[1], output_dir=output, params=MEAN)

    assert not abandoned.exists()
    assert not empty.exists()
    assert backup.read_bytes() == b"recovery copy"
    assert (unrelated / "keep").read_bytes() == b"not a private staging name"
