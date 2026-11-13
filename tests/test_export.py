"""Disk-backed derived-image export keeps values, prefixes, geometry and cleanup."""

import json
from pathlib import Path
from typing import Never, cast

import numpy as np
import pytest
import SimpleITK as sitk
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings, export, export_files, extract_files
from mlxradiomics.config.transforms import (
    ImageType,
    LoGSettings,
    TransformSettings,
    WaveletSettings,
)
from mlxradiomics.imaging.io import Volume, read_image
from mlxradiomics.medmlx import run


def _inputs(root: Path) -> tuple[Path, Path, NDArray[np.float64], NDArray[np.uint8]]:
    """Write paired HCC-TACE voxels on a controlled export-test grid."""
    root.mkdir(parents=True, exist_ok=True)
    values, labels, _ = hcc_crop(np.s_[103:112, 214:224, 60:71])
    image = sitk.GetImageFromArray(values)
    mask = sitk.GetImageFromArray(labels)
    image.SetSpacing((1.5, 1.0, 0.8))
    image.SetOrigin((3.0, -2.0, 5.0))
    mask.CopyInformation(image)
    paths = root / "image.nii.gz", root / "mask.nii.gz"
    sitk.WriteImage(image, paths[0].as_posix())
    sitk.WriteImage(mask, paths[1].as_posix())
    return paths[0], paths[1], values, labels


def test_python_and_medmlx_exports_share_reports_and_physical_map_values(tmp_path: Path) -> None:
    paths = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    ct, labels, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    for values, path in zip(
        (ct, labels), paths, strict=True,
    ):
        image = sitk.GetImageFromArray(values)
        image.SetSpacing((3.0, 2.0))
        image.SetOrigin((7.0, 11.0))
        sitk.WriteImage(image, str(path))
    exported = export_files(
        *paths, tmp_path / "python", settings=Settings(features=("firstorder_Mean",)),
    )
    adapted = run(
        image_path=paths[0], mask_path=paths[1], output_dir=tmp_path / "medmlx", voxel=True,
        params={"featureClass": {"firstorder": ["Mean"]}},
    )
    reports = [json.loads(path.read_text()) for path in (
        exported.report, Path(adapted["files"]["features_json"]),
    )]
    assert reports[0]["details"] == reports[1]["details"]
    for report in reports:
        assert report["schema_version"] == 2
        assert report["mode"] == "voxel"
        assert report["details"]["feature_count"] == 1
        values = read_image(report["maps"]["original_firstorder_Mean"])
        # Every clipped radius-one neighborhood contains the whole 2x2 liver crop.
        np.testing.assert_array_equal(values.array, np.full(ct.shape, ct.mean(), dtype=np.float32))
        assert values.spacing == (2.0, 3.0)
        assert values.origin == (7.0, 11.0)


def test_export_derived_maps_match_in_memory_values(tmp_path: Path) -> None:
    image_path, mask_path, _image, _mask = _inputs(tmp_path / "inputs")
    settings = Settings(
        features=("firstorder_Mean", "glcm_Contrast"),
        normalize=True, normalize_scale=100.0, remove_outliers=2.0,
    )
    transforms = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.WAVELET, ImageType.LOG, ImageType.LBP2D),
        wavelet=WaveletSettings(level=2),
        log=LoGSettings(sigma=(1.0,)),
    )
    expected = extract_files(
        image_path, mask_path, settings=settings, transforms=transforms,
        mode="voxel",
    )
    result = export_files(
        image_path, mask_path, tmp_path / "out", settings=settings, transforms=transforms,
    )
    report = json.loads(result.report.read_text())
    assert report["schema_version"] == 2
    assert report["maps"] == {name: str(path) for name, path in result.maps.items()}
    assert result.metadata["settings"]["normalize"] is True
    assert result.metadata["settings"]["remove_outliers"] == 2.0
    assert report["details"]["execution"]["settings"]["normalize"] is True
    assert report["details"]["execution"]["settings"]["remove_outliers"] == 2.0

    assert set(result.maps) == set(expected.features)
    assert any(name.startswith("wavelet-") for name in result.maps)
    assert any(name.startswith("log-sigma-") for name in result.maps)
    assert any(name.startswith("lbp-2D_") for name in result.maps)
    source = read_image(image_path)
    for name, path in result.maps.items():
        written = read_image(path)
        np.testing.assert_allclose(
            written.array.astype(np.float64), expected.features[name],
            rtol=1e-4, atol=1e-4, equal_nan=True,
        )
        assert written.spacing == source.spacing
        assert written.origin == source.origin
        assert written.direction == source.direction
        assert written.array.dtype == np.float32


def test_export_refuses_overwrite_and_keeps_previous_outputs(tmp_path: Path) -> None:
    image_path, mask_path, _image, _mask = _inputs(tmp_path / "inputs")
    settings = Settings(features=("firstorder_Mean",))
    transforms = TransformSettings(image_types=(ImageType.ORIGINAL, ImageType.SQUARE))
    output = tmp_path / "out"
    first = export_files(
        image_path, mask_path, output, settings=settings, transforms=transforms,
    )
    previous = {path: Path(path).read_bytes() for path in first.maps.values()}
    with pytest.raises(InvalidInputError, match="overwrite=True"):
        export_files(
            image_path, mask_path, output, settings=settings, transforms=transforms,
        )
    assert {path: Path(path).read_bytes() for path in first.maps.values()} == previous
    # A different selection still refuses to leave a mixed output directory.
    with pytest.raises(InvalidInputError, match="overwrite=True"):
        export_files(
            image_path, mask_path, output,
            settings=Settings(features=("firstorder_Variance",)),
        )


def test_export_writer_failure_removes_staging_and_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _image_path, _mask_path, image, mask = _inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    def fail_write(path: str | Path, array: NDArray[np.float32], *, reference: Volume) -> None:
        raise OSError("injected map write failure")

    monkeypatch.setattr("mlxradiomics.workflows.reporting.write_feature_map", fail_write)
    with pytest.raises(OSError, match="injected map write failure"):
        export(
            image, mask, output, settings=Settings(features=("firstorder_Mean",)),
            spacing=(1.5, 1.0, 0.8),
        )
    assert not (output / "features.json").exists()
    assert not list(output.glob(".mlxradiomics-*"))
    if (output / "maps").exists():
        assert not list((output / "maps").glob("*.nii.gz"))


def test_export_rejects_shape_features_before_writing(tmp_path: Path) -> None:
    _, _, image, mask = _inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="shape"):
        export(image, mask, output, settings=Settings(features=("shape_MeshVolume",)))
    assert not output.exists()


def test_export_ibsi_maps_preserve_lattice_values_and_geometry(tmp_path: Path) -> None:
    """GLDZM, intensity-histogram and IVH maps keep their values and geometry in NIfTI."""
    values, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62:64])
    # Each neighborhood contains all eight liver voxels. Their 43..106 HU values
    # form one connected zone in the anchored [0, 256) bin, while IVH uses raw HU.
    assert np.all(mask == 1) and np.all((values >= 0) & (values < 256))
    expected = {
        "gldzm_ZoneDistanceEntropy": 0.0,
        "intensityhistogram_Mean": 1.0,
        "ivh_VolumeAtIntensityFraction10": float(np.mean(values >= 20)),
        "ivh_VolumeAtIntensityFraction90": float(np.mean(values >= 180)),
    }
    source = sitk.GetImageFromArray(values)
    source.SetSpacing((0.7, 1.2, 2.3))
    source.SetOrigin((4.0, -3.0, 9.0))
    source.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    labels = sitk.GetImageFromArray(mask)
    labels.CopyInformation(source)
    image_path, mask_path = tmp_path / "image.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(source, str(image_path))
    sitk.WriteImage(labels, str(mask_path))
    result = export_files(
        image_path, mask_path, tmp_path / "out",
        settings=Settings(
            features=tuple(expected), bin_width=256,
            bin_anchor=0, ivh_intensity_range=(0, 200), scratch_mib=1,
        ),
    )
    for name, value in expected.items():
        written = sitk.ReadImage(str(result.maps[f"original_{name}"]))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(written), value * mask)
        np.testing.assert_allclose(written.GetSpacing(), source.GetSpacing(), rtol=1e-6)
        np.testing.assert_allclose(written.GetOrigin(), source.GetOrigin())
        np.testing.assert_allclose(written.GetDirection(), source.GetDirection())


def test_export_files_rejects_nonboolean_overwrite_and_preserves_outputs(tmp_path: Path) -> None:
    image_path, mask_path, _image, _mask = _inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    (output / "maps").mkdir(parents=True)
    previous = {
        output / "features.json": b"previous report",
        output / "maps" / "original_firstorder_Mean.nii.gz": b"previous map",
    }
    for path, value in previous.items():
        path.write_bytes(value)

    with pytest.raises(InvalidInputError, match="overwrite must be a boolean"):
        export_files(
            image_path, mask_path, output,
            settings=Settings(features=("firstorder_Mean",)),
            overwrite=cast(bool, cast(object, "false")),
        )

    assert {path: path.read_bytes() for path in previous} == previous


@pytest.mark.parametrize("problem", ["missing_directory", "dense_budget"])
def test_export_rejects_invalid_scratch_before_deriving_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    crop = (np.s_[100:165, 180:245, 25:90] if problem == "dense_budget"
            else np.s_[105:109, 215:220, 64])
    image, mask, _ = hcc_crop(crop)
    scratch, output = tmp_path / "scratch", tmp_path / "out"
    if problem != "missing_directory":
        scratch.mkdir()

    def forbidden_transforms(*args: object, **kwargs: object) -> Never:
        raise AssertionError("scratch admission must precede derived-image processing")

    monkeypatch.setattr("mlxradiomics.workflows.derived.feature_images", forbidden_transforms)
    with pytest.raises(InvalidInputError):
        export(
            image, mask, output, settings=Settings(features=("firstorder_Mean",)),
            transforms=TransformSettings(image_types=(ImageType.SQUARE,)),
            temporary_dir=scratch, max_temporary_mib=1,
        )

    assert not (output / "features.json").exists()
    assert not list(output.glob(".mlxradiomics-*"))
    assert not list((output / "maps").glob("*.nii.gz"))
    if scratch.exists():
        assert not list(scratch.iterdir())


@pytest.mark.parametrize("spacing,settings", [
    ((-1.0, 1.0, 1.0), Settings()),
    ((1.0, 1.0, 1.0), Settings(bin_width=0)),
])
def test_export_images_reports_invalid_requests_as_input_errors(
    tmp_path: Path, spacing: tuple[float, ...], settings: Settings,
) -> None:
    image, mask, _ = hcc_crop(np.s_[105:111, 215:221, 62:68])
    with pytest.raises(InvalidInputError):
        export(image, mask, tmp_path / "out", spacing=spacing, settings=settings)


def test_array_export_warns_about_synthetic_geometry(tmp_path: Path) -> None:
    image, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    settings = Settings(features=("firstorder_Mean",))
    with pytest.warns(UserWarning, match="synthetic geometry"):
        warned = export(image, mask, tmp_path / "warned", settings=settings, spacing=(2.0, 3.0))
    accepted = export(
        image, mask, tmp_path / "accepted", settings=settings, spacing=(2.0, 3.0),
        allow_synthetic_geometry=True,
    )
    for result in (warned, accepted):
        volume = read_image(result.maps["original_firstorder_Mean"])
        assert volume.origin == (0.0, 0.0)
        assert volume.spacing == (2.0, 3.0)
        assert volume.direction == (1.0, 0.0, 0.0, 1.0)
        np.testing.assert_array_equal(volume.array, np.full(image.shape, image.mean()))
