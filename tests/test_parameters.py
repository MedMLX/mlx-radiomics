"""Advanced parameter files route through the same qualified extraction pipeline."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import SimpleITK as sitk
from medmlx_core.errors import InvalidInputError
from medmlx_core.outputs import Artifact, validate_outputs
from radiomics_reference.real_images import hcc_crop

from mlxradiomics.config.filtering import FilterPadding, FourierGrid, RieszBase
from mlxradiomics.config.parameters import load_params, parse_params, settings_from_params
from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings, Interpolator
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.imaging.io import read_image
from mlxradiomics.medmlx import RunResult, run
from mlxradiomics.workflows.extraction import extract_files


def _inputs(
    root: Path, *, direction: tuple[float, ...] | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    values, liver, _ = hcc_crop(np.s_[103:113, 214:225, 60:72])
    # Select a central part of the source annotation so zero-padding crops leave
    # a smaller output grid; keep its actual liver boundary and tumor exclusions.
    labels = np.zeros_like(liver)
    labels[2:9, 2:9, 3:10] = liver[2:9, 2:9, 3:10]
    image = sitk.GetImageFromArray(values.astype(np.int16))
    mask = sitk.GetImageFromArray(labels)
    image.SetSpacing((1.5, 1.0, 0.8))
    image.SetOrigin((3.0, -2.0, 5.0))
    if direction is not None:
        image.SetDirection(direction)
    mask.CopyInformation(image)
    paths = root / "image.nii.gz", root / "mask.nii.gz"
    sitk.WriteImage(image, paths[0].as_posix())
    sitk.WriteImage(mask, paths[1].as_posix())
    return paths


def _params(*, image_types: dict[str, object], setting: dict[str, object]) -> dict[str, object]:
    return {
        "setting": setting,
        "imageType": image_types,
        "featureClass": {"firstorder": ["Mean"], "glcm": ["Contrast"]},
    }


def _scalar_fields(result: RunResult) -> list[Artifact]:
    artifacts = validate_outputs(result).artifacts
    return [entry for entry in artifacts.values() if entry["kind"] == "scalar_field"]


def _maps(result: RunResult) -> dict[str, str]:
    """Return the voxel-map paths: the ``outputs`` roles whose artifact is a scalar field."""
    artifacts = validate_outputs(result).artifacts
    return {
        role: path for role, path in result["outputs"].items()
        if artifacts[role]["kind"] == "scalar_field"
    }


def _geometry(result: RunResult) -> tuple[dict[str, object], dict[str, object]]:
    execution = cast(dict[str, object], result["details"]["execution"])
    return (
        cast(dict[str, object], execution["image_geometry"]),
        cast(dict[str, object], execution["source_geometry"]),
    )


@pytest.mark.parametrize(
    "problem,params",
    [
        ("unknown_section", {"pipeline": {}}),
        ("geometry_tolerance", {"setting": {"geometryTolerance": 1e-6}}),
        ("unknown_image", {"imageType": {"Unknown": {}}}),
        ("log_without_sigma", {"imageType": {"LoG": {}}}),
        ("original_override", {"imageType": {"Original": {"binWidth": 5}}}),
        ("bad_interpolator", {"setting": {"interpolator": "cubic"}}),
        ("unknown_lbp_method", {"setting": {"lbp2DMethod": "nope"}}),
        ("bad_spacing_length", {"setting": {"resampledPixelSpacing": [1.0, 1.0]}}),
    ],
)
def test_parameter_files_reject_unsupported_controls(
    problem: str, params: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        parse_params(params, ndim=3)


@pytest.mark.parametrize(
    "params,context",
    [
        ({1: {}, "unknown": {}}, "params"),
        ({"setting": {1: 2, "unknown": 3}}, "setting"),
        ({"featureClass": {1: ["Mean"]}}, "featureClass"),
        ({"imageType": {1: {}, "Mean": {}}}, "imageType"),
        ({"imageType": {"Mean": {1: 2, "unknown": 3}}}, r"imageType\.Mean"),
    ],
)
def test_parameter_parsers_reject_non_string_keys(params: object, context: str) -> None:
    mapping = cast(Mapping[str, object], params)
    with pytest.raises(ValueError, match=f"'{context}' keys must be strings"):
        parse_params(mapping, ndim=3)


def test_parameter_file_matches_equivalent_python_call(tmp_path: Path) -> None:
    image_path, mask_path = _inputs(tmp_path / "inputs")
    params = _params(
        image_types={"Original": {}, "Wavelet": {"level": 1}},
        setting={
            "binWidth": 25, "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "interpolator": "sitkLinear", "padDistance": 1,
        },
    )
    result = run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "run",
                 params=params, voxel=True)
    expected = extract_files(
        image_path, mask_path,
        settings=Settings(features=("firstorder_Mean", "glcm_Contrast")),
        image_settings=ImageSettings(
            resampled_spacing=(1.0, 1.0, 1.0), interpolator=Interpolator.LINEAR,
            pad_distance=1,
        ),
        transforms=TransformSettings(image_types=(ImageType.ORIGINAL, ImageType.WAVELET)),
        mode="voxel",
    )
    payload = json.loads(Path(result["outputs"]["features_json"]).read_text())
    assert result["details"]["shape"] == [9, 11, 15]
    assert result["details"]["spacing"] == [1.0, 1.0, 1.0]
    assert payload["details"]["execution"]["source_geometry"]["shape"] == [10, 11, 12]
    assert payload["details"]["execution"]["image_settings"]["pad_distance"] == 1
    fields = _scalar_fields(result)
    assert fields
    assert {(entry["coordinate_space"], entry.get("grid_of")) for entry in fields} == {
        ("image_physical", None)
    }
    maps = _maps(result)
    assert set(maps) == set(expected.features)
    for name, path in maps.items():
        written = sitk.GetArrayFromImage(sitk.ReadImage(path))
        np.testing.assert_allclose(
            written.astype(np.float64), expected.features[name], rtol=1e-4, atol=1e-4,
            equal_nan=True,
        )


def test_parameter_file_correct_mask_and_effective_settings(tmp_path: Path) -> None:
    image_path, _ = _inputs(tmp_path / "inputs")
    label_array = np.zeros((12, 13, 14), dtype=np.uint8)
    label_array[3:9, 4:10, 4:11] = 1
    mismatched = sitk.GetImageFromArray(label_array)
    mismatched.SetSpacing((1.5, 1.0, 0.8))
    mismatched.SetOrigin((3.0, -2.0, 5.0))
    mask_path = tmp_path / "inputs" / "mismatched.nii.gz"
    sitk.WriteImage(mismatched, mask_path.as_posix())

    params = _params(image_types={"Original": {}, "Square": {}}, setting={"correctMask": True})
    with pytest.raises(InvalidInputError, match="geometry mismatch"):
        run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "refused",
            params={"setting": {}, "featureClass": {"firstorder": ["Mean"]}})
    result = run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
                 params=params)
    execution = cast(dict[str, object], result["details"]["execution"])
    source = cast(dict[str, object], execution["source_geometry"])
    image_settings = cast(dict[str, object], execution["image_settings"])
    assert source["shape"] == [10, 11, 12]
    assert image_settings["correct_mask"] is True
    features = json.loads(Path(result["outputs"]["features_json"]).read_text())["features"]
    assert set(features) == {
        "original_firstorder_Mean", "original_glcm_Contrast",
        "square_firstorder_Mean", "square_glcm_Contrast",
    }
    assert result["details"]["feature_count"] == len(features) == 4


def test_map_descriptor_claims_the_input_voxel_grid_only_when_geometry_matches(
    tmp_path: Path,
) -> None:
    image_path, mask_path = _inputs(tmp_path / "inputs")
    result = run(
        image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
        params=_params(image_types={"Original": {}}, setting={}), voxel=True,
    )
    image_geometry, source_geometry = _geometry(result)
    assert image_geometry == source_geometry
    fields = _scalar_fields(result)
    assert len(fields) == 2
    assert {(entry["coordinate_space"], entry.get("grid_of")) for entry in fields} == {
        ("image_voxel", "image_path"),
    }


def test_map_descriptor_reports_a_cropped_grid_as_physical(tmp_path: Path) -> None:
    image_path, mask_path = _inputs(tmp_path / "inputs")
    # Native spacing still crops around the ROI plus one voxel of padding.
    params = _params(
        image_types={"Original": {}},
        setting={"resampledPixelSpacing": [1.5, 1.0, 0.8], "padDistance": 1},
    )
    result = run(
        image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
        params=params, voxel=True,
    )
    image_geometry, source_geometry = _geometry(result)
    assert source_geometry["shape"] == [10, 11, 12]
    assert image_geometry["shape"] != source_geometry["shape"]
    fields = _scalar_fields(result)
    assert fields
    assert {(entry["coordinate_space"], entry.get("grid_of")) for entry in fields} == {
        ("image_physical", None),
    }
    written = read_image(next(iter(_maps(result).values())))
    np.testing.assert_allclose(
        written.spacing, cast(list[float], image_geometry["spacing"]), rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        written.origin, cast(list[float], image_geometry["origin"]), rtol=0, atol=1e-5,
    )
    np.testing.assert_allclose(
        written.direction, cast(list[float], image_geometry["direction"]), atol=1e-6,
    )


def test_map_descriptor_preserves_oblique_direction_through_crop(tmp_path: Path) -> None:
    angle = np.pi / 6
    direction = (
        float(np.cos(angle)), float(-np.sin(angle)), 0.0,
        float(np.sin(angle)), float(np.cos(angle)), 0.0,
        0.0, 0.0, 1.0,
    )
    image_path, mask_path = _inputs(tmp_path / "inputs", direction=direction)
    unchanged = run(
        image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "unchanged",
        params=_params(image_types={"Original": {}}, setting={}), voxel=True,
    )
    assert {entry["coordinate_space"] for entry in _scalar_fields(unchanged)} == {"image_voxel"}

    params = _params(
        image_types={"Original": {}},
        setting={"resampledPixelSpacing": [1.5, 1.0, 0.8], "padDistance": 1},
    )
    cropped = run(
        image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "cropped",
        params=params, voxel=True,
    )
    image_geometry, source_geometry = _geometry(cropped)
    source_direction = cast(list[float], source_geometry["direction"])
    assert abs(source_direction[1]) > 0.4
    assert image_geometry["shape"] != source_geometry["shape"]
    np.testing.assert_allclose(
        cast(list[float], image_geometry["direction"]), source_direction, atol=1e-12,
    )
    assert {(entry["coordinate_space"], entry.get("grid_of"))
            for entry in _scalar_fields(cropped)} == {("image_physical", None)}


def test_force2d_override_distinguishes_absent_false_and_true() -> None:
    base = Settings(force2d_axis=1)
    assert settings_from_params({}, base=base).force2d_axis == 1
    cleared = settings_from_params({"setting": {"force2D": False}}, base=base)
    assert cleared.force2d_axis is None
    enabled = settings_from_params({"setting": {"force2D": True}}, base=base)
    assert enabled.force2d_axis == 0
    axis = settings_from_params(
        {"setting": {"force2D": True, "force2Ddimension": 2}}, base=base,
    )
    assert axis.force2d_axis == 2


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_force2d_dimension_requires_an_integer_axis(value: object) -> None:
    with pytest.raises(ValueError, match="force2Ddimension"):
        settings_from_params({"setting": {"force2D": True, "force2Ddimension": value}})
    with pytest.raises(ValueError, match="force2d_axis"):
        Settings(force2d_axis=cast(int | None, value)).validate(3)


def test_malformed_params_yaml_is_a_value_error_with_parser_context(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text("setting: [", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid params YAML") as error:
        load_params(params)
    assert "line 1" in str(error.value)


@pytest.mark.parametrize(
    "image_types,ndim,message",
    [
        ({"Laws": {"kernel": "E3L3S3"}}, 2, "three-axis Laws"),
        ({"Gabor": {"sigma": [1], "wavelength": [2], "theta": [0], "slice_axis": 0}},
         2, "slice_axis"),
        ({"Simoncelli": {"padding": "mirror"}}, 3, "periodic"),
        ({"Mean": {"support": 4}}, 3, "positive odd"),
        ({"Gabor": {"sigma": 1, "wavelength": [2], "theta": [0]}}, 3, "list"),
        ({"Laws": {"normalization": "none"}}, 3, "unsupported"),
        ({"NonseparableWavelet": {"wavelet": "shannon"}}, 3, "unsupported"),
        ({"Riesz": {}}, 3, "explicit riesz_orders"),
        ({"Riesz": {"orders": [[1, 0, 0]]}}, 3, "riesz_log_sigma"),
        ({"Riesz": {"orders": []}}, 3, "integer lists"),
        ({"Riesz": {"orders": [[1, -1, 0]], "log_sigma": [3.0]}}, 3, "nonnegative"),
        ({"Riesz": {"orders": [[1, 0]], "log_sigma": [3.0]}}, 3,
         "one entry per image axis"),
        ({"Riesz": {"base": "simoncelli", "orders": [[0, 2, 0]], "padding": "mirror"}},
         3, "periodic"),
        ({"Riesz": {"base": "nope", "orders": [[1, 0, 0]], "log_sigma": [3.0]}}, 3,
         "RieszBase"),
        ({"Riesz": {"orders": [[1, 0, 0]], "log_sigma": [3.0], "grid": "nope"}}, 3,
         "FourierGrid"),
        ({"LoG": {"sigma": [3.0], "planar": True}}, 3, "require log_kernel"),
        ({"SeparableWavelet": {"wavelet": "haar"}}, 3, "separable_wavelet_bands"),
        ({"SeparableWavelet": {"bands": ["LH"]}}, 3, "one letter per filtered axis"),
        ({"SeparableWavelet": {"bands": ["LLH"], "planar": True}}, 3,
         "one letter per filtered axis"),
        ({"Riesz": {"orders": [[2, 0, 0]], "log_sigma": [3.0], "steering": True}}, 3,
         "requires riesz_tensor_sigma"),
        ({"Riesz": {"orders": [[2, 0, 0]], "log_sigma": [3.0], "tensor_sigma": 2.0}}, 3,
         "requires riesz_steering"),
        ({"Riesz": {"orders": [[0, 0, 0]], "log_sigma": [3.0], "steering": True,
                    "tensor_sigma": 2.0}}, 3, "positive sum"),
    ],
)
def test_filter_parameters_refuse_unsupported_scientific_controls(
    image_types: dict[str, object], ndim: int, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_params({"imageType": image_types}, ndim=ndim)


def test_planar_separable_wavelet_parameter_file_runs_on_3d_input(tmp_path: Path) -> None:
    # Planar bands name the two filtered in-plane axes; parsing and naming must agree.
    image_path, mask_path = _inputs(tmp_path / "inputs")
    params = _params(
        image_types={"SeparableWavelet": {"wavelet": "haar", "bands": ["LH"], "planar": True}},
        setting={},
    )
    result = run(image_path=image_path, mask_path=mask_path, output_dir=tmp_path / "out",
                 params=params)
    features = json.loads(Path(result["outputs"]["features_json"]).read_text())["features"]
    assert any(name.startswith("separablewavelet-haar-L1-LH-") and "-2D_firstorder_Mean" in name
               for name in features)


def test_parameter_file_parses_riesz_and_simoncelli_grid_controls() -> None:
    parsed = parse_params({
        "imageType": {
            "Riesz": {
                "base": "log", "orders": [[1, 0, 0], [0, 2, 0]],
                "log_sigma": [3.0], "log_cutoff": 3.5, "padding": "constant",
                "constant": 0.0, "grid": "centered",
            },
            "NonseparableWavelet": {"levels": [2], "grid": "centered"},
        },
    }, ndim=3)
    transforms = parsed.transforms
    assert transforms.image_types == (ImageType.RIESZ, ImageType.SIMONCELLI)
    assert transforms.riesz.base is RieszBase.LOG
    assert transforms.riesz.orders == ((1, 0, 0), (0, 2, 0))
    assert transforms.riesz.log_sigma == (3.0,)
    assert transforms.riesz.log_cutoff == 3.5
    assert transforms.riesz.padding is FilterPadding.CONSTANT
    assert transforms.riesz.constant == 0.0
    assert transforms.riesz.grid is FourierGrid.CENTERED
    assert transforms.simoncelli.grid is FourierGrid.CENTERED


def test_parameter_file_sets_each_convention() -> None:
    params = {"setting": {
        "covarianceDivisor": "n", "ellipsoidArea": "exact", "kurtosis": "raw",
        "zeroVarianceMoments": "nan", "ngtdmSingularCases": "unadjusted",
        "neighborhoodSupport": "distances", "binOrigin": "zero-lattice",
        "constantRoiBin": "middle", "maskExtrapolation": "nearest", "splineBoundary": "mirror",
    }}
    parsed = settings_from_params(params)
    assert parsed == Settings(
        covariance_divisor="n", ellipsoid_area="exact", kurtosis="raw",
        zero_variance_moments="nan", ngtdm_singular_cases="unadjusted",
        neighborhood_support="distances", bin_origin="zero-lattice", constant_roi_bin="middle",
        mask_extrapolation="nearest", spline_boundary="mirror",
    )
    parsed.validate(3)
    assert settings_from_params({"setting": {"kurtosis": "raw"}}) == Settings(kurtosis="raw")
    with pytest.raises(ValueError, match="kurtosis must be one of excess, raw"):
        settings_from_params({"setting": {"kurtosis": "other"}}).validate(3)
