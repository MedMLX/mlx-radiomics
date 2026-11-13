"""The public API shares results, geometry and streaming contracts across inputs."""

import subprocess
import sys
import textwrap
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, assert_type, cast

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.typing import NDArray
from radiomics_reference.mathematics import masked_neighborhood_mean
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import (
    ImageType,
    Result,
    Settings,
    TransformSettings,
    VoxelSink,
    extract,
    extract_files,
    list_features,
)
from mlxradiomics.config.transforms import MeanSettings


def test_array_file_and_streamed_features_share_values_and_preserve_geometry(
    tmp_path: Path,
) -> None:
    image, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    assert np.all(mask == 1)
    expected_mean = float(image.mean())
    settings = Settings(features=("firstorder_Mean",))
    transforms = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.MEAN), mean=MeanSettings(support=1)
    )
    names = ("original_firstorder_Mean", "mean-M1-mirror_firstorder_Mean")
    paths = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    for values, path in zip((image, mask), paths, strict=True):
        volume = sitk.GetImageFromArray(values)
        volume.SetSpacing((3.0, 2.0))
        volume.SetOrigin((7.0, 11.0))
        sitk.WriteImage(volume, str(path))

    array_result = extract(
        image,
        mask,
        spacing=(2.0, 3.0),
        settings=settings,
        transforms=transforms,
    )
    file_result = extract_files(
        str(paths[0]),
        paths[1],
        settings=settings,
        transforms=transforms,
    )
    assert_type(array_result, Result[float])
    assert_type(file_result, Result[float])
    for result in (array_result, file_result):
        assert result.features == {name: expected_mean for name in names}
    assert file_result.metadata.get("image_geometry") == {
        "shape": (2, 2),
        "spacing": (2.0, 3.0),
        "origin": (7.0, 11.0),
        "direction": (1.0, 0.0, 0.0, 1.0),
    }
    compact = extract_files(
        paths[0], paths[1], settings=settings, transforms=transforms,
        mode="voxel", map_dtype="float32",
    )
    assert_type(compact, Result[NDArray[np.float32]])
    assert compact.metadata.get("map_dtype") == "float32"
    assert compact.metadata.get("image_geometry") == file_result.metadata.get("image_geometry")
    for values in compact.features.values():
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, np.full(image.shape, expected_mean, dtype=np.float32))

    collected = {name: np.zeros((2, 2), dtype=np.float32) for name in names}
    lifetime: list[tuple[str, str]] = []

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        for name, values in batch.items():
            collected[name][tuple(centers.T)] = values

    @contextmanager
    def sink_factory(image_name: str) -> Generator[VoxelSink]:
        lifetime.append(("open", image_name))
        try:
            yield consume
        finally:
            lifetime.append(("close", image_name))

    streamed = extract_files(
        paths[0],
        str(paths[1]),
        settings=settings,
        transforms=transforms,
        mode="voxel",
        sink_factory=sink_factory,
    )
    assert_type(streamed, Result[NDArray[np.float64]])
    assert streamed.features == {}
    assert lifetime == [
        ("open", "original"),
        ("close", "original"),
        ("open", "mean-M1-mirror"),
        ("close", "mean-M1-mirror"),
    ]
    # Every radius-one neighborhood covers all four liver pixels.
    for name, values in collected.items():
        np.testing.assert_array_equal(values, np.full(image.shape, expected_mean, dtype=np.float32))
        assert streamed.metadata["feature_validity"][name]["finite"] == 4


def test_invalid_mode_and_roi_sink_are_rejected_before_file_io(tmp_path: Path) -> None:
    image, mask = tmp_path / "absent-image.nii.gz", tmp_path / "absent-mask.nii.gz"
    invalid_mode = cast(Literal["roi", "voxel"], "volume")
    with pytest.raises(ValueError, match="mode must be 'roi' or 'voxel'"):
        extract_files(image, mask, mode=invalid_mode)

    def consume(_centers: NDArray[np.int32], _batch: Mapping[str, NDArray[np.float32]]) -> None:
        raise AssertionError("an invalid request must not reach the sink")

    with pytest.raises(ValueError, match="require mode='voxel'"):
        extract_files(image, mask, sink=consume)
    with pytest.raises(ValueError, match="map_dtype"):
        extract_files(image, mask, mode="voxel", map_dtype=cast(Literal["float32"], "float16"))


@pytest.mark.parametrize("features", [
    ("firstorder_Mean",),
    ("firstorder_Entropy",),
    ("ivh_AreaUnderIVHCurve",),
    ("firstorder_Mean", "firstorder_Entropy", "ivh_AreaUnderIVHCurve", "glcm_Contrast"),
])
def test_compact_retained_maps_preserve_values_and_background(features: tuple[str, ...]) -> None:
    image, mask, _ = hcc_crop(np.s_[103:107, 214:219, 64])
    assert mask.any() and not mask.all()
    settings = Settings(features=features, bin_width=25)
    transforms = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.MEAN), mean=MeanSettings(support=1),
    )
    legacy = extract(image, mask, settings=settings, transforms=transforms,
                     spacing=(1.0, 1.0), mode="voxel")
    compact = extract(image, mask, settings=settings, transforms=transforms,
                      spacing=(1.0, 1.0), mode="voxel", map_dtype="float32")
    assert_type(legacy, Result[NDArray[np.float64]])
    assert_type(compact, Result[NDArray[np.float32]])
    assert legacy.metadata.get("map_dtype") == "float64"
    assert compact.metadata.get("map_dtype") == "float32"
    assert compact.metadata["feature_validity"] == legacy.metadata["feature_validity"]
    for name, values in compact.features.items():
        original = legacy.features[name]
        assert values.dtype == np.float32 and original.dtype == np.float64
        assert 2 * values.nbytes == original.nbytes
        np.testing.assert_array_equal(values.astype(np.float64), original)
        np.testing.assert_array_equal(np.signbit(values), np.signbit(original))
        np.testing.assert_array_equal(values[mask == 0], 0)
    if "firstorder_Mean" in features:
        np.testing.assert_allclose(compact.features["original_firstorder_Mean"],
                                   masked_neighborhood_mean(image, mask == 1), rtol=1e-6)


def test_compact_retained_maps_preserve_nonfinite_values_and_zero_signs() -> None:
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 1:3] = 1
    settings = Settings(
        features=("firstorder_Energy", "firstorder_Skewness", "firstorder_Entropy"),
        bin_count=1, zero_variance_moments="nan",
    )
    background, _, _ = hcc_crop(np.s_[0:3, 0:4, 64])
    assert np.all(background == -2048)
    # Scale the scan's constant stored background into the feature-overflow domain.
    image = background * (-1e20 / 2048)
    legacy = extract(image, mask, settings=settings, spacing=(1.0, 1.0), mode="voxel")
    compact = extract(image, mask, settings=settings, spacing=(1.0, 1.0),
                      mode="voxel", map_dtype="float32")
    assert np.isposinf(compact.features["original_firstorder_Energy"][mask == 1]).all()
    assert np.isnan(compact.features["original_firstorder_Skewness"][mask == 1]).all()
    assert np.signbit(compact.features["original_firstorder_Entropy"][mask == 1]).all()
    for name, values in compact.features.items():
        original = legacy.features[name]
        np.testing.assert_array_equal(values.astype(np.float64), original)
        np.testing.assert_array_equal(np.signbit(values), np.signbit(original))
        assert not np.signbit(values[mask == 0]).any()


def test_configuration_loads_without_execution_dependencies() -> None:
    # A fresh interpreter prevents the rest of the suite's MLX imports hiding coupling.
    program = textwrap.dedent("""
        import importlib.abc
        import sys

        class NoExecutionRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"mlx", "SimpleITK"}:
                    raise AssertionError(f"configuration loaded execution dependency: {fullname}")
                return None

        sys.meta_path.insert(0, NoExecutionRuntime())
        from mlxradiomics import ImageSettings, TransformSettings
        from mlxradiomics.config.parameters import parse_params

        parsed = parse_params({"setting": {"label": 2}}, ndim=3)
        assert parsed.settings.label == 2
        assert ImageSettings().correct_mask is False
        assert TransformSettings().image_types
    """)
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_feature_discovery_builds_supported_extraction_selections() -> None:
    image, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    assert np.all(mask == 1)
    expected_mean = float(image.mean())
    names = list_features("firstorder", "shape", "shape2D", "localintensity", ndim=2)
    result = extract(image, mask, spacing=(1.0, 1.0), settings=Settings(features=names))
    assert result.features["original_firstorder_Mean"] == expected_mean
    assert result.features["original_shape2D_PixelSurface"] == 4.0
    voxel_names = list_features("firstorder", "shape", "shape2D", "localintensity", mode="voxel")
    voxel = extract(
        image, mask, spacing=(1.0, 1.0), mode="voxel", settings=Settings(features=voxel_names),
    )
    np.testing.assert_array_equal(voxel.features["original_firstorder_Mean"],
                                  np.full(image.shape, expected_mean))
    with pytest.raises(ValueError, match="unknown feature families"):
        list_features("glmc")


def test_array_extraction_warns_before_assuming_unit_spacing() -> None:
    image, mask, _ = hcc_crop(np.s_[105:107, 217:219, 62])
    settings = Settings(features=("firstorder_TotalEnergy",))
    with pytest.warns(UserWarning, match="assumes 1 mm spacing"):
        assumed = extract(image, mask, settings=settings)
    explicit = extract(image, mask, settings=settings, spacing=(2.0, 3.0))
    energy = float(np.sum(image**2))
    assert assumed.features["original_firstorder_TotalEnergy"] == energy
    assert explicit.features["original_firstorder_TotalEnergy"] == energy * 6
