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
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones((2, 2), dtype=np.uint8)
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
        assert result.features == {name: 2.5 for name in names}
    assert file_result.metadata.get("image_geometry") == {
        "shape": (2, 2),
        "spacing": (2.0, 3.0),
        "origin": (7.0, 11.0),
        "direction": (1.0, 0.0, 0.0, 1.0),
    }

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
    # Every radius-one neighborhood covers all four pixels; each mean is 2.5.
    for name, values in collected.items():
        np.testing.assert_array_equal(values, np.full((2, 2), 2.5, dtype=np.float32))
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
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones((2, 2), dtype=np.uint8)
    names = list_features("firstorder", "shape", "shape2D", "localintensity", ndim=2)
    result = extract(image, mask, spacing=(1.0, 1.0), settings=Settings(features=names))
    assert result.features["original_firstorder_Mean"] == 2.5
    assert result.features["original_shape2D_PixelSurface"] == 4.0
    voxel_names = list_features("firstorder", "shape", "shape2D", "localintensity", mode="voxel")
    voxel = extract(
        image, mask, spacing=(1.0, 1.0), mode="voxel", settings=Settings(features=voxel_names),
    )
    np.testing.assert_array_equal(voxel.features["original_firstorder_Mean"], np.full((2, 2), 2.5))
    with pytest.raises(ValueError, match="unknown feature families"):
        list_features("glmc")


def test_array_extraction_warns_before_assuming_unit_spacing() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones((2, 2), dtype=np.uint8)
    settings = Settings(features=("firstorder_TotalEnergy",))
    with pytest.warns(UserWarning, match="assumes 1 mm spacing"):
        assumed = extract(image, mask, settings=settings)
    explicit = extract(image, mask, settings=settings, spacing=(2.0, 3.0))
    assert assumed.features["original_firstorder_TotalEnergy"] == 30.0
    assert explicit.features["original_firstorder_TotalEnergy"] == 180.0
