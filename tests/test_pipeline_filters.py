"""Public filter extraction preserves categorical source decisions and file geometry."""

from collections.abc import Mapping
from fractions import Fraction
from math import pi
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from conftest import requires_fft
from numpy.typing import NDArray

from mlxradiomics.config.filtering import LBPMethod, LogKernel
from mlxradiomics.config.parameters import parse_params
from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import (
    ImageType,
    LBP2DSettings,
    LoGSettings,
    MeanSettings,
    TransformSettings,
)
from mlxradiomics.contracts.metadata import transform_payload
from mlxradiomics.imaging.transforms import feature_image_names, feature_images
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.workflows.derived import filter_source_words
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract, extract_files


def test_filter_source_words_reconstruct_source_bits_with_a_required_third_word() -> None:
    image = np.array([
        [1.0 + 2.0**-26 + 2.0**-52, -1.0 - 2.0**-26 - 2.0**-52, -0.0],
        [0.1, np.nextafter(0.1, np.inf), float(np.finfo(np.float32).max)],
    ], dtype=np.float64)
    high = image.astype(np.float32)
    low, tail = filter_source_words(image, high, tail=True)
    assert tail is not None
    assert low[0, 0] == 2.0**-26 and tail[0, 0] == 2.0**-52
    assert low[0, 1] == -(2.0**-26) and tail[0, 1] == -(2.0**-52)
    for source, a, b, c in zip(image.flat, high.flat, low.flat, tail.flat, strict=True):
        assert Fraction(float(source)) == sum(Fraction(float(word)) for word in (a, b, c))


def test_filter_source_words_refuse_loss_below_all_three_words() -> None:
    image = np.array([[1.0, 2.0**-200]], dtype=np.float64)
    with pytest.raises(ValueError, match="represented by three float32 words"):
        filter_source_words(image, image.astype(np.float32), tail=True)


def test_public_feature_images_still_validate_the_complete_input() -> None:
    with pytest.raises(ValueError, match="feature image intensities must be finite"):
        next(feature_images(
            mx.array([[1.0, float("nan")], [2.0, 3.0]], dtype=mx.float32), (1.0, 1.0),
            TransformSettings(image_types=(ImageType.SQUARE,)),
        ))


@pytest.mark.parametrize("derived", [False, True])
def test_retained_maps_admit_the_complete_payload_before_allocation(
    monkeypatch: pytest.MonkeyPatch, derived: bool,
) -> None:
    image = np.arange(125, dtype=np.float64).reshape(5, 5, 5)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    # One map fits exactly. A second feature, or eight wavelet bands accumulated
    # across individually small extractions, must refuse the full retained selection.
    actual_device = mx.device_info(mx.gpu)

    def device_info(_: object = None) -> dict[str, str | int]:
        return {**actual_device, "memory_size": image.size * 8 * 4}

    monkeypatch.setattr(mx, "device_info", device_info)
    options = TransformSettings()
    if derived:
        options = TransformSettings(image_types=(ImageType.ORIGINAL, ImageType.WAVELET))
        settings = Settings(features=("firstorder_Mean",))
        with pytest.raises(MemoryError, match="9 float64 voxel maps"):
            extract(image, mask, settings=settings, transforms=options, mode="voxel")
    else:
        settings = Settings(features=("firstorder_Mean", "glcm_Contrast"))
        with pytest.raises(MemoryError, match="2 float64 voxel maps"):
            extract_original(image, mask, settings=settings, voxel=True)
    result = extract_original(image, mask,
        settings=Settings(features=("firstorder_Mean",)), voxel=True)
    assert np.asarray(result.features["original_firstorder_Mean"])[2, 2, 2] == 62
    # Streamed batches use no full-map payload even when the selection exceeds it.
    rows = 0

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        nonlocal rows
        rows += len(centers)
        assert all(values.shape == (len(centers),) for values in batch.values())

    streamed = extract(
        image, mask, settings=settings,
        transforms=options if derived else None, mode="voxel", sink=consume,
    )
    assert not streamed.features
    assert rows == 27 * (9 if derived else 1)


@pytest.mark.parametrize("direction,expected", [(0.0, 241), (2.0, 255)])
def test_public_extraction_distinguishes_adjacent_float64_source_values(
    direction: float, expected: int,
) -> None:
    image = np.ones((3, 3), dtype=np.float64)
    image[0, 1] = np.nextafter(1.0, direction)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1, 1] = 1
    result = extract(
        image,
        mask,
        settings=Settings(features=("firstorder_Mean",)),
        transforms=TransformSettings(
            image_types=(ImageType.LBP2D,),
            lbp2d=LBP2DSettings(samples=8, radius=0.5, method=LBPMethod.DEFAULT),
        ),
    )
    assert result.features["lbp-2D_firstorder_Mean"] == expected


@requires_fft
def test_parameter_driven_riesz_images_are_named_and_finite(tmp_path: Path) -> None:
    rng = np.random.default_rng(53)
    values = rng.normal(20.0, 4.0, (7, 8, 9))
    mask = np.zeros(values.shape, dtype=np.uint8)
    mask[2:5, 2:6, 3:7] = 1
    image, labels = sitk.GetImageFromArray(values), sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))
    labels.CopyInformation(image)
    image_path, mask_path = tmp_path / "image.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(labels, str(mask_path))
    parsed = parse_params({
        "featureClass": {"firstorder": ["Mean"]},
        "imageType": {
            "Riesz": {
                "orders": [[1, 0, 0], [0, 2, 0]], "log_sigma": [2.0],
                "padding": "constant", "constant": 0.0, "grid": "centered",
            },
        },
    }, ndim=3)
    result = extract_files(
        image_path, mask_path, settings=parsed.settings,
        image_settings=parsed.image_settings, transforms=parsed.transforms,
    )
    assert set(result.features) == {
        "riesz-log-s2mm-c4-l1-0-0-constant0-centered_firstorder_Mean",
        "riesz-log-s2mm-c4-l0-2-0-constant0-centered_firstorder_Mean",
    }
    assert all(np.isfinite(value) for value in result.features.values())
    assert result.metadata.get("transforms") == transform_payload(parsed.transforms)


def test_parameter_driven_file_filters_keep_geometry_and_physical_controls(tmp_path: Path) -> None:
    values = np.full((5, 7, 9), 3.0, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    image, labels = sitk.GetImageFromArray(values), sitk.GetImageFromArray(mask)
    image.SetSpacing((1.5, 1.0, 0.8))
    image.SetOrigin((4.0, -2.0, 7.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    labels.CopyInformation(image)
    image_path, mask_path = tmp_path / "image.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(labels, str(mask_path))
    parsed = parse_params({
        "featureClass": {"firstorder": ["Mean"]},
        "imageType": {
            "Mean": {"support": 5, "padding": "periodic"},
            "Laws": {"kernel": "E3L3", "padding": "periodic",
                     "pooling": "max", "energy_distance": 1},
            "Gabor": {"sigma": [0.7], "wavelength": [3.0], "theta": [0.0],
                      "slice_axis": 0, "padding": "periodic"},
        },
    }, ndim=3)
    result = extract_files(image_path, mask_path, settings=parsed.settings,
                           image_settings=parsed.image_settings, transforms=parsed.transforms)
    # A constant's Gabor response equals its intensity times the discrete kernel
    # DC gain. Theta=0,gamma=1 separates the physical x/y sums analytically.
    x, y = np.arange(-4, 5) * 1.5, np.arange(-3, 4) * 1.0
    x_gain = np.sum(np.exp(-x*x / (2*0.7**2)) * np.cos(2*pi*x / 3.0))
    y_gain = np.sum(np.exp(-y*y / (2*0.7**2)))
    expected = {
        "mean-M5-periodic_firstorder_Mean": 3.0,
        "laws-E3L3-periodic-pool-max-energy-d1_firstorder_Mean": 0.0,
        "gabor-s0p7mm-l3mm-g1-t0-modulus-periodic-axis0_firstorder_Mean":
            3.0 * abs(x_gain * y_gain),
    }
    assert set(result.features) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(result.features[name], value, rtol=2e-5, atol=2e-5)
    assert "image_geometry" in result.metadata
    geometry = result.metadata["image_geometry"]
    assert geometry["shape"] == values.shape
    np.testing.assert_allclose(geometry["spacing"], (0.8, 1.0, 1.5))
    np.testing.assert_allclose(geometry["origin"], image.GetOrigin())
    np.testing.assert_allclose(geometry["direction"], image.GetDirection())
    assert result.metadata.get("transforms") == transform_payload(parsed.transforms)


@pytest.mark.parametrize("resegment_shape,expected", [(False, 0.25), (True, 1.0)])
def test_identity_filter_preserves_the_gldzm_morphology_mask(
    resegment_shape: bool, expected: float,
) -> None:
    image = np.zeros((5, 5), dtype=np.float64)
    image[1:4, 1:4] = 1.0
    mask = np.ones((5, 5), dtype=np.uint8)
    settings = Settings(
        features=("gldzm_SmallDistanceEmphasis",), resegment_range=(1.0,),
        resegment_shape=resegment_shape,
    )
    transforms = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.MEAN), mean=MeanSettings(support=1)
    )
    result = extract(image, mask, settings=settings, transforms=transforms)
    # A unit-support mean is the identity, so it must not change a GLDZM value: the
    # intensity ROI stays at the central 3x3 while morphology keeps the original 5x5
    # mask (resegment_shape=False) or its resegmented 3x3 support (True).
    for key in (
        "original_gldzm_SmallDistanceEmphasis",
        "mean-M1-mirror_gldzm_SmallDistanceEmphasis",
    ):
        assert result.features[key] == pytest.approx(expected, rel=1e-6), key


@pytest.mark.parametrize("mode", ["roi", "retained", "streamed"])
def test_adjacent_sampled_log_sigmas_survive_every_delivery_path(mode: str) -> None:
    image = np.random.default_rng(61).normal(20.0, 4.0, (6, 6, 6))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    settings = Settings(bin_width=2.0, features=("firstorder_Mean",))
    transforms = TransformSettings(
        image_types=(ImageType.LOG,),
        log=LoGSettings(sigma=(1.0000001, 1.0000002), kernel=LogKernel.SAMPLED),
    )
    prefixes = feature_image_names(image.shape, (1.0, 1.0, 1.0), transforms)
    expected = {f"{prefix}_firstorder_Mean" for prefix in prefixes}
    assert len(expected) == 2
    collected: set[str] = set()

    def consume(
        centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]],
    ) -> None:
        collected.update(batch)

    voxel = mode != "roi"
    result = extract(
        image, mask, settings=settings, transforms=transforms, mode=("voxel" if voxel else "roi"),
        sink=consume if mode == "streamed" else None,
    )
    if mode == "streamed":
        assert not result.features
        assert collected == expected
    else:
        assert set(result.features) == expected
        if voxel:
            for values in result.features.values():
                assert np.asarray(values).shape == image.shape


