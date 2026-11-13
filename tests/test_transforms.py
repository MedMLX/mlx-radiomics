"""Independent mathematical checks for native Metal image transforms."""

import dataclasses
import warnings
from collections.abc import Mapping
from math import pi
from typing import cast

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.typing import NDArray
from radiomics_reference.mathematics import gradient_magnitude, masked_neighborhood_mean
from radiomics_reference.real_images import hcc_crop, hcc_patch
from skimage.feature import local_binary_pattern

from mlxradiomics.config.filtering import LBPMethod, LogKernel, OrientationPooling
from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import (
    GaborSettings,
    ImageType,
    LBP2DSettings,
    LBP3DSettings,
    LoGSettings,
    RieszSettings,
    SeparableWaveletSettings,
    TransformSettings,
)
from mlxradiomics.contracts.types import HostArray
from mlxradiomics.imaging.transforms import feature_image_names, feature_images
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.workflows.extraction import extract


@pytest.mark.parametrize("sigma", [True, 1.0, [1.0], (True,), ("1",), (None,),
                                  (0.0,), (-1.0,), (float("nan"),), (float("inf"),)])
def test_log_scales_require_a_tuple_of_positive_real_numbers(sigma: object) -> None:
    with pytest.raises(ValueError, match="sigma values must be finite and positive"):
        LoGSettings(sigma=cast(tuple[float, ...], sigma))


@pytest.mark.parametrize("settings_type", [LBP2DSettings, LBP3DSettings])
@pytest.mark.parametrize("radius", [True, "1", None, 0.0, -1.0, float("nan"), float("inf")])
def test_lbp_radius_requires_a_positive_real_number(
    settings_type: type[LBP2DSettings] | type[LBP3DSettings], radius: object,
) -> None:
    with pytest.raises(ValueError, match="LBP radii must be finite and positive"):
        settings_type(radius=cast(float, radius))


@pytest.mark.parametrize("amplitude", [1e-30, 1e-12, 1., 1e20, 1e38])
@pytest.mark.parametrize("kind", [ImageType.SQUARE_ROOT, ImageType.LOGARITHM,
                                  ImageType.EXPONENTIAL])
def test_elementary_transform_keeps_representable_constant_endpoint(
    amplitude: float, kind: ImageType,
) -> None:
    background, _, _ = hcc_crop(np.s_[0:2, 0:3, 64])
    assert np.all(background == -2048)
    image = (background * (-amplitude / 2048)).astype(np.float32)
    with mx.stream(mx.gpu):
        output = next(feature_images(mx.array(image), (1., 1.),
                                    TransformSettings(image_types=(kind,))))
        actual = np.asarray(output.image, dtype=np.float32)
    np.testing.assert_allclose(actual, image, rtol=5e-6, atol=0)


def _assert_close(
    actual: mx.array, expected: NDArray[np.float64], name: str = "",
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), expected,
        rtol=1e-4, atol=1e-4, equal_nan=True, err_msg=name,
    )


def _elementary(kind: ImageType, image: HostArray) -> NDArray[np.float64]:
    """Apply the declared scaled transform formula directly to the source intensities."""
    values = np.asarray(image, dtype=np.float64)
    maximum = float(np.max(np.abs(values)))
    if kind is ImageType.SQUARE:
        return (values / np.sqrt(maximum)) ** 2
    if kind is ImageType.SQUARE_ROOT:
        return np.sign(values) * np.sqrt(np.abs(values) * maximum)
    if kind is ImageType.LOGARITHM:
        return np.sign(values) * np.log1p(np.abs(values)) * maximum / np.log1p(maximum)
    if kind is ImageType.EXPONENTIAL:
        return np.exp(values * np.log(maximum) / maximum)
    raise ValueError(f"no elementary formula for {kind}")


@pytest.mark.parametrize(
    "kind", [ImageType.SQUARE, ImageType.SQUARE_ROOT, ImageType.LOGARITHM,
             ImageType.EXPONENTIAL, ImageType.GRADIENT],
)
@pytest.mark.parametrize("integer", [False, True])
def test_elementary_and_gradient_images_follow_direct_formulas(
    kind: ImageType, integer: bool,
) -> None:
    image, _, _ = hcc_crop(np.s_[103:109, 214:221, 60:68])
    image = image.astype(np.int16 if integer else np.float64)
    spacing = (1.3, 0.9, 2.0)
    options = TransformSettings(image_types=(kind,))
    with mx.stream(mx.gpu):
        actual = list(
            feature_images(mx.array(image.astype(np.float32)), spacing, options,
                           source_dtype=image.dtype)
        )
    assert len(actual) == 1
    assert actual[0].name == kind.value
    expected = (gradient_magnitude(image, spacing) if kind is ImageType.GRADIENT
                else _elementary(kind, image))
    _assert_close(actual[0].image, expected, kind.value)


@pytest.mark.parametrize(
    "row", [(0.0, 1e-30, 2e-30), (0.0, 1e20, 2e20), (-2e38, 2e38, 2e38)],
    ids=["tiny-ramp", "large-ramp", "opposite-limits"],
)
def test_gradient_magnitude_is_exact_across_the_float32_range(
    row: tuple[float, float, float],
) -> None:
    # Each expected magnitude is a normal float32 value although squaring the components
    # or subtracting the opposite-sign endpoints would leave the float32 range.
    source = hcc_patch((3, 3))[0]
    # Calibrate CT contrast to the requested stress range; the repeated endpoint
    # case clips its upper half to retain a sharp plateau at the positive limit.
    unit = (source - source.min()) / np.ptp(source)
    if row[1] == row[2]:
        unit = np.minimum(unit * 2, 1)
    image = (row[0] + unit * (row[-1] - row[0])).astype(np.float32)
    with mx.stream(mx.gpu):
        output = next(feature_images(mx.array(image), (1.0, 1.0),
                                     TransformSettings(image_types=(ImageType.GRADIENT,))))
        actual = np.asarray(output.image, dtype=np.float64)
    expected = gradient_magnitude(image, (1.0, 1.0))
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_log_images_dispatch_each_sigma_with_physical_spacing() -> None:
    ct, _, _ = hcc_crop(np.s_[103:111, 214:223, 60:70])
    image = ct.astype(np.float32)
    spacing = (0.7, 1.2, 2.0)
    options = TransformSettings(image_types=(ImageType.LOG,), log=LoGSettings(sigma=(1.0, 2.5)))
    source = sitk.GetImageFromArray(image.astype(np.float64))
    source.SetSpacing(spacing[::-1])
    with mx.stream(mx.gpu):
        actual = list(feature_images(mx.array(image), spacing, options))
    assert [item.name for item in actual] == [
        "log-sigma-1-0-mm-3D", "log-sigma-2-5-mm-3D",
    ]
    for item, sigma in zip(actual, options.log.sigma, strict=True):
        reference = sitk.LaplacianRecursiveGaussianImageFilter()
        reference.SetSigma(sigma)
        reference.SetNormalizeAcrossScale(True)
        expected = np.asarray(sitk.GetArrayFromImage(reference.Execute(source)), dtype=np.float64)
        _assert_close(item.image, expected, item.name)


@pytest.mark.parametrize("pooling", [OrientationPooling.NONE, OrientationPooling.MAX])
def test_gabor_names_keep_lossless_angles_and_the_pooled_orientation_set(
    pooling: OrientationPooling,
) -> None:
    options = TransformSettings(
        image_types=(ImageType.GABOR,),
        gabor=GaborSettings(
            sigma=(1.0,), wavelength=(2.0,), theta=(1.0000001, 1.0000002), pooling=pooling
        ),
    )
    names = feature_image_names((5, 5), (1.0, 1.0), options)
    if pooling is OrientationPooling.NONE:
        assert len(names) == 2
        assert len(set(names)) == 2
        assert "t1p0000001" in names[0] and "t1p0000002" in names[1]
    else:
        # The pooled map names its orientation count; exact angles are metadata.
        assert names == ("gabor-s1mm-l2mm-g1-o2-modulus-mirror-pool-max",)
        sixteen = dataclasses.replace(
            options,
            gabor=dataclasses.replace(
                options.gabor, theta=tuple(index * pi / 8 for index in range(16))
            ),
        )
        # Exported maps are <name>_<feature>.nii.gz; keep well inside 255 bytes.
        (pooled,) = feature_image_names((5, 5), (1.0, 1.0), sixteen)
        assert len(f"{pooled}_glrlm_ShortRunLowGrayLevelEmphasis.nii.gz") < 255


def test_redundant_duplicate_transform_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="sigma values must be unique"):
        TransformSettings(
            image_types=(ImageType.LOG,),
            log=LoGSettings(sigma=(1.0, 1.0), kernel=LogKernel.SAMPLED),
        )
    with pytest.raises(ValueError, match="gabor_theta values must be unique"):
        TransformSettings(
            image_types=(ImageType.GABOR,),
            gabor=GaborSettings(sigma=(1.0,), wavelength=(2.0,), theta=(0.5, 0.5)),
        )
    with pytest.raises(ValueError, match="gabor sigma/wavelength pairs must be unique"):
        TransformSettings(
            image_types=(ImageType.GABOR,),
            gabor=GaborSettings(sigma=(1.0, 1.0), wavelength=(2.0, 2.0), theta=(0.5,)),
        )


def _lbp3d_reference(
    image: HostArray, mask: HostArray, *, levels: int, radius: float, subdivision: int
) -> list[NDArray[np.float64]]:
    """CPU reference for the spherical-harmonic band norms and sampled kurtosis."""
    from scipy.ndimage import map_coordinates
    from scipy.spatial import SphericalVoronoi
    from scipy.special import sph_harm_y
    from trimesh.creation import icosphere

    vertices = np.asarray(icosphere(subdivisions=subdivision, radius=radius).vertices)
    areas = SphericalVoronoi(vertices, radius=radius).calculate_areas() / radius**2
    centers = np.argwhere(mask)
    sampled = map_coordinates(
        image,
        (centers[:, None, :] + vertices[None, :, :]).transpose(2, 0, 1),
        order=3,
        mode="constant",
        cval=0.0,
    )
    sampled_float = sampled.astype(np.float64)
    binary = (sampled_float >= image.astype(np.float64)[tuple(centers.T)][:, None]).astype(
        np.float64
    )
    theta = np.arccos(vertices[:, 2] / radius)
    phi = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2 * np.pi)
    expected: list[NDArray[np.float64]] = []
    for degree in range(levels):
        y = np.stack(
            [sph_harm_y(degree, m, theta, phi) for m in range(-degree, degree + 1)], axis=1
        )
        coefficients = (binary * areas) @ np.conj(y)
        expected.append(np.sqrt(np.sum(np.abs(coefficients) ** 2, axis=1)))
    centered = sampled_float - np.mean(sampled_float, axis=1, keepdims=True)
    variance = np.mean(centered**2, axis=1)
    expected.append(np.mean(centered**4, axis=1) / variance**2 - 3)
    return expected


@pytest.mark.parametrize("integer", [False, True])
def test_lbp3d_matches_independent_spherical_projection(integer: bool) -> None:
    from scipy.ndimage import map_coordinates
    from trimesh.creation import icosphere

    image, _, _ = hcc_crop(np.s_[103:112, 214:223, 60:69])
    image = image.astype(np.int16 if integer else np.float64)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 1
    mask[0, 0, 0] = 1
    centers = np.argwhere(mask)
    vertices = np.asarray(icosphere(subdivisions=1, radius=1.0).vertices)
    sampled = np.asarray(map_coordinates(
        image.astype(np.float64),
        (centers[:, None, :] + vertices[None, :, :]).transpose(2, 0, 1),
        order=3, mode="constant", cval=0.0,
    ), dtype=np.float64)
    threshold = image[tuple(centers.T)].astype(np.float64) - (0.5 if integer else 0)
    # The public contract rejects ambiguous spline decisions. Compare accepted
    # centers whose independent source samples clear their decision by 0.01 HU,
    # exceeding the float32 roundoff bound throughout this CT crop.
    stable = np.all(np.abs(sampled - threshold[:, None]) > 0.01, axis=1)
    assert stable.any()
    mask[tuple(centers[~stable].T)] = 0
    levels = 3
    options = TransformSettings(
        image_types=(ImageType.LBP3D,), lbp3d=LBP3DSettings(levels=levels, subdivision=1)
    )
    with mx.stream(mx.gpu):
        actual = list(
            feature_images(
                mx.array(image.astype(np.float32)),
                (1.0, 1.0, 1.0),
                options,
                mask=mx.array(mask.astype(bool)),
                source_dtype=image.dtype,
            )
        )
    expected = _lbp3d_reference(image, mask.astype(bool), levels=levels, radius=1.0,
                                subdivision=1)
    assert [output.name for output in actual] == [
        *(f"lbp-3D-m{degree}" for degree in range(1, levels + 1)), "lbp-3D-k"
    ]
    selection = mask.astype(bool)
    for output, reference_values in zip(actual, expected, strict=True):
        values = np.asarray(output.image, dtype=np.float64)[selection]
        np.testing.assert_allclose(
            values, reference_values, rtol=1e-4, atol=1e-4, err_msg=output.name
        )


@pytest.mark.parametrize("baseline", [8.0, 10000.0])
@pytest.mark.parametrize("method", ["default", "ror", "uniform", "nri_uniform", "var"])
def test_lbp2d_methods(method: str, baseline: float) -> None:
    ct, _, _ = hcc_crop(np.s_[103:110, 214:223, 64])
    image = baseline + (ct - ct.mean()) / 100
    options = TransformSettings(
        image_types=(ImageType.LBP2D,), lbp2d=LBP2DSettings(method=LBPMethod(method))
    )
    with mx.stream(mx.gpu):
        actual = next(feature_images(mx.array(image.astype(np.float32)), (1.0, 1.0), options,
                                    source_residual=mx.array(image - image.astype(np.float32))))
    assert actual.name == ImageType.LBP2D.value
    # scikit-image warns for floating-point inputs; this comparison is defined on the
    # original source values rather than their float32 approximations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = np.asarray(
            local_binary_pattern(image, P=8, R=1.0, method=method), dtype=np.float64,
        )
    _assert_close(actual.image, expected, method)


def test_lbp2d_variance_keeps_fractional_values_across_integer_wrappers() -> None:
    ct, _, _ = hcc_crop(np.s_[105:108, 215:218, 64])
    slice_values = ct.astype(np.int16)
    # At P=4,R=1 the samples are the four cardinal neighbors. Their variance
    # is fractional even for integer CT, and must survive both dimensional wrappers.
    expected = float(np.var(ct[([1, 0, 1, 2], [2, 1, 0, 1])]))
    assert expected % 1 != 0
    options = TransformSettings(
        image_types=(ImageType.LBP2D,), lbp2d=LBP2DSettings(samples=4, method=LBPMethod.VARIANCE)
    )
    cases = (
        (slice_values, 2),
        (slice_values[None], 3),
        (slice_values.astype(np.float64)[None], 3),
    )
    actual: list[float] = []
    for image, ndim in cases:
        with mx.stream(mx.gpu):
            output = next(feature_images(
                mx.array(image.astype(np.float32)),
                (1.0,) * ndim,
                options,
                source_dtype=image.dtype,
            ))
        values = np.asarray(output.image, dtype=np.float64)
        actual.append(float(values[(1, 1) if ndim == 2 else (0, 1, 1)]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", ["roi", "voxel", "streamed"])
def test_filtered_feature_pipeline_prefixes_each_image_type(mode: str) -> None:
    image, mask, _ = hcc_crop(np.s_[103:109, 214:221, 60:68])
    settings = Settings(bin_width=2, features=("firstorder_Mean",))
    options = TransformSettings(image_types=(ImageType.SQUARE, ImageType.EXPONENTIAL))
    collected: dict[str, NDArray[np.float32]] = {}

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        # One consumer receives several image types; their feature names must stay distinct.
        for name, values in batch.items():
            if name not in collected:
                collected[name] = np.zeros(image.shape, dtype=np.float32)
            collected[name][tuple(centers.T)] = values

    voxel = mode != "roi"
    actual = extract(image, mask, settings=settings, transforms=options,
        mode=("voxel" if voxel else "roi"),
                            sink=consume if mode == "streamed" else None)
    features: Mapping[str, float | NDArray[np.float64] | NDArray[np.float32]] = (
        collected if mode == "streamed" else actual.features
    )
    if mode == "streamed":
        assert not actual.features
    transforms = tuple((kind, _elementary(kind, image)) for kind in options.image_types)
    expected: dict[str, float | NDArray[np.float64]] = {}
    for kind, derived in transforms:
        name = f"{kind.value}_firstorder_Mean"
        expected[name] = (masked_neighborhood_mean(derived, mask == 1) if voxel
                          else float(np.mean(derived[mask == 1])))
    assert features.keys() == expected.keys()
    assert actual.metadata["feature_validity"].keys() == expected.keys()
    transform_metadata = actual.metadata.get("transforms")
    assert transform_metadata is not None
    assert transform_metadata["image_types"] == options.image_types
    for name, value in expected.items():
        np.testing.assert_allclose(features[name], value, rtol=1e-4, atol=1e-4)
        counted = np.asarray(value)[mask == 1] if voxel else np.asarray(value)
        counts = actual.metadata["feature_validity"][name]
        assert counts["finite"] == int(np.count_nonzero(np.isfinite(counted)))
        assert counts["valid_zero"] == int(np.count_nonzero(counted == 0))
        assert counts["excluded"] == (image.size - int(np.count_nonzero(mask)) if voxel else 0)


@pytest.mark.parametrize("label", [2**31 - 1, 2**31, 2**32 - 1, 2**32,
                                  2**53 + 1, 2**63, 2**64 - 1])
@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "streamed"])
def test_derived_images_preserve_large_integer_label_selection(label: int, voxel: bool) -> None:
    image, liver, tumor = hcc_crop(np.s_[105:110, 215:221, 64])
    mask = np.where(liver != 0, np.uint64(label), np.uint64(7) * tumor).astype(np.uint64)
    options = Settings(label=label, features=("firstorder_Mean",))
    key = "square_firstorder_Mean"
    collected = np.zeros(image.shape, dtype=np.float32)

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        collected[tuple(centers.T)] = batch[key]

    actual = extract(
        image, mask, settings=options,
        transforms=TransformSettings(image_types=(ImageType.SQUARE,)),
        mode=("voxel" if voxel else "roi"), sink=consume if voxel else None,
    )
    support = mask == label
    square = _elementary(ImageType.SQUARE, image)
    expected = (masked_neighborhood_mean(square, support) if voxel
                else float(np.mean(square[support])))
    value = collected if voxel else actual.features[key]
    np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-4)


def test_unsupported_log_requests_fail_before_any_image_is_extracted() -> None:
    """The pre-flight rejects LoG grids the filter would refuse after Original ran."""
    recursive = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.LOG), log=LoGSettings(sigma=(1.0,))
    )
    with pytest.raises(ValueError, match="3D image"):
        feature_image_names((20, 20), (1.0, 1.0), recursive)
    with pytest.raises(ValueError, match="sigma / spacing <= 200"):
        feature_image_names((8, 8, 8), (1.0, 1.0, 0.001), recursive)
    sampled = TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.LOG),
        log=LoGSettings(sigma=(1.0,), kernel=LogKernel.SAMPLED),
    )
    with pytest.raises(ValueError, match="isotropic"):
        feature_image_names((8, 8, 8), (1.0, 1.0, 3.0), sampled)
    assert feature_image_names((8, 8, 8), (1.0, 1.0, 1.0), sampled)[0] == "original"


def test_options_of_unrequested_filters_do_not_reject_a_request() -> None:
    options = TransformSettings(
        image_types=(ImageType.LAWS,),
        riesz=RieszSettings(orders=((1, 0),)),
        separable_wavelet=SeparableWaveletSettings(bands=("LH",)),
    )
    names = feature_image_names((6, 6, 6), (1.0, 1.0, 1.0), options)
    assert len(names) == 1 and names[0].startswith("laws")
    wavelet = dataclasses.replace(options, image_types=(ImageType.SEPARABLE_WAVELET,))
    with pytest.raises(ValueError, match="one letter per filtered axis"):
        feature_image_names((6, 6, 6), (1.0, 1.0, 1.0), wavelet)
