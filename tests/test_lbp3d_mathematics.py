"""Independent mathematical checks for the 3-D spherical-harmonic LBP descriptor."""

from collections.abc import Mapping
from math import pi, sqrt
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics import Settings
from mlxradiomics.config.transforms import ImageType, LBP3DSettings, TransformSettings
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.workflows.extraction import extract


@pytest.mark.parametrize("voxel,map_dtype", [
    (False, "float64"), (True, "float64"), (True, "float32"),
])
def test_undefined_kurtosis_keeps_harmonic_features_and_validity(
    voxel: bool, map_dtype: Literal["float64", "float32"],
) -> None:
    image = np.ones((7, 7, 7), dtype=np.float64)
    mask = np.zeros(image.shape, np.uint8)
    mask[0, 0, 0] = 1
    mask[3:5, 3:5, 3:5] = 1
    options = TransformSettings(
        image_types=(ImageType.LBP3D,), lbp3d=LBP3DSettings(levels=1, subdivision=0)
    )
    settings = Settings(features=("firstorder_Mean",), bin_count=2)
    result = extract(image, mask, settings=settings, transforms=options,
        mode=("voxel" if voxel else "roi"), map_dtype=map_dtype)
    harmonic = result.features["lbp-3D-m1_firstorder_Mean"]
    kurtosis = result.features["lbp-3D-k_firstorder_Mean"]
    counts = result.metadata["feature_validity"]["lbp-3D-k_firstorder_Mean"]
    if voxel:
        assert isinstance(harmonic, np.ndarray) and isinstance(kurtosis, np.ndarray)
        assert harmonic.dtype == np.dtype(map_dtype) and kurtosis.dtype == np.dtype(map_dtype)
        assert harmonic[0, 0, 0] == pytest.approx(sqrt(pi) / 2, rel=1e-6)
        np.testing.assert_allclose(harmonic[3:5, 3:5, 3:5], sqrt(4 * pi), rtol=1e-6)
        assert kurtosis[0, 0, 0] == pytest.approx(-2 / 3, rel=1e-6)
        assert np.isnan(kurtosis[3:5, 3:5, 3:5]).all()
        assert counts["finite"] == 1 and counts["nan"] == 8
        streamed: dict[str, NDArray[np.float32]] = {}

        def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
            for name, values in batch.items():
                output = streamed.setdefault(name, np.zeros(image.shape, np.float32))
                output[tuple(centers.T)] = values

        collected = extract(image, mask, settings=settings, transforms=options,
                                   mode="voxel", sink=consume)
        assert collected.features == {}
        assert collected.metadata["feature_validity"] == result.metadata["feature_validity"]
        for name, values in result.features.items():
            np.testing.assert_allclose(streamed[name], values, rtol=0, atol=0, equal_nan=True)
    else:
        assert harmonic == pytest.approx((sqrt(pi) / 2 + 8 * sqrt(4 * pi)) / 9, rel=1e-6)
        assert np.isnan(kurtosis)
        assert counts["nan"] == 1


@pytest.mark.parametrize("value", [1e-30, 1e-12, 1.0, 0.0, -1.0, 5e9, 1e10, 1e30])
def test_constant_field_uses_zero_extension_at_image_corner(value: float) -> None:
    image = np.full((7, 7, 7), value)
    mask = np.zeros(image.shape, bool)
    mask[0, 0, 0] = True
    outputs = _lbp3d(image, mask, levels=1, subdivision=0)
    expected = sqrt(pi) / 2 if value > 0 else sqrt(4 * pi)
    assert outputs["lbp-3D-m1"][0, 0, 0] == pytest.approx(expected, rel=1e-6)
    if value == 0:
        assert np.isnan(outputs["lbp-3D-k"][0, 0, 0])
    else:
        assert outputs["lbp-3D-k"][0, 0, 0] == pytest.approx(-2 / 3, rel=1e-6)


def _lbp3d(
    image: NDArray[np.float64], mask: NDArray[np.bool_], *, levels: int, subdivision: int
) -> dict[str, NDArray[np.float64]]:
    options = TransformSettings(
        image_types=(ImageType.LBP3D,),
        lbp3d=LBP3DSettings(levels=levels, subdivision=subdivision, radius=1.0),
    )
    with mx.stream(mx.gpu):
        outputs = list(
            feature_images(
                mx.array(image.astype(np.float32)),
                (1.0, 1.0, 1.0),
                options,
                mask=mx.array(mask.astype(bool)),
                source_dtype=image.dtype,
                source_residual=mx.array(image - image.astype(np.float32)),
            )
        )
    return {output.name: np.asarray(output.image, dtype=np.float64) for output in outputs}


def _first_degree_harmonics(vertices: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Cartesian Y_1^-1, Y_1^0, Y_1^1 columns for unit vectors ``vertices``."""
    x, y, z = vertices.T
    radius = np.linalg.norm(vertices, axis=1)
    return np.column_stack(
        [
            np.sqrt(3 / (8 * pi)) * (x - 1j * y) / radius,
            np.sqrt(3 / (4 * pi)) * z / radius,
            -np.sqrt(3 / (8 * pi)) * (x + 1j * y) / radius,
        ]
    )


@pytest.mark.parametrize("subdivision", [0, 1, 2])
@pytest.mark.parametrize("value", [0.1, 3.25])
def test_all_set_bits_give_sqrt_four_pi_independent_of_subdivision(
    subdivision: int, value: float,
) -> None:
    image = np.full((7, 7, 7), value, dtype=np.float64)
    mask = np.zeros(image.shape, dtype=bool)
    mask[3, 3, 3] = True
    levels = 3
    outputs = _lbp3d(image, mask, levels=levels, subdivision=subdivision)

    assert list(outputs) == [f"lbp-3D-m{level}" for level in range(1, levels + 1)] + [
        "lbp-3D-k"
    ]
    np.testing.assert_allclose(outputs["lbp-3D-m1"][mask], sqrt(4 * pi), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["lbp-3D-m2"][mask], 0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(outputs["lbp-3D-m3"][mask], 0, rtol=0, atol=1e-5)
    assert np.isnan(outputs["lbp-3D-k"][mask]).all()


@pytest.mark.parametrize("subdivision", [1, 2])
def test_first_degree_band_matches_half_space_projection(subdivision: int) -> None:
    from scipy.spatial import SphericalVoronoi
    from trimesh.creation import icosphere

    shape = (7, 7, 7)
    z, y, x = np.indices(shape, dtype=np.float64)
    normal = np.array([1.0, sqrt(2.0), pi])
    # Icosphere vertices are sampled in array-axis order (Z, Y, X), so the Cartesian
    # normal components multiply the indices in the same order.
    image = 1000.0 + 1000.0 * (normal[0] * z + normal[1] * y + normal[2] * x)
    mask = np.zeros(shape, dtype=bool)
    mask[3, 3, 3] = True

    vertices = np.asarray(icosphere(subdivisions=subdivision, radius=1.0).vertices)
    areas = SphericalVoronoi(vertices, radius=1.0).calculate_areas()
    binary = (vertices @ normal >= 0).astype(np.float64)
    y00 = 1 / (2 * sqrt(pi))
    expected_m1 = np.sqrt(((binary * areas).sum() * y00) ** 2)
    coefficients = (binary * areas) @ np.conj(_first_degree_harmonics(vertices))
    expected_m2 = np.sqrt(np.sum(np.abs(coefficients) ** 2))

    outputs = _lbp3d(image, mask, levels=2, subdivision=subdivision)
    np.testing.assert_allclose(outputs["lbp-3D-m1"][mask], expected_m1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["lbp-3D-m2"][mask], expected_m2, rtol=1e-5, atol=1e-5)
    # The half-space covers half the sphere, so the degree-0 band is exactly sqrt(pi).
    np.testing.assert_allclose(outputs["lbp-3D-m1"][mask], sqrt(pi), rtol=1e-5, atol=1e-5)
    # First-order Voronoi quadrature of a half-space approaches the continuous first-degree
    # band norm sqrt(3*pi)/2 from above; the residual shrinks with subdivision.
    analytic = sqrt(3 * pi) / 2
    assert abs(outputs["lbp-3D-m2"][mask][0] - analytic) < 0.02 * analytic


def test_source_variation_collapsed_by_float32_is_not_constant() -> None:
    image = np.full((7, 7, 7), 1e8, dtype=np.float64)
    image[3, 3, 4] += 1
    mask = np.zeros(image.shape, dtype=bool)
    mask[3, 3, 3] = True
    with pytest.raises(ValueError, match="threshold is ambiguous"):
        _lbp3d(image, mask, levels=2, subdivision=1)


@pytest.mark.parametrize("amplitude", [1e36, -1e36])
def test_near_float32_limit_field_keeps_the_unit_amplitude_descriptor(amplitude: float) -> None:
    # Bits and kurtosis are amplitude invariant, so a near-limit finite field must give
    # the descriptor of its unit-amplitude field, not NaN-derived bits.
    z, y, x = np.meshgrid(*[np.arange(7.0)] * 3, indexing="ij")
    field = 100 + z + np.sqrt(2) * y + np.pi * x
    mask = np.zeros(field.shape, bool)
    mask[3, 3, 3] = True
    sign = 1.0 if amplitude > 0 else -1.0
    reference = _lbp3d(sign * field, mask, levels=1, subdivision=0)
    scaled = (amplitude * field).astype(np.float32).astype(np.float64)
    outputs = _lbp3d(scaled, mask, levels=1, subdivision=0)
    assert outputs["lbp-3D-m1"][3, 3, 3] == reference["lbp-3D-m1"][3, 3, 3]
    assert outputs["lbp-3D-k"][3, 3, 3] == pytest.approx(
        reference["lbp-3D-k"][3, 3, 3], rel=1e-5)


def test_integer_rounding_tie_at_center_threshold_is_refused() -> None:
    # A 0/1 checkerboard along x sampled at this radius places four spline samples
    # exactly at 0.5 (mirrored coefficients [-1, 2, -1, 2, -1], half-point weights
    # [1, 23, 23, 1] / 48). Float32 roundoff splits them around the tie, which would
    # silently clear three of twelve bits.
    image = (np.indices((5, 5, 5))[2] % 2).astype(np.uint8)
    mask = np.zeros(image.shape, bool)
    mask[2, 2, 3] = True
    options = TransformSettings(
        image_types=(ImageType.LBP3D,),
        lbp3d=LBP3DSettings(levels=1, subdivision=0, radius=0.5877852522924731),
    )
    with mx.stream(mx.gpu), pytest.raises(ValueError, match="rounding threshold is ambiguous"):
        list(feature_images(mx.array(image.astype(np.float32)), (1.0, 1.0, 1.0), options,
                            mask=mx.array(mask), source_dtype=image.dtype))
