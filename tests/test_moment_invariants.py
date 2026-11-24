"""Moment invariants of the HCC tumor at the public extraction boundary.

Expected values come from an independent float64 evaluation of the definition: the
normalized central moments of PySERA 2.1.5, taken over physical voxel-center
coordinates. On an isotropic grid this is PySERA's own voxel-index calculation.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings, extract, list_features

FEATURES = list_features("momentinvariant")
# The tumor of HCC_063 with a one-voxel margin: 18,114 voxels between 12 and 150 HU.
_TUMOR = np.s_[97:150, 209:301, 56:75]
_SECOND = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))
_THIRD = (
    (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
    (1, 2, 0), (1, 0, 2), (0, 2, 1), (0, 1, 2), (1, 1, 1),
)


def _invariants(
    density: NDArray[np.float64], spacing: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return I1, F2, I2, I3 and F3 of a density sampled at voxel centers."""
    support = density != 0
    weights = density[support]
    positions = np.argwhere(support) * np.asarray(spacing)
    voxel_volume = float(np.prod(spacing))
    mass = weights.sum() * voxel_volume
    offsets = positions - (weights[:, None] * positions).sum(axis=0) * voxel_volume / mass

    def eta(p: int, q: int, r: int) -> float:
        central = (weights * offsets[:, 0]**p * offsets[:, 1]**q * offsets[:, 2]**r).sum()
        return float(central * voxel_volume / mass ** ((p + q + r) / 3 + 1))

    xx, yy, zz, xy, xz, yz = (eta(*order) for order in _SECOND)
    third = [eta(*order) for order in _THIRD]
    return np.array([
        xx + yy + zz,
        xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2),
        xx * yy + xx * zz + yy * zz - xy**2 - xz**2 - yz**2,
        xx * yy * zz + 2 * xy * xz * yz - zz * xy**2 - yy * xz**2 - xx * yz**2,
        sum(v**2 for v in third[:3]) + 3 * sum(v**2 for v in third[3:9]) + 6 * third[9]**2,
    ])


def _expected(
    image: NDArray[np.float64],
    intensity_mask: NDArray[np.bool_],
    morphology_mask: NDArray[np.bool_],
    spacing: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Intensity invariants, then shape invariants, in catalog order."""
    samples = image[intensity_mask]
    density = np.zeros(image.shape)
    density[intensity_mask] = (samples - samples.min()) / (samples.mean() - samples.min())
    return np.concatenate([
        _invariants(density, spacing),
        _invariants(morphology_mask.astype(np.float64), spacing),
    ])


def _extract(
    image: NDArray[np.float64],
    mask: NDArray[np.uint8],
    spacing: tuple[float, float, float],
    resegment_range: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    settings = Settings(features=FEATURES, resegment_range=resegment_range)
    result = extract(image, mask, spacing=spacing, settings=settings)
    return np.array([result.features[f"original_{name}"] for name in FEATURES])


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.78, 0.78, 2.5)])
@pytest.mark.parametrize("resegment_range", [None, (40.0, 120.0)])
def test_invariants_follow_the_definition_on_each_mask(
    spacing: tuple[float, float, float], resegment_range: tuple[float, float] | None,
) -> None:
    image, _, tumor = hcc_crop(_TUMOR)
    morphology = tumor.astype(np.bool_)
    intensity = morphology.copy()
    if resegment_range is not None:
        # Resegmentation narrows the intensity density and leaves the shape intact.
        intensity &= (image >= resegment_range[0]) & (image <= resegment_range[1])
        assert 0 < np.count_nonzero(intensity) < np.count_nonzero(morphology)
    computed = _extract(image, tumor, spacing, resegment_range=resegment_range)
    # Feature arithmetic is float32; the invariants are well conditioned on this ROI.
    np.testing.assert_allclose(
        computed, _expected(image, intensity, morphology, spacing), rtol=1e-6, atol=0,
    )


def test_invariants_survive_rotation_scaling_and_intensity_gain() -> None:
    image, _, tumor = hcc_crop(_TUMOR)
    spacing = (0.78, 0.78, 2.5)
    reference = _extract(image, tumor, spacing)
    # A quarter turn with a reflection, a uniform 3.5x magnification and an affine
    # intensity change describe the same object.
    turned = _extract(
        np.ascontiguousarray(image.transpose(2, 0, 1)[::-1]) * 0.25 + 900,
        np.ascontiguousarray(tumor.transpose(2, 0, 1)[::-1]),
        (2.5 * 3.5, 0.78 * 3.5, 0.78 * 3.5),
    )
    np.testing.assert_allclose(turned, reference, rtol=2e-6, atol=0)


def test_constant_intensity_leaves_only_the_shape_defined() -> None:
    image, _, tumor = hcc_crop(_TUMOR)
    computed = _extract(np.full_like(image, 37.0), tumor, (1.0, 1.0, 1.0))
    assert np.all(np.isnan(computed[:5]))
    np.testing.assert_allclose(
        computed[5:], _invariants(tumor.astype(np.float64), (1.0, 1.0, 1.0)), rtol=1e-6, atol=0,
    )


def test_voxel_mode_refuses_moment_invariants() -> None:
    image, _, tumor = hcc_crop(_TUMOR)
    with pytest.raises(ValueError, match="moment-invariant"):
        extract(
            image, tumor, spacing=(1.0, 1.0, 1.0), mode="voxel",
            settings=Settings(features=("momentinvariant_ShapeI1",)),
        )
