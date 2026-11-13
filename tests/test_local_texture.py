"""Fused voxel-map texture kernels against independent definitions and the dense path."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import settings_for
from numpy.typing import NDArray
from radiomics_reference import catalogue
from radiomics_reference.real_images import hcc_crop

from mlxradiomics import Settings, extract
from mlxradiomics.config.settings import FAMILIES
from mlxradiomics.imaging.preprocessing import prepare
from mlxradiomics.voxel import extraction as voxel_extraction

NAMES = tuple(
    f"{family}_{name}" for family in ("glcm", "glrlm", "glszm", "gldm", "ngtdm")
    for name in FAMILIES[family]
)
_CONTINUOUS, _HOLES, _ = hcc_crop(np.s_[103:110, 214:222, 60:69])
_LEVELS = np.floor(_CONTINUOUS / 25) * 25
_PLANE, _PLANE_MASK, _ = hcc_crop(np.s_[103:117, 214:229, 64])
_BACKGROUND, _, _ = hcc_crop(np.s_[0:5, 0:5, 60:65])

CASES = {
    "levels-holes": (_LEVELS, _HOLES, Settings(bin_width=25)),
    "continuous": (_CONTINUOUS, _HOLES, Settings(bin_count=16)),
    "asymmetric": (_LEVELS, _HOLES, Settings(bin_width=25, symmetrical_glcm=False)),
    "unmasked": (_LEVELS, _HOLES, Settings(bin_width=25, masked_kernel=False)),
    "distances": (_LEVELS, _HOLES, settings_for("pyradiomics",
        bin_width=25, distances=(1, 2), gldm_alpha=1)),
    "force2d": (_LEVELS, _HOLES, settings_for("pyradiomics", bin_width=25, force2d_axis=0)),
    "plane-radius2": (_PLANE, _PLANE_MASK, Settings(bin_width=25, kernel_radius=2)),
    "flat": (
        _BACKGROUND,
        np.ones((5, 5, 5), dtype=np.uint8), settings_for("pyradiomics"),
    ),
}


def _ineligible(*_: object) -> bool:
    return False


def _maps(image: NDArray[np.float64], mask: NDArray[np.uint8], settings: Settings,
          monkeypatch: pytest.MonkeyPatch | None = None) -> dict[str, NDArray[np.float64]]:
    if monkeypatch is not None:
        monkeypatch.setattr(voxel_extraction, "fused_glcm_eligible", _ineligible)
        monkeypatch.setattr(voxel_extraction, "fused_texture_eligible", _ineligible)
    result = extract(image, mask, settings=replace(settings, features=NAMES), mode="voxel")
    return {name.removeprefix("original_"): np.asarray(value)
            for name, value in result.features.items()}


def _oracle(image: NDArray[np.float64], mask: NDArray[np.uint8], settings: Settings,
            centers: NDArray[np.intp]) -> dict[str, NDArray[np.float64]]:
    """Independent local definitions on each center's clipped neighborhood."""
    data = prepare(image, mask, settings, voxel=True)
    assert data.bins is not None
    values: dict[str, list[float]] = {name: [] for name in NAMES}
    for center in centers:
        spans = [0 if settings.force2d_axis == a else settings.kernel_radius
                 for a in range(image.ndim)]
        axes: list[NDArray[np.intp]] = [
            np.arange(int(c) - r, int(c) + r + 1, dtype=np.intp)
            for c, r in zip(center, spans, strict=True)
        ]
        grid: tuple[NDArray[np.intp], ...] = tuple(np.meshgrid(*axes, indexing="ij"))
        inside = np.logical_and.reduce(
            [(g >= 0) & (g < s) for g, s in zip(grid, image.shape, strict=True)])
        clipped = tuple(np.clip(g, 0, s - 1) for g, s in zip(grid, image.shape, strict=True))
        local = catalogue.intensity_features(
            data.image[clipped], data.bins[clipped].astype(np.int32),
            data.sampling_mask[clipped] & inside, settings, (1.0,) * image.ndim,
            data.gray_levels,
        )
        for name in NAMES:
            values[name].append(local[name])
    return {name: np.array(value) for name, value in values.items()}


def _assert_close(actual: NDArray[np.float64], expected: NDArray[np.float64], name: str) -> None:
    for check in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(check(actual), check(expected), err_msg=name)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-5, atol=2e-6,
                               err_msg=name)


@pytest.mark.parametrize("case", CASES)
def test_fused_maps_match_independent_local_definitions(
    case: str,
) -> None:
    image, mask, settings = CASES[case]
    actual = _maps(image, mask, settings)
    centers = np.argwhere(mask > 0)
    centers = centers[np.random.default_rng(1).choice(len(centers), 60, replace=False)]
    expected = _oracle(image, mask, settings, centers)
    for name in NAMES:
        _assert_close(actual[name][tuple(centers.T)], expected[name], name)


@pytest.mark.parametrize("case", ["levels-holes", "unmasked", "force2d", "plane-radius2"])
def test_fused_maps_match_dense_maps(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask, settings = CASES[case]
    fused = _maps(image, mask, settings)
    dense = _maps(image, mask, settings, monkeypatch)
    roi = mask > 0
    for name in NAMES:
        _assert_close(fused[name][roi], dense[name][roi], name)

