"""Exact physical peaks for bounded sums, ties and separated candidate support."""

from math import fsum, pi

import numpy as np
import pytest
from radiomics_reference.real_images import hcc_background, hcc_patch

from mlxradiomics import Settings, extract


@pytest.mark.parametrize("scale", [1.0, 0.125, 32768.0])
@pytest.mark.parametrize("selected", [
    ("LocalIntensityPeak",), ("GlobalIntensityPeak",),
    ("LocalIntensityPeak", "GlobalIntensityPeak"),
])
def test_physical_peaks_match_exact_sums_with_cancellation_and_truncated_rows(
    scale: float, selected: tuple[str, ...],
) -> None:
    ct, _, _ = hcc_patch((4, 5, 6))
    image = (ct - ct.mean()) * scale
    mask = np.zeros(image.shape, dtype=np.bool_)
    mask[0, 0, 0] = mask[2, 2, 2] = mask[3, 4, 5] = True
    image[mask] = image.max()
    spacing = (3.0, 2.0, 4.0)
    radius_squared = (3000 / (4 * pi)) ** (2 / 3)
    means = [
        fsum(samples) / len(samples)
        for center in zip(*np.nonzero(mask), strict=True)
        for samples in [[
            float(image[point]) for point in np.ndindex(image.shape)
            if sum(((a - b) * step) ** 2
                   for a, b, step in zip(center, point, spacing, strict=True)) <= radius_squared
        ]]
    ]
    # Every ROI center is tied for the source maximum, so both peaks select from
    # the same independently enumerated physical means, including image background.
    expected = float(np.float32(max(means)))
    result = extract(image, mask, spacing=spacing, settings=Settings(
        features=tuple(f"localintensity_{name}" for name in selected),
    ))
    for name in selected:
        assert result.features[f"original_localintensity_{name}"] == expected


def test_local_peak_uses_all_tied_maximum_neighborhoods_with_separated_support() -> None:
    image, _, _ = hcc_patch((1, 1, 30))
    mask = np.zeros(image.shape, dtype=np.bool_)
    mask[0, 0, 1] = mask[0, 0, 28] = True
    image[mask] = image.max()
    if image[0, 0, :3].mean() > image[0, 0, 27:].mean():
        image = image[..., ::-1].copy()
    # Place the stronger real-source neighborhood at the later tied candidate;
    # a first-maximum shortcut must not choose the weaker one.
    assert image[0, 0, 27:].mean() > image[0, 0, :3].mean()
    expected = float(np.float32(image[0, 0, 27:].mean()))
    result = extract(image, mask, spacing=(1.0, 1.0, 6.0), settings=Settings(
        features=("localintensity_LocalIntensityPeak",),
    ))
    assert result.features["original_localintensity_LocalIntensityPeak"] == expected


@pytest.mark.parametrize("selected", [
    ("LocalIntensityPeak",), ("GlobalIntensityPeak",),
    ("LocalIntensityPeak", "GlobalIntensityPeak"),
])
def test_finite_float32_sources_keep_large_peaks_without_residuals(
    selected: tuple[str, ...],
) -> None:
    # The three-voxel sum exceeds float32, but its mean is representable. Exact
    # float32 sources have no residual, including on the convolution path.
    value = float(np.float32(2e38))
    image = hcc_background((1, 1, 3)) * (-value / 2048)
    result = extract(image, np.ones(image.shape, dtype=np.bool_), spacing=(1.0, 1.0, 6.0),
                     settings=Settings(features=tuple(f"localintensity_{n}" for n in selected)))
    for name in selected:
        assert result.features[f"original_localintensity_{name}"] == pytest.approx(value, rel=3e-6)


@pytest.mark.parametrize("name", ["LocalIntensityPeak", "GlobalIntensityPeak"])
def test_nonfinite_background_inside_a_candidate_sphere_is_still_refused(name: str) -> None:
    image, _, _ = hcc_patch((1, 1, 3))
    image[0, 0, 2] = np.inf
    mask = np.array([[[False, True, False]]], dtype=np.bool_)
    with pytest.raises(ValueError, match="spheres require finite float32-range samples"):
        extract(image, mask, spacing=(1.0, 1.0, 6.0), settings=Settings(
            features=(f"localintensity_{name}",),
        ))
