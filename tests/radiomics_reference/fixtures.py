"""Deterministic synthetic cases for feature contract tests; not patient data."""

from dataclasses import dataclass, replace

import numpy as np

from mlxradiomics.config.settings import Settings
from mlxradiomics.contracts.types import HostArray

CASE_SEED = 104729
# The generic case catalog uses a small first-order selection; tests may replace it.
CATALOGUE_FEATURES = (
    "firstorder_Mean", "firstorder_Minimum", "firstorder_Maximum", "firstorder_Range",
)


@dataclass(frozen=True)
class Case:
    """A named synthetic image, mask, settings and spacing (mm per array axis)."""

    name: str
    image: HostArray
    mask: HostArray
    settings: Settings
    spacing: tuple[float, ...]


def cases() -> list[Case]:
    """Return deterministic synthetic cases covering material extraction paths.

    They cover a CT-like image with negative intensities, a hole and a second label; an
    unmasked neighborhood; ``force2d_axis``; larger distances with a kernel radius and
    intensity shift; values at bin boundaries in 2-D; a flat image; and a single-voxel ROI.
    """
    rng = np.random.default_rng(CASE_SEED)
    image = rng.integers(-1024, 512, size=(5, 6, 7)).astype(np.float64)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1:4, 1:5, 1:6] = 2
    mask[2, 2:4, 2:4] = 0
    mask[0, 0, 0] = 1
    ct = Settings(features=CATALOGUE_FEATURES, label=2, bin_width=100, scratch_mib=1)
    boundary = np.array([[-25.00000001, -25, -24.99999999, 0], [0, 24.99999999, 25, 25.00000001],
                         [50, 25, 0, -25]])
    tiny = np.zeros((3, 3, 3), dtype=np.uint8)
    tiny[1, 1, 1] = 1
    return [
        Case("negative_ct_multilabel", image, mask, ct, (3.0, 1.4, 0.7)),
        Case("unmasked_neighborhood", image, mask, replace(ct, masked_kernel=False), (1, 1, 1)),
        Case("force2d", image, mask, replace(ct, force2d_axis=0), (2, 1, 1)),
        Case("larger_shifted_kernel", image, mask,
             replace(ct, distances=(1, 2), kernel_radius=2, voxel_array_shift=1024,
                     symmetrical_glcm=False), (3.0, 1.4, 0.7)),
        Case("bin_boundaries_2d", boundary, np.ones(boundary.shape, dtype=np.uint8),
             Settings(features=CATALOGUE_FEATURES, bin_width=25, symmetrical_glcm=False),
             (1.3, 0.8)),
        Case("flat", np.full((3, 4), -17.0), np.ones((3, 4), dtype=np.uint8),
             Settings(features=CATALOGUE_FEATURES), (1, 1)),
        Case("singleton", np.full((3, 3, 3), 4.0), tiny,
             Settings(features=CATALOGUE_FEATURES), (1, 1, 1)),
    ]
