"""HCC-TACE image cases for feature contract tests."""

from dataclasses import dataclass, replace

import numpy as np

from mlxradiomics.config.settings import Settings
from mlxradiomics.contracts.types import HostArray
from radiomics_reference.real_images import hcc_crop

# The generic case catalog uses a small first-order selection; tests may replace it.
CATALOGUE_FEATURES = (
    "firstorder_Mean", "firstorder_Minimum", "firstorder_Maximum", "firstorder_Range",
)


@dataclass(frozen=True)
class Case:
    """A named image, mask, settings and spacing (mm per array axis)."""

    name: str
    image: HostArray
    mask: HostArray
    settings: Settings
    spacing: tuple[float, ...]


def cases() -> list[Case]:
    """Return real-image cases covering material extraction paths.

    Cases cover a multilabel liver/tumor crop, an unmasked neighborhood,
    ``force2d_axis``, a larger kernel, a 2D CT slice, stored background, and one voxel.
    """
    image, liver, tumor = hcc_crop(np.s_[105:110, 215:221, 62:69])
    if not liver.any() or not tumor.any() or np.any((liver != 0) & (tumor != 0)):
        raise ValueError("Expected nonempty, mutually exclusive liver and tumor segments")
    mask = np.where(tumor != 0, 1, np.where(liver != 0, 2, 0)).astype(np.uint8)
    ct = Settings(features=CATALOGUE_FEATURES, label=2, bin_width=100, scratch_mib=1)
    plane = image[:, :, 3]
    background, _, _ = hcc_crop(np.s_[0:3, 0:4, 40])
    singleton, _, real_tumor = hcc_crop(np.s_[122:125, 238:241, 65:68])
    tiny = np.zeros((3, 3, 3), dtype=np.uint8)
    tiny[1, 1, 1] = real_tumor[1, 1, 1]
    if not tiny.any():
        raise ValueError("Expected a tumor voxel in the singleton crop")
    spacing = (0.859375, 0.859375, 2.5)
    return [
        Case("ct_liver_tumor", image, mask, ct, spacing),
        Case("unmasked_neighborhood", image, mask, replace(ct, masked_kernel=False), spacing),
        Case("force2d", image, mask, replace(ct, force2d_axis=0), spacing),
        Case("larger_shifted_kernel", image, mask,
             replace(ct, distances=(1, 2), kernel_radius=2, voxel_array_shift=1024,
                     symmetrical_glcm=False), spacing),
        Case("ct_slice_2d", plane, np.ones(plane.shape, dtype=np.uint8),
             Settings(features=CATALOGUE_FEATURES, bin_width=25, symmetrical_glcm=False),
             spacing[:2]),
        Case("flat_background", background, np.ones(background.shape, dtype=np.uint8),
             Settings(features=CATALOGUE_FEATURES), spacing[:2]),
        Case("singleton", singleton, tiny,
             Settings(features=CATALOGUE_FEATURES), spacing),
    ]
