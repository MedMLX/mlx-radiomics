"""Stream derived-image voxel maps to disk with bounded host storage.

    python examples/derived_voxel_export.py image.nii.gz mask.nii.gz outputs/maps --label 1

Each image type gets its own prefix (``original_``, ``wavelet-LLH_``, ...). One
disk-backed map is written at a time; the full map set is never retained in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlxradiomics import ImageType, Settings, TransformSettings, export_files
from mlxradiomics.config.transforms import WaveletSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", type=int, default=1)
    args = parser.parse_args()
    result = export_files(
        args.image,
        args.mask,
        args.output,
        settings=Settings(
            label=args.label,
            features=("firstorder_Mean", "glcm_Contrast", "glszm_ZoneEntropy"),
        ),
        transforms=TransformSettings(
            image_types=(ImageType.ORIGINAL, ImageType.WAVELET), wavelet=WaveletSettings(level=1)
        ),
        overwrite=True,
    )
    print(f"wrote {len(result.maps)} maps")
    print(result.report)


if __name__ == "__main__":
    main()
