"""ROI extraction from one NIfTI/NRRD image and integer label mask.

    python examples/roi_from_files.py image.nii.gz mask.nii.gz outputs/roi --label 1

Extracts five first-order and GLCM features. The report is strict JSON schema 2;
non-finite values are named strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxradiomics.medmlx import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--bin-width", type=float, default=25.0)
    args = parser.parse_args()
    result = run(
        image_path=args.image, mask_path=args.mask, output_dir=args.output,
        label=args.label,
        params={
            "setting": {"binWidth": args.bin_width},
            "featureClass": {
                "firstorder": ["Mean", "Variance", "Entropy"],
                "glcm": ["Contrast", "JointEntropy"],
            },
        },
        overwrite=True,
    )
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
