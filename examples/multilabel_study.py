"""Run one or more cases and labels with explicit per-combination statuses.

    python examples/multilabel_study.py outputs/study \
      --case case-a image-a.nii.gz mask-a.nii.gz \
      --case case-b image-b.nii.gz mask-b.nii.gz --labels 1 2

Outputs are written under ``<output>/<case>/label-<n>``. A failed label does not stop
the remaining combinations; inspect every returned status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxradiomics.workflows.study import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case", nargs=3, action="append", metavar=("ID", "IMAGE", "MASK"),
        required=True,
    )
    parser.add_argument("--labels", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    cases = [
        {"case_id": case_id, "image_path": image, "mask_path": mask}
        for case_id, image, mask in args.case
    ]
    outcomes = run_study(
        cases, args.output, labels=args.labels,
        params={"featureClass": {"firstorder": ["Mean"], "glcm": ["Contrast"]}},
        overwrite=True,
    )
    print(json.dumps(outcomes, indent=2))
    if any(outcome["status"] == "failed" for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
