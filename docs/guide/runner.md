# MedMLX runner

{func}`mlxradiomics.medmlx.run` is the registered [MedMLX](https://github.com/MedMLX/medmlx)
entry point. It reads files, writes a report and returns JSON-compatible
descriptions of the outputs:

```python
from mlxradiomics.medmlx import run

result = run(image_path="image.nii.gz", mask_path="mask.nii.gz", output_dir="features")
result["files"]["features_json"]  # path to the report
```

## Use through MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md)
installs this runtime and lets an AI agent describe, check and run `mlxradiomics`:

```sh
uv tool install --python 3.13 "medmlx[vision,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`.
Other MCP clients take the same command and arguments. The same install provides the
`medmlx` CLI, for example `medmlx describe mlxradiomics`. The MCP `run_model` tool
takes the arguments of `run`.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `image_path` | required | One 2D or 3D image: `.nii`, `.nii.gz`, `.nrrd` or `.nhdr` |
| `mask_path` | required | Integer label mask on the same grid. A geometry mismatch is refused unless `params` enable `correctMask` or `resampledPixelSpacing` |
| `output_dir` | required | Created if missing |
| `params` / `params_path` | none | PyRadiomics-style settings as a mapping or YAML file, at most one; see [Parameter files](parameters.md) |
| `label` | params label, else 1 | Mask label to extract |
| `voxel` | `False` | `False`: one value per feature. `True`: one map per feature |
| `overwrite` | `False` | Refuse to replace existing outputs unless `True` |
| `temporary_dir` | system temp | Existing directory for voxel-mode scratch |
| `max_temporary_mib` | none | Cap on voxel-mode scratch, checked before extraction and each batch. Requires `voxel=True`. Not a RAM cap |

## Outputs

| Mode | Files in `output_dir` |
| --- | --- |
| ROI | `features.json`, `features.csv` |
| Voxel | `features.json`, `maps/<feature>.nii.gz` (float32, processed grid, zero outside the ROI) |

The return value has `outputs`, `artifacts`, `files` and `details`. `outputs`
maps each file's role to its path: `features_json`, `features_csv` in ROI mode,
and one role per map named by its feature. `artifacts` gives each role's MedMLX
kind and coordinate space. `files` groups the same paths as `features_json`,
`features_csv` and `maps`. A map references `image_path` as its grid only when
the processed grid is unchanged; cropped or resampled maps use physical
coordinates.

`features.json` uses report schema 2. Finite values are JSON numbers; NaN and
infinities are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`. CSV files and
maps keep the numeric values. `details.execution` records:

| Field | Content |
| --- | --- |
| `runtime` | Package versions, Python, platform, Metal device, `MLX_ENABLE_TF32` (null when unset) |
| `settings`, `transforms`, `image_settings` | Effective extraction, derived-image and file-resampling options |
| `array_shape`, `array_dtype`, `spacing` | Processed shape, input dtype and spacing, NumPy axis order |
| `source_geometry`, `image_geometry` | Input and processed grids; origin and direction in ITK LPS order |
| `roi_voxels`, `excluded_voxels` | Intensity-ROI size and array voxels outside it |
| `feature_validity` | Per feature: `finite`, `valid_zero` (subset of finite), `nan`, `positive_infinity`, `negative_infinity`, `excluded` counts |

## Errors

| Error | Raised for |
| --- | --- |
| `InvalidInputError` | Bad paths, formats, geometry, labels, parameters or unsupported options; raised before extraction |
| `ModelExecutionError` | Runtime or optional-dependency failures during extraction; incomplete outputs or failed publication rollback |
| `OSError` | File I/O failures |

Unexpected programming errors propagate without being converted to a case failure.

## Writing files

Read output files after `run` returns. Failed runs restore previous outputs and
remove new ones. Concurrent runs must use different output directories.
Detached inputs such as `.nhdr` always need a new directory, even with
`overwrite=True`.
