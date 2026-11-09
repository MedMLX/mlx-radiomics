# Getting started

## Requirements

- An Apple Silicon Mac with Metal.
- Python 3.13.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "mlxradiomics>=1.0.1"
```

Install `"mlxradiomics[lbp]"` to compute LBP3D derived images; it adds SciPy and
trimesh for the spherical sampling geometry.

## Extract ROI features

`extract_files` reads a NIfTI or NRRD image and an integer label mask, and returns
one value per feature for the selected label:

```python
from mlxradiomics import Settings, extract_files

settings = Settings(label=1)  # the 93 default PyRadiomics intensity and texture features
result = extract_files("image.nii.gz", "mask.nii.gz", settings=settings)

result.features["original_firstorder_Mean"]   # float
result.metadata["feature_validity"]           # finite / NaN / infinity counts per feature
```

Feature keys are `<image>_<family>_<name>`, such as `original_glcm_Contrast` or
`wavelet-HHL_glcm_Contrast`. `mlxradiomics.FEATURES` lists all 190 selectable
names; see [Features](features.md).

## Arrays instead of files

`extract` takes aligned NumPy arrays. Spacing is in millimeters and in NumPy axis
order:

```python
import numpy as np
from mlxradiomics import Settings, extract

image = np.random.default_rng(0).normal(size=(16, 32, 32)).astype(np.float32)
mask = np.zeros(image.shape, dtype=np.uint8)
mask[4:12, 8:24, 8:24] = 1

result = extract(image, mask, spacing=(3.0, 1.0, 1.0),
                 settings=Settings(features=("firstorder_Mean", "glcm_Contrast")))
```

## Voxel maps

`mode="voxel"` computes each feature in a neighborhood around every ROI voxel and
returns one map per feature. `export_files` writes the maps to disk as float32
NIfTI files without holding them all in memory:

```python
from mlxradiomics import Settings, export_files

saved = export_files("image.nii.gz", "mask.nii.gz", "outputs/maps",
                     settings=Settings(features=("firstorder_Mean",)))
saved.maps["original_firstorder_Mean"]  # pathlib.Path to the NIfTI map
```

## Next steps

- [Python API](python-api.md): the four core operations, derived images and streaming.
- [MedMLX runner](runner.md): the file-in, file-out entry point used by MedMLX.
- [Parameter files](parameters.md): reuse PyRadiomics YAML settings.
- Runnable scripts are in
  [`examples/`](https://github.com/MedMLX/mlx-radiomics/tree/main/examples).
