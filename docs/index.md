# mlxradiomics

Extract radiomics features and voxel maps on the Apple Silicon GPU with native
[MLX](https://github.com/ml-explore/mlx).

```{figure} feature-maps.png
:alt: A lung CT slice with a tumour ROI, and six radiomics voxel maps computed inside it

A lung tumour on CT (NSCLC-Radiomics LUNG1-023) and six of its feature maps,
computed with `extract(..., mode="voxel")` and a kernel radius of 2. Each panel is
titled with its feature key.
```

| Input | Output |
| :--- | :--- |
| Aligned image + integer label mask | ROI features (JSON/CSV) or one float32 NIfTI map per feature |

mlxradiomics uses PyRadiomics feature names and parameter files, with IBSI
conventions by default. Feature arithmetic runs in float32 on Metal. It needs
Apple Silicon and has no CPU fallback. It uses no model weights.

```python
from mlxradiomics import Settings, extract_files

result = extract_files(
    "image.nii.gz", "mask.nii.gz",
    settings=Settings(features=("firstorder_Mean", "glcm_Contrast")),
)
result.features["original_glcm_Contrast"]
```

New users should start with [Getting started](guide/getting-started.md). The
[API reference](reference/index.md) documents every public function and class.

```{toctree}
:caption: User guide
:hidden:

guide/getting-started
guide/python-api
guide/visualization
guide/runner
guide/parameters
guide/features
guide/limits
```

```{toctree}
:caption: API reference
:hidden:

reference/index
```

```{toctree}
:caption: Release notes
:hidden:

release-notes
```
