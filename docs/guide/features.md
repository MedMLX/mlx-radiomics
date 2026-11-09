# Features

`mlxradiomics.FEATURES` lists all 190 selectable names, including all 116
non-deprecated PyRadiomics features. Without a `features` selection, extraction
returns the 93 PyRadiomics intensity and texture features marked as defaults
below. Shape and the other families must be requested by name.

| Family | Count | Modes |
| --- | ---: | --- |
| `firstorder` | 21 | ROI, voxel |
| `intensityhistogram` | 23 | ROI, voxel |
| `glcm` | 26 | ROI, voxel |
| `gldm` (NGLDM) | 17 | ROI, voxel |
| `ngtdm` | 5 | ROI, voxel |
| `glrlm` | 16 | ROI, voxel |
| `glszm` | 16 | ROI, voxel |
| `gldzm` | 16 | ROI, voxel |
| `ivh` | 7 | ROI, voxel |
| `localintensity` | 2 | 3D ROI |
| `shape` | 32 | 3D ROI |
| `shape2D` | 9 | 2D ROI, or one slice of a 3D ROI |

Select features with their family prefix:

```python
from mlxradiomics import Settings

Settings(features=("shape_MeshVolume", "firstorder_Mean", "glcm_Contrast"))
```

Calculation conventions are separate `Settings` fields with IBSI defaults.
Select alternative definitions individually (see [Conventions](python-api.md#conventions)). The catalogue
below maps public feature names to IBSI identifiers.

## Catalogue

Generated from the package's feature list and IBSI manifest. A blank IBSI column
marks a PyRadiomics feature or extension without an IBSI definition.

```{include} ../_generated/feature-catalog.md
```

## Select families in Python

```python
from mlxradiomics import Settings, list_features

settings = Settings(features=list_features("firstorder", "glcm"))
map_features = list_features(mode="voxel", ndim=3)
```

The helper filters the catalogue by mode and optional input dimension. It does
not change the default selection or discretization. ROI-only families are
excluded in voxel mode; unknown family names raise `ValueError`.
