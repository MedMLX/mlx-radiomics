# Features

`mlxradiomics.FEATURES` lists all 200 selectable names, including all 116
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
| `momentinvariant` | 10 | 3D ROI |
| `shape2D` | 9 | 2D ROI, or one slice of a 3D ROI |

Select features with their family prefix:

```python
from mlxradiomics import Settings

Settings(features=("shape_MeshVolume", "firstorder_Mean", "glcm_Contrast"))
```

Calculation conventions are separate `Settings` fields with IBSI defaults.
Select alternative definitions individually (see [Conventions](python-api.md#conventions)). The catalog
below maps public feature names to IBSI identifiers.

## Catalog

Generated from the package's feature list and IBSI manifest. A blank IBSI column
marks a PyRadiomics feature or extension without an IBSI definition.

```{include} ../_generated/feature-catalog.md
```

## Moment invariants

`momentinvariant` holds five invariants of the ROI shape and the same five of its
intensity, as defined in PySERA. They are dimensionless and do not change when the ROI
is moved, rotated or uniformly scaled.

Each set is computed from a density at the voxel centers. The shape density is 1 on the
morphological mask. The intensity density is `(I - min) / (mean - min)` on the intensity
mask, so the intensity invariants also ignore an intensity offset or positive gain, and
they are NaN for a constant ROI. With mass `m000` and central moments `mu_pqr`, the
normalized moments are `eta_pqr = mu_pqr / m000^((p + q + r) / 3 + 1)`.

| Suffix | Definition |
| --- | --- |
| `I1` | Trace of the second-order tensor, `eta200 + eta020 + eta002` |
| `F2` | Sum of the squared elements of the second-order tensor |
| `I2` | Second principal invariant of the second-order tensor, `eta200 eta020 + eta200 eta002 + eta020 eta002 - eta110^2 - eta101^2 - eta011^2` |
| `I3` | Determinant of the second-order tensor |
| `F3` | Sum of the squared elements of the third-order tensor |

Names are `momentinvariant_Shape<suffix>` and `momentinvariant_Intensity<suffix>`, such
as `momentinvariant_ShapeI1`. Moments use physical coordinates, so anisotropic spacing
is honored. PySERA uses voxel indices, and the two agree on an isotropic grid. On an HCC
tumor at 1 mm spacing, all ten values match PySERA 2.1.5 within 5e-8 relative. PySERA
reports each connected lesion of a mask separately, and here the whole mask is one ROI.
Like shape, the family is computed once on the original image.

## Select families in Python

```python
from mlxradiomics import Settings, list_features

settings = Settings(features=list_features("firstorder", "glcm"))
map_features = list_features(mode="voxel", ndim=3)
```

The helper filters the catalog by mode and optional input dimension. It does
not change the default selection or discretization. ROI-only families are
excluded in voxel mode; unknown family names raise `ValueError`.
