# Parameter files

The runner, studies and {func}`~mlxradiomics.config.parameters.parse_params`
accept PyRadiomics-style settings as a mapping or a YAML file. The supported
sections are `setting`, `imageType` and `featureClass`. Unsupported keys, image
types and per-image overrides are rejected, not ignored.

Mapping keys must be strings at every level. Numeric settings require numbers;
YAML `true` and `false` are accepted only for boolean settings.

```yaml
setting:
  binWidth: 25
  label: 1
  resampledPixelSpacing: [1.0, 1.0, 1.0]
  interpolator: sitkLinear
  padDistance: 1
imageType:
  Original: {}
  Wavelet:
    level: 1
featureClass:
  firstorder: [Mean, Entropy]
  glcm: [Contrast, JointEntropy]
```

`resampledPixelSpacing` uses ITK x-first order, as in PyRadiomics. The Python
`ImageSettings.resampled_spacing` uses NumPy axis order instead.

## Image types

`imageType` keys: `Original`, `Square`, `SquareRoot`, `Logarithm`, `Exponential`,
`Gradient`, `Wavelet`, `LoG`, `LBP2D`, `LBP3D`, `Mean`, `Laws`, `Gabor`,
`SeparableWavelet`. `Simoncelli` (alias `NonseparableWavelet`) and `Riesz` are
recognized but cannot run on this release; see [Limits](limits.md).

{func}`~mlxradiomics.config.parameters.parse_params` documents every per-filter key.
{mod}`mlxradiomics.config.transforms` documents the typed filter options. See
[Conventions](python-api.md#conventions) for the corresponding definition settings.

## From Python

```python
from mlxradiomics.config.parameters import load_params, parse_params

parsed = parse_params(load_params("params.yaml"), ndim=3)
parsed.settings, parsed.image_settings, parsed.transforms
```

The three parsed objects can be passed straight to
{func}`~mlxradiomics.extract_files` and {func}`~mlxradiomics.export_files`.
