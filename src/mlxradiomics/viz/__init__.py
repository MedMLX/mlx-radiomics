"""Optional CPU figure export of already evaluated NumPy feature maps.

Install ``mlxradiomics[viz]``. Importing the package root does not load this
module or Matplotlib. Figures are created without opening GUI windows or
changing Matplotlib's configured backend.
"""

from importlib.util import find_spec

if find_spec("matplotlib") is None:
    raise ImportError(
        'Feature-map figures require Matplotlib; install "mlxradiomics[viz]".'
    )

from mlxradiomics.viz.rendering import COLORMAPS, plot_feature_maps

__all__ = ["COLORMAPS", "plot_feature_maps"]
