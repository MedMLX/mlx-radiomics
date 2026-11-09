"""Sphinx configuration for the mlxradiomics documentation site.

MLX is mocked for autodoc because it only installs on macOS arm64; the docs build
imports the package for its docstrings and never runs Metal code. Mocking it locally
too keeps local builds identical to the Linux Read the Docs builders.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from sphinx.application import Sphinx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "mlxradiomics"
author = "Joseph Sandoval and Alfredo Sandoval Jr."
copyright = "2026, Joseph Sandoval and Alfredo Sandoval Jr."

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "_generated", "_templates", "requirements.txt"]
templates_path = ["../_templates"]

autodoc_mock_imports = ["mlx"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"members": True, "show-inheritance": False}
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Keep "On this page" to section headings; dataclass fields would flood it.
toc_object_entries = False

html_theme = "furo"
html_title = "mlxradiomics"
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
pygments_dark_style = "github-dark"

# Dark mode uses a black background, as on medmlx.org; light mode keeps Furo's defaults.
html_theme_options = {
    "source_repository": "https://github.com/MedMLX/mlx-radiomics",
    "source_branch": "main",
    "source_directory": "docs/",
    "dark_css_variables": {
        "color-background-primary": "#000000",
        "color-background-secondary": "#0b0b0c",
        "color-background-hover": "#1c1c1e",
        "color-background-border": "rgba(255, 255, 255, 0.14)",
        "color-foreground-primary": "#f5f5f7",
        "color-foreground-secondary": "#a1a1a6",
        "color-foreground-muted": "#86868b",
        "color-brand-primary": "#f5f5f7",
        "color-brand-content": "#2997ff",
        "color-brand-visited": "#2997ff",
        "color-code-background": "#161617",
        "color-inline-code-background": "#1c1c1e",
    },
}


def _skip_reexported(
    app: Sphinx, what: str, name: str, obj: object, skip: bool, options: object,
) -> bool | None:
    """Document top-level ``mlxradiomics`` names once, on the API reference page.

    Their defining modules would otherwise document them again, which makes every
    type-hint cross-reference to them ambiguous.
    """
    import mlxradiomics

    if what == "module" and name in mlxradiomics.__all__ and getattr(mlxradiomics, name) is obj:
        return True
    return None


def setup(app: Sphinx) -> None:
    """Register the re-export filter."""
    app.connect("autodoc-skip-member", _skip_reexported)


def _load_runtime_without_metal() -> None:
    """Import ``mlxradiomics.numerics.runtime`` bound to a mocked ``mlx.core``.

    The runtime module refuses hosts without Apple Silicon Metal when imported, and
    every Metal-using module imports ``mx`` from it. Patching only that host check
    lets autodoc read signatures and docstrings on any builder; no kernel can run.
    ``array`` and ``Dtype`` become plain classes because annotations such as
    ``mx.array | None`` are evaluated at import and mock instances cannot form unions.
    """
    from unittest import mock

    from sphinx.ext.autodoc.mock import mock as mock_modules

    with mock_modules(["mlx"]):
        import mlx.core

        for name in ("array", "Dtype"):
            setattr(mlx.core, name, type(name, (), {"__module__": "mlx.core"}))

        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("medmlx_core.import_mlx", return_value=mlx.core),
        ):
            import mlxradiomics.numerics.runtime  # noqa: F401


def _feature_catalog() -> str:
    """Render every public feature name with its IBSI mapping as Markdown tables."""
    from mlxradiomics.config.settings import DEFAULT_FEATURES, FAMILIES

    manifest = cast(
        Mapping[str, object],
        json.loads((ROOT / "docs/ibsi-features.json").read_text()),
    )
    ibsi: dict[str, tuple[str, str, str]] = {}
    for entry in cast(Sequence[Mapping[str, object]], manifest["definitions"]):
        for name in cast(Sequence[str], entry.get("public_names", ())):
            ibsi[name] = (str(entry["id"]), str(entry["title"]), str(entry.get("units", "")))
    defaults = set(DEFAULT_FEATURES)
    lines = [f"{manifest['attribution']} ([CC BY 4.0]({manifest['source_license']}))", ""]
    for family, names in FAMILIES.items():
        lines += [f"### `{family}`", "", "| Name | Default | IBSI ID | IBSI definition | Units |",
                  "| --- | :---: | --- | --- | --- |"]
        for name in names:
            key = f"{family}_{name}"
            identifier, title, units = ibsi.get(key, ("", "", ""))
            mark = "yes" if key in defaults else ""
            units = f"`{units}`" if units else ""
            lines.append(f"| `{name}` | {mark} | {identifier} | {title} | {units} |")
        lines.append("")
    return "\n".join(lines)


_load_runtime_without_metal()
generated = ROOT / "docs/_generated"
generated.mkdir(exist_ok=True)
(generated / "feature-catalog.md").write_text(_feature_catalog())
