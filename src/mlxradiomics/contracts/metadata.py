"""Serialization adapters for stable report metadata."""

from dataclasses import asdict
from importlib.metadata import version
from typing import cast

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import ImageSettings
from mlxradiomics.config.transforms import TransformSettings
from mlxradiomics.contracts.json import json_safe
from mlxradiomics.contracts.types import (
    ExecutionMetadata,
    ImageGeometryPayload,
    ImageSettingsPayload,
    RunDetails,
    SettingsPayload,
    TransformPayload,
)
from mlxradiomics.imaging.io import Volume


def file_metadata(
    metadata: ExecutionMetadata,
    reference: Volume,
    *,
    settings: Settings | None,
    source_geometry: ImageGeometryPayload | None,
    image_settings: ImageSettings | None,
) -> ExecutionMetadata:
    """Attach the processed grid and effective file settings to extraction provenance."""
    result = metadata.copy()
    if settings is not None:
        result["settings"] = cast(SettingsPayload, asdict(settings))
    if source_geometry is not None:
        result["source_geometry"] = source_geometry
    result["image_geometry"] = {
        "shape": reference.array.shape,
        "spacing": reference.spacing,
        "origin": reference.origin,
        "direction": reference.direction,
    }
    if image_settings is not None:
        result["image_settings"] = cast(ImageSettingsPayload, asdict(image_settings))
        runtime = result["runtime"].copy()
        runtime["versions"] = {
            **runtime["versions"],
            **{name: version(name) for name in ("SimpleITK", "nibabel")},
        }
        result["runtime"] = runtime
        result["preprocessing"] = "Host SimpleITK I/O/spatial preprocessing and float64 binning"
    return result


def report_details(metadata: ExecutionMetadata, *, count: int) -> RunDetails:
    """Build the shared report details from the completed extraction metadata."""
    safe = json_safe(metadata)
    assert isinstance(safe, dict)
    geometry = metadata.get("image_geometry")
    return {
        "voxel": metadata["voxel"],
        "label": metadata["settings"]["label"],
        "shape": list(geometry["shape"] if geometry else metadata["array_shape"]),
        "spacing": list(geometry["spacing"] if geometry else metadata["spacing"]),
        "feature_count": count,
        "execution": safe,
    }


def transform_payload(options: TransformSettings) -> TransformPayload:
    """Preserve the schema-2 flat transform fields independently of Python settings."""
    return {
        "scratch_mib": options.scratch_mib,
        "image_types": options.image_types,
        "wavelet": options.wavelet.name,
        "wavelet_level": options.wavelet.level,
        "wavelet_start_level": options.wavelet.start_level,
        "sigma": options.log.sigma,
        "log_kernel": options.log.kernel,
        "log_cutoff": options.log.cutoff,
        "log_padding": options.log.padding,
        "log_constant": options.log.constant,
        "log_planar": options.log.planar,
        "lbp2d_radius": options.lbp2d.radius,
        "lbp2d_samples": options.lbp2d.samples,
        "lbp2d_method": options.lbp2d.method,
        "lbp3d_levels": options.lbp3d.levels,
        "lbp3d_radius": options.lbp3d.radius,
        "lbp3d_subdivision": options.lbp3d.subdivision,
        "gradient_use_spacing": options.gradient.use_spacing,
        "mean_support": options.mean.support,
        "mean_padding": options.mean.padding,
        "mean_constant": options.mean.constant,
        "mean_planar": options.mean.planar,
        "laws_kernel": options.laws.kernel,
        "laws_padding": options.laws.padding,
        "laws_constant": options.laws.constant,
        "laws_pooling": options.laws.pooling,
        "laws_energy_distance": options.laws.energy_distance,
        "gabor_sigma": options.gabor.sigma,
        "gabor_wavelength": options.gabor.wavelength,
        "gabor_gamma": options.gabor.gamma,
        "gabor_theta": options.gabor.theta,
        "gabor_response": options.gabor.response,
        "gabor_pooling": options.gabor.pooling,
        "gabor_slice_axis": options.gabor.slice_axis,
        "gabor_padding": options.gabor.padding,
        "gabor_constant": options.gabor.constant,
        "simoncelli_levels": options.simoncelli.levels,
        "simoncelli_padding": options.simoncelli.padding,
        "simoncelli_grid": options.simoncelli.grid,
        "simoncelli_planar": options.simoncelli.planar,
        "riesz_base": options.riesz.base,
        "riesz_orders": options.riesz.orders,
        "riesz_log_sigma": options.riesz.log_sigma,
        "riesz_log_cutoff": options.riesz.log_cutoff,
        "riesz_simoncelli_levels": options.riesz.simoncelli_levels,
        "riesz_padding": options.riesz.padding,
        "riesz_constant": options.riesz.constant,
        "riesz_grid": options.riesz.grid,
        "riesz_steering": options.riesz.steering,
        "riesz_tensor_sigma": options.riesz.tensor_sigma,
        "separable_wavelet": options.separable_wavelet.name,
        "separable_wavelet_bands": options.separable_wavelet.bands,
        "separable_wavelet_level": options.separable_wavelet.level,
        "separable_wavelet_padding": options.separable_wavelet.padding,
        "separable_wavelet_constant": options.separable_wavelet.constant,
        "separable_wavelet_pooling": options.separable_wavelet.pooling,
        "separable_wavelet_planar": options.separable_wavelet.planar,
    }
