"""Parse PyRadiomics-style parameter mappings into extraction, spatial and filter settings.

The registered runner accepts the supported subset of the ``setting``, ``imageType`` and
``featureClass`` sections. Unknown keys, image types, unsupported per-image overrides
and unimplemented spatial controls are rejected before feature extraction.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from mlxradiomics.config.filtering import (
    FilterPadding,
    FourierGrid,
    GaborResponse,
    LBPMethod,
    LogKernel,
    OrientationPooling,
    RieszBase,
)
from mlxradiomics.config.settings import (
    CONVENTIONS,
    DirectionalAggregation,
    MatrixAggregation,
    Settings,
)
from mlxradiomics.config.spatial import ImageSettings, Interpolator
from mlxradiomics.config.transforms import (
    ImageType,
    TransformSettings,
    WaveletSettings,
    validate_filter_shape,
)
from mlxradiomics.config.validation import (
    require_bool,
    require_float,
    require_float_tuple,
    require_int,
    require_int_tuple,
    require_mapping,
)


@dataclass(frozen=True)
class ExtractionParameters:
    """Effective extraction, spatial preprocessing and derived-image settings."""

    settings: Settings
    image_settings: ImageSettings
    transforms: TransformSettings


_IMAGE_TYPE_NAMES: dict[str, ImageType] = {
    "Original": ImageType.ORIGINAL,
    "Square": ImageType.SQUARE,
    "SquareRoot": ImageType.SQUARE_ROOT,
    "Logarithm": ImageType.LOGARITHM,
    "Exponential": ImageType.EXPONENTIAL,
    "Gradient": ImageType.GRADIENT,
    "Wavelet": ImageType.WAVELET,
    "LoG": ImageType.LOG,
    "LBP2D": ImageType.LBP2D,
    "LBP3D": ImageType.LBP3D,
    "Mean": ImageType.MEAN,
    "Laws": ImageType.LAWS,
    "Gabor": ImageType.GABOR,
    "Simoncelli": ImageType.SIMONCELLI,
    "NonseparableWavelet": ImageType.SIMONCELLI,
    "Riesz": ImageType.RIESZ,
    "SeparableWavelet": ImageType.SEPARABLE_WAVELET,
}

# Parameter-file name of each image type, used in error messages.
_IMAGE_TYPE_KEYS: dict[ImageType, str] = {
    kind: name for name, kind in reversed(_IMAGE_TYPE_NAMES.items())
}

_INTERPOLATORS: dict[str, Interpolator] = {
    "sitkNearestNeighbor": Interpolator.NEAREST,
    "sitkLinear": Interpolator.LINEAR,
    "sitkBSpline": Interpolator.BSPLINE,
    "nearest": Interpolator.NEAREST,
    "linear": Interpolator.LINEAR,
    "bspline": Interpolator.BSPLINE,
}

_LBP_METHODS: dict[str, LBPMethod] = {method.value: method for method in LBPMethod}

_ADVANCED_SETTING_KEYS = frozenset(
    (
        "resampledPixelSpacing", "interpolator", "padDistance", "correctMask",
        "lbp2DRadius", "lbp2DSamples", "lbp2DMethod",
        "lbp3DLevels", "lbp3DIcosphereRadius", "lbp3DIcosphereSubdivision",
        "gradientUseSpacing",
    )
)


def _as_orders(value: object, key: str) -> tuple[tuple[int, ...], ...]:
    """Parse a nonempty list of integer lists, such as Riesz orders."""
    if isinstance(value, bool) or not isinstance(value, list | tuple):
        raise ValueError(f"setting '{key}' must be a list of integer lists")
    items = tuple(cast(list[object] | tuple[object, ...], value))
    if not items:
        raise ValueError(f"setting '{key}' must be a list of integer lists")
    return tuple(require_int_tuple(item, key) for item in items)


def _image_settings(setting: Mapping[str, object], ndim: int) -> ImageSettings:
    """Build ``ImageSettings`` from the ``setting`` section.

    ``resampledPixelSpacing`` is given in ITK x-first order and reversed into
    NumPy array-axis order. Unknown interpolator names are rejected.
    """
    spacing_value = setting.get("resampledPixelSpacing")
    resampled: tuple[float, ...] | None = None
    if spacing_value is not None:
        values = require_float_tuple(spacing_value, "resampledPixelSpacing")
        if len(values) != ndim:
            raise ValueError("resampledPixelSpacing must contain one value per image axis")
        # Parameter files use ITK's x-first order; ImageSettings follows array axes.
        resampled = tuple(reversed(values))
    interpolator = Interpolator.BSPLINE
    if "interpolator" in setting:
        name = setting["interpolator"]
        if not isinstance(name, str) or name not in _INTERPOLATORS:
            raise ValueError(
                "setting 'interpolator' must be sitkNearestNeighbor, sitkLinear or sitkBSpline"
            )
        interpolator = _INTERPOLATORS[name]
    pad_distance = (
        require_int(setting["padDistance"], "padDistance") if "padDistance" in setting else 5
    )
    correct_mask = (
        require_bool(setting["correctMask"], "correctMask") if "correctMask" in setting else False
    )
    return ImageSettings(
        resampled_spacing=resampled, interpolator=interpolator,
        correct_mask=correct_mask, pad_distance=pad_distance,
    )


def _filter_overrides(
    options: TransformSettings, kind: ImageType, custom: Mapping[str, object],
) -> TransformSettings:
    """Parse explicit IBSI filter controls at the parameter-file boundary."""
    allowed = {
        ImageType.MEAN: {"support", "padding", "constant", "planar"},
        ImageType.LAWS: {"kernel", "padding", "constant", "pooling", "energy_distance"},
        ImageType.GABOR: {"sigma", "wavelength", "gamma", "theta", "response", "pooling",
                          "slice_axis", "padding", "constant"},
        ImageType.SIMONCELLI: {"levels", "padding", "grid", "planar"},
        ImageType.RIESZ: {"base", "orders", "log_sigma", "log_cutoff",
                          "simoncelli_levels", "padding", "constant", "grid",
                          "steering", "tensor_sigma"},
        ImageType.SEPARABLE_WAVELET: {"wavelet", "bands", "level", "padding", "constant",
                                      "pooling", "planar"},
    }[kind]
    name = _IMAGE_TYPE_KEYS[kind]
    group = "separable_wavelet" if kind is ImageType.SEPARABLE_WAVELET else kind.value
    unexpected = set(custom) - allowed
    if unexpected:
        raise ValueError(f"unsupported {name} overrides: {sorted(unexpected)}")
    changes: dict[str, object] = {}
    for key, raw in custom.items():
        context = f"imageType.{name}.{key}"
        if key in {"padding", "pooling", "response", "kernel", "base", "grid"}:
            if not isinstance(raw, str):
                raise ValueError(f"{context} must be a string")
            if key == "padding":
                value: object = FilterPadding(raw)
            elif key == "pooling":
                value = OrientationPooling(raw)
            elif key == "response":
                value = GaborResponse(raw)
            elif key == "base":
                value = RieszBase(raw)
            elif key == "grid":
                value = FourierGrid(raw)
            else:
                value = raw
            changes[key] = value
        elif key in {"support", "energy_distance", "slice_axis"}:
            changes[key] = (
                None if raw is None and key != "support" else require_int(raw, context)
            )
        elif key in {"levels", "simoncelli_levels"}:
            changes[key] = require_int_tuple(raw, context)
        elif key == "wavelet":
            if not isinstance(raw, str):
                raise ValueError(f"{context} must be a string")
            changes["name"] = raw
        elif key == "bands":
            if not isinstance(raw, list) or not all(
                isinstance(band, str) for band in cast(list[object], raw)
            ):
                raise ValueError(f"{context} must be a list of band strings")
            changes[key] = tuple(cast(list[str], raw))
        elif key == "level":
            changes[key] = require_int(raw, context)
        elif key == "orders":
            changes[key] = _as_orders(raw, context)
        elif key in {"steering", "planar"}:
            changes[key] = require_bool(raw, context)
        elif key in {"sigma", "wavelength", "theta", "log_sigma"}:
            changes[key] = require_float_tuple(raw, context)
        else:
            changes[key] = require_float(raw, context)
    groups = {
        ImageType.MEAN: options.mean, ImageType.LAWS: options.laws,
        ImageType.GABOR: options.gabor, ImageType.SIMONCELLI: options.simoncelli,
        ImageType.RIESZ: options.riesz, ImageType.SEPARABLE_WAVELET: options.separable_wavelet,
    }
    return dataclasses.replace(options, **{group: dataclasses.replace(groups[kind], **changes)})


def _transforms(params: Mapping[str, object]) -> TransformSettings:
    """Build ``TransformSettings`` from the ``imageType`` section.

    A missing section selects ``Original`` only. Per-type override mappings are
    applied on top of the defaults; unknown types and overrides are rejected.
    """
    image_type = params.get("imageType")
    if image_type is None:
        return TransformSettings()
    entries = require_mapping(image_type, "imageType")
    kinds: list[ImageType] = []
    overrides: dict[ImageType, Mapping[str, object]] = {}
    for name, raw in entries.items():
        kind = _IMAGE_TYPE_NAMES.get(name)
        if kind is None:
            raise ValueError(f"unsupported image type: {name}")
        kinds.append(kind)
        overrides[kind] = {} if raw is None else require_mapping(raw, f"imageType.{name}")
    if not kinds:
        raise ValueError("imageType must enable at least one image type")

    options = TransformSettings()
    for kind in kinds:
        custom = overrides[kind]
        if kind == ImageType.WAVELET:
            allowed = {"wavelet", "level", "start_level"}
            unexpected = set(custom) - allowed
            if unexpected:
                raise ValueError(f"unsupported Wavelet overrides: {sorted(unexpected)}")
            selected = custom.get("wavelet", options.wavelet.name)
            if not isinstance(selected, str):
                raise ValueError("Wavelet 'wavelet' must be a string")
            options = dataclasses.replace(
                options,
                wavelet=WaveletSettings(
                    name=selected,
                    level=require_int(custom.get("level", options.wavelet.level), "level"),
                    start_level=require_int(
                        custom.get("start_level", options.wavelet.start_level), "start_level"
                    ),
                ),
            )
        elif kind == ImageType.LOG:
            unexpected = set(custom) - {"sigma", "kernel", "cutoff", "padding", "constant",
                                        "planar"}
            if unexpected:
                raise ValueError(f"unsupported LoG overrides: {sorted(unexpected)}")
            if "sigma" not in custom:
                raise ValueError("LoG requires explicit sigma values")
            log_changes: dict[str, object] = {
                "sigma": require_float_tuple(custom["sigma"], "sigma"),
            }
            for key in ("kernel", "padding"):
                if key in custom:
                    raw = custom[key]
                    if not isinstance(raw, str):
                        raise ValueError(f"imageType.LoG.{key} must be a string")
                    log_changes[key] = (
                        LogKernel(raw) if key == "kernel" else FilterPadding(raw)
                    )
            for key in ("cutoff", "constant"):
                if key in custom:
                    log_changes[key] = require_float(custom[key], f"imageType.LoG.{key}")
            if "planar" in custom:
                log_changes["planar"] = require_bool(custom["planar"], "imageType.LoG.planar")
            options = dataclasses.replace(
                options,
                log=dataclasses.replace(options.log, **log_changes),
            )
        elif kind in {
            ImageType.MEAN, ImageType.LAWS, ImageType.GABOR, ImageType.SIMONCELLI,
            ImageType.RIESZ, ImageType.SEPARABLE_WAVELET,
        }:
            options = _filter_overrides(options, kind, custom)
        elif custom:
            raise ValueError(f"imageType.{_IMAGE_TYPE_KEYS[kind]} overrides are unsupported")
    return dataclasses.replace(options, image_types=tuple(kinds))


def _with_gradient_and_lbp(
    options: TransformSettings, setting: Mapping[str, object],
) -> TransformSettings:
    """Apply gradient-spacing and LBP2D/LBP3D keys from ``setting``."""
    gradient = options.gradient.use_spacing
    radius2d = options.lbp2d.radius
    samples2d = options.lbp2d.samples
    method2d = options.lbp2d.method
    levels3d = options.lbp3d.levels
    radius3d = options.lbp3d.radius
    subdivision3d = options.lbp3d.subdivision
    if "gradientUseSpacing" in setting:
        gradient = require_bool(setting["gradientUseSpacing"], "gradientUseSpacing")
    if "lbp2DRadius" in setting:
        radius2d = require_float(setting["lbp2DRadius"], "lbp2DRadius")
    if "lbp2DSamples" in setting:
        samples2d = require_int(setting["lbp2DSamples"], "lbp2DSamples")
    if "lbp2DMethod" in setting:
        method = setting["lbp2DMethod"]
        if not isinstance(method, str) or method not in _LBP_METHODS:
            values = ", ".join(sorted(_LBP_METHODS))
            raise ValueError(f"setting 'lbp2DMethod' must be one of {values}")
        method2d = _LBP_METHODS[method]
    if "lbp3DLevels" in setting:
        levels3d = require_int(setting["lbp3DLevels"], "lbp3DLevels")
    if "lbp3DIcosphereRadius" in setting:
        radius3d = require_float(setting["lbp3DIcosphereRadius"], "lbp3DIcosphereRadius")
    if "lbp3DIcosphereSubdivision" in setting:
        subdivision3d = require_int(
            setting["lbp3DIcosphereSubdivision"], "lbp3DIcosphereSubdivision",
        )
    return dataclasses.replace(
        options,
        gradient=dataclasses.replace(options.gradient, use_spacing=gradient),
        lbp2d=dataclasses.replace(
            options.lbp2d, radius=radius2d, samples=samples2d, method=method2d
        ),
        lbp3d=dataclasses.replace(
            options.lbp3d, levels=levels3d, radius=radius3d, subdivision=subdivision3d
        ),
    )


def parse_params(params: Mapping[str, object], *, ndim: int) -> ExtractionParameters:
    """Translate a supported PyRadiomics-style parameter mapping.

    Reads the ``setting``, ``imageType`` and ``featureClass`` sections; the accepted
    keys and per-image-type overrides are listed in ``docs/guide/parameters.md``.
    Spatial controls use PyRadiomics names, and ``resampledPixelSpacing`` is given in
    ITK x-first order.

    Raises:
        ValueError: On unknown sections, keys or image types, unsupported per-image
            overrides, wrong value types, or settings invalid for ``ndim``.
    """
    if ndim not in (2, 3):
        raise ValueError("parameter files require a 2D or 3D image")
    params = require_mapping(params, "params")
    unexpected = set(params) - {"setting", "featureClass", "imageType"}
    if unexpected:
        raise ValueError(f"unsupported params sections: {sorted(unexpected)}")
    setting = require_mapping(params.get("setting", {}), "setting")
    transforms = _with_gradient_and_lbp(_transforms(params), setting)
    if ndim != 3 and ImageType.LBP3D in transforms.image_types:
        raise ValueError("LBP3D requires a 3D image")
    for kind in transforms.image_types:
        validate_filter_shape(ndim, transforms, kind)
    stage = dict(params)
    stage.pop("imageType", None)
    stage["setting"] = {
        key: value for key, value in setting.items() if key not in _ADVANCED_SETTING_KEYS
    }
    settings = settings_from_params(stage)
    settings.validate(ndim)
    return ExtractionParameters(
        settings=settings,
        image_settings=_image_settings(setting, ndim),
        transforms=transforms,
    )


def _as_resegment_mode(value: object, key: str) -> Literal["absolute", "relative", "sigma"]:
    """Validate a ``resegmentMode`` value as absolute, relative or sigma."""
    if value == "absolute":
        return "absolute"
    if value == "relative":
        return "relative"
    if value == "sigma":
        return "sigma"
    raise ValueError(f"setting '{key}' must be one of absolute, relative, sigma")


# PyRadiomics ``setting`` keys mapped onto feature Settings. The full parser handles
# spatial and image-type controls separately, before calling settings_from_params.
_KNOWN_SETTING_KEYS = frozenset(
    (
        "binWidth", "binCount", "label", "distances", "symmetricalGLCM",
        "force2D", "force2Ddimension", "maskedKernel", "kernelRadius", "voxelArrayShift",
        "gldm_a", "normalize", "normalizeScale", "removeOutliers",
        "resegmentRange", "resegmentMode", "resegmentShape",
        "binAnchor", "resegmentOutlierSigma", "sliceAxis",
        "neighborhoodRadius", "glcmAggregation", "glrlmAggregation", "glszmAggregation",
        "gldzmAggregation", "ngtdmAggregation", "gldmAggregation",
        "ivhBinWidth", "ivhBinCount", "ivhIntensityRange",
        "covarianceDivisor", "ellipsoidArea", "kurtosis", "zeroVarianceMoments",
        "ngtdmSingularCases", "neighborhoodSupport", "binOrigin", "constantRoiBin",
        "maskExtrapolation", "splineBoundary",
    )
)
# Parameter-file key of each convention field of ``Settings``.
_CONVENTION_KEYS = {
    "covariance_divisor": "covarianceDivisor", "ellipsoid_area": "ellipsoidArea",
    "kurtosis": "kurtosis", "zero_variance_moments": "zeroVarianceMoments",
    "ngtdm_singular_cases": "ngtdmSingularCases", "neighborhood_support": "neighborhoodSupport",
    "bin_origin": "binOrigin", "constant_roi_bin": "constantRoiBin",
    "mask_extrapolation": "maskExtrapolation", "spline_boundary": "splineBoundary",
}
assert _CONVENTION_KEYS.keys() == CONVENTIONS.keys()


def settings_from_params(params: Mapping[str, object], *, base: Settings | None = None) -> Settings:
    """Translate a PyRadiomics-style params mapping onto :class:`Settings`.

    Recognized ``setting`` keys are ``binWidth``, ``binCount``, ``label``, ``distances``,
    ``symmetricalGLCM``, ``force2D`` with ``force2Ddimension`` (which sets
    ``force2d_axis``, default axis 0), ``maskedKernel``, ``kernelRadius``,
    ``voxelArrayShift``, ``gldm_a``, ``normalize``, ``normalizeScale``, ``removeOutliers``,
    ``resegmentRange``, ``resegmentMode`` and ``resegmentShape``. Extension keys name
    each convention field of :class:`Settings` (``covarianceDivisor``, ``ellipsoidArea``,
    ``kurtosis``, ``zeroVarianceMoments``, ``ngtdmSingularCases``, ``neighborhoodSupport``,
    ``binOrigin``, ``constantRoiBin``, ``maskExtrapolation`` and ``splineBoundary``); an
    absent key keeps the value of ``base``. Set alternative conventions explicitly on
    ``base`` or in the parameter file. An absent ``force2D``
    retains ``base.force2d_axis``; an explicit ``false`` clears it to ``None``; ``true``
    sets the integer axis from ``force2Ddimension`` (default 0). ``featureClass`` must map
    each family to an explicit list of feature names, with at least one name overall.
    Image types are handled by :func:`parse_params`.

    Args:
        params: The params mapping (``setting``, ``featureClass``, ``imageType`` sections).
        base: Settings supplying every field the params do not set; defaults to
            ``Settings()``.

    Raises:
        ValueError: On unknown setting keys, values of the wrong type, or an empty or
            non-list feature selection. Field-level validation of the result happens later, in
            ``Settings.validate``.
    """

    params = require_mapping(params, "params")
    source = base or Settings()
    setting = require_mapping(params.get("setting", {}), "setting")
    unsupported_keys = set(setting) - _KNOWN_SETTING_KEYS
    if unsupported_keys:
        raise ValueError(f"unsupported setting keys for this runner: {sorted(unsupported_keys)}")

    bin_width = (
        require_float(setting["binWidth"], "binWidth")
        if "binWidth" in setting else source.bin_width
    )
    bin_count = (
        require_int(setting["binCount"], "binCount") if "binCount" in setting else source.bin_count
    )
    gldm_alpha = (
        require_int(setting["gldm_a"], "gldm_a") if "gldm_a" in setting else source.gldm_alpha
    )
    normalize = (
        require_bool(setting["normalize"], "normalize")
        if "normalize" in setting else source.normalize
    )
    normalize_scale = (
        require_float(setting["normalizeScale"], "normalizeScale")
        if "normalizeScale" in setting else source.normalize_scale
    )
    remove_outliers = (
        require_float(setting["removeOutliers"], "removeOutliers")
        if "removeOutliers" in setting else source.remove_outliers
    )
    resegment_range = (
        require_float_tuple(setting["resegmentRange"], "resegmentRange")
        if "resegmentRange" in setting else source.resegment_range
    )
    resegment_mode = (
        _as_resegment_mode(setting["resegmentMode"], "resegmentMode")
        if "resegmentMode" in setting else source.resegment_mode
    )
    resegment_shape = (
        require_bool(setting["resegmentShape"], "resegmentShape")
        if "resegmentShape" in setting else source.resegment_shape
    )
    label = require_int(setting["label"], "label") if "label" in setting else source.label
    distances = (
        require_int_tuple(setting["distances"], "distances")
        if "distances" in setting else source.distances
    )
    symmetrical_glcm = (
        require_bool(setting["symmetricalGLCM"], "symmetricalGLCM")
        if "symmetricalGLCM" in setting else source.symmetrical_glcm
    )
    masked_kernel = (
        require_bool(setting["maskedKernel"], "maskedKernel")
        if "maskedKernel" in setting else source.masked_kernel
    )
    kernel_radius = (
        require_int(setting["kernelRadius"], "kernelRadius")
        if "kernelRadius" in setting else source.kernel_radius
    )
    voxel_array_shift = (
        require_float(setting["voxelArrayShift"], "voxelArrayShift")
        if "voxelArrayShift" in setting else source.voxel_array_shift
    )
    force2d_axis = source.force2d_axis
    if "force2D" in setting:
        force_2d = require_bool(setting["force2D"], "force2D")
        # An explicit false removes a base restriction; only a true value reads the axis.
        force2d_axis = (
            require_int(setting.get("force2Ddimension", 0), "force2Ddimension")
            if force_2d else None
        )

    features = source.features
    feature_class = params.get("featureClass")
    if feature_class is not None:
        names: list[str] = []
        for family, selection in require_mapping(feature_class, "featureClass").items():
            if not isinstance(selection, list):
                raise ValueError(
                    f"featureClass '{family}' must list feature names explicitly for this runner"
                )
            for name in cast(list[object], selection):
                if not isinstance(name, str):
                    raise ValueError(f"featureClass '{family}' entries must be strings")
                names.append(f"{family}_{name}")
        if not names:
            raise ValueError("featureClass must select at least one feature")
        features = tuple(names)

    def text_option(key: str, default: str) -> str:
        value = setting.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"setting '{key}' must be a string")
        return value

    ivh_range = source.ivh_intensity_range
    if "ivhIntensityRange" in setting:
        bounds = require_float_tuple(setting["ivhIntensityRange"], "ivhIntensityRange")
        if len(bounds) != 2:
            raise ValueError("ivhIntensityRange must contain two bounds")
        ivh_range = (bounds[0], bounds[1])
    return dataclasses.replace(
        source,
        bin_width=bin_width,
        bin_count=bin_count,
        gldm_alpha=gldm_alpha,
        normalize=normalize,
        normalize_scale=normalize_scale,
        remove_outliers=remove_outliers,
        resegment_range=resegment_range,
        resegment_mode=resegment_mode,
        resegment_shape=resegment_shape,
        label=label,
        distances=distances,
        symmetrical_glcm=symmetrical_glcm,
        masked_kernel=masked_kernel,
        kernel_radius=kernel_radius,
        voxel_array_shift=voxel_array_shift,
        force2d_axis=force2d_axis,
        features=features,
        bin_anchor=(require_float(setting["binAnchor"], "binAnchor")
                    if "binAnchor" in setting else source.bin_anchor),
        resegment_outlier_sigma=(
            require_float(setting["resegmentOutlierSigma"], "resegmentOutlierSigma")
            if "resegmentOutlierSigma" in setting else source.resegment_outlier_sigma
        ),
        **{
            field: text_option(key, getattr(source, field))
            for field, key in _CONVENTION_KEYS.items()
        },
        slice_axis=(
            require_int(setting["sliceAxis"], "sliceAxis")
            if "sliceAxis" in setting else source.slice_axis
        ),
        neighborhood_radius=(require_int(setting["neighborhoodRadius"], "neighborhoodRadius")
                             if "neighborhoodRadius" in setting else source.neighborhood_radius),
        glcm_aggregation=cast(
            DirectionalAggregation, text_option("glcmAggregation", source.glcm_aggregation),
        ),
        glrlm_aggregation=cast(
            DirectionalAggregation, text_option("glrlmAggregation", source.glrlm_aggregation),
        ),
        glszm_aggregation=cast(
            MatrixAggregation, text_option("glszmAggregation", source.glszm_aggregation),
        ),
        gldzm_aggregation=cast(
            MatrixAggregation, text_option("gldzmAggregation", source.gldzm_aggregation),
        ),
        ngtdm_aggregation=cast(
            MatrixAggregation, text_option("ngtdmAggregation", source.ngtdm_aggregation),
        ),
        gldm_aggregation=cast(
            MatrixAggregation, text_option("gldmAggregation", source.gldm_aggregation),
        ),
        ivh_bin_width=(require_float(setting["ivhBinWidth"], "ivhBinWidth")
                       if "ivhBinWidth" in setting else source.ivh_bin_width),
        ivh_bin_count=(require_int(setting["ivhBinCount"], "ivhBinCount")
                       if "ivhBinCount" in setting else source.ivh_bin_count),
        ivh_intensity_range=ivh_range,
    )


def load_params(path: str | Path) -> Mapping[str, object]:
    """Load a PyRadiomics-style params YAML file as a plain mapping.

    Malformed YAML is translated to ``ValueError`` with the parser's location and
    description, so runner and study callers classify it as invalid input rather than an
    unexpected execution failure.

    Raises:
        ValueError: If the file is missing, is not valid YAML or UTF-8, or does not contain
            a mapping.
    """
    import yaml

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"missing params file: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid params YAML: {source}: {error}") from error
    return require_mapping(loaded, "params")
