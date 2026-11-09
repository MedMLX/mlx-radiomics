"""Typed transform configuration and validation without a Metal dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from mlxradiomics.config.filtering import (
    MAX_STEERING_ORDER,
    FilterPadding,
    FourierGrid,
    GaborResponse,
    LBPMethod,
    LogKernel,
    OrientationPooling,
    RieszBase,
    validate_filter_constant,
    validate_laws_kernel,
)
from mlxradiomics.config.wavelets import wavelet_filters


class ImageType(StrEnum):
    """Image types accepted in configuration and extraction requests.

    ``SIMONCELLI`` and ``RIESZ`` are FFT-based and need MLX 0.32.5 or later;
    every extraction or planning request for them is refused on an older MLX
    release; see
    :data:`mlxradiomics.filters.FFT_DISABLED_REASON`.

    ``WAVELET`` yields several images per level, ``LOG`` one per sigma, ``GABOR``
    one or more per scale/orientation, ``SIMONCELLI`` one per level and ``RIESZ``
    one per base-filter scale/level and order. ``MEAN`` and ``LAWS`` follow the
    IBSI 2 definitions. The filter values are chosen so the existing
    PyRadiomics-compatible names stay unchanged.
    """

    ORIGINAL = "original"
    SQUARE = "square"
    SQUARE_ROOT = "squareroot"
    LOGARITHM = "logarithm"
    EXPONENTIAL = "exponential"
    GRADIENT = "gradient"
    WAVELET = "wavelet"
    LOG = "log"
    LBP2D = "lbp-2D"
    LBP3D = "lbp-3D"
    MEAN = "mean"
    LAWS = "laws"
    GABOR = "gabor"
    SIMONCELLI = "simoncelli"
    RIESZ = "riesz"
    SEPARABLE_WAVELET = "separablewavelet"


@dataclass(frozen=True)
class WaveletSettings:
    """Stationary wavelet decomposition and output levels.

    Attributes:
        name: Name of a discrete wavelet in ``wavelet_tables.json`` (PyWavelets names such
            as ``"coif1"``, ``"haar"`` or ``"db3"``).
        level: Number of decomposition levels to return.
        start_level: Number of levels decomposed first and not returned; returned levels
            are numbered from 1 regardless.
    """

    name: str = "coif1"
    level: int = 1
    start_level: int = 0

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.level) is not int or self.level < 1:
            raise ValueError("wavelet_level must be a positive integer")
        if type(self.start_level) is not int or self.start_level < 0:
            raise ValueError("wavelet_start_level must be a nonnegative integer")


@dataclass(frozen=True)
class LoGSettings:
    """Laplacian-of-Gaussian scales and boundary conditions.

    Attributes:
        sigma: LoG Gaussian sigmas in mm (same unit as the spacing); required for
            ``LOG``.
        kernel: ``RECURSIVE`` keeps the PyRadiomics/ITK recursive LoG. ``SAMPLED`` uses
            the IBSI 2 spatial kernel with the four options below, which the recursive
            kernel rejects when changed from their defaults.
        cutoff: Sampled-kernel size cutoff as a multiple of sigma.
        padding: Boundary-extension method for the sampled kernel.
        constant: Extension value for constant sampled-LoG padding.
        planar: Filter each array-axis-0 plane of a 3-D image with a 2-D sampled kernel
            instead of one 3-D kernel.
    """

    sigma: tuple[float, ...] = ()
    kernel: LogKernel = LogKernel.RECURSIVE
    cutoff: float = 4.0
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0
    planar: bool = False

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.sigma) is not tuple or any(
            type(s) not in (int, float) or not isfinite(s) or s <= 0 for s in self.sigma
        ):
            raise ValueError("sigma values must be finite and positive")
        if len(set(self.sigma)) != len(self.sigma):
            raise ValueError("sigma values must be unique")
        if type(self.kernel) is not LogKernel:
            raise ValueError("log_kernel must be a typed LogKernel")
        if type(self.padding) is not FilterPadding:
            raise ValueError("log_padding must be a typed FilterPadding")
        if type(self.planar) is not bool:
            raise ValueError("log_planar must be a boolean")
        if type(self.cutoff) not in (int, float) or not isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("log_cutoff must be finite and positive")
        validate_filter_constant(self.constant, "log_constant")
        if self.kernel is LogKernel.RECURSIVE and (
            self.planar
            or self.cutoff != 4.0
            or self.padding is not FilterPadding.MIRROR
            or (self.constant != 0.0)
        ):
            raise ValueError(
                "log_cutoff, log_padding, log_constant and log_planar require "
                "log_kernel=LogKernel.SAMPLED"
            )


@dataclass(frozen=True)
class LBP2DSettings:
    """Planar local binary patterns and their coding method.

    Attributes:
        radius: LBP2D sampling-circle radius in voxels.
        samples: Number of LBP2D sample points, 1 to 24.
        method: LBP2D coding method.
    """

    radius: float = 1.0
    samples: int = 8
    method: LBPMethod = LBPMethod.UNIFORM

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.method) is not LBPMethod:
            raise ValueError("lbp2d_method must be a typed LBPMethod")
        if type(self.samples) is not int or not 1 <= self.samples <= 24:
            raise ValueError("lbp2d_samples must be an integer from 1 to 24")
        if type(self.radius) not in (int, float) or not isfinite(self.radius) or self.radius <= 0:
            raise ValueError("LBP radii must be finite and positive")


@dataclass(frozen=True)
class LBP3DSettings:
    """Spherical local binary patterns and harmonic sampling.

    Attributes:
        levels: Number of spherical-harmonic levels for LBP3D.
        radius: LBP3D sampling-sphere radius in voxels.
        subdivision: Icosphere subdivisions for LBP3D, 0 to 4.
    """

    levels: int = 2
    radius: float = 1.0
    subdivision: int = 1

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.radius) not in (int, float) or not isfinite(self.radius) or self.radius <= 0:
            raise ValueError("LBP radii must be finite and positive")
        if type(self.levels) is not int or self.levels < 1:
            raise ValueError("lbp3d_levels must be a positive integer")
        if type(self.subdivision) is not int or not 0 <= self.subdivision <= 4:
            raise ValueError("lbp3d_subdivision must be an integer from zero to four")


@dataclass(frozen=True)
class GradientSettings:
    """Physical-unit scaling for image gradients.

    Attributes:
        use_spacing: If true, gradient derivatives are taken per mm rather than per
            voxel.
    """

    use_spacing: bool = True

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.use_spacing) is not bool:
            raise ValueError("gradient_use_spacing must be boolean")


@dataclass(frozen=True)
class MeanSettings:
    """Mean-filter support and boundary conditions.

    Attributes:
        support: Odd isotropic mean-filter support in voxels.
        padding: IBSI boundary condition for the mean filter.
        constant: Extension value for constant mean-filter padding.
        planar: Filter each array-axis-0 plane of a 3-D image with the 2-D ``M x M``
            mean kernel.
    """

    support: int = 3
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0
    planar: bool = False

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.planar) is not bool:
            raise ValueError("mean_planar must be a boolean")
        if type(self.support) is not int or self.support < 1 or self.support % 2 == 0:
            raise ValueError("mean_support must be a positive odd integer")
        if type(self.padding) is not FilterPadding:
            raise ValueError("mean_padding must be a typed FilterPadding")
        validate_filter_constant(self.constant, "mean_constant")


@dataclass(frozen=True)
class LawsSettings:
    """Laws texture kernels, orientation pooling and local energy.

    Attributes:
        kernel: Concatenated Laws kernel names in IBSI ``(k1, k2, k3)`` order. Two names
            apply a 2-D kernel to each ``(k2, k1)`` plane of a 3-D image; three names
            apply a 3-D kernel.
        padding: IBSI boundary condition for the Laws response.
        constant: Extension value for constant Laws padding.
        pooling: Pooling over right-angle rotations of the Laws kernel.
        energy_distance: Chebyshev distance in voxels for the Laws texture energy image;
            ``None`` returns the response map instead.
    """

    kernel: str = "L5S5"
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0
    pooling: OrientationPooling = OrientationPooling.NONE
    energy_distance: int | None = None

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.padding) is not FilterPadding:
            raise ValueError("laws_padding must be a typed FilterPadding")
        if type(self.kernel) is not str:
            raise ValueError("laws_kernel must be a string")
        validate_laws_kernel(self.kernel)
        if type(self.pooling) is not OrientationPooling:
            raise ValueError("laws_pooling must be a typed OrientationPooling")
        if self.energy_distance is not None and (
            type(self.energy_distance) is not int or self.energy_distance < 0
        ):
            raise ValueError("laws_energy_distance must be a nonnegative integer or None")
        validate_filter_constant(self.constant, "laws_constant")


@dataclass(frozen=True)
class GaborSettings:
    """Gabor scales, orientations and complex-response selection.

    Attributes:
        sigma: Gabor envelope standard deviations in mm.
        wavelength: Gabor oscillation wavelengths in mm. Lengths of ``sigma`` and
            ``wavelength`` must match, or one may have length one and broadcast over the
            other.
        gamma: Gabor spatial aspect ratio ``a1 / a2``.
        theta: In-plane orientations in radians, clockwise from ``k1``.
        response: Modulus, real or imaginary complex response component.
        pooling: Pooling over Gabor orientations.
        slice_axis: Array-axis normal for 2-D Gabor filtering of a 3-D image. ``None``
            filters and averages all three orthogonal plane families.
        padding: IBSI boundary condition for Gabor plane convolution.
        constant: Extension value for constant Gabor padding.
    """

    sigma: tuple[float, ...] = ()
    wavelength: tuple[float, ...] = ()
    gamma: float = 1.0
    theta: tuple[float, ...] = ()
    response: GaborResponse = GaborResponse.MODULUS
    pooling: OrientationPooling = OrientationPooling.NONE
    slice_axis: int | None = None
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.padding) is not FilterPadding:
            raise ValueError("gabor_padding must be a typed FilterPadding")
        if type(self.response) is not GaborResponse:
            raise ValueError("gabor_response must be a typed GaborResponse")
        if type(self.pooling) is not OrientationPooling:
            raise ValueError("gabor_pooling must be a typed OrientationPooling")
        for label, values in (("gabor_sigma", self.sigma), ("gabor_wavelength", self.wavelength)):
            if type(values) is not tuple or any(
                type(value) not in (int, float) or not isfinite(value) or value <= 0
                for value in values
            ):
                raise ValueError(f"{label} must contain finite positive values")
        if type(self.gamma) not in (int, float) or not isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gabor_gamma must be finite and positive")
        if type(self.theta) is not tuple or any(
            type(value) not in (int, float) or not isfinite(value) for value in self.theta
        ):
            raise ValueError("gabor_theta must contain finite values")
        if self.slice_axis is not None and (
            type(self.slice_axis) is not int or self.slice_axis not in range(3)
        ):
            raise ValueError("gabor_slice_axis must be None or an array axis from zero to two")
        validate_filter_constant(self.constant, "gabor_constant")


@dataclass(frozen=True)
class SimoncelliSettings:
    """Simoncelli levels and Fourier sampling (needs MLX 0.32.5 or later).

    Attributes:
        levels: Positive one-based Simoncelli band levels.
        padding: Must be periodic; the full-image Fourier Hadamard product has no
            spatial padding option.
        grid: Fourier sampling convention for the Simoncelli band.
        planar: Filter each array-axis-0 plane of a 3-D image with the 2-D band and a
            per-plane 2-D FFT.
    """

    levels: tuple[int, ...] = (1,)
    padding: FilterPadding = FilterPadding.PERIODIC
    grid: FourierGrid = FourierGrid.DFT
    planar: bool = False

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.planar) is not bool:
            raise ValueError("simoncelli_planar must be a boolean")
        if type(self.padding) is not FilterPadding:
            raise ValueError("simoncelli_padding must be a typed FilterPadding")
        if self.padding is not FilterPadding.PERIODIC:
            raise ValueError("Simoncelli filtering is defined with periodic boundaries only")
        if (
            type(self.levels) is not tuple
            or not self.levels
            or any(type(level) is not int or level < 1 for level in self.levels)
        ):
            raise ValueError("simoncelli_levels must contain positive integers")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("simoncelli_levels must be unique")
        if type(self.grid) is not FourierGrid:
            raise ValueError("simoncelli_grid must be a typed FourierGrid")


@dataclass(frozen=True)
class RieszSettings:
    """Riesz components, base filters and optional tensor steering (needs MLX 0.32.5 or later).

    Attributes:
        base: Base filter for ``RIESZ``: the IBSI 2 LoG kernel or a Simoncelli B-map.
        orders: Riesz order tuples in the manual's IBSI ``(k1, k2, k3)`` order, so
            ``order[i]`` applies to native array axis ``ndim - 1 - i``. Each tuple has
            one nonnegative integer per image axis and a positive sum.
        log_sigma: LoG scales in mm for a ``LOG`` Riesz base.
        log_cutoff: LoG kernel-size cutoff as a multiple of sigma.
        simoncelli_levels: Positive one-based Simoncelli band levels for a
            ``SIMONCELLI`` Riesz base.
        padding: Boundary-extension method for a ``LOG`` Riesz base. A ``SIMONCELLI``
            base is periodic by construction and requires
            :attr:`FilterPadding.PERIODIC`.
        constant: Extension value for constant Riesz LoG padding.
        grid: Fourier sampling convention for the Riesz multiplier and a Simoncelli
            base.
        steering: Align each requested order with the local structure-tensor frame (IBSI
            2 section 4.7.1). The full order-``L`` bank and the first-order bank are
            computed and steered per voxel; only the requested components are returned.
            ``False`` keeps axis-aligned maps. Steering is qualified for orders up to 16
            (see ``config.filtering.MAX_STEERING_ORDER``); higher orders are refused.
        tensor_sigma: Gaussian orientation-window scale in mm for the structure tensor;
            required with steering and rejected without it. The window uses ``padding``
            and ``constant``.
    """

    base: RieszBase = RieszBase.LOG
    orders: tuple[tuple[int, ...], ...] = ()
    log_sigma: tuple[float, ...] = ()
    log_cutoff: float = 4.0
    simoncelli_levels: tuple[int, ...] = (1,)
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0
    grid: FourierGrid = FourierGrid.DFT
    steering: bool = False
    tensor_sigma: float | None = None

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.base) is not RieszBase:
            raise ValueError("riesz_base must be a typed RieszBase")
        if type(self.grid) is not FourierGrid:
            raise ValueError("riesz_grid must be a typed FourierGrid")
        if type(self.padding) is not FilterPadding:
            raise ValueError("riesz_padding must be a typed FilterPadding")
        if self.base is RieszBase.SIMONCELLI and self.padding is not FilterPadding.PERIODIC:
            raise ValueError("Simoncelli-based Riesz filtering requires periodic padding")
        if type(self.orders) is not tuple or any(
            type(order) is not tuple
            or not order
            or any(type(power) is not int or power < 0 for power in order)
            or (sum(order) < 1)
            for order in self.orders
        ):
            raise ValueError(
                "riesz_orders must contain nonempty tuples of nonnegative integers "
                "with a positive sum"
            )
        if len(set(self.orders)) != len(self.orders):
            raise ValueError("riesz_orders must be unique")
        if type(self.log_sigma) is not tuple or any(
            type(value) not in (int, float) or not isfinite(value) or value <= 0
            for value in self.log_sigma
        ):
            raise ValueError("riesz_log_sigma must contain finite positive values")
        if (
            type(self.log_cutoff) not in (int, float)
            or not isfinite(self.log_cutoff)
            or self.log_cutoff <= 0
        ):
            raise ValueError("riesz_log_cutoff must be finite and positive")
        if (
            type(self.simoncelli_levels) is not tuple
            or not self.simoncelli_levels
            or any(type(level) is not int or level < 1 for level in self.simoncelli_levels)
        ):
            raise ValueError("riesz_simoncelli_levels must contain positive integers")
        if len(set(self.simoncelli_levels)) != len(self.simoncelli_levels):
            raise ValueError("riesz_simoncelli_levels must be unique")
        validate_filter_constant(self.constant, "riesz_constant")
        if type(self.steering) is not bool:
            raise ValueError("riesz_steering must be a boolean")
        if self.tensor_sigma is not None and (
            type(self.tensor_sigma) not in (int, float)
            or not isfinite(self.tensor_sigma)
            or self.tensor_sigma <= 0
        ):
            raise ValueError("riesz_tensor_sigma must be finite and positive")
        if self.steering and self.tensor_sigma is None:
            raise ValueError("riesz_steering requires riesz_tensor_sigma")
        if self.steering and any(sum(order) > MAX_STEERING_ORDER for order in self.orders):
            raise ValueError(f"riesz_steering supports Riesz orders up to {MAX_STEERING_ORDER}")
        if not self.steering and self.tensor_sigma is not None:
            raise ValueError("riesz_tensor_sigma requires riesz_steering")


@dataclass(frozen=True)
class SeparableWaveletSettings:
    """IBSI separable wavelet bands and boundary conditions.

    Attributes:
        name: Wavelet name for ``SEPARABLE_WAVELET``, the IBSI 2 undecimated separable
            transform; taps come from ``wavelet_tables.json``.
        bands: Band codes such as ``"LHL"``, one ``L``/``H`` per image axis in IBSI
            ``(k1, k2, k3)`` order; required for the image type.
        level: One-based decomposition level of the returned bands.
        padding: Boundary-extension method.
        constant: Extension value for constant padding.
        pooling: Pooling over the right-angle kernel rotations.
        planar: Filter each array-axis-0 plane of a 3-D image with the 2-D transform;
            bands then have two letters, for ``(k1, k2)``.
    """

    name: str = "haar"
    bands: tuple[str, ...] = ()
    level: int = 1
    padding: FilterPadding = FilterPadding.MIRROR
    constant: float = 0.0
    pooling: OrientationPooling = OrientationPooling.NONE
    planar: bool = False

    def __post_init__(self) -> None:
        """Reject option values outside the documented ranges with ``ValueError``."""
        if type(self.planar) is not bool:
            raise ValueError("separable_wavelet_planar must be a boolean")
        if type(self.bands) is not tuple or any(
            type(band) is not str or not band or set(band) - {"L", "H"} for band in self.bands
        ):
            raise ValueError("separable_wavelet_bands must contain L/H band codes")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("separable_wavelet_bands must be unique")
        if type(self.level) is not int or self.level < 1:
            raise ValueError("separable_wavelet_level must be a positive integer")
        if type(self.padding) is not FilterPadding:
            raise ValueError("separable_wavelet_padding must be a typed FilterPadding")
        if type(self.pooling) is not OrientationPooling:
            raise ValueError("separable_wavelet_pooling must be a typed OrientationPooling")
        validate_filter_constant(self.constant, "separable_wavelet_constant")


@dataclass(frozen=True)
class TransformSettings:
    """Ordered image selection with independently validated options for each filter.

    ``scratch_mib`` bounds LBP3D batching. Options do not load MLX or SimpleITK.
    """

    image_types: tuple[ImageType, ...] = (ImageType.ORIGINAL,)
    scratch_mib: int = 64
    wavelet: WaveletSettings = WaveletSettings()
    log: LoGSettings = LoGSettings()
    lbp2d: LBP2DSettings = LBP2DSettings()
    lbp3d: LBP3DSettings = LBP3DSettings()
    gradient: GradientSettings = GradientSettings()
    mean: MeanSettings = MeanSettings()
    laws: LawsSettings = LawsSettings()
    gabor: GaborSettings = GaborSettings()
    simoncelli: SimoncelliSettings = SimoncelliSettings()
    riesz: RieszSettings = RieszSettings()
    separable_wavelet: SeparableWaveletSettings = SeparableWaveletSettings()

    def __post_init__(self) -> None:
        """Validate option types and the inputs each selected image type requires.

        Raises ``ValueError`` for duplicate or untyped image types, unknown wavelets, and
        missing LoG sigmas, separable-wavelet bands, Gabor scales or Riesz orders.
        """
        if type(self.wavelet) is not WaveletSettings:
            raise ValueError("wavelet must be WaveletSettings")
        if type(self.log) is not LoGSettings:
            raise ValueError("log must be LoGSettings")
        if type(self.lbp2d) is not LBP2DSettings:
            raise ValueError("lbp2d must be LBP2DSettings")
        if type(self.lbp3d) is not LBP3DSettings:
            raise ValueError("lbp3d must be LBP3DSettings")
        if type(self.gradient) is not GradientSettings:
            raise ValueError("gradient must be GradientSettings")
        if type(self.mean) is not MeanSettings:
            raise ValueError("mean must be MeanSettings")
        if type(self.laws) is not LawsSettings:
            raise ValueError("laws must be LawsSettings")
        if type(self.gabor) is not GaborSettings:
            raise ValueError("gabor must be GaborSettings")
        if type(self.simoncelli) is not SimoncelliSettings:
            raise ValueError("simoncelli must be SimoncelliSettings")
        if type(self.riesz) is not RieszSettings:
            raise ValueError("riesz must be RieszSettings")
        if type(self.separable_wavelet) is not SeparableWaveletSettings:
            raise ValueError("separable_wavelet must be SeparableWaveletSettings")
        if type(self.scratch_mib) is not int or self.scratch_mib < 1:
            raise ValueError("scratch_mib must be a positive integer")
        if not self.image_types or any(type(t) is not ImageType for t in self.image_types):
            raise ValueError("image_types must contain typed ImageType entries")
        if len(set(self.image_types)) != len(self.image_types):
            raise ValueError("image_types must be unique")
        if ImageType.WAVELET in self.image_types:
            wavelet_filters(self.wavelet.name)
        if ImageType.LOG in self.image_types and (not self.log.sigma):
            raise ValueError("LoG requires explicit positive sigma values")
        if ImageType.SEPARABLE_WAVELET in self.image_types:
            wavelet_filters(self.separable_wavelet.name)
            if not self.separable_wavelet.bands:
                raise ValueError("separable wavelets require explicit separable_wavelet_bands")
        if ImageType.GABOR in self.image_types:
            if not self.gabor.sigma or not self.gabor.wavelength or (not self.gabor.theta):
                raise ValueError(
                    "Gabor requires explicit gabor_sigma, gabor_wavelength and gabor_theta"
                )
            if len(self.gabor.sigma) != len(self.gabor.wavelength) and 1 not in (
                len(self.gabor.sigma),
                len(self.gabor.wavelength),
            ):
                raise ValueError(
                    "gabor_sigma and gabor_wavelength must have equal lengths or one value"
                )
            if len(set(self.gabor.theta)) != len(self.gabor.theta):
                raise ValueError("gabor_theta values must be unique")
            pairs = gabor_scale_pairs(self)
            if len(set(pairs)) != len(pairs):
                raise ValueError("gabor sigma/wavelength pairs must be unique")
        if ImageType.RIESZ in self.image_types:
            if not self.riesz.orders:
                raise ValueError("Riesz requires explicit riesz_orders")
            if self.riesz.base is RieszBase.LOG and (not self.riesz.log_sigma):
                raise ValueError("a LoG Riesz base requires explicit riesz_log_sigma values")


def gabor_scale_pairs(options: TransformSettings) -> tuple[tuple[float, float], ...]:
    """Pair Gabor sigmas with wavelengths, broadcasting a single value."""
    sigmas, wavelengths = (options.gabor.sigma, options.gabor.wavelength)
    if len(sigmas) == 1 and len(wavelengths) > 1:
        sigmas = sigmas * len(wavelengths)
    if len(wavelengths) == 1 and len(sigmas) > 1:
        wavelengths = wavelengths * len(sigmas)
    return tuple(zip(sigmas, wavelengths, strict=True))


def validate_filter_shape(ndim: int, options: TransformSettings, kind: ImageType) -> None:
    """Check one requested filter's options against the image dimensionality.

    Shared by parameter-file parsing and derived-image naming so both accept the same
    requests; options of unrequested filters are never checked. Planar separable wavelet
    bands name the two filtered in-plane axes, not every image axis.
    """
    if kind == ImageType.LAWS and ndim == 2 and (len(options.laws.kernel) == 6):
        raise ValueError("a three-axis Laws kernel requires a 3-D image")
    if kind == ImageType.GABOR and ndim == 2 and options.gabor.slice_axis is not None:
        raise ValueError("gabor_slice_axis is only valid for a 3-D image")
    if kind == ImageType.RIESZ and any(len(order) != ndim for order in options.riesz.orders):
        raise ValueError("Riesz order tuples must have one entry per image axis")
    if kind == ImageType.SEPARABLE_WAVELET:
        if options.separable_wavelet.planar and ndim != 3:
            raise ValueError("planar separable wavelets require a 3-D image")
        axes = 2 if options.separable_wavelet.planar else ndim
        if any(len(band) != axes for band in options.separable_wavelet.bands):
            raise ValueError("separable wavelet bands must have one letter per filtered axis")
