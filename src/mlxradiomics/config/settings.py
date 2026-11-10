"""Feature catalog and extraction settings.

Unsupported settings raise ``ValueError``; nothing falls back to another backend.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, get_args

# PyRadiomics feature names form the default selection; IBSI definitions and
# explicit mathematical aliases extend them.
_PYRADIOMICS_FIRST_ORDER = (
    "Mean", "Minimum", "Maximum", "Range", "Variance", "Skewness", "Kurtosis",
    "Energy", "TotalEnergy", "RootMeanSquared", "Entropy", "Uniformity",
    "10Percentile", "90Percentile", "Median", "InterquartileRange",
    "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation",
)
_PYRADIOMICS_GLCM = (
    "Autocorrelation", "Contrast", "Correlation", "DifferenceAverage",
    "JointEnergy", "JointEntropy", "Id", "Idm", "JointAverage", "ClusterProminence",
    "ClusterShade", "ClusterTendency", "DifferenceEntropy", "DifferenceVariance", "Imc1",
    "Imc2", "Idmn", "Idn", "InverseVariance", "MaximumProbability", "SumAverage",
    "SumEntropy", "SumSquares", "MCC",
)
_PYRADIOMICS_GLDM = (
    "SmallDependenceEmphasis", "LargeDependenceEmphasis", "GrayLevelNonUniformity",
    "DependenceNonUniformity", "DependenceNonUniformityNormalized", "GrayLevelVariance",
    "DependenceVariance", "DependenceEntropy", "LowGrayLevelEmphasis", "HighGrayLevelEmphasis",
    "SmallDependenceLowGrayLevelEmphasis", "SmallDependenceHighGrayLevelEmphasis",
    "LargeDependenceLowGrayLevelEmphasis", "LargeDependenceHighGrayLevelEmphasis",
)
NGTDM = ("Coarseness", "Contrast", "Busyness", "Complexity", "Strength")
GLRLM = (
    "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance", "RunVariance",
    "RunEntropy", "LowGrayLevelRunEmphasis", "HighGrayLevelRunEmphasis",
    "ShortRunLowGrayLevelEmphasis", "ShortRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis", "LongRunHighGrayLevelEmphasis",
)
GLSZM = (
    "SmallAreaEmphasis", "LargeAreaEmphasis", "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized", "SizeZoneNonUniformity",
    "SizeZoneNonUniformityNormalized", "ZonePercentage", "GrayLevelVariance", "ZoneVariance",
    "ZoneEntropy", "LowGrayLevelZoneEmphasis", "HighGrayLevelZoneEmphasis",
    "SmallAreaLowGrayLevelEmphasis", "SmallAreaHighGrayLevelEmphasis",
    "LargeAreaLowGrayLevelEmphasis", "LargeAreaHighGrayLevelEmphasis",
)
_PYRADIOMICS_SHAPE = ("MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio",
         "Sphericity",
         "Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterColumn",
         "Maximum2DDiameterRow", "MajorAxisLength", "MinorAxisLength", "LeastAxisLength",
         "Elongation", "Flatness")
SHAPE2D = ("MeshSurface", "PixelSurface", "Perimeter", "PerimeterSurfaceRatio", "Sphericity",
           "MaximumDiameter", "MajorAxisLength", "MinorAxisLength", "Elongation")
# The default selection is the PyRadiomics set; physical local-intensity and morphology
# additions have separate dimensional/sampling requirements and are opt-in.
DEFAULT_FEATURES = tuple(
    f"{family}_{name}"
    for family, names in (("firstorder", _PYRADIOMICS_FIRST_ORDER),
                          ("glcm", _PYRADIOMICS_GLCM), ("gldm", _PYRADIOMICS_GLDM),
                          ("ngtdm", NGTDM), ("glrlm", GLRLM), ("glszm", GLSZM))
    for name in names
)
FIRST_ORDER = _PYRADIOMICS_FIRST_ORDER + (
    "MedianAbsoluteDeviation", "CoefficientOfVariation", "QuartileCoefficientOfDispersion",
)
GLCM = _PYRADIOMICS_GLCM + ("Dissimilarity", "SumVariance")
GLDM = _PYRADIOMICS_GLDM + (
    "GrayLevelNonUniformityNormalized", "DependenceCountEnergy", "DependencePercentage",
)
GLDZM = (
    "SmallDistanceEmphasis", "LargeDistanceEmphasis", "LowGrayLevelZoneEmphasis",
    "HighGrayLevelZoneEmphasis", "SmallDistanceLowGrayLevelEmphasis",
    "SmallDistanceHighGrayLevelEmphasis", "LargeDistanceLowGrayLevelEmphasis",
    "LargeDistanceHighGrayLevelEmphasis", "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized", "ZoneDistanceNonUniformity",
    "ZoneDistanceNonUniformityNormalized", "ZonePercentage", "GrayLevelVariance",
    "ZoneDistanceVariance", "ZoneDistanceEntropy",
)
INTENSITY_HISTOGRAM = (
    "Mean", "Variance", "Skewness", "Kurtosis", "Median", "Minimum", "10Percentile",
    "90Percentile", "Maximum", "Mode", "InterquartileRange", "Range",
    "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation", "MedianAbsoluteDeviation",
    "CoefficientOfVariation", "QuartileCoefficientOfDispersion", "Entropy", "Uniformity",
    "MaximumGradient", "MaximumGradientGrayLevel", "MinimumGradient", "MinimumGradientGrayLevel",
)
LOCAL_INTENSITY = ("LocalIntensityPeak", "GlobalIntensityPeak")
IVH = ("VolumeAtIntensityFraction10", "VolumeAtIntensityFraction90",
       "IntensityAtVolumeFraction10", "IntensityAtVolumeFraction90",
       "VolumeFractionDifference", "IntensityFractionDifference", "AreaUnderIVHCurve")
SHAPE = _PYRADIOMICS_SHAPE + (
    "Compactness1", "Compactness2", "SphericalDisproportion", "Asphericity",
    "VolumeDensityAxisAlignedBoundingBox", "AreaDensityAxisAlignedBoundingBox",
    "VolumeDensityApproximateEnclosingEllipsoid", "AreaDensityApproximateEnclosingEllipsoid",
    "VolumeDensityOrientedMinimumBoundingBox", "AreaDensityOrientedMinimumBoundingBox",
    "VolumeDensityMinimumVolumeEnclosingEllipsoid",
    "AreaDensityMinimumVolumeEnclosingEllipsoid",
    "VolumeDensityConvexHull", "AreaDensityConvexHull", "CentreOfMassShift",
    "IntegratedIntensity", "MoransI", "GearysC",
)
FAMILIES: dict[str, tuple[str, ...]] = {
    "firstorder": FIRST_ORDER, "glcm": GLCM, "gldm": GLDM, "ngtdm": NGTDM,
    "glrlm": GLRLM, "glszm": GLSZM, "shape": SHAPE, "shape2D": SHAPE2D,
    "gldzm": GLDZM, "intensityhistogram": INTENSITY_HISTOGRAM,
    "localintensity": LOCAL_INTENSITY, "ivh": IVH,
}
# Selectable feature names, "<family>_<Name>"; result keys add an image-type prefix.
FEATURES = tuple(f"{family}_{name}" for family, names in FAMILIES.items() for name in names)


def list_features(
    *families: str,
    mode: Literal["roi", "voxel"] = "roi",
    ndim: int | None = None,
) -> tuple[str, ...]:
    """List selectable names, optionally restricted to families, mode and dimension.

    With no families, list the full catalog, not the 93-feature default selection.
    Unknown families raise ``ValueError``. Voxel mode excludes shape and local intensity.
    ``ndim=2`` excludes 3-D shape and local intensity; ``ndim=3`` excludes shape2D,
    whose single-slice 3-D use requires explicit selection and ``force2d_axis``.
    The returned tuple can be passed directly to ``Settings(features=...)``.
    Names retain catalog order and occur once even when a family is repeated.
    """
    if mode not in ("roi", "voxel"):
        raise ValueError("mode must be 'roi' or 'voxel'")
    if ndim is not None and (type(ndim) is not int or ndim not in (2, 3)):
        raise ValueError("ndim must be 2, 3 or None")
    unknown = set(families) - FAMILIES.keys()
    if unknown:
        raise ValueError(
            f"unknown feature families: {sorted(unknown)}; choose from {tuple(FAMILIES)}"
        )
    selected = set(families) if families else set(FAMILIES)
    if mode == "voxel":
        selected -= {"shape", "shape2D", "localintensity"}
    if ndim == 2:
        selected -= {"shape", "localintensity"}
    elif ndim == 3:
        selected.discard("shape2D")
    return tuple(name for name in FEATURES if name.split("_", 1)[0] in selected)

type DirectionalAggregation = Literal[
    "native", "2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_merge",
    "3d_average", "3d_merge",
]
type MatrixAggregation = Literal["native", "2d_average", "2.5d_merge", "3d"]
type CovarianceDivisor = Literal["n-1", "n"]
type EllipsoidArea = Literal["ibsi-series", "exact"]
type Kurtosis = Literal["excess", "raw"]
type ZeroVarianceMoments = Literal["zero", "nan"]
type NgtdmSingularCases = Literal["ibsi-caps", "unadjusted"]
type NeighborhoodSupport = Literal["closed-radius", "distances"]
type BinOrigin = Literal["resegment-or-minimum", "zero-lattice"]
type ConstantRoiBin = Literal["first", "middle"]
type MaskExtrapolation = Literal["zero", "nearest"]
type SplineBoundary = Literal["nearest", "mirror"]


# Convention fields with their allowed values; the first value is the IBSI default.
CONVENTIONS: dict[str, tuple[str, ...]] = {
    "covariance_divisor": get_args(CovarianceDivisor.__value__),
    "ellipsoid_area": get_args(EllipsoidArea.__value__),
    "kurtosis": get_args(Kurtosis.__value__),
    "zero_variance_moments": get_args(ZeroVarianceMoments.__value__),
    "ngtdm_singular_cases": get_args(NgtdmSingularCases.__value__),
    "neighborhood_support": get_args(NeighborhoodSupport.__value__),
    "bin_origin": get_args(BinOrigin.__value__),
    "constant_roi_bin": get_args(ConstantRoiBin.__value__),
    "mask_extrapolation": get_args(MaskExtrapolation.__value__),
    "spline_boundary": get_args(SplineBoundary.__value__),
}


def _is_real(value: object) -> bool:
    """Whether ``value`` is an ``int`` or ``float`` (``bool`` and strings are not)."""
    return type(value) in (int, float)


@dataclass(frozen=True)
class Settings:
    """Feature selection and preprocessing options for one extraction.

    Field names correspond to PyRadiomics settings (``bin_width`` is ``binWidth``,
    ``gldm_alpha`` is ``gldm_a``, and so on). Intensities are in the image's own units
    (HU for CT). ``distances`` and ``kernel_radius`` are in voxels, not physical units.
    ``bin_anchor``, ``resegment_outlier_sigma``, the convention settings and the
    aggregation controls are extensions also accepted by this package's parameter-file
    parser; pinned PyRadiomics does not define these settings.

    Calculation conventions are independent named fields with IBSI defaults (see the
    convention attributes). Select alternative definitions explicitly for the intended
    scientific method.

    Attributes:
        gldm_alpha: A neighbor counts as dependent when its gray level differs from the
            center by at most this many bins (GLDM alpha).
        bin_width: Fixed bin width. With the default ``bin_anchor=None``, bin edges are
            set by ``bin_origin``: by default the lower bound of an absolute
            ``resegment_range`` (IBSI X_gl,min), or the sampled minimum when no absolute
            range is set. Ignored, and not validated, when ``bin_count`` is set.
        bin_count: Fixed number of bins between the sampled minimum and maximum. Takes
            precedence over ``bin_width``; an explicit ``max_gray_levels`` limits the
            used lattice. Cannot be combined with ``bin_anchor``.
        bin_anchor: Optional fixed-bin-size origin. ``None`` uses the origin chosen
            by ``bin_origin``. A finite value defines the first edge and gray
            level 1 at the anchor itself, even when no selected value falls in that bin;
            selected values below the anchor are rejected when discretization is performed.
            The anchor is in the same units as the binned (possibly normalized) intensities
            and is the explicit IBSI fixed-bin-size minimum (for example ``-1000`` HU with
            a 25 HU width). The extension parameter-file key is ``binAnchor``.
        normalize: Normalize the whole image (not just the ROI) to zero mean and unit
            sample standard deviation, before resegmentation and binning.
        normalize_scale: Factor applied after normalization (and after outlier clipping).
        remove_outliers: Requires ``normalize``. Clip normalized values to plus or minus
            this many standard deviations before scaling.
        resegment_range: One or two thresholds, inclusive. One keeps voxels at or above
            it; two keep voxels between them. Applied to the label's voxels after
            normalization.
        resegment_mode: ``"absolute"`` uses the thresholds as given, ``"relative"``
            multiplies them by the maximum ROI intensity, and ``"sigma"`` uses
            ``mean + threshold * std`` of the ROI intensities.
        resegment_outlier_sigma: Optional second, independent resegmentation filter.
            Requires ``resegment_range`` and ``resegment_mode="absolute"``. When set,
            keep voxels within this many population standard deviations of the mean of
            the label's original selected values, computed before ``resegment_range``
            is applied, and intersect that window with the absolute range. The sigma
            statistics are never recomputed from the already clipped ROI. With
            normalization enabled the values, and therefore both filters, use the
            normalized intensities after ``remove_outliers`` clipping and
            ``normalize_scale``. ``None`` disables the second filter. Its extension
            parameter-file key is ``resegmentOutlierSigma``; unlike ``remove_outliers`` it filters
            only the selected ROI rather than clipping the whole image.
        resegment_shape: If true, shape features use the fully resegmented mask (absolute
            range and any outlier-sigma window); otherwise they use the original label
            mask.
        label: Mask value selecting the ROI; the mask is compared by equality.
        distances: GLCM, GLDM and NGTDM neighbor distances. Each is a Chebyshev
            (maximum-norm) distance in voxels, so distance 1 in 3-D gives 26 neighbors.
            GLRLM and GLSZM always use distance 1, as PyRadiomics does.
        symmetrical_glcm: Add each GLCM to its transpose before computing features.
        force2d_axis: If set, neighbor directions stay in the plane perpendicular to this
            NumPy axis (PyRadiomics ``force2D`` with ``force2Ddimension``), and
            ``shape2D`` on a 3-D array uses the single slice along it. Must be a real
            integer; booleans and floats are rejected by validation.
        masked_kernel: Voxel mode only. If true, local neighborhoods sample only ROI
            voxels. If false, they sample the whole image, and gray-level discretization
            also uses the whole image; output centers still lie in the ROI.
        kernel_radius: Voxel mode only. Half-width in voxels of the cubic neighborhood
            around each center (flattened along ``force2d_axis`` if set).
        voxel_array_shift: Added to intensities before Energy, TotalEnergy and
            RootMeanSquared only.
        scratch_mib: Working-memory budget in MiB used to size voxel batches and shape-mesh
            chunks, and bound each selected primary dense ROI count buffer. NGTDM partial
            tables may also scale linearly with their input domain. Combined live arrays
            and statistic intermediates can exceed this budget; it is not an OS limit.
        max_gray_levels: Optional caller-imposed upper bound on the used discretization
            lattice. None preserves the requested bin width without an arbitrary
            gray-level cap; indices must still fit the backend's signed 32-bit range.
            This setting does not constrain features that do not use discretization.
        features: Names from ``FEATURES``. The default is the original 93-feature selection;
            shape features must be listed explicitly. Voxel-mode MCC keeps this
            selection and returns NaN at non-flat centers with an empty retained
            GLCM direction; it does not abort the remaining features or centers.
        covariance_divisor: Divisor of the voxel-center covariance behind the
            shape principal axes and the approximate-enclosing-ellipsoid semi-axes:
            ``"n-1"`` (IBSI, default) or ``"n"`` (PyRadiomics). A single voxel keeps zero
            axes either way.
        ellipsoid_area: Surface area of the approximate enclosing ellipsoid in its area
            density: ``"ibsi-series"`` (default) evaluates the series printed in the IBSI
            manual, which reproduces the published reference; ``"exact"`` evaluates the
            converged Legendre series.
        kurtosis: ``"excess"`` (IBSI, default) subtracts three; ``"raw"`` reports the
            plain fourth-moment ratio (PyRadiomics).
        zero_variance_moments: Skewness and kurtosis of a zero-variance ROI: ``"zero"``
            (IBSI, default) or ``"nan"`` for an undefined standardized moment.
        ngtdm_singular_cases: ``"ibsi-caps"`` (default) applies the IBSI degenerate
            NGTDM conventions: coarseness capped at 1e6, busyness zero with one occupied
            gray level, strength zero when the differences sum to zero. ``"unadjusted"``
            returns the formulas as evaluated, including infinities and NaNs.
        neighborhood_support: Support of the NGLDM and NGTDM neighborhood. ``"closed-radius"``
            (default) uses every voxel within Chebyshev ``neighborhood_radius``;
            ``"distances"`` uses the explicitly selected ``distances`` (PyRadiomics).
        bin_origin: Origin of a fixed-bin-size lattice when ``bin_anchor`` is ``None``:
            ``"resegment-or-minimum"`` (IBSI, default) starts at the lower bound of an
            absolute ``resegment_range`` or else the sampled minimum, with gray level 1 in
            the first bin; ``"zero-lattice"`` uses edges at multiples of ``bin_width``
            counted from zero, numbered from the minimum bin (PyRadiomics).
        constant_roi_bin: Gray level of a constant ROI under ``bin_count``: ``"first"``
            (IBSI, default) is level 1; ``"middle"`` is ``bin_count // 2 + 1`` (PyRadiomics).
        mask_extrapolation: How IBSI centered-grid interpolation (``IbsiInterpolation``)
            fills the mask where the resampled grid lies outside the source domain:
            ``"zero"`` (default, as in MIRP) leaves it outside the ROI; ``"nearest"``
            copies the edge occupancy, so the ROI grows at the image border.
            Image intensities always use the nearest source intensity there.
            PyRadiomics-style resampling is not affected.
        spline_boundary: Boundary handling of the cubic B-spline image interpolator
            (``Interpolator.BSPLINE``) under ``IbsiInterpolation``: ``"nearest"``
            (default, as ``scipy.ndimage.map_coordinates(mode="nearest")`` in MIRP)
            replicates the edge voxels, ``"mirror"`` uses ITK's mirror condition.
            Linear interpolation is unaffected.
        glcm_aggregation: Directional matrix aggregation: 2d_average, 2d_slice_merge,
            2.5d_direction_merge, 2.5d_merge, 3d_average or 3d_merge. ``native`` resolves
            to 3d_average, or 2.5d_direction_merge with force2d_axis, or 2d_average in 2D.
            ``2d_average`` gives every nonempty slice/direction matrix equal weight;
            it does not first equalize slices with different numbers of valid directions.
        glrlm_aggregation: The same six directional choices, independently for runs.
        glszm_aggregation: 2d_average, 2.5d_merge, or 3d. ``native`` selects 3d unless
            restricted to planes, in which case their matrices are merged.
        gldzm_aggregation: The same three matrix choices, for distance zones.
        ngtdm_aggregation: The same three choices, merging both occurrences and differences.
        gldm_aggregation: The same three choices, for dependence counts.
        slice_axis: NumPy axis used by explicit planar aggregation on 3D input when
            force2d_axis is absent. Empty slices are excluded from feature averages.
        neighborhood_radius: IBSI NGTDM/NGLDM Chebyshev radius, including every inner
            shell. Used when ``neighborhood_support="closed-radius"``.
        ivh_bin_width: Separate IVH physical intensity interval; reports bin centers.
        ivh_bin_count: Separate IVH bin count for arbitrary intensity units. With neither
            IVH discretizer, input intensities must be calibrated integers.
        ivh_intensity_range: Optional closed IVH domain. Without one, use absolute
            resegmentation bounds when available, otherwise the sampled range.
    """

    gldm_alpha: int = 0
    bin_width: float = 25.0
    bin_count: int | None = None
    normalize: bool = False
    normalize_scale: float = 1.0
    remove_outliers: float | None = None
    resegment_range: tuple[float, ...] | None = None
    resegment_mode: Literal["absolute", "relative", "sigma"] = "absolute"
    resegment_shape: bool = False
    label: int = 1
    distances: tuple[int, ...] = (1,)
    symmetrical_glcm: bool = True
    force2d_axis: int | None = None
    masked_kernel: bool = True
    kernel_radius: int = 1
    voxel_array_shift: float = 0.0
    scratch_mib: int = 64
    max_gray_levels: int | None = None
    features: tuple[str, ...] = DEFAULT_FEATURES
    bin_anchor: float | None = None
    resegment_outlier_sigma: float | None = None
    covariance_divisor: CovarianceDivisor = "n-1"
    ellipsoid_area: EllipsoidArea = "ibsi-series"
    kurtosis: Kurtosis = "excess"
    zero_variance_moments: ZeroVarianceMoments = "zero"
    ngtdm_singular_cases: NgtdmSingularCases = "ibsi-caps"
    neighborhood_support: NeighborhoodSupport = "closed-radius"
    bin_origin: BinOrigin = "resegment-or-minimum"
    constant_roi_bin: ConstantRoiBin = "first"
    mask_extrapolation: MaskExtrapolation = "zero"
    spline_boundary: SplineBoundary = "nearest"
    glcm_aggregation: DirectionalAggregation = "native"
    glrlm_aggregation: DirectionalAggregation = "native"
    glszm_aggregation: MatrixAggregation = "native"
    gldzm_aggregation: MatrixAggregation = "native"
    ngtdm_aggregation: MatrixAggregation = "native"
    gldm_aggregation: MatrixAggregation = "native"
    slice_axis: int = 0
    neighborhood_radius: int = 1
    ivh_bin_width: float | None = None
    ivh_bin_count: int | None = None
    ivh_intensity_range: tuple[float, float] | None = None

    def validate(self, ndim: int) -> None:
        """Check the supported settings and raise ``ValueError`` for detected violations.

        When ``bin_count`` is set, ``bin_width`` is unused and is not validated. An
        explicit ``bin_anchor`` requires ``bin_count=None``. ``resegment_outlier_sigma``
        requires a finite positive value, a nonempty ``resegment_range`` and absolute
        resegmentation. This checks settings, not image contents or whether a resegmented
        ROI is nonempty.

        Args:
            ndim: Dimensionality of the image the settings will be applied to; only used
                to check that ``force2d_axis`` indexes an array axis.
        """
        self._validate_preprocessing()
        self._validate_definitions(ndim)
        if type(self.gldm_alpha) is not int or self.gldm_alpha < 0:
            raise ValueError("gldm_alpha must be a nonnegative integer")
        if self.bin_count is None and (
            not _is_real(self.bin_width) or not isfinite(self.bin_width) or self.bin_width <= 0
        ):
            raise ValueError("bin_width must be finite and positive")
        if type(self.label) is not int or self.label <= 0:
            raise ValueError("label must be a positive integer")
        if not self.distances or any(type(x) is not int or x < 1 for x in self.distances):
            raise ValueError("distances must contain positive integers")
        if len(set(self.distances)) != len(self.distances):
            raise ValueError("distances must be unique")
        if self.force2d_axis is not None and (
            type(self.force2d_axis) is not int or self.force2d_axis not in range(ndim)
        ):
            raise ValueError("force2d_axis must be an integer indexing a NumPy image axis")
        if type(self.masked_kernel) is not bool or type(self.symmetrical_glcm) is not bool:
            raise ValueError("masked_kernel and symmetrical_glcm must be booleans")
        for name in ("kernel_radius", "scratch_mib"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_gray_levels is not None and (
            type(self.max_gray_levels) is not int or self.max_gray_levels < 1
        ):
            raise ValueError("max_gray_levels must be a positive integer or None")
        if not _is_real(self.voxel_array_shift) or not isfinite(self.voxel_array_shift):
            raise ValueError("voxel_array_shift must be finite")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be a nonempty unique selection")
        unsupported = set(self.features) - set(FEATURES)
        if unsupported:
            raise ValueError(f"unsupported features: {sorted(unsupported)}")

    def _validate_definitions(self, ndim: int) -> None:
        """Validate explicit definitions, matrix aggregation and the separate IVH lattice."""
        for name, choices in CONVENTIONS.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {', '.join(choices)}")
        directional = get_args(DirectionalAggregation.__value__)
        matrix = get_args(MatrixAggregation.__value__)
        for family in ("glcm", "glrlm", "glszm", "gldzm", "ngtdm", "gldm"):
            mode = getattr(self, f"{family}_aggregation")
            if mode not in (directional if family in {"glcm", "glrlm"} else matrix):
                raise ValueError(f"unsupported {family} aggregation: {mode}")
            if (any(name.startswith(f"{family}_") for name in self.features)
                    and mode.startswith("3d")
                    and (ndim != 3 or self.force2d_axis is not None)):
                raise ValueError("3d aggregation requires 3D input without force2d_axis")
        if type(self.slice_axis) is not int or self.slice_axis not in range(ndim):
            raise ValueError("slice_axis must be an integer indexing an image axis")
        if type(self.neighborhood_radius) is not int or self.neighborhood_radius < 1:
            raise ValueError("neighborhood_radius must be a positive integer")
        if self.ivh_bin_width is not None and (
            not _is_real(self.ivh_bin_width)
            or not isfinite(self.ivh_bin_width)
            or self.ivh_bin_width <= 0
        ):
            raise ValueError("ivh_bin_width must be finite and positive")
        if self.ivh_bin_count is not None and (
            type(self.ivh_bin_count) is not int or self.ivh_bin_count < 1
        ):
            raise ValueError("ivh_bin_count must be a positive integer")
        if self.ivh_bin_width is not None and self.ivh_bin_count is not None:
            raise ValueError("IVH accepts either bin width or bin count")
        if self.ivh_intensity_range is not None and (
            len(self.ivh_intensity_range) != 2
            or any(not _is_real(v) or not isfinite(v) for v in self.ivh_intensity_range)
            or self.ivh_intensity_range[0] > self.ivh_intensity_range[1]
        ):
            raise ValueError("ivh_intensity_range must be finite ordered lower/upper bounds")

    def _validate_preprocessing(self) -> None:
        """Validate binning, normalization and resegmentation options."""
        if self.bin_anchor is not None:
            if type(self.bin_anchor) not in (int, float) or not isfinite(self.bin_anchor):
                raise ValueError("bin_anchor must be a finite number or None")
            if self.bin_count is not None:
                raise ValueError("bin_anchor cannot be combined with bin_count")
        if self.bin_count is not None and (
            type(self.bin_count) is not int or not 1 <= self.bin_count < 2**31
        ):
            raise ValueError("bin_count must be a positive integer below 2**31")
        if type(self.normalize) is not bool or type(self.resegment_shape) is not bool:
            raise ValueError("normalize and resegment_shape must be booleans")
        if (
            not _is_real(self.normalize_scale)
            or not isfinite(self.normalize_scale)
            or self.normalize_scale <= 0
        ):
            raise ValueError("normalize_scale must be finite and positive")
        if self.remove_outliers is not None and (
            not _is_real(self.remove_outliers)
            or not isfinite(self.remove_outliers)
            or self.remove_outliers <= 0
            or not self.normalize
        ):
            raise ValueError("remove_outliers requires normalization and a finite positive value")
        if self.resegment_mode not in {"absolute", "relative", "sigma"}:
            raise ValueError("resegment_mode must be absolute, relative, or sigma")
        if self.resegment_range is not None and (
            len(self.resegment_range) not in (1, 2)
            or any(not _is_real(value) or not isfinite(value) for value in self.resegment_range)
        ):
            raise ValueError("resegment_range must contain one or two finite thresholds")
        if self.resegment_outlier_sigma is not None:
            if (
                type(self.resegment_outlier_sigma) not in (int, float)
                or not isfinite(self.resegment_outlier_sigma)
                or self.resegment_outlier_sigma <= 0
            ):
                raise ValueError(
                    "resegment_outlier_sigma must be a finite positive number or None"
                )
            if self.resegment_range is None or self.resegment_mode != "absolute":
                raise ValueError(
                    "resegment_outlier_sigma requires resegment_range and "
                    'resegment_mode="absolute"'
                )
