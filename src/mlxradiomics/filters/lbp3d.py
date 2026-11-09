"""3-D spherical-harmonic LBP (PyRadiomics ``lbp-3D``): host coefficients, Metal sampling."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.protocols import IndexedAddition
from mlxradiomics.numerics.reduction import sum_rows
from mlxradiomics.numerics.runtime import evaluate, mx
from mlxradiomics.numerics.spline import cubic_samples, spline_coefficients

if TYPE_CHECKING:
    from mlxradiomics.config.transforms import TransformSettings


@lru_cache(maxsize=32)
def _sphere_coefficients(
    subdivision: int,
    radius: float,
    levels: int,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.complex128], ...], NDArray[np.float64]]:
    """Return the icosphere sampling geometry and spherical-harmonic basis.

    Returns the icosphere vertices (offsets in voxels at ``radius``), one matrix of
    orthonormal spherical harmonics ``Y_n^m`` (``m = -n..n``) per degree
    ``n < levels``, and the spherical Voronoi solid angle of every vertex. Requires
    the optional ``lbp`` extra (SciPy and trimesh); results are cached.
    """
    try:
        from scipy.spatial import SphericalVoronoi
        from scipy.special import sph_harm_y
        from trimesh.creation import icosphere
    except ImportError as error:
        raise ImportError("LBP3D coefficient generation requires mlxradiomics[lbp]") from error
    vertices = np.asarray(icosphere(subdivisions=subdivision, radius=radius).vertices)
    # SciPy's sph_harm_y(n, m, theta, phi) takes theta as the polar (colatitudinal)
    # angle and phi as the azimuth. PyRadiomics passes these arguments in the opposite
    # order to the legacy sph_harm, which exchanged the two angles.
    theta = np.arccos(vertices[:, 2] / radius)
    phi = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2 * np.pi)
    harmonics = tuple(
        np.stack([sph_harm_y(n, m, theta, phi) for m in range(-n, n + 1)], axis=1)
        for n in range(levels)
    )
    # Spherical Voronoi cells give a first-order solid-angle quadrature:
    # sum(areas) == 4*pi on the unit sphere. Vertex samples are treated as constant
    # over their cells. The degree-zero coefficient of a constant pattern is exact;
    # higher-degree integrals and nonconstant patterns retain quadrature error.
    areas = SphericalVoronoi(vertices / radius, radius=1.0).calculate_areas()
    return vertices, harmonics, areas


def _scatter(values: mx.array, centers: mx.array, shape: tuple[int, ...]) -> mx.array:
    """Place one value per ``(n, 3)`` center into a zero float32 volume of ``shape``."""
    strides = mx.array([shape[1] * shape[2], shape[2], 1], dtype=mx.int32)
    indexes = mx.sum(centers * strides, axis=1)
    output = mx.zeros((int(np.prod(shape)),), dtype=mx.float32)
    output = cast(IndexedAddition, output.at[indexes]).add(values)
    return mx.reshape(output, shape)


def lbp3d_images(
    image: mx.array,
    mask: mx.array,
    options: TransformSettings,
    source_dtype: np.dtype[np.generic],
    source_residual: mx.array | None = None,
) -> Iterator[tuple[str, mx.array]]:
    """Yield the 3-D LBP images ``lbp-3D-m1`` ... ``lbp-3D-m<levels>`` and ``lbp-3D-k``.

    For every masked voxel, the image is sampled at the icosphere vertices around it (radius
    ``lbp3d_radius`` voxels) by cubic B-spline interpolation. Each sample is compared with the
    center intensity (bit set where sample >= center). Let ``f_j`` be that bit at vertex ``j``,
    ``Y_n^m`` the orthonormal complex spherical harmonics (SciPy's polar/azimuth convention)
    and ``dOmega_j`` the spherical Voronoi solid angle of the vertex. The degree-``n`` band
    coefficients are

    ``c_nm = sum_j dOmega_j f_j conj(Y_n^m(theta_j, phi_j))``

    and ``lbp-3D-m<k>`` (``k = 1..levels``, degree ``n = k - 1``) is the rotation-invariant
    band norm ``sqrt(sum_m |c_nm|^2)``. With exact coefficients, Parseval's identity
    makes this the L2 norm of the degree-``n`` projection; sampled quadrature retains
    discretization error and only approximate rotation invariance. An all-set bit vector gives
    ``sqrt(4*pi)`` for ``m1`` at every subdivision. This is a deliberate scale change from
    PyRadiomics' unnormalized sampled reconstruction, whose values grow with vertex count.
    Voronoi quadrature is first-order and approximate for nonconstant bit patterns, whose
    bit field is known only at the vertices. ``lbp-3D-k`` is the excess kurtosis (Fisher
    definition) of the sampled values. Voxels outside ``mask`` are 0.

    Args:
        image: Float32 array ``(Z, Y, X)`` in NumPy axis order.
        mask: Boolean ROI mask of the same shape; only masked voxels are computed.
        options: Supplies the levels, radius, icosphere subdivision and scratch budget.
        source_dtype: dtype of the original image. Integer types round the interpolated
            samples to the nearest integer (halves away from zero) and clip them to the
            dtype range; floating types are checked for ambiguous thresholds (below).
        source_residual: Float32 source-minus-image residual, required to recognize a
            constant float64 source when distinct values could collapse on upload.

    Yields:
        ``(name, volume)`` pairs, float32 volumes of the image's shape.

    Raises:
        ValueError: If the image is not 3-D, shapes differ, the mask is empty, a spline
            sample is not finite (floating images at or above ``2**120`` are first
            rescaled by an exact power of two, which preserves every output), an integer
            dtype is too wide to be exact in float32 (anything beyond 8/16-bit), an
            integer sample lies within the same roundoff bound of its center's rounding
            threshold ``center - 0.5`` (exact spline ties cannot be certified there), or a
            nonconstant floating-point image has a sample within
            ``64 * float32_eps * max(|sample|, |center|, 1)`` of its center, where float32
            spline roundoff could change the bit. A constant floating-point image is
            sampled exactly at its constant value inside the source domain and at zero
            outside it, consistent with the spline sampler's boundary rule.
        ImportError: If the ``lbp`` extra (SciPy, trimesh) is not installed.
    """
    if image.ndim != 3 or image.shape != mask.shape:
        raise ValueError("LBP3D requires a 3D image and an aligned boolean mask")
    coordinates = np.argwhere(np.asarray(mask, dtype=np.bool_)).astype(np.int32)
    if not len(coordinates):
        raise ValueError("LBP3D mask is empty")
    vertices, harmonics, areas = _sphere_coefficients(
        options.lbp3d.subdivision, options.lbp3d.radius, options.lbp3d.levels
    )
    offsets = mx.array(vertices, dtype=mx.float32)
    weights = mx.array(areas, dtype=mx.float32)
    # Analysis matrices: conj(Y_n^m) scaled by each vertex's solid angle. Conjugating the
    # analysis basis makes the projection the adjoint of the synthesis basis Y, i.e. the
    # quadrature approximation to the projection onto the degree-n harmonic band.
    analysis = tuple(
        mx.conj(mx.array(harmonic, dtype=mx.complex64)) * weights[:, None] for harmonic in harmonics
    )
    if source_dtype.kind == "f":
        # The prefilter multiplies samples by six and accumulates a geometric series,
        # so magnitudes near the float32 limit overflow to inf/NaN coefficients. Bits
        # compare samples with their center and kurtosis is amplitude invariant, so an
        # exact power-of-two rescale changes neither; only large images are rescaled to
        # keep subnormal flushing of tiny values unchanged elsewhere.
        peak = float(cast(float, mx.max(mx.abs(image)).item()))
        if np.isfinite(peak) and peak >= 2.0**120:
            image = image * (2.0 ** (120 - int(np.frexp(peak)[1])))
    coefficients = spline_coefficients(image)
    memory_per_center = max(4096, len(vertices) * (64 + 16 * options.lbp3d.levels))
    batch_size = max(1, options.scratch_mib * 1024**2 // memory_per_center)
    by_level: list[list[mx.array]] = [[] for _ in harmonics]
    kurtosis: list[mx.array] = []
    centers = mx.array(coordinates)
    strides = mx.array([image.shape[1] * image.shape[2], image.shape[2], 1])
    # A constant field has exact in-domain samples; spline roundoff must not perturb
    # their ties. Outside the domain, the sampler's zero extension still applies.
    constant_image = (
        source_dtype.kind == "f"
        and (source_dtype.itemsize <= 4 or source_residual is not None)
        and bool(mx.all(image == mx.reshape(image, (-1,))[0]).item())
        and (
            source_residual is None
            or bool(mx.all(source_residual == mx.reshape(source_residual, (-1,))[0]).item())
        )
    )
    for start in range(0, len(coordinates), batch_size):
        chunk = centers[start : start + batch_size]
        samples = cubic_samples(coefficients, chunk, offsets)
        original = mx.take(mx.reshape(image, (-1,)), mx.sum(chunk * strides, axis=1))
        # NaN compares false against the center, which would encode a legitimate
        # all-clear bit pattern; invalid interpolation must stop before any decision.
        if not bool(mx.all(mx.isfinite(samples)).item()):
            raise ValueError("LBP3D spline samples are not finite for this image")
        if constant_image:
            positions = chunk[:, None, :] + offsets[None, :, :]
            inside = mx.all((positions >= 0) & (positions <= mx.array(image.shape) - 1), axis=2)
            samples = mx.where(inside, original[:, None], 0)
            # In-domain samples tie even with a residual; outside bits depend on sign.
            low = (mx.array(0) if source_residual is None
                   else mx.reshape(source_residual, (-1,))[0])
            nonpositive = (original < 0) | ((original == 0) & (low <= 0))
            binary = (inside | nonpositive[:, None]).astype(mx.complex64)
        else:
            if source_dtype.kind in "iu":
                limits = np.iinfo(source_dtype.name)
                if max(abs(int(limits.min)), abs(int(limits.max))) > 2**24:
                    raise ValueError("LBP3D integer dtype exceeds exact float32 integer range")
                # With an integer center c, the rounded sample's bit flips exactly at
                # c - 0.5. Exact spline ties there (for example a 0/1 checkerboard at
                # half-voxel radius) land on either side under float32 roundoff, so a
                # sample within the roundoff bound of that threshold is refused rather
                # than snapped: an epsilon would misclassify genuinely lower samples.
                threshold_margin = mx.abs(samples - (original[:, None] - 0.5))
                threshold_bound = 64 * 1.1920928955078125e-7 * mx.maximum(
                    mx.maximum(mx.abs(samples), mx.abs(original[:, None])), 1,
                )
                if bool(mx.any(threshold_margin <= threshold_bound).item()):
                    raise ValueError(
                        "LBP3D integer rounding threshold is ambiguous at float32 "
                        "precision; spline roundoff can change its binary code"
                    )
                samples = mx.clip(
                    mx.sign(samples) * mx.floor(mx.abs(samples) + 0.5),
                    int(limits.min), int(limits.max),
                )
            binary = (samples >= original[:, None]).astype(mx.complex64)
        # Center before reduction so a constant ring has exactly zero variance even
        # when summing its repeated value and dividing would move the mean by an ulp.
        sample_offsets = samples - samples[:, :1]
        # Kurtosis is invariant to amplitude. Scaling before powers preserves finite
        # ratios even when the unscaled second/fourth moments overflow or underflow.
        amplitude = mx.max(mx.abs(sample_offsets), axis=1, keepdims=True)
        sample_offsets = sample_offsets / mx.where(amplitude > 0, amplitude, 1)
        mean = sum_rows(sample_offsets) / samples.shape[1]
        centered = sample_offsets - mean[:, None]
        variance = sum_rows(centered**2) / samples.shape[1]
        k = sum_rows(centered**4) / samples.shape[1] / variance**2 - 3
        margin = mx.abs(samples - original[:, None])
        bound = (
            64
            * 1.1920928955078125e-7
            * mx.maximum(mx.maximum(mx.abs(samples), mx.abs(original[:, None])), 1)
        )
        if (
            source_dtype.kind == "f"
            and not constant_image
            and bool(mx.any(margin <= bound).item())
        ):
            raise ValueError(
                "LBP3D threshold is ambiguous at float32 precision; "
                "source float64 spline roundoff can change its binary code"
            )
        for degree, columns in enumerate(analysis):
            c = binary @ columns
            power = mx.real(c) ** 2 + mx.imag(c) ** 2
            result = mx.sqrt(mx.sum(power, axis=1))
            evaluate(result)
            by_level[degree].append(result)
        evaluate(k)
        kurtosis.append(k)
    for degree, parts in enumerate(by_level, 1):
        yield f"lbp-3D-m{degree}", _scatter(mx.concatenate(parts), centers, image.shape)
    yield "lbp-3D-k", _scatter(mx.concatenate(kurtosis), centers, image.shape)
