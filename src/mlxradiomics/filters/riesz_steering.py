"""Local Riesz alignment from a regularised structure tensor (IBSI 2 ``1SD3``).

The IBSI 2 filtering manual (Depeursinge et al., arXiv:2006.05470v9, section 4.7.1
and Eqs. 4.12-4.15) aligns the order-``L`` Riesz filter bank with the eigenvectors
of a Gaussian-regularised structure tensor built from the first-order Riesz
responses. Because the Riesz bank is steerable, the aligned responses are a
per-voxel linear combination of the axis-aligned ones; no extra convolution is
needed. The manual provides no consensus response maps for aligned Riesz, so this
module's correctness rests on the exact algebra of the steering matrix and on the
closed-form eigendecomposition.

Conventions that the API depends on:

* Multi-indices ``l = (l_1, ..., l_D)`` follow the manual's IBSI axis order
  ``(k_1, ..., k_D)`` and are enumerated in descending lexicographic order. They
  index the Riesz component space, not array axes: the native array axis order is
  reversed relative to IBSI, but a component multi-index always refers to the
  ``k``-axis with the same position in the tuple.
* ``U`` has the analysis directions as rows ``u_1..u_D``. Aligned component ``n``
  is the Riesz transform whose Fourier multiplier is
  ``(-j)^L sqrt(L!/n!) (U nu)^n / |nu|^L``, the kernel rotated so that its
  principal direction is ``u_n``. Evaluating Eq. steeringMatrix3D with the rows of
  ``U`` in this convention makes the order-one steering matrix equal to ``U``
  itself, and composition follows ``S(U_1 U_2) = S(U_1) S(U_2)``.
* Eigenvectors of the structure tensor are ordered by descending Rayleigh
  quotient, so ``u_1`` maximises the energy of ``R_1`` and ``u_2`` maximises the
  residual energy of ``R_2``. The sign of ``u_1`` and ``u_2`` is fixed so that the
  largest-magnitude component (lowest index on ties) is positive, and
  ``u_3 = u_1 x u_2`` keeps ``det U = +1``; consequently ``u_3`` does not obey the
  sign rule itself. Sign choices only matter for odd orders, where a flipped
  axis negates the corresponding steered component.
* A positive image scale multiplies the structure tensor by a positive constant
  and leaves its eigenvectors unchanged. The response products use one common
  bank scale, and every nonzero tensor is divided by its largest absolute entry
  before the solve. A voxel with a nonzero response whose complete smoothed tensor
  underflows after this scaling is rejected instead of silently given the identity.
* Degenerate spectra have no unique frame. A tensor whose spectral spread is at
  float32 rounding level relative to its eigenvalue magnitude, including the zero
  tensor, returns the identity rotation. Otherwise the 3x3 solver takes the
  eigenvector of the eigenvalue farther from the middle one, which is always well
  separated, and diagonalises the remaining pair in its orthogonal plane in
  closed form. No eigenvalue-gap threshold is involved: a resolvable pair keeps
  its true eigenvectors, and an exactly repeated pair keeps the deterministic
  in-plane basis built from the coordinate axis where the separated eigenvector is
  smallest.

Voxel-wise work is batched in bounded blocks: the steering matrix has ``K x K``
entries per voxel with ``K = binom(L + D - 1, D - 1)``, and the block size follows
the intermediate polynomial tables so resident memory stays bounded.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, factorial, isfinite, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.config.filtering import MAX_STEERING_ORDER, FilterPadding
from mlxradiomics.filters.convolution import pad_image
from mlxradiomics.numerics.runtime import mx

# A near-isotropic tensor has no meaningful orientation: its spectral spread is at
# float32 rounding level relative to its eigenvalue magnitude. Tensor entries carry
# a few float32 ulps of error, so eigenvectors resolved above 64 eps (2**-17) are
# accurate to about 1/64 radian or better.
_ISOTROPIC_TOLERANCE = 2.0**-17
# Upper bounds for the batch sizes of the closed-form eigen solver and of the
# steering polynomial expansion, in voxels and float32 elements respectively.
_EIGEN_BLOCK_VOXELS = 1 << 14
_STEERING_BLOCK_FLOATS = 1 << 21
_SMALLEST_NORM = 1e-30


def _multi_indices(ndim: int, order: int) -> tuple[tuple[int, ...], ...]:
    """All multi-indices of the given total order, descending lexicographically."""
    if ndim == 1:
        return ((order,),)
    return tuple(
        (head,) + tail
        for head in range(order, -1, -1)
        for tail in _multi_indices(ndim - 1, order - head)
    )


@lru_cache(maxsize=32)
def riesz_multi_indices(ndim: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Multi-indices of the order-``order`` Riesz components in canonical order.

    The enumeration is descending lexicographic over IBSI axis order, matching the
    manual's expansion of ``(U nu)^n``; for three dimensions and order two it is
    ``(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)``.

    Args:
        ndim: Spatial dimension ``D``, either two or three.
        order: Riesz order ``L >= 1``.

    Returns:
        Tuple of ``binom(L + D - 1, D - 1)`` multi-indices.

    Raises:
        ValueError: If ``ndim`` is not two or three, or ``order`` is below one.
    """
    if ndim not in (2, 3):
        raise ValueError("Riesz multi-indices are defined for two or three dimensions")
    if order < 1:
        raise ValueError("Riesz order must be at least one")
    return _multi_indices(ndim, order)


@dataclass(frozen=True)
class _SteeringPlan:
    """Host-side index tables that expand the steering polynomials for one bank.

    ``shifts[step][axis]`` maps every monomial of total degree ``step + 1`` (plus a
    trailing zero slot) to the position of that monomial divided by ``nu_axis`` in
    the degree-``step`` layer. ``rows`` names the rotation row used by each of the
    ``order`` factors of every output component, and ``factors`` holds the
    ``sqrt(m! / n!)`` normalisation of Eq. steeringMatrix3D.
    """

    count: int
    order: int
    rows: NDArray[np.int32]
    shifts: tuple[tuple[NDArray[np.int32], ...], ...]
    factors: NDArray[np.float32]

    @property
    def elements_per_voxel(self) -> int:
        """Largest number of polynomial coefficients held for one voxel."""
        widest = max(shift.shape[0] for step in self.shifts for shift in step)
        return self.count * widest


@lru_cache(maxsize=16)
def _steering_plan(ndim: int, order: int) -> _SteeringPlan:
    """Build the cached steering index tables for one dimensionality and Riesz order."""
    indices = riesz_multi_indices(ndim, order)
    layers = [_multi_indices(ndim, degree) for degree in range(order + 1)]
    shifts: list[tuple[NDArray[np.int32], ...]] = []
    for degree in range(order):
        sources = {monomial: index for index, monomial in enumerate(layers[degree])}
        targets = layers[degree + 1]
        columns: list[NDArray[np.int32]] = []
        for axis in range(ndim):
            column = np.empty(len(targets) + 1, dtype=np.int32)
            for index, monomial in enumerate(targets):
                if monomial[axis] == 0:
                    column[index] = len(layers[degree])
                else:
                    reduced = monomial[:axis] + (monomial[axis] - 1,) + monomial[axis + 1:]
                    column[index] = sources[reduced]
            # The trailing slot is never a target, so it stays zero through the
            # expansion and can serve as the out-of-range gather value.
            column[-1] = len(layers[degree])
            columns.append(column)
        shifts.append(tuple(columns))
    rows = np.empty((len(indices), order), dtype=np.int32)
    for index, monomial in enumerate(indices):
        rows[index] = np.repeat(np.arange(ndim, dtype=np.int32), monomial)
    factors = np.empty((len(indices), len(indices)), dtype=np.float32)
    for row, output in enumerate(indices):
        output_factorial = 1
        for exponent in output:
            output_factorial *= factorial(exponent)
        for column, source in enumerate(indices):
            source_factorial = 1
            for exponent in source:
                source_factorial *= factorial(exponent)
            factors[row, column] = sqrt(source_factorial / output_factorial)
    return _SteeringPlan(
        count=len(indices), order=order, rows=rows, shifts=tuple(shifts), factors=factors,
    )


def _steering(rotations: mx.array, plan: _SteeringPlan) -> mx.array:
    """Evaluate the steering polynomials for a batch of rotation matrices."""
    batch = rotations.shape[:-2]
    count = plan.count
    # Every output component starts from the constant polynomial one; the extra
    # trailing column is the permanent zero slot used for out-of-range gathers.
    polynomials = mx.concatenate(
        [
            mx.ones((*batch, count, 1), dtype=mx.float32),
            mx.zeros((*batch, count, 1), dtype=mx.float32),
        ],
        axis=-1,
    )
    for step, shifts in enumerate(plan.shifts):
        rows = mx.array(plan.rows[:, step])
        previous = polynomials
        terms = [
            mx.take(rotations[..., axis], rows, axis=-1)[..., :, None]
            * mx.take(previous, mx.array(column), axis=-1)
            for axis, column in enumerate(shifts)
        ]
        polynomials = terms[0]
        for term in terms[1:]:
            polynomials = polynomials + term
    return polynomials[..., :count] * mx.array(plan.factors)


def steering_matrices(rotations: mx.array, order: int) -> mx.array:
    """Steering matrices that rotate the order-``order`` Riesz bank onto ``U``.

    Entry ``(n, m)`` multiplies component ``m`` of the axis-aligned bank to give
    component ``n`` of the bank steered by ``U``, following Eq. steeringMatrix3D
    with ``U = (u_1, ..., u_D)^T``. The manual formula is evaluated as the
    multinomial expansion of ``(U nu)^n`` over the monomials of ``nu`` and then
    rescaled by ``sqrt(m! / n!)``, which is algebraically identical and avoids
    enumerating the nested partition sums explicitly. Steering is a group
    representation: ``S(U_1 U_2) = S(U_1) S(U_2)`` and ``S(U) S(U)^T`` is the
    identity, so for order one the matrix is ``U`` itself.

    Args:
        rotations: Float32 array ``(..., D, D)`` whose rows ``u_1..u_D`` form the
            new analysis directions. Orthogonality is assumed, as produced by
            :func:`structure_tensor_rotations`; it is not re-verified per voxel.
        order: Riesz order ``L >= 1``.

    Returns:
        Float32 array ``(..., K, K)`` with ``K = binom(L + D - 1, D - 1)``, rows
        and columns in :func:`riesz_multi_indices` order.

    Raises:
        ValueError: If ``order`` is below one or above :data:`MAX_STEERING_ORDER`, or
            ``rotations`` is not a batch of 2x2 or 3x3 matrices.
    """
    if order < 1:
        raise ValueError("steering matrices require a Riesz order of at least one")
    if order > MAX_STEERING_ORDER:
        raise ValueError(
            f"Riesz steering is qualified up to order {MAX_STEERING_ORDER}; order {order} "
            "is numerically unstable in float32"
        )
    if (
        rotations.ndim < 2
        or rotations.shape[-1] != rotations.shape[-2]
        or rotations.shape[-1] not in (2, 3)
    ):
        raise ValueError("steering matrices require batches of 2x2 or 3x3 rotation matrices")
    return _steering(rotations, _steering_plan(rotations.shape[-1], order))


def _gaussian_taps(sigma_mm: float, spacing_mm: float) -> NDArray[np.float32]:
    """Unit-sum sampled Gaussian taps truncated at four standard deviations."""
    sigma = sigma_mm / spacing_mm
    radius = ceil(4.0 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) / sigma
    taps = np.exp(-0.5 * offsets * offsets)
    return (taps / taps.sum()).astype(np.float32)


def _smooth_gaussian(
    image: mx.array,
    spacing: tuple[float, ...],
    sigma_mm: float,
    padding: FilterPadding,
) -> mx.array:
    """Separable Gaussian smoothing with per-axis spacing in millimetres.

    The kernel is the unit-sum sampled Gaussian bounded by ``ceil(4 sigma / h)``
    voxels on each axis, applied by the manual's boundary-extension method, with
    zero as the ``CONSTANT`` extension (see :func:`structure_tensor_rotations`). The
    taps are symmetric, so MLX's cross-correlation is the requested convolution.
    """
    taps = [_gaussian_taps(sigma_mm, step) for step in spacing]
    widths = [(radius, radius) for radius in (ceil(4.0 * sigma_mm / step) for step in spacing)]
    padded = pad_image(image, widths, padding, 0.0)
    if image.ndim == 2:
        planes = padded[None, :, :, None]
        for axis, weights in enumerate(taps):
            shape = (1, weights.size, 1, 1) if axis == 0 else (1, 1, weights.size, 1)
            planes = mx.conv2d(planes, mx.array(weights.reshape(shape)))
        return planes[0, :, :, 0]
    volume = padded[None, :, :, :, None]
    for axis, weights in enumerate(taps):
        shape = [1, 1, 1, 1, 1]
        shape[axis + 1] = weights.size
        volume = mx.conv3d(volume, mx.array(weights.reshape(tuple(shape))))
    return volume[0, :, :, :, 0]


def _scaled_norm(vector: mx.array) -> mx.array:
    """Euclidean norm of ``(..., k)`` rows from a largest-component scaling.

    Forming ``sum(vector * vector)`` directly can overflow or underflow while the
    vector itself is representable. Dividing by the largest absolute component
    first keeps the squared sum near one; multiplying the scale back afterwards
    returns the norm with the original magnitude.
    """
    scale = mx.max(mx.abs(vector), axis=-1, keepdims=True)
    safe = mx.where(scale > 0, scale, mx.ones_like(scale))
    scaled = vector / safe
    return scale * mx.sqrt(mx.sum(scaled * scaled, axis=-1, keepdims=True))


def _normalised(vector: mx.array) -> mx.array:
    """Scale rows of unit-last-axis vectors to unit length without dividing by zero."""
    norm = _scaled_norm(vector)
    return vector / mx.maximum(norm, mx.array(_SMALLEST_NORM, dtype=mx.float32))


def _positive_leading(vector: mx.array) -> mx.array:
    """Make the largest-magnitude component positive, keeping the lowest index on ties."""
    leading = mx.take_along_axis(
        vector, mx.argmax(mx.abs(vector), axis=-1, keepdims=True), axis=-1,
    )
    return mx.where(leading < 0, -vector, vector)


def _cross(first: mx.array, second: mx.array) -> mx.array:
    """Row-wise cross product of ``(..., 3)`` vectors."""
    return mx.stack(
        [
            first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
            first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0],
        ],
        axis=-1,
    )


def _null_vector(matrix: mx.array, eigenvalue: mx.array) -> mx.array:
    """Unit null vector of ``matrix - eigenvalue I`` from the largest row cross product.

    Rows of a symmetric ``M = J - lambda I`` are orthogonal to every null vector, so
    the cross product of any independent row pair spans the null space. The pair
    with the largest cross-product norm is the best conditioned; ties keep the
    earliest pair. The caller passes a well-separated eigenvalue, so ``M`` has rank
    two unless the tensor is isotropic, where the zero vector is replaced by the
    identity frame.
    """
    first = mx.stack(
        [matrix[:, 0, 0] - eigenvalue, matrix[:, 0, 1], matrix[:, 0, 2]], axis=-1,
    )
    second = mx.stack(
        [matrix[:, 0, 1], matrix[:, 1, 1] - eigenvalue, matrix[:, 1, 2]], axis=-1,
    )
    third = mx.stack(
        [matrix[:, 0, 2], matrix[:, 1, 2], matrix[:, 2, 2] - eigenvalue], axis=-1,
    )
    candidates = (_cross(first, second), _cross(first, third), _cross(second, third))
    # The scaled norm keeps the comparison finite even for a tensor whose entries
    # are large enough that the raw cross-product squares would overflow.
    norms = [_scaled_norm(candidate)[:, 0] for candidate in candidates]
    use_second = (norms[1] > norms[0]) & (norms[1] >= norms[2])
    use_third = (norms[2] > norms[0]) & (norms[2] > norms[1])
    chosen = mx.where(
        use_second[:, None],
        candidates[1],
        mx.where(use_third[:, None], candidates[2], candidates[0]),
    )
    return _normalised(chosen)


def _complement(vector: mx.array) -> mx.array:
    """Deterministic unit vector orthogonal to each ``(..., 3)`` row of ``vector``.

    The smallest-magnitude component names the coordinate axis used for the cross
    product, which keeps the construction well conditioned and reproducible.
    """
    axis = mx.argmin(mx.abs(vector), axis=-1)
    basis = mx.take(mx.array(np.eye(3, dtype=np.float32)), axis, axis=0)
    return _normalised(_cross(vector, basis))


def _bilinear(matrix: mx.array, left: mx.array, right: mx.array) -> mx.array:
    """Batched ``left^T J right`` for ``(n, 3)`` vectors and ``(n, 3, 3)`` tensors."""
    # A broadcast product: MLX's ``@`` treats a (n, D) operand as a plain matrix and
    # would expand the batch.
    product = mx.sum(left[..., :, None] * matrix, axis=-2)
    return mx.sum(product * right, axis=-1)


def _sorted_by_rayleigh(matrix: mx.array, rows: mx.array) -> mx.array:
    """Sort the three orthonormal rows by descending Rayleigh quotient.

    Eigenvectors are ordered by eigenvalue, but a repeated pair is only resolved to
    the accuracy of the closed-form eigenvalues. Sorting by the quotient of the
    actual tensor enforces the documented ordering even inside a degenerate plane
    and is a no-op when the spectrum is well separated.
    """
    first, second, third = rows[:, 0], rows[:, 1], rows[:, 2]

    q_first, q_second, q_third = (
        _bilinear(matrix, vector, vector) for vector in (first, second, third)
    )
    exchange = (q_second > q_first)[:, None]
    first, second = mx.where(exchange, second, first), mx.where(exchange, first, second)
    q_first, q_second = mx.maximum(q_first, q_second), mx.minimum(q_first, q_second)
    exchange = (q_third > q_second)[:, None]
    second, third = mx.where(exchange, third, second), mx.where(exchange, second, third)
    q_second, q_third = mx.maximum(q_second, q_third), mx.minimum(q_second, q_third)
    exchange = (q_second > q_first)[:, None]
    first, second = mx.where(exchange, second, first), mx.where(exchange, first, second)
    return mx.stack([first, second, third], axis=-2)


def _rotations_3x3(matrix: mx.array) -> mx.array:
    """Closed-form eigenvectors of a batch of symmetric 3x3 matrices.

    Eigenvalues use the trigonometric solution of the characteristic cubic in the
    shifted and scaled form of Smith's method, which needs no iteration and avoids
    the CPU-stream ``mx.linalg`` primitives. The arccosine resolves a close pair
    only to about ``sqrt(eps)`` of the spread, but the eigenvalue farther from the
    middle one sits where the cosine is stationary and keeps float32 accuracy.
    Its eigenvector, a cross product of rows of ``J - lambda I``, is therefore well
    conditioned; its gap to both others is at least half the spread. The other two
    eigenvectors diagonalise the restriction of ``J`` to the orthogonal plane with
    the closed-form 2x2 angle ``atan2(2 b, a - c) / 2``, whose accuracy depends on
    the tensor entries rather than on the trigonometric eigenvalues (Eberly, "A
    Robust Eigensolver for 3 x 3 Symmetric Matrices", 2014).
    """
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 0, 2]
    d = matrix[:, 1, 1]
    e = matrix[:, 1, 2]
    f = matrix[:, 2, 2]
    mean = (a + d + f) / 3
    off_diagonal = b * b + c * c + e * e
    spread = (
        (a - mean) * (a - mean)
        + (d - mean) * (d - mean)
        + (f - mean) * (f - mean)
        + 2 * off_diagonal
    )
    radius = mx.sqrt(mx.maximum(spread, 0.0) / 6)
    safe_radius = mx.where(radius > 0, radius, mx.ones_like(radius))
    shifted_a = (a - mean) / safe_radius
    shifted_b = b / safe_radius
    shifted_c = c / safe_radius
    shifted_d = (d - mean) / safe_radius
    shifted_e = e / safe_radius
    shifted_f = (f - mean) / safe_radius
    # det(J - mean I) / radius**3 lies in [-2, 2]; the half-angle cosine argument
    # is its halved and clamped value.
    determinant = (
        shifted_a * (shifted_d * shifted_f - shifted_e * shifted_e)
        - shifted_b * (shifted_b * shifted_f - shifted_e * shifted_c)
        + shifted_c * (shifted_b * shifted_e - shifted_d * shifted_c)
    )
    angle = mx.arccos(mx.clip(determinant / 2, -1.0, 1.0)) / 3
    largest = mean + 2 * radius * mx.cos(angle)
    smallest = mean + 2 * radius * mx.cos(angle + 2 * pi / 3)
    middle = 3 * mean - largest - smallest

    top_isolated = (largest - middle) >= (middle - smallest)
    isolated = _null_vector(matrix, mx.where(top_isolated, largest, smallest))
    # Orthonormal basis (p, q) of the plane holding the remaining eigenvalue pair.
    p = _complement(isolated)
    q = _cross(isolated, p)
    plane_a = _bilinear(matrix, p, p)
    plane_b = _bilinear(matrix, p, q)
    plane_c = _bilinear(matrix, q, q)
    # The larger in-plane eigenvalue's eigenvector; atan2(0, 0) = 0 keeps p for an
    # exactly repeated pair.
    plane_angle = mx.arctan2(2 * plane_b, plane_a - plane_c) / 2
    upper = mx.cos(plane_angle)[:, None] * p + mx.sin(plane_angle)[:, None] * q
    lower = _normalised(_cross(isolated, upper))
    upper = _normalised(upper)
    top = top_isolated[:, None]
    rows = _sorted_by_rayleigh(matrix, mx.stack([
        mx.where(top, isolated, upper),
        mx.where(top, upper, lower),
        mx.where(top, lower, isolated),
    ], axis=-2))
    first = _positive_leading(rows[:, 0])
    second = _positive_leading(rows[:, 1])
    rotation = mx.stack([first, second, _cross(first, second)], axis=-2)

    magnitude = mx.abs(largest) + mx.abs(middle) + mx.abs(smallest)
    isotropic = (largest - smallest) <= _ISOTROPIC_TOLERANCE * magnitude
    identity = mx.zeros_like(rotation) + mx.array(np.eye(3, dtype=np.float32))
    return mx.where(isotropic[:, None, None], identity, rotation)


def _rotations_2x2(matrix: mx.array) -> mx.array:
    """Closed-form eigenvectors of a batch of symmetric 2x2 matrices.

    The eigenvalues are ``(a + c) / 2 +- sqrt(((a - c) / 2)**2 + b**2)``. The
    eigenvector is read from the better-conditioned row of ``J - lambda I`` as its
    perpendicular, and the second row is a right-handed quarter turn of the first.
    """
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 1, 1]
    half = (a - c) / 2
    delta = mx.sqrt(half * half + b * b)
    largest = (a + c) / 2 + delta
    first_row = mx.stack([a - largest, b], axis=-1)
    second_row = mx.stack([b, c - largest], axis=-1)
    from_first = mx.stack([b, largest - a], axis=-1)
    from_second = mx.stack([c - largest, -b], axis=-1)
    use_first = mx.sum(first_row * first_row, axis=-1) >= mx.sum(
        second_row * second_row, axis=-1,
    )
    first = _positive_leading(
        _normalised(mx.where(use_first[:, None], from_first, from_second)),
    )
    rotation = mx.stack([first, mx.stack([-first[:, 1], first[:, 0]], axis=-1)], axis=-2)
    isotropic = delta <= _ISOTROPIC_TOLERANCE * (mx.abs(a) + mx.abs(b) + mx.abs(c))
    identity = mx.zeros_like(rotation) + mx.array(np.eye(2, dtype=np.float32))
    return mx.where(isotropic[:, None, None], identity, rotation)


def symmetric_rotations(matrix: mx.array) -> mx.array:
    """Rotation field whose rows are the descending-eigenvalue eigenvectors of ``J``.

    Closed-form symmetric eigendecomposition of batched 2x2 or 3x3 tensors, used by
    :func:`structure_tensor_rotations` and exposed for callers that already hold a
    structure tensor. No iterative solver and no CPU-stream ``mx.linalg`` call is
    involved. Each tensor is divided by its largest absolute entry before the
    solve; that positive scale leaves the eigenvectors unchanged and keeps the
    closed-form products clear of float32 overflow and underflow. Rows are ordered
    by descending Rayleigh quotient, each of ``u_1`` and ``u_2`` has its
    largest-magnitude component positive, and ``u_3`` closes the right-handed
    frame. Spectra that are degenerate within float32 resolution use the
    deterministic completions documented in the module overview.

    Args:
        matrix: Float32 symmetric tensors with shape ``(n, D, D)``.

    Returns:
        Float32 ``(n, D, D)`` batches with orthonormal rows and ``det = +1``.

    Raises:
        ValueError: If ``matrix`` is not a batch of symmetric 2x2 or 3x3 tensors.
    """
    if (
        matrix.ndim != 3
        or matrix.shape[1] != matrix.shape[2]
        or matrix.shape[1] not in (2, 3)
    ):
        raise ValueError("structure tensors must be batched 2x2 or 3x3 matrices")
    ndim = matrix.shape[1]
    if matrix.shape[0] == 0:
        return mx.zeros((0, ndim, ndim), dtype=mx.float32)
    solver = _rotations_3x3 if ndim == 3 else _rotations_2x2
    # One positive scale per voxel preserves that voxel's eigenvectors. A zero
    # tensor keeps its zero matrix (scale one), which the solver maps to identity.
    magnitude = mx.max(
        mx.abs(mx.reshape(matrix, (matrix.shape[0], ndim * ndim))),
        axis=1, keepdims=True,
    )
    scale = mx.reshape(
        mx.where(magnitude > 0, magnitude, mx.ones_like(magnitude)),
        (-1, 1, 1),
    )
    normalised = matrix / scale
    blocks = [
        solver(normalised[start:start + _EIGEN_BLOCK_VOXELS])
        for start in range(0, matrix.shape[0], _EIGEN_BLOCK_VOXELS)
    ]
    return blocks[0] if len(blocks) == 1 else mx.concatenate(blocks, axis=0)


def _validated_responses(
    first_order: Sequence[mx.array],
    spacing: tuple[float, ...],
    tensor_sigma: float,
) -> tuple[int, ...]:
    """Validate the first-order Riesz bank and its physical parameters."""
    if len(first_order) not in (2, 3):
        raise ValueError("Riesz alignment requires the two or three first-order responses")
    ndim = len(first_order)
    if len(spacing) != ndim:
        raise ValueError("spacing must provide one value per array axis")
    for step in spacing:
        if not isfinite(step) or step <= 0:
            raise ValueError("spacing values must be positive and finite")
    if not isfinite(tensor_sigma) or tensor_sigma <= 0:
        raise ValueError("tensor_sigma must be positive and finite")
    shape = tuple(first_order[0].shape)
    if len(shape) != ndim:
        raise ValueError("first-order Riesz responses must have one axis per rotation axis")
    for response in first_order[1:]:
        if tuple(response.shape) != shape:
            raise ValueError("first-order Riesz responses must share one spatial shape")
    return shape


def structure_tensor_rotations(
    first_order: Sequence[mx.array],
    spacing: tuple[float, ...],
    tensor_sigma: float,
    padding: FilterPadding,
) -> mx.array:
    """Per-voxel rotation whose rows are the structure-tensor eigenvectors.

    The regularised structure tensor of Eq. structureTensor has entries
    ``J[m, n]`` equal to the element-wise products ``R_m R_n`` of the first-order
    Riesz responses convolved with a Gaussian window of ``tensor_sigma``
    millimetres on every axis. That window sets the scale of the orientation
    estimate independently of the Riesz analysis scale. Rows of the returned
    rotation are sorted by descending eigenvalue, so ``u_1`` maximises the energy
    of ``R_1``.

    The window extends each product ``R_m R_n`` by ``padding``. ``CONSTANT``
    extends it by zero: first-order Riesz filters annihilate constants, so the
    responses of a constant image extension vanish away from the image. Every
    extended sample is then an outer product of one response vector, or zero, and
    the nonnegative Gaussian window keeps ``J`` positive semidefinite. Reusing a
    signed image extension value would add a negative constant to the diagonal.

    Before forming the products, the whole first-order bank is divided by one
    common maximum-magnitude scale. That positive factor multiplies ``J`` by its
    square and therefore preserves every eigenvector, while keeping the outer
    products representable for uniformly scaled images near either end of the
    float32 range. Products below ``2**-126`` of the squared bank maximum flush to
    zero; that is accepted while a product in the voxel's window survives, and a
    responding voxel whose entire smoothed tensor underflows is rejected.
    Scaling each response separately would instead reweight directions and voxels.

    Args:
        first_order: The ``D`` first-order Riesz responses of one base-filtered
            image, in :func:`riesz_multi_indices` order, all with one spatial
            shape whose ``D`` axes match the array-axis order.
        spacing: Voxel spacing in millimetres per array axis.
        tensor_sigma: Standard deviation of the orientation window in millimetres.
        padding: Boundary-extension method for the Gaussian window.

    Returns:
        Float32 array ``(..., D, D)`` with the spatial shape followed by the
        rotation whose rows are ``u_1..u_D`` and whose determinant is ``+1``.
        Isotropic or zero tensors return the identity rotation.

    Raises:
        ValueError: If the response bank, spacing or ``tensor_sigma`` are
            inconsistent, or a voxel with a nonzero response has a smoothed
            tensor that underflows to zero in float32.
    """
    responses = tuple(first_order)
    shape = _validated_responses(responses, spacing, tensor_sigma)
    ndim = len(responses)
    # One common scale across the bank leaves J proportional to the unscaled tensor
    # (by the square of the scale) and keeps R_m R_n inside the float32 range.
    amplitude = mx.max(mx.abs(responses[0]))
    for response in responses[1:]:
        amplitude = mx.maximum(amplitude, mx.max(mx.abs(response)))
    safe_amplitude = mx.where(amplitude > 0, amplitude, mx.ones_like(amplitude))
    scaled = tuple(response / safe_amplitude for response in responses)
    # Only the upper triangle is convolved; the tensor is symmetric by construction,
    # so both halves share one array and never disagree through rounding.
    smoothed = {
        (row, column): _smooth_gaussian(
            scaled[row] * scaled[column], spacing, tensor_sigma, padding,
        )
        for row in range(ndim)
        for column in range(row, ndim)
    }
    entries = [
        [smoothed[min(row, column), max(row, column)] for column in range(ndim)]
        for row in range(ndim)
    ]
    # Products below 2**-126 of the squared bank maximum flush under the common scale.
    # Ordinary banks (for example near the zero crossings of a wave) keep surviving
    # products in every window. A voxel with a nonzero response whose whole smoothed
    # tensor is zero has lost its orientation; refuse it rather than use the identity.
    trace = smoothed[0, 0]
    for axis in range(1, ndim):
        trace = trace + smoothed[axis, axis]
    responding = responses[0] != 0
    for response in responses[1:]:
        responding = responding | (response != 0)
    if bool(mx.any(responding & (trace == 0)).item()):
        raise ValueError("Riesz structure-tensor products exceed float32 dynamic range")
    tensor = mx.stack([mx.stack(line, axis=-1) for line in entries], axis=-2)
    rotations = symmetric_rotations(mx.reshape(tensor, (-1, ndim, ndim)))
    return mx.reshape(rotations, (*shape, ndim, ndim))


def _alignment_block(plan: _SteeringPlan) -> int:
    """Voxel count that keeps the steering expansion within the element budget."""
    return max(1, _STEERING_BLOCK_FLOATS // max(1, plan.elements_per_voxel))


def align_riesz(
    components: Sequence[mx.array],
    first_order: Sequence[mx.array],
    order: int,
    spacing: tuple[float, ...],
    tensor_sigma: float,
    padding: FilterPadding,
) -> list[mx.array]:
    """Align the order-``order`` Riesz bank with the local structure-tensor frame.

    Implements Eq. steerability, ``R{f}_U = S_U R{f}``, at every voxel: response
    ``n`` is the linear combination of the axis-aligned responses weighted by row
    ``n`` of the steering matrix, so no further convolution is required. Voxels are
    processed in bounded blocks because the steering matrix is ``K x K`` each.

    Args:
        components: All ``K`` order-``order`` Riesz responses of the base-filtered
            image, in :func:`riesz_multi_indices` order.
        first_order: The ``D`` first-order Riesz responses of the same base-filtered
            image, used for the structure tensor.
        order: Riesz order ``L >= 1`` of ``components``.
        spacing: Voxel spacing in millimetres per array axis.
        tensor_sigma: Standard deviation of the orientation window in millimetres.
        padding: Boundary-extension method for the orientation window; see
            :func:`structure_tensor_rotations`.

    Returns:
        List of ``K`` float32 aligned response maps with the input spatial shape,
        in the same order as ``components``.

    Raises:
        ValueError: If ``order`` is below one, the component count does not match
            ``order``, or shapes, spacing or ``tensor_sigma`` are inconsistent.
    """
    if order < 1:
        raise ValueError("Riesz alignment requires an order of at least one")
    responses = tuple(first_order)
    shape = _validated_responses(responses, spacing, tensor_sigma)
    ndim = len(responses)
    indices = riesz_multi_indices(ndim, order)
    values = tuple(components)
    if len(values) != len(indices):
        raise ValueError(
            f"order-{order} alignment in {ndim} dimensions needs {len(indices)} "
            f"components, received {len(values)}"
        )
    for value in values:
        if tuple(value.shape) != shape:
            raise ValueError("Riesz components must share the first-order spatial shape")
    rotations = structure_tensor_rotations(responses, spacing, tensor_sigma, padding)
    plan = _steering_plan(ndim, order)
    flat_rotations = mx.reshape(rotations, (-1, ndim, ndim))
    flat_values = [mx.reshape(value, (-1,)) for value in values]
    block = _alignment_block(plan)
    pieces: list[list[mx.array]] = [[] for _ in indices]
    for start in range(0, flat_rotations.shape[0], block):
        stop = start + block
        matrices = _steering(flat_rotations[start:stop], plan)
        axis_aligned = mx.stack([value[start:stop] for value in flat_values], axis=-1)
        aligned = (matrices @ axis_aligned[..., None])[..., 0]
        for index in range(len(indices)):
            pieces[index].append(aligned[:, index])
    return [mx.reshape(mx.concatenate(part), shape) for part in pieces]
