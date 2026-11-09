"""Independent checks of Riesz steering and structure-tensor alignment.

Every reference here is rebuilt from the manual's equations in float64 NumPy: the
Riesz multipliers on the DFT grid, the exact 90-degree grid rotations, the sampled
Gaussian window, and ``numpy.linalg.eigh`` for the structure tensor. Production
code is only exercised through its public entry points.
"""

import dataclasses
from math import ceil, factorial, prod, sqrt

import numpy as np
import pytest
from conftest import requires_fft
from numpy.typing import NDArray

from mlxradiomics.config.filtering import FilterPadding, RieszBase
from mlxradiomics.config.transforms import ImageType, RieszSettings, TransformSettings
from mlxradiomics.filters.riesz_steering import (
    align_riesz,
    steering_matrices,
    structure_tensor_rotations,
    symmetric_rotations,
)
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import mx

_SPECTRA: dict[str, dict[int, tuple[float, ...]]] = {
    "distinct": {2: (2.0, -1.0), 3: (3.0, 0.5, -1.5)},
    "near": {2: (2.0, 1.99), 3: (2.0, 1.99, 0.3)},
    "close": {2: (1.0009, 1.0), 3: (1.0009, 1.0, 0.0)},
    "close_low": {2: (1.0, 0.9991), 3: (1.0, 0.0009, 0.0)},
    "repeated": {2: (2.0, 2.0), 3: (2.0, 2.0, -1.0)},
    "isotropic": {2: (2.0, 2.0), 3: (2.0, 2.0, 2.0)},
    "zero": {2: (0.0, 0.0), 3: (0.0, 0.0, 0.0)},
}


def _multi_indices(ndim: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Oracle enumeration of the Riesz multi-indices in descending lexicographic order."""
    if ndim == 1:
        return ((order,),)
    return tuple(
        (head,) + tail
        for head in range(order, -1, -1)
        for tail in _multi_indices(ndim - 1, order - head)
    )


def _rotation(ndim: int, seed: int) -> NDArray[np.float64]:
    """Deterministic random proper rotation from a QR factorisation."""
    generator = np.random.default_rng(seed)
    factor, triangular = np.linalg.qr(generator.normal(size=(ndim, ndim)))
    factor = factor * np.sign(np.diag(triangular))
    if np.linalg.det(factor) < 0:
        factor[:, 0] = -factor[:, 0]
    return np.asarray(factor, dtype=np.float64)


def _riesz_fft(
    image: NDArray[np.float64], order: int, rotation: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    """Riesz responses of the periodic image on the DFT grid (manual Eq. 4.9).

    With ``rotation`` the multiplier uses ``(U nu)^n``, i.e. the kernel rotated so
    that component ``n`` points along row ``n`` of ``U``. Written as
    ``(a . nu)^n`` by the multinomial expansion this is exactly the steered bank.
    """
    ndim = image.ndim
    grids = np.meshgrid(*(np.fft.fftfreq(size) * size for size in image.shape), indexing="ij")
    frequency = np.stack(grids)
    rows = frequency if rotation is None else np.einsum("ij,j...->i...", rotation, frequency)
    norm = np.sqrt(np.sum(frequency * frequency, axis=0))
    safe = np.where(norm == 0.0, 1.0, norm)
    spectrum = np.fft.fftn(image)
    responses: list[NDArray[np.float64]] = []
    for index in _multi_indices(ndim, order):
        multiplier = (-1j) ** order * np.sqrt(
            factorial(order) / prod(factorial(item) for item in index)
        )
        for axis, power in enumerate(index):
            multiplier = multiplier * rows[axis] ** power
        responses.append(np.real(np.fft.ifftn(multiplier / safe**order * spectrum)))
    return responses


def _warp(array: NDArray[np.float64], rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sample ``array`` at ``U x`` for a signed permutation ``U`` (exact on the grid)."""
    offsets = np.rint(rotation).astype(np.int64)
    samples = np.einsum("ij,j...->i...", offsets, np.indices(array.shape))
    samples = samples % np.array(array.shape, dtype=np.int64).reshape((-1,) + (1,) * array.ndim)
    return np.asarray(array.reshape(-1)[np.ravel_multi_index(tuple(samples), array.shape)],
                      dtype=np.float64)


def _taps(sigma_mm: float, spacing_mm: float) -> NDArray[np.float64]:
    sigma = sigma_mm / spacing_mm
    radius = ceil(4.0 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) / sigma
    weights = np.exp(-0.5 * offsets * offsets)
    return weights / weights.sum()


def _boundary(array: NDArray[np.float64], widths: list[tuple[int, int]],
              padding: FilterPadding) -> NDArray[np.float64]:
    if padding is FilterPadding.CONSTANT:
        return np.pad(array, widths, mode="constant", constant_values=0.0)
    if padding is FilterPadding.NEAREST:
        return np.pad(array, widths, mode="edge")
    if padding is FilterPadding.MIRROR:
        return np.pad(array, widths, mode="symmetric")
    return np.pad(array, widths, mode="wrap")


def _smooth(
    array: NDArray[np.float64],
    spacing: tuple[float, ...],
    sigma: float,
    padding: FilterPadding,
) -> NDArray[np.float64]:
    """Separable Gaussian window written as explicit weighted shifts.

    The taps are symmetric, so the moving sum is both the correlation and the
    convolution of the manual's Eq. structureTensor.
    """
    result = np.array(array, dtype=np.float64)
    for axis, step in enumerate(spacing):
        weights = _taps(sigma, step)
        radius = weights.size // 2
        widths = [(radius, radius) if index == axis else (0, 0) for index in range(result.ndim)]
        padded = _boundary(result, widths, padding)
        length = padded.shape[axis] - 2 * radius
        smoothed = np.zeros(
            tuple(length if index == axis else size for index, size in enumerate(padded.shape)),
            dtype=np.float64,
        )
        for offset, weight in enumerate(weights):
            window = [slice(None)] * padded.ndim
            window[axis] = slice(offset, offset + length)
            smoothed += weight * padded[tuple(window)]
        result = smoothed
    return result


def _structure_tensor(
    responses: list[NDArray[np.float64]],
    spacing: tuple[float, ...],
    sigma: float,
    padding: FilterPadding,
) -> NDArray[np.float64]:
    count = len(responses)
    tensor = np.empty((*responses[0].shape, count, count), dtype=np.float64)
    for row in range(count):
        for column in range(row, count):
            value = _smooth(responses[row] * responses[column], spacing, sigma, padding)
            tensor[..., row, column] = tensor[..., column, row] = value
    return tensor


@pytest.mark.parametrize("ndim", [2, 3])
def test_order_one_steering_matrix_is_the_rotation(ndim: int) -> None:
    rotation = mx.array(_rotation(ndim, seed=ndim).astype(np.float32))
    with mx.stream(mx.gpu):
        matrices = np.asarray(steering_matrices(rotation, 1), dtype=np.float64)
    # Component n of the steered bank is the Riesz transform along row n of U.
    np.testing.assert_array_equal(matrices, np.asarray(rotation, dtype=np.float64))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 16])
def test_steering_matrix_is_orthogonal(ndim: int, order: int) -> None:
    rotations = mx.array(
        np.stack([_rotation(ndim, seed=100 * ndim + order + step) for step in range(4)]).astype(
            np.float32
        )
    )
    with mx.stream(mx.gpu):
        matrices = np.asarray(steering_matrices(rotations, order), dtype=np.float64)
    count = len(_multi_indices(ndim, order))
    assert matrices.shape == (4, count, count)
    gram = matrices @ matrices.transpose(0, 2, 1)
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(count), gram.shape), rtol=0, atol=1e-5)


@pytest.mark.parametrize("ndim,shape", [(2, (7, 6)), (3, (5, 6, 4))])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_steering_matches_an_independent_rotated_kernel(
    ndim: int, shape: tuple[int, ...], order: int,
) -> None:
    generator = np.random.default_rng(37 + ndim + order)
    image = generator.normal(size=shape)
    rotation = _rotation(ndim, seed=5 + order)
    components = np.stack(_riesz_fft(image, order))
    expected = np.stack(_riesz_fft(image, order, rotation))
    with mx.stream(mx.gpu):
        matrices = np.asarray(
            steering_matrices(mx.array(rotation.astype(np.float32)), order), dtype=np.float64,
        )
    # The steered bank is the axis-aligned bank evaluated with the (U nu)^n kernel.
    actual = np.tensordot(matrices, components, axes=([1], [0]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize(
    "rotation",
    [
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    ],
    ids=["quarter-turn", "half-turn", "cycle"],
)
@pytest.mark.parametrize("order", [1, 2])
def test_steering_rotates_responses_with_the_image(
    rotation: NDArray[np.float64], order: int,
) -> None:
    # Odd sizes avoid the Nyquist plane, where a signed grid permutation fixes the
    # frequency and the discrete Riesz transform is not exactly equivariant.
    image = np.random.default_rng(19 + order).normal(size=(7,) * rotation.shape[0])
    moved = _warp(image, rotation)
    original = np.stack(_riesz_fft(image, order))
    moved_components = np.stack(_riesz_fft(moved, order))
    with mx.stream(mx.gpu):
        matrices = np.asarray(
            steering_matrices(mx.array(rotation.astype(np.float32)), order), dtype=np.float64,
        )
    # S_U R{f_U} = R{f}_U: steering replaces re-analysing the rotated image.
    actual = np.tensordot(matrices, moved_components, axes=([1], [0]))
    expected = np.stack([_warp(value, rotation) for value in original])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "spectrum", ["distinct", "near", "close", "close_low", "repeated", "isotropic", "zero"],
)
def test_closed_form_eigenvectors_match_numpy_eigh(ndim: int, spectrum: str) -> None:
    eigenvalues = np.array(_SPECTRA[spectrum][ndim], dtype=np.float64)
    frames = np.stack([_rotation(ndim, seed=200 + step) for step in range(16)])
    tensors = np.einsum("nij,j,nkj->nik", frames, eigenvalues, frames)
    with mx.stream(mx.gpu):
        solved = np.asarray(symmetric_rotations(mx.array(tensors.astype(np.float32))),
                             dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))

    orthonormal = np.einsum("nij,nkj->nik", solved, solved)
    np.testing.assert_allclose(orthonormal, np.broadcast_to(np.eye(ndim), orthonormal.shape),
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(solved), 1.0, rtol=0, atol=1e-5)
    rebuilt = np.einsum("nij,njk,nlk->nil", solved, tensors, solved)
    diagonal = np.einsum("nii->ni", rebuilt)
    off_diagonal = rebuilt - diagonal[:, :, None] * np.eye(ndim)
    np.testing.assert_allclose(off_diagonal, 0.0, rtol=0, atol=2e-3 * scale)
    # Rows are ordered by descending Rayleigh quotient, matching descending eigenvalues.
    assert np.all(np.diff(diagonal, axis=1) <= 2e-3 * scale)
    reference = np.linalg.eigvalsh(tensors)[:, ::-1]
    np.testing.assert_allclose(diagonal, reference, rtol=0, atol=2e-3 * scale)

    if spectrum in ("distinct", "near", "close", "close_low"):
        # Separated eigenvalues, including pairs 9e-4 apart, have unique eigenvectors.
        # The reference solves the float32-rounded tensors the solver receives;
        # eigh returns eigenvectors as columns, so row n is eigenvector n.
        rounded = tensors.astype(np.float32).astype(np.float64)
        reference_vectors = np.linalg.eigh(rounded)[1][:, :, ::-1].transpose(0, 2, 1)
        overlap = np.abs(np.sum(solved * reference_vectors, axis=-1))
        np.testing.assert_allclose(overlap, 1.0, rtol=0, atol=2e-5)
    if spectrum in ("isotropic", "zero"):
        np.testing.assert_array_equal(solved, np.broadcast_to(np.eye(ndim), solved.shape))


@pytest.mark.parametrize("scale", [1e-12, 1e10])
def test_well_conditioned_tensor_frame_survives_positive_scaling(scale: float) -> None:
    # A positive scale must not change eigenvectors.
    eigenvalues = np.array([3.0, 0.5, -1.5], dtype=np.float64)
    frames = np.stack([_rotation(3, seed=300 + step) for step in range(8)])
    tensors = np.einsum("nij,j,nkj->nik", frames, eigenvalues, frames)
    scaled = (scale * tensors).astype(np.float32)
    with mx.stream(mx.gpu):
        solved = np.asarray(symmetric_rotations(mx.array(scaled)), dtype=np.float64)

    orthonormal = solved @ solved.transpose(0, 2, 1)
    np.testing.assert_allclose(
        orthonormal, np.broadcast_to(np.eye(3), orthonormal.shape), rtol=0, atol=1e-5,
    )
    np.testing.assert_allclose(np.linalg.det(solved), 1.0, rtol=0, atol=1e-5)
    # eigh returns eigenvectors as columns; the frame stores them as rows, in
    # descending eigenvalue order. The reference receives the same float32 data.
    reference = np.linalg.eigh(scaled.astype(np.float64))[1][:, :, ::-1].transpose(0, 2, 1)
    overlap = np.abs(np.sum(solved * reference, axis=-1))
    np.testing.assert_allclose(overlap, 1.0, rtol=0, atol=2e-5)


@pytest.mark.parametrize("ndim", [2, 3])
def test_empty_tensor_batch_has_an_empty_rotation_field(ndim: int) -> None:
    # Zero tensors have zero frames, so the field is (0, D, D).
    with mx.stream(mx.gpu):
        solved = symmetric_rotations(mx.zeros((0, ndim, ndim), dtype=mx.float32))
        host = np.asarray(solved, dtype=np.float64)
    assert solved.dtype == mx.float32
    assert host.shape == (0, ndim, ndim)


def test_close_principal_pair_keeps_its_eigenvector() -> None:
    # Eigenvalues 1.0009, 1 and 0: the principal direction is (1, 1, 0) / sqrt(2),
    # not an arbitrary vector of the near-repeated plane.
    tensor = np.array([[1.00045, 0.00045, 0.0], [0.00045, 1.00045, 0.0], [0.0, 0.0, 0.0]])
    with mx.stream(mx.gpu):
        solved = np.asarray(
            symmetric_rotations(mx.array(tensor[None].astype(np.float32))), dtype=np.float64,
        )[0]
    np.testing.assert_allclose(solved[0], np.array([1.0, 1.0, 0.0]) / sqrt(2.0),
                               rtol=0, atol=1e-3)


@pytest.mark.parametrize("padding", list(FilterPadding))
def test_structure_tensor_diagonalises_the_oracle_tensor(padding: FilterPadding) -> None:
    image = np.random.default_rng(41).normal(size=(6, 7, 5))
    spacing = (0.8, 1.1, 1.7)
    sigma = 1.4
    responses = _riesz_fft(image, 1)
    expected = _structure_tensor(responses, spacing, sigma, padding)
    eigenvalues = np.linalg.eigvalsh(expected)
    with mx.stream(mx.gpu):
        rotations = np.asarray(
            structure_tensor_rotations(
                [mx.array(value.astype(np.float32)) for value in responses],
                spacing, sigma, padding,
            ),
            dtype=np.float64,
        )
    assert rotations.shape == (6, 7, 5, 3, 3)
    rebuilt = np.einsum("...ij,...jk,...lk->...il", rotations, expected, rotations)
    diagonal = np.einsum("...ii->...i", rebuilt)
    off_diagonal = rebuilt * (1.0 - np.eye(3))
    scale = float(np.max(np.abs(eigenvalues)))
    np.testing.assert_allclose(off_diagonal, 0.0, rtol=0, atol=5e-5 * scale)
    np.testing.assert_allclose(diagonal, eigenvalues[..., ::-1], rtol=0, atol=5e-4 * scale)


def test_first_order_alignment_smooths_to_the_top_eigenvalue() -> None:
    # A wave packet along one direction has a locally constant orientation, which is
    # the regime where g * (u_1 . R)^2 equals u_1^T J u_1 = lambda_1 exactly.
    wave = np.array([1.0, 2.0, 1.0])
    direction = wave / np.linalg.norm(wave)
    phase = sum(
        (2 * np.pi / 16) * wave[axis] * np.indices((16, 16, 16))[axis] for axis in range(3)
    )
    image = np.cos(phase) + 0.6 * np.cos(2 * phase + 0.4) + 0.3 * np.cos(3 * phase + 1.1)
    spacing = (1.0, 1.0, 1.0)
    sigma = 2.0
    responses = _riesz_fft(image, 1)
    stacked = np.stack(responses)
    # Every Riesz component of a wave packet along d is d_n times one scalar field.
    np.testing.assert_allclose(stacked, direction[:, None, None, None] * stacked[0] / direction[0],
                               rtol=0, atol=1e-12)
    tensor = _structure_tensor(responses, spacing, sigma, FilterPadding.PERIODIC)
    top_eigenvalue = np.linalg.eigvalsh(tensor)[..., -1]
    with mx.stream(mx.gpu):
        aligned = [
            np.asarray(value, dtype=np.float64)
            for value in align_riesz(
                [mx.array(response.astype(np.float32)) for response in responses],
                [mx.array(response.astype(np.float32)) for response in responses],
                1, spacing, sigma, FilterPadding.PERIODIC,
            )
        ]
    smoothed = _smooth(aligned[0] ** 2, spacing, sigma, FilterPadding.PERIODIC)
    np.testing.assert_allclose(smoothed, top_eigenvalue, rtol=0,
                               atol=1e-4 * float(np.max(np.abs(top_eigenvalue))))
    energy = float(np.sum(stacked**2))
    assert float(np.sum(aligned[1] ** 2)) / energy < 1e-6
    assert float(np.sum(aligned[2] ** 2)) / energy < 1e-6


def test_planar_wave_alignment_concentrates_energy_in_the_first_component() -> None:
    wave = np.array([1.0, 2.0, 3.0])
    direction = wave / np.linalg.norm(wave)
    indices = np.indices((16, 16, 16))
    phase = sum((2 * np.pi / 16) * wave[axis] * indices[axis] for axis in range(3))
    first_order = _riesz_fft(np.cos(phase), 1)
    spacing = (1.0, 1.0, 1.0)
    with mx.stream(mx.gpu):
        rotations = np.asarray(
            structure_tensor_rotations(
                [mx.array(response.astype(np.float32)) for response in first_order],
                spacing, 1.0, FilterPadding.PERIODIC,
            ),
            dtype=np.float64,
        )
    # The single plane wave makes J rank one with u_1 = +-d wherever it has energy.
    stacked = np.stack(first_order)
    active = np.sum(stacked**2, axis=0) > 1e-6 * float(np.sum(stacked**2))
    cosine = np.sum(rotations[..., 0, :] * direction, axis=-1)
    np.testing.assert_allclose(np.abs(cosine[active]), 1.0, rtol=0, atol=1e-5)
    for order in (1, 2):
        components = _riesz_fft(np.cos(phase), order)
        with mx.stream(mx.gpu):
            aligned = [
                np.asarray(value, dtype=np.float64)
                for value in align_riesz(
                    [mx.array(response.astype(np.float32)) for response in components],
                    [mx.array(response.astype(np.float32)) for response in first_order],
                    order, spacing, 1.0, FilterPadding.PERIODIC,
                )
            ]
        # Every order-L component of a plane wave is one scalar field times d^l, so
        # the aligned component along u_1 = d holds the whole response energy.
        energy = float(np.sum(np.stack(components) ** 2))
        assert float(np.sum(aligned[0] ** 2)) / energy > 1.0 - 1e-5
        for response in aligned[1:]:
            assert float(np.sum(response**2)) / energy < 1e-6


@requires_fft
def test_public_first_order_riesz_bank_scales_with_the_image() -> None:
    # Eigenvectors must not overflow or underflow at extreme image amplitudes.
    generator = np.random.default_rng(8271)
    image = generator.normal(size=(7, 7, 7)).astype(np.float32)
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            orders=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            log_sigma=(1.0,),
            steering=True,
            tensor_sigma=0.3,
        ),
    )

    def bank(values: NDArray[np.float32]) -> NDArray[np.float64]:
        with mx.stream(mx.gpu):
            produced = feature_images(mx.array(values), (1.0, 1.0, 1.0), options)
            return np.stack(
                [np.asarray(item.image, dtype=np.float64) for item in produced],
            )

    baseline = bank(image)
    assert np.all(np.isfinite(baseline))
    baseline_energy = float(np.sum(baseline * baseline))
    assert baseline_energy > 0.0
    for scale in (1e-6, 1e5):
        actual = bank(np.asarray(image * np.float32(scale), dtype=np.float32))
        expected = baseline * scale
        assert np.all(np.isfinite(actual))
        # Amplitude covariance with the baseline scaled by the image factor.
        covariance = float(np.sum(actual * expected)) / float(np.sum(expected * expected))
        np.testing.assert_allclose(covariance, 1.0, rtol=2e-4, atol=0)
        # The steering matrices are orthogonal, so the whole bank's energy scales
        # quadratically with the image amplitude.
        np.testing.assert_allclose(
            float(np.sum(actual * actual)), scale * scale * baseline_energy,
            rtol=2e-4, atol=0,
        )


def test_tensor_rejects_remote_amplitude_that_erases_local_products() -> None:
    # A large remote response must not erase a small nonzero orientation merely
    # because both responses participate in the same bank-wide normalization.
    first = np.full((3, 3), 1e-30, dtype=np.float32)
    first[0, 0] = 1e30
    second = np.full((3, 3), 2e-30, dtype=np.float32)
    with pytest.raises(ValueError, match="float32 dynamic range"):
        structure_tensor_rotations(
            [mx.array(first), mx.array(second)], (1.0, 1.0), 0.1,
            FilterPadding.PERIODIC,
        )


@requires_fft
def test_steered_pipeline_response_rotates_with_a_planar_wave() -> None:
    # An odd periodic cube has no Nyquist bin, so a 90-degree grid rotation maps the
    # DFT grid and the periodic tensor window onto themselves exactly.
    size = 15
    indices = np.indices((size, size, size))
    wave = (1.0, 2.0, 1.0)
    phase = sum((2 * np.pi / size) * wave[axis] * indices[axis] for axis in range(3))
    image = np.cos(phase) + 0.25 * np.cos(2 * phase)
    rotated = np.rot90(image, 1, axes=(1, 2))
    options = TransformSettings(
        image_types=(ImageType.RIESZ,),
        riesz=RieszSettings(
            base=RieszBase.SIMONCELLI,
            orders=((2, 0, 0),),
            simoncelli_levels=(1,),
            padding=FilterPadding.PERIODIC,
        ),
    )
    steered = dataclasses.replace(
        options, riesz=dataclasses.replace(options.riesz, steering=True, tensor_sigma=1.0)
    )

    def response(values: NDArray[np.float64], settings: TransformSettings) -> NDArray[np.float64]:
        with mx.stream(mx.gpu):
            (result,) = feature_images(
                mx.array(values.astype(np.float32)), (1.0, 1.0, 1.0), settings,
            )
            return np.asarray(result.image, dtype=np.float64)

    original = response(image, steered)
    turned = response(rotated, steered)
    # The (2, 0, 0) aligned component follows u_1 only, so it is sign-invariant and
    # must rotate with the image; the axis-aligned k1 component does not.
    scale = float(np.max(np.abs(original)))
    np.testing.assert_allclose(turned, np.rot90(original, 1, axes=(1, 2)),
                               rtol=0, atol=1e-4 * scale)
    axis_aligned = response(image, options)
    assert not np.allclose(
        response(rotated, options), np.rot90(axis_aligned, 1, axes=(1, 2)),
        rtol=0, atol=1e-2 * scale,
    )


def test_steering_refuses_orders_beyond_the_qualified_bound() -> None:
    # Beyond order 16 the steering matrices lose orthogonality in float32.
    rotation = mx.array(_rotation(2, seed=7).astype(np.float32)[None])
    with pytest.raises(ValueError, match="qualified up to order 16"):
        steering_matrices(rotation, 17)
    with pytest.raises(ValueError, match="orders up to 16"):
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(
                orders=((17, 0),), log_sigma=(2.0,), steering=True, tensor_sigma=1.0
            ),
        )
