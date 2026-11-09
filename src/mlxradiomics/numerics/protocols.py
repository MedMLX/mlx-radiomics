"""Typed interfaces for MLX evaluation, indexed updates and custom Metal kernels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mlx.core import Dtype, array


class Evaluate(Protocol):
    """``mx.eval``: evaluate the given arrays."""

    def __call__(self, *arrays: array) -> None:
        """Evaluate ``arrays`` and block until they are computed."""
        ...


class EvaluationRuntime(Protocol):
    """A module exposing ``eval``, such as ``mlx.core``."""

    @property
    def eval(self) -> Evaluate:
        """Return the evaluation function."""
        ...


class IndexedAddition(Protocol):
    """The ``array.at[indices]`` view used for scatter-add."""

    def add(self, value: array) -> array:
        """Return a copy with ``value`` added at the indexed positions."""
        ...


class IndexedMinimum(Protocol):
    """The ``array.at[indices]`` view used for scatter-minimum."""

    def minimum(self, value: array) -> array:
        """Return a copy holding the minimum of each indexed position and ``value``."""
        ...


class MetalKernel(Protocol):
    """A compiled ``mx.fast.metal_kernel``."""

    def __call__(
        self,
        *,
        inputs: Sequence[array],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[Dtype],
        init_value: float | None = None,
    ) -> list[array]:
        """Dispatch the kernel over ``grid`` and return its outputs."""
        ...
