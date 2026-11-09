"""Strict type checks for values parsed from PyRadiomics-style parameter mappings.

YAML and JSON booleans are ints in Python, so every numeric check rejects ``bool``
explicitly. Each helper raises ``ValueError`` naming the offending key.
"""

from collections.abc import Mapping
from typing import cast


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    """Return ``value`` as a string-keyed mapping, or reject it under ``context``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"'{context}' must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"'{context}' keys must be strings")
    return cast(Mapping[str, object], mapping)


def require_float(value: object, key: str) -> float:
    """Return an int or float setting as ``float``; ``bool`` is rejected."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"setting '{key}' must be a number")
    return float(value)


def require_int(value: object, key: str) -> int:
    """Return an ``int`` setting unchanged; ``bool`` and floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"setting '{key}' must be an integer")
    return value


def require_bool(value: object, key: str) -> bool:
    """Return a ``bool`` setting unchanged; integers are rejected."""
    if not isinstance(value, bool):
        raise ValueError(f"setting '{key}' must be a boolean")
    return value


def require_int_tuple(value: object, key: str) -> tuple[int, ...]:
    """Return a nonempty list or tuple of integers as a tuple."""
    if isinstance(value, bool) or not isinstance(value, list | tuple):
        raise ValueError(f"setting '{key}' must be a list of integers")
    items = tuple(cast(list[object] | tuple[object, ...], value))
    if not items or any(isinstance(v, bool) or not isinstance(v, int) for v in items):
        raise ValueError(f"setting '{key}' must be a list of integers")
    return cast(tuple[int, ...], items)


def require_float_tuple(value: object, key: str) -> tuple[float, ...]:
    """Return a nonempty list or tuple of numbers as a float tuple."""
    if isinstance(value, bool) or not isinstance(value, list | tuple):
        raise ValueError(f"setting '{key}' must be a list of numbers")
    items = tuple(cast(list[object] | tuple[object, ...], value))
    if not items or any(isinstance(v, bool) or not isinstance(v, int | float) for v in items):
        raise ValueError(f"setting '{key}' must be a list of numbers")
    return tuple(float(cast(int | float, v)) for v in items)
