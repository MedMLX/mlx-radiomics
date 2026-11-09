"""Packaged wavelet coefficients, available without the execution runtime."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from math import isfinite
from typing import cast


@dataclass(frozen=True)
class Wavelet:
    """Decomposition filter pair (low-pass and high-pass taps) of one discrete wavelet."""

    low: tuple[float, ...]
    high: tuple[float, ...]


def wavelet_coefficients(value: object) -> tuple[float, ...]:
    """Parse a decoded JSON filter into a tuple of floats.

    Raises:
        ValueError: If ``value`` is not a nonempty list of finite ints or floats.
    """
    if not isinstance(value, list):
        raise ValueError("wavelet coefficients require a numeric list")
    result: list[float] = []
    for v in cast(list[object], value):
        if type(v) not in (int, float) or not isfinite(cast(float, v)):
            raise ValueError("wavelet coefficient must be finite numeric")
        result.append(float(cast(float, v)))
    if not result:
        raise ValueError("wavelet filter must not be empty")
    return tuple(result)


@lru_cache(maxsize=128)
def wavelet_filters(name: str) -> Wavelet:
    """Load the named wavelet's low- and high-pass filters from ``wavelet_tables.json``."""
    raw: object = json.loads(
        files("mlxradiomics.imaging").joinpath("wavelet_tables.json").read_text()
    )
    if not isinstance(raw, dict):
        raise ValueError("invalid wavelet table schema")
    data = cast(Mapping[str, object], raw)
    tables = data.get("wavelets")
    if data.get("schema_version") != 1 or not isinstance(tables, dict):
        raise ValueError("invalid wavelet table schema")
    table = cast(Mapping[str, object], tables).get(name)
    if not isinstance(table, dict):
        raise ValueError(f"unsupported discrete wavelet: {name}")
    filters = cast(Mapping[str, object], table)
    wavelet = Wavelet(wavelet_coefficients(filters["low"]), wavelet_coefficients(filters["high"]))
    if len(wavelet.low) != len(wavelet.high) or len(wavelet.low) % 2:
        raise ValueError("wavelet requires equally sized even-length filters")
    return wavelet
