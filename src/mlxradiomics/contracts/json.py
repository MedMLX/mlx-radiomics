"""Strict JSON value conversion shared by result reports."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, isnan
from typing import cast

type JsonValue = bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]


def json_safe(value: object) -> JsonValue:
    """Encode non-finite floats as named strings, including nested mappings and tuples."""
    if isinstance(value, float) and not isfinite(value):
        return "NaN" if isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value).items()
        return {str(key): json_safe(item) for key, item in items}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in cast(tuple[object, ...], value)]
    raise ValueError(f"cannot serialize value of type {type(value).__name__}")
