"""Adapter for project-adapted exPreCast normalized-reflectivity forecasts.

Exports are loaded lazily so ``python -m exprecast_adapter.adapter`` does not
import its command module before :mod:`runpy` executes it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FIELD_LEADS",
    "TARGET_LEADS",
    "WINDOW_INDICES",
    "extract_station_patches",
    "gather_scene_windows",
    "validate_field_axes",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import adapter

    return getattr(adapter, name)
