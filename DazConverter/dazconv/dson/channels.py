"""Helpers for reading DSON channel values.

Every animatable property in DSON is a "channel" object::

    {"id": "x", "type": "float", "value": 0, "current_value": 12.5}

`current_value` wins when present (it is the value after the scene applied
its modifiers); otherwise `value`, otherwise the type default.  Channels also
appear bare (a plain number) in a few older assets.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def channel_value(chan: Any, default: Any = 0.0) -> Any:
    if chan is None:
        return default
    if isinstance(chan, (int, float, str, bool)):
        return chan
    if isinstance(chan, list):
        return chan
    if isinstance(chan, dict):
        inner = chan.get("channel")
        if isinstance(inner, dict):
            return channel_value(inner, default)
        for key in ("current_value", "value"):
            if key in chan and chan[key] is not None:
                return chan[key]
    return default


def as_float(chan: Any, default: float = 0.0) -> float:
    v = channel_value(chan, default)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def as_bool(chan: Any, default: bool = False) -> bool:
    v = channel_value(chan, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    try:
        return bool(v)
    except (TypeError, ValueError):
        return default


def as_color(chan: Any, default: Tuple[float, float, float] = (1.0, 1.0, 1.0)
             ) -> Tuple[float, float, float]:
    v = channel_value(chan, None)
    if v is None:
        return default
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, f, f)
    if isinstance(v, str):
        return _parse_hex_color(v, default)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, ValueError):
            return default
    return default


def _parse_hex_color(s: str, default):
    s = s.strip().lstrip("#")
    if len(s) == 6:
        try:
            return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
        except ValueError:
            pass
    return default


def axis_triplet(channels: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                 ) -> Tuple[float, float, float]:
    """Read an x/y/z channel array such as `translation` or `center_point`.

    Accepts either the list-of-channels form::

        [{"id": "x", "value": 1}, {"id": "y", ...}, {"id": "z", ...}]

    or a bare `[x, y, z]` array (used in `preview` blocks).
    """
    if channels is None:
        return default
    if isinstance(channels, dict):
        channels = [channels]
    if not isinstance(channels, (list, tuple)):
        return default
    if channels and all(isinstance(c, (int, float)) for c in channels[:3]):
        vals = list(channels[:3]) + [0.0] * (3 - len(channels[:3]))
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    out = {"x": default[0], "y": default[1], "z": default[2]}
    for i, chan in enumerate(channels):
        if not isinstance(chan, dict):
            continue
        cid = str(chan.get("id", "")).lower()
        if cid not in out:
            cid = "xyz"[i] if i < 3 else ""
        if cid in out:
            out[cid] = as_float(chan, out[cid])
    return (out["x"], out["y"], out["z"])


def find_channel(container: Any, channel_id: str) -> Optional[Dict[str, Any]]:
    """Find a channel by (case-insensitive) id inside a list or dict."""
    if container is None:
        return None
    target = channel_id.strip().lower()
    if isinstance(container, dict):
        container = container.values()
    for item in container:
        if not isinstance(item, dict):
            continue
        chan = item.get("channel") if "channel" in item else item
        if isinstance(chan, dict) and str(chan.get("id", "")).strip().lower() == target:
            return chan
    return None


def extra_channels(node_or_material: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collect channels from `extra` blocks keyed by lowercase channel id.

    Iray Uber materials put nearly everything (Base Color, Metallic Weight,
    Glossy Roughness, ...) inside ``extra -> studio_material_channels``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    extras = node_or_material.get("extra")
    if not isinstance(extras, list):
        return out
    for block in extras:
        if not isinstance(block, dict):
            continue
        channels = block.get("channels")
        if not isinstance(channels, list):
            continue
        for item in channels:
            chan = item.get("channel") if isinstance(item, dict) else None
            if isinstance(chan, dict) and chan.get("id"):
                out[str(chan["id"]).strip().lower()] = chan
    return out


def extra_types(node_or_material: Dict[str, Any]) -> List[str]:
    out = []
    extras = node_or_material.get("extra")
    if isinstance(extras, list):
        for block in extras:
            if isinstance(block, dict) and block.get("type"):
                out.append(str(block["type"]))
    return out


def image_of(chan: Any) -> Optional[str]:
    """Return the `image_file` of a channel, if it has one."""
    if not isinstance(chan, dict):
        return None
    inner = chan.get("channel")
    if isinstance(inner, dict) and inner.get("image_file"):
        return str(inner["image_file"])
    if chan.get("image_file"):
        return str(chan["image_file"])
    # Some assets use an `image` reference into the image_library instead.
    if chan.get("image"):
        return str(chan["image"])
    return None
