"""Daz material channels -> neutral material description.

Daz ships two shapes of material data:

* the plain DSON ``material`` channels (``diffuse``, ``specular``, ``opacity``,
  ``normal``, ...) used by ``material_library`` entries
* an ``extra`` block of type ``studio_material_channels`` carrying the Iray
  uber shader channels (``Base Color``, ``Metallic Weight``, ``Glossy
  Roughness``, ``Normal Map``, ``Cutout Opacity``, ...)

Both are collected here; texture references are Daz content URLs and are
resolved to real files by :mod:`dazconv.library`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

Color = tuple


@dataclass
class Material:
    id: str
    name: str = ""
    label: str = ""
    base_color: Color = (0.8, 0.8, 0.8)
    base_color_map: str = ""
    specular_color: Color = (0.2, 0.2, 0.2)
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    opacity_map: str = ""
    normal_map: str = ""
    bump_map: str = ""
    emissive_color: Color = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0
    double_sided: bool = False
    #: everything else, preserved for inspection / round-tripping
    channels: Dict[str, object] = field(default_factory=dict)
    #: resolved absolute texture paths, filled in by the converter
    texture_paths: Dict[str, str] = field(default_factory=dict)
    geometry_ref: str = ""
    groups: List[str] = field(default_factory=list)
    #: file the material was read from (used to resolve relative texture names)
    source_path: str = ""

    # ------------------------------------------------------------ loading --
    @classmethod
    def from_dson(cls, data: dict) -> "Material":
        material = cls(
            id=data.get("id") or "material",
            name=data.get("name") or data.get("id") or "",
            label=data.get("label") or "",
            geometry_ref=data.get("geometry") or "",
            groups=list(data.get("groups") or []),
        )
        flat = _flatten_channels(data)
        material.channels = flat

        material.base_color = _first_color(flat, ("base color", "diffuse color", "diffuse")) or material.base_color
        material.base_color_map = _first_image(flat, ("base color", "diffuse color", "diffuse"))
        material.specular_color = _first_color(flat, ("glossy color", "specular color", "specular")) or material.specular_color
        material.roughness = _first_float(flat, ("glossy roughness", "roughness"), material.roughness)
        material.metallic = _first_float(flat, ("metallic weight", "metallic"), material.metallic)
        material.normal_map = _first_image(flat, ("normal map",))
        material.bump_map = _first_image(flat, ("bump", "height"))
        material.opacity = _first_float(flat, ("cutout opacity", "opacity"), material.opacity)
        material.opacity_map = _first_image(flat, ("cutout opacity", "opacity"))
        material.emissive_color = _first_color(flat, ("emission color", "emissive")) or material.emissive_color
        material.emissive_strength = _first_float(flat, ("emission strength", "luminance"), material.emissive_strength)
        material.double_sided = bool(_first_value(flat, ("two sided", "double sided", "smooth on")) is True)
        return material

    # ------------------------------------------------------------ helpers --
    def has_textures(self) -> bool:
        return bool(
            self.base_color_map or self.normal_map or self.opacity_map or self.bump_map
        )

    def used_textures(self) -> Dict[str, str]:
        """Slot -> resolved path, for slots that resolved to an existing file."""
        return {slot: path for slot, path in self.texture_paths.items() if path}

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": self.label or self.name or self.id,
            "base_color": [round(c, 4) for c in self.base_color],
            "base_color_map": self.base_color_map,
            "roughness": round(self.roughness, 4),
            "metallic": round(self.metallic, 4),
            "opacity": round(self.opacity, 4),
            "normal_map": self.normal_map,
            "opacity_map": self.opacity_map,
            "double_sided": self.double_sided,
            "resolved_textures": len(self.used_textures()),
        }


# ------------------------------------------------------------ extraction --
def _flatten_channels(material: dict) -> Dict[str, dict]:
    """Collect every channel dict in a material, keyed by lower-cased name.

    Later definitions win, which means scene ``materials`` entries override the
    shared ``material_library`` defaults they were derived from.
    """
    flat: Dict[str, dict] = {}

    def add(channel: dict) -> None:
        if not isinstance(channel, dict):
            return
        key = str(channel.get("label") or channel.get("name") or channel.get("id") or "")
        key = key.strip().lower()
        if not key:
            return
        if key not in flat or channel.get("image_file") or "current_value" in channel:
            flat[key] = channel

    # Standard DSON material channels appear directly on the material object.
    for key, value in material.items():
        if key == "extra":
            continue
        if isinstance(value, dict) and ("channel" in value or "type" in value):
            channel = value.get("channel") if isinstance(value.get("channel"), dict) else value
            add(channel)
        elif isinstance(value, dict) and "image_file" in value:
            add(value)

    # Iray/uber channels live under extra -> studio_material_channels.
    for extra in material.get("extra") or []:
        if not isinstance(extra, dict):
            continue
        for wrapper in extra.get("channels") or []:
            if isinstance(wrapper, dict):
                add(wrapper.get("channel") or {})
    return flat


def _first_value(flat: Dict[str, dict], names):
    for name in names:
        channel = flat.get(name)
        if channel is None:
            continue
        for key in ("current_value", "value"):
            if channel.get(key) is not None:
                return channel[key]
    return None


def _first_float(flat: Dict[str, dict], names, default: float) -> float:
    value = _first_value(flat, names)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_color(flat: Dict[str, dict], names) -> Optional[Color]:
    value = _first_value(flat, names)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


def _first_image(flat: Dict[str, dict], names) -> str:
    for name in names:
        channel = flat.get(name)
        if not channel:
            continue
        for key in ("image_file", "current_image_file", "value", "current_value"):
            candidate = channel.get(key)
            if isinstance(candidate, str) and candidate:
                # A colour channel can carry a float value; only accept paths.
                if "/" in candidate or "\\" in candidate:
                    return candidate
    return ""


def sanitize_material_name(name: str, used: Optional[set] = None) -> str:
    """Make a name safe for FBX (no quotes, backslashes or newlines)."""
    cleaned = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in (name or "Material"))
    cleaned = cleaned.strip(" .") or "Material"
    if used is None:
        return cleaned
    base = cleaned
    index = 1
    while cleaned in used:
        index += 1
        cleaned = f"{base}_{index}"
    used.add(cleaned)
    return cleaned


def texture_basename(url: str) -> str:
    """``/Runtime/Textures/foo/bar.png`` -> ``bar.png``."""
    return os.path.basename(url.replace("\\", "/").split("?")[0])
