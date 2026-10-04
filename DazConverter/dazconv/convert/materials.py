"""DSON materials -> IR PBR materials.

Daz ships three main shader families and they name things very differently:

* **Iray Uber** (``studio/material/uber_iray``) -- everything of interest lives
  in ``extra -> studio_material_channels`` with human names like
  "Base Color", "Metallic Weight", "Glossy Roughness", "Normal Map".
* **PBRSkin** (``studio/material/daz_shader``) -- same idea, different names
  ("Dual Lobe Specular Weight", "Specular Lobe 1 Roughness", ...).
* **3Delight / DAZ Studio Default** -- the legacy top-level channels
  (``diffuse``, ``specular``, ``glossiness``, ``bump``, ``normal_map``, ...).

This module normalises all three onto one metallic/roughness PBR material so
the exporters never have to care which shader the artist used.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dson import channels as ch
from ..ir import Material, TextureRef

# Channel name aliases, searched in order.  Lowercased.
BASE_COLOR = ("base color", "diffuse color", "diffuse")
METALLIC = ("metallic weight", "metallicity", "metallic")
ROUGHNESS = ("glossy roughness", "specular lobe 1 roughness", "roughness",
             "diffuse roughness")
GLOSSINESS = ("glossiness", "glossy layered weight", "glossy weight")
SPECULAR = ("glossy reflectivity", "specular strength", "specular",
            "dual lobe specular weight")
NORMAL = ("normal map", "detail normal map")
BUMP = ("bump strength", "bump", "detail weight")
OPACITY = ("cutout opacity", "opacity strength", "opacity", "transparency")
EMISSION = ("emission color", "emission", "ambient color", "ambient")
EMISSION_STR = ("luminance", "emission strength", "ambient strength")
TRANSLUCENCY = ("translucency weight", "translucency")
TRANSLUCENCY_COLOR = ("translucency color", "sss reflectance tint",
                      "transmitted color")
THIN_WALLED = ("thin walled",)
H_TILES = ("horizontal tiles", "u_scale", "u scale")
V_TILES = ("vertical tiles", "v_scale", "v scale")
H_OFFSET = ("horizontal offset", "u_offset", "u offset")
V_OFFSET = ("vertical offset", "v_offset", "v offset")

# Surfaces whose names suggest they should be alpha-blended rather than opaque.
TRANSPARENT_HINTS = ("eyemoisture", "cornea", "tear", "glass", "lashes",
                     "eyelash", "eyebrow", "brows", "sclera", "moisture")


class MaterialBuilder:
    """Builds IR materials, resolving texture paths through a ContentLibrary."""

    def __init__(self, library, base_file: Optional[str] = None,
                 warn=None):
        self.library = library
        self.base_file = base_file
        self._warn = warn or (lambda msg: None)
        self._cache: Dict[str, Material] = {}
        self.missing_textures: List[str] = []

    # ------------------------------------------------------------------ API
    def build(self, material_def: Dict[str, Any],
              instance: Optional[Dict[str, Any]] = None,
              base_file: Optional[str] = None,
              name: Optional[str] = None) -> Material:
        """Merge a library material with its scene instance and convert it."""
        merged = merge_material(material_def, instance)
        mat_name = name or str(merged.get("name") or merged.get("id") or "Material")
        base = base_file or self.base_file

        extras = ch.extra_channels(merged)
        types = ch.extra_types(merged)
        shader = next((t for t in types if "material" in t or "shader" in t), "")

        mat = Material(name=sanitize(mat_name), source_type=shader)

        # ---- base color -------------------------------------------------
        chan = self._find(merged, extras, BASE_COLOR)
        if chan is not None:
            mat.base_color = ch.as_color(chan, (0.8, 0.8, 0.8))
            mat.base_color_map = self._texture(chan, base, "basecolor", gamma=2.2)
        strength = self._find(merged, extras, ("diffuse strength", "diffuse weight"))
        if strength is not None:
            s = ch.as_float(strength, 1.0)
            if 0.0 <= s <= 1.0 and s != 1.0:
                mat.base_color = tuple(c * s for c in mat.base_color)  # type: ignore

        # ---- metallic ----------------------------------------------------
        chan = self._find(merged, extras, METALLIC)
        if chan is not None:
            mat.metallic = clamp(ch.as_float(chan, 0.0))
            mat.metallic_map = self._texture(chan, base, "metallic", gamma=1.0)

        # ---- roughness ---------------------------------------------------
        mat.roughness, mat.roughness_map = self._roughness(merged, extras, base)

        # ---- specular ----------------------------------------------------
        chan = self._find(merged, extras, SPECULAR)
        if chan is not None:
            mat.specular = clamp(ch.as_float(chan, 0.5))

        # ---- normal / bump -----------------------------------------------
        chan = self._find(merged, extras, NORMAL)
        if chan is not None:
            mat.normal_map = self._texture(chan, base, "normal", gamma=1.0)
            if mat.normal_map:
                mat.normal_strength = ch.as_float(chan, 1.0)
        chan = self._find(merged, extras, BUMP)
        if chan is not None:
            mat.bump_map = self._texture(chan, base, "bump", gamma=1.0)
            if mat.bump_map:
                mat.bump_strength = ch.as_float(chan, 1.0)

        # ---- opacity -----------------------------------------------------
        chan = self._find(merged, extras, OPACITY)
        if chan is not None:
            val = ch.as_float(chan, 1.0)
            is_transparency = _channel_id(chan) == "transparency"
            mat.opacity = clamp(1.0 - val if is_transparency else val)
            mat.opacity_map = self._texture(chan, base, "opacity", gamma=1.0)
        mat.alpha_blend = (mat.opacity < 0.999 or mat.opacity_map is not None
                           or any(h in mat.name.lower() for h in TRANSPARENT_HINTS))

        # ---- emission ----------------------------------------------------
        chan = self._find(merged, extras, EMISSION)
        if chan is not None:
            col = ch.as_color(chan, (0.0, 0.0, 0.0))
            emap = self._texture(chan, base, "emission", gamma=2.2)
            strength_chan = self._find(merged, extras, EMISSION_STR)
            lum = ch.as_float(strength_chan, 0.0) if strength_chan is not None else 0.0
            if any(c > 0.0 for c in col) and (lum > 0.0 or emap):
                mat.emissive = col
                mat.emissive_map = emap
                # Luminance is in cd/m^2; normalise to something sane for DCCs.
                mat.emissive_strength = min(10.0, lum / 1000.0) if lum > 1.0 else lum

        # ---- translucency / SSS -------------------------------------------
        chan = self._find(merged, extras, TRANSLUCENCY)
        if chan is not None:
            mat.translucency = clamp(ch.as_float(chan, 0.0))
        chan = self._find(merged, extras, TRANSLUCENCY_COLOR)
        if chan is not None:
            mat.translucency_color = ch.as_color(chan, (1.0, 1.0, 1.0))
            mat.translucency_map = self._texture(chan, base, "translucency", gamma=2.2)
            mat.sss_color = mat.translucency_color
        chan = self._find(merged, extras, THIN_WALLED)
        if chan is not None:
            mat.thin_walled = ch.as_bool(chan, False)

        # ---- tiling --------------------------------------------------------
        mat.horizontal_tiles = ch.as_float(self._find(merged, extras, H_TILES), 1.0) or 1.0
        mat.vertical_tiles = ch.as_float(self._find(merged, extras, V_TILES), 1.0) or 1.0
        hoff = ch.as_float(self._find(merged, extras, H_OFFSET), 0.0)
        voff = ch.as_float(self._find(merged, extras, V_OFFSET), 0.0)
        for tex in mat.textures():
            tex.tiling = (mat.horizontal_tiles, mat.vertical_tiles)
            tex.offset = (hoff, voff)

        mat.extra["groups"] = list(merged.get("groups") or [])
        mat.extra["uv_set"] = merged.get("uv_set")
        return mat

    # ------------------------------------------------------------- internals
    def _roughness(self, merged, extras, base) -> Tuple[float, Optional[TextureRef]]:
        chan = self._find(merged, extras, ROUGHNESS)
        if chan is not None:
            rough = clamp(ch.as_float(chan, 0.5))
            tex = self._texture(chan, base, "roughness", gamma=1.0)
            return rough, tex
        chan = self._find(merged, extras, GLOSSINESS)
        if chan is not None:
            gloss = clamp(ch.as_float(chan, 0.5))
            tex = self._texture(chan, base, "roughness", gamma=1.0)
            if tex:
                tex.name += "_fromGloss"
            return clamp(1.0 - gloss), tex
        return 0.5, None

    def _find(self, merged: Dict[str, Any], extras: Dict[str, Any],
              names: Sequence[str]) -> Optional[Dict[str, Any]]:
        for n in names:
            key = n.lower()
            if key in extras:
                return extras[key]
            # top-level 3Delight channels use snake_case ids
            snake = key.replace(" ", "_")
            for candidate in (snake, key):
                val = merged.get(candidate)
                if isinstance(val, dict):
                    return val.get("channel") if isinstance(val.get("channel"), dict) else val
        return None

    def _texture(self, chan: Any, base_file: Optional[str], role: str,
                 gamma: float = 0.0) -> Optional[TextureRef]:
        img = ch.image_of(chan)
        if not img:
            return None
        resolved = None
        if self.library is not None:
            resolved = self.library.resolve_image(img, base_file)
        if not resolved:
            if img not in self.missing_textures:
                self.missing_textures.append(img)
                self._warn("texture not found: %s" % img)
            resolved = img  # keep the original path so DCCs can relink
        return TextureRef(path=resolved,
                          name=sanitize(os.path.splitext(os.path.basename(resolved))[0]),
                          gamma=gamma)


def _channel_id(chan: Any) -> str:
    if isinstance(chan, dict):
        inner = chan.get("channel")
        if isinstance(inner, dict):
            return str(inner.get("id", "")).lower()
        return str(chan.get("id", "")).lower()
    return ""


def merge_material(library_def: Optional[Dict[str, Any]],
                   instance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Overlay a scene material instance on top of its library definition."""
    out: Dict[str, Any] = dict(library_def or {})
    if not instance:
        return out
    for key, val in instance.items():
        if key == "extra":
            out["extra"] = _merge_extras(out.get("extra"), val)
        elif key == "url":
            continue
        else:
            out[key] = val
    return out


def _merge_extras(base: Any, override: Any) -> List[Any]:
    """Merge studio_material_channels blocks, instance channels winning."""
    base_list = list(base) if isinstance(base, list) else []
    over_list = list(override) if isinstance(override, list) else []
    if not base_list:
        return over_list
    if not over_list:
        return base_list

    def index(blocks):
        out = {}
        for b in blocks:
            if isinstance(b, dict) and isinstance(b.get("channels"), list):
                for item in b["channels"]:
                    chan = item.get("channel") if isinstance(item, dict) else None
                    if isinstance(chan, dict) and chan.get("id"):
                        out[str(chan["id"]).lower()] = item
        return out

    merged_channels = index(base_list)
    merged_channels.update(index(over_list))
    others = [b for b in base_list + over_list
              if isinstance(b, dict) and not isinstance(b.get("channels"), list)]
    seen_types = set()
    uniq_others = []
    for b in others:
        t = b.get("type")
        if t in seen_types:
            continue
        seen_types.add(t)
        uniq_others.append(b)
    return uniq_others + [{"type": "studio_material_channels",
                           "channels": list(merged_channels.values())}]


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def sanitize(name: str) -> str:
    out = []
    for c in str(name):
        out.append(c if (c.isalnum() or c in "_- .") else "_")
    return "".join(out).strip() or "Material"


def default_material(name: str = "Default") -> Material:
    return Material(name=sanitize(name), base_color=(0.8, 0.8, 0.8), roughness=0.5)
