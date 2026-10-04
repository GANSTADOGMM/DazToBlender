"""DSON morph modifiers -> IR blend shapes.

A morph modifier carries a sparse delta list::

    {"morph": {"vertex_count": 16556,
               "deltas": {"count": 812, "values": [[idx, dx, dy, dz], ...]}}}

HD morphs (``*_HD_div?``) encode detail at a subdivision level the base cage
does not have; we skip their sub-level deltas and keep the base ones.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ir import MorphTarget

HD_RE = re.compile(r"_hd_div\d+$", re.IGNORECASE)


def parse_morph(modifier: Dict[str, Any], vertex_count: int,
                name: Optional[str] = None) -> Optional[MorphTarget]:
    """Build a MorphTarget from a DSON modifier, or None if it has no deltas."""
    morph = modifier.get("morph")
    if not isinstance(morph, dict):
        return None
    deltas_block = morph.get("deltas")
    values = deltas_block.get("values") if isinstance(deltas_block, dict) else deltas_block
    if not isinstance(values, list) or not values:
        return None

    mname = name or str(modifier.get("name") or modifier.get("id") or "Morph")
    channel = modifier.get("channel") if isinstance(modifier.get("channel"), dict) else {}
    label = str(channel.get("label") or channel.get("name") or mname)

    target = MorphTarget(name=sanitize_name(mname), label=label)
    declared = morph.get("vertex_count")
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        declared = -1
    # vertex_count of -1 means "same as base mesh".
    limit = vertex_count

    for entry in values:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            continue
        try:
            idx = int(entry[0])
            dx, dy, dz = float(entry[1]), float(entry[2]), float(entry[3])
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= limit:
            continue
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        target.indices.append(idx)
        target.deltas.append((dx, dy, dz))

    if not target.indices:
        return None

    from ..dson import channels as ch
    target.default_value = ch.as_float(channel, 0.0) if channel else 0.0
    return target


def is_hd_morph(name: str) -> bool:
    return bool(HD_RE.search(name or ""))


def sanitize_name(name: str) -> str:
    """Make a morph name safe for FBX/glTF/engine consumption."""
    base = os.path.splitext(str(name).strip())[0]
    out = []
    for c in base:
        out.append(c if (c.isalnum() or c in "_-") else "_")
    s = "".join(out).strip("_")
    return s or "Morph"


def filter_morphs(morphs: Sequence[MorphTarget],
                  include: Optional[Sequence[str]] = None,
                  exclude_hd: bool = True,
                  min_deltas: int = 1) -> List[MorphTarget]:
    """Select which morphs survive into the export."""
    patterns = [p.strip().lower() for p in (include or []) if p.strip()]
    out: List[MorphTarget] = []
    for m in morphs:
        if exclude_hd and is_hd_morph(m.name):
            continue
        if len(m) < min_deltas:
            continue
        if patterns:
            low = m.name.lower()
            label = (m.label or "").lower()
            if not any(_match(p, low) or _match(p, label) for p in patterns):
                continue
        out.append(m)
    return out


def _match(pattern: str, value: str) -> bool:
    if "*" in pattern or "?" in pattern:
        import fnmatch
        return fnmatch.fnmatch(value, pattern)
    return pattern in value


def apply_morph(vertices: List[Tuple[float, float, float]],
                morph: MorphTarget, weight: float = 1.0) -> List[Tuple[float, float, float]]:
    """Return a new vertex list with the morph applied at `weight`."""
    verts = list(vertices)
    for idx, d in zip(morph.indices, morph.deltas):
        if 0 <= idx < len(verts):
            v = verts[idx]
            verts[idx] = (v[0] + d[0] * weight,
                          v[1] + d[1] * weight,
                          v[2] + d[2] * weight)
    return verts


def collect_from_library(doc: Dict[str, Any], vertex_count: int,
                         geometry_id: Optional[str] = None) -> List[MorphTarget]:
    """Pull every morph in a document's modifier_library."""
    out: List[MorphTarget] = []
    for mod in doc.get("modifier_library") or []:
        if not isinstance(mod, dict) or "morph" not in mod:
            continue
        if geometry_id:
            parent = str(mod.get("parent") or "")
            if parent and geometry_id not in parent:
                continue
        t = parse_morph(mod, vertex_count)
        if t:
            out.append(t)
    return out
