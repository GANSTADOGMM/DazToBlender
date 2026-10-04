"""Morph (shape key) loading and application.

A Daz morph lives in ``modifier_library`` with a ``morph`` member::

    "morph": {
        "vertex_count": 17418,
        "deltas": {"count": 14796, "values": [[vertex_index, dx, dy, dz], ...]}
    }

Deltas are sparse -- only vertices that actually move are listed -- and are
additive in the geometry's own units (centimetres).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .geometry import Geometry
from .mathutil import Vec3
from . import dson


@dataclass
class Morph:
    id: str
    name: str = ""
    label: str = ""
    group: str = ""
    region: str = ""
    parent_ref: str = ""
    vertex_count: int = 0
    #: vertex_index -> delta
    deltas: Dict[int, Vec3] = field(default_factory=dict)
    min_value: float = 0.0
    max_value: float = 1.0
    default_value: float = 0.0
    icon: str = ""
    source_path: str = ""

    @classmethod
    def from_dson(cls, modifier: dict, source_path: str = "") -> Optional["Morph"]:
        morph = modifier.get("morph")
        if not isinstance(morph, dict):
            return None
        deltas: Dict[int, Vec3] = {}
        for entry in (morph.get("deltas") or {}).get("values") or []:
            if len(entry) < 4:
                continue
            deltas[int(entry[0])] = (float(entry[1]), float(entry[2]), float(entry[3]))
        channel = modifier.get("channel") or {}
        presentation = modifier.get("presentation") or {}
        return cls(
            id=modifier.get("id") or "",
            name=modifier.get("name") or "",
            label=presentation.get("label") or channel.get("label") or modifier.get("id") or "",
            group=modifier.get("group") or "",
            region=modifier.get("region") or "",
            parent_ref=modifier.get("parent") or "",
            vertex_count=int(morph.get("vertex_count") or 0),
            deltas=deltas,
            min_value=float(channel.get("min", 0.0)),
            max_value=float(channel.get("max", 1.0)),
            default_value=float(channel.get("value", 0.0)),
            icon=presentation.get("icon_large") or "",
            source_path=source_path,
        )

    # ---------------------------------------------------------- geometry --
    def affected_count(self) -> int:
        return len(self.deltas)

    def apply(self, geometry: Geometry, weight: float = 1.0) -> List[Vec3]:
        """Return a new vertex list with this morph applied at ``weight``."""
        if weight == 0.0 or not self.deltas:
            return list(geometry.vertices)
        out = list(geometry.vertices)
        for vertex, delta in self.deltas.items():
            if 0 <= vertex < len(out):
                base = out[vertex]
                out[vertex] = (
                    base[0] + delta[0] * weight,
                    base[1] + delta[1] * weight,
                    base[2] + delta[2] * weight,
                )
        return out

    def delta_tuples(self) -> List[tuple]:
        """``(vertex_index, dx, dy, dz)`` pairs sorted by vertex index."""
        return [(v, *self.deltas[v]) for v in sorted(self.deltas)]

    def matches_geometry(self, geometry: Geometry) -> bool:
        return self.vertex_count == 0 or self.vertex_count == geometry.vertex_count

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "group": self.group,
            "region": self.region,
            "affected_vertices": self.affected_count(),
            "vertex_count": self.vertex_count,
            "min": self.min_value,
            "max": self.max_value,
            "source": os.path.basename(self.source_path) if self.source_path else "",
        }


def apply_morphs(
    geometry: Geometry, selections: Sequence[tuple], skip_mismatched: bool = True
) -> tuple:
    """Apply ``(morph, weight)`` pairs in order.

    Returns ``(vertices, applied, skipped)``.
    """
    vertices = list(geometry.vertices)
    applied: List[tuple] = []
    skipped: List[str] = []
    for morph, weight in selections:
        if not isinstance(morph, Morph):
            skipped.append(f"{morph!r} is not a Morph")
            continue
        if weight == 0.0:
            continue
        if skip_mismatched and not morph.matches_geometry(geometry):
            skipped.append(
                f"{morph.id} expects {morph.vertex_count} vertices, "
                f"geometry has {geometry.vertex_count}"
            )
            continue
        for vertex, delta in morph.deltas.items():
            if 0 <= vertex < len(vertices):
                base = vertices[vertex]
                vertices[vertex] = (
                    base[0] + delta[0] * weight,
                    base[1] + delta[1] * weight,
                    base[2] + delta[2] * weight,
                )
        applied.append((morph, weight))
    return vertices, applied, skipped


def load_morphs_from_file(path: str) -> List[Morph]:
    """Load every morph defined by one ``.dsf`` file."""
    try:
        document = dson.load_dson_file(path)
    except dson.DsonError:
        return []
    morphs: List[Morph] = []
    for modifier in document.get("modifier_library") or []:
        if not isinstance(modifier, dict):
            continue
        morph = Morph.from_dson(modifier, source_path=path)
        if morph is not None:
            morphs.append(morph)
    return morphs


def discover_morphs(
    directories: Iterable[str], limit: Optional[int] = None
) -> List[Morph]:
    """Scan folders for morph assets (``Morphs`` sub-trees are the usual home)."""
    found: List[Morph] = []
    seen_paths = set()
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if not name.lower().endswith((".dsf", ".duf")):
                    continue
                path = os.path.join(root, name)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                found.extend(load_morphs_from_file(path))
                if limit and len(found) >= limit:
                    return found
    return found


def group_morphs(morphs: Iterable[Morph]) -> Dict[str, List[Morph]]:
    """Group morphs by their Daz property group for the UI."""
    grouped: Dict[str, List[Morph]] = {}
    for morph in morphs:
        grouped.setdefault(morph.group or "Ungrouped", []).append(morph)
    for bucket in grouped.values():
        bucket.sort(key=lambda m: (m.label or m.id).lower())
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))
