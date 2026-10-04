"""UV set handling for DSON ``uv_set_library`` entries.

Daz stores UVs in a vertex-ordered array plus a sparse override table:
``polygon_vertex_indices`` entries are ``[polygon_index, corner_index,
uv_index]``.  A face corner that has no override simply uses the UV stored at
its own vertex index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .geometry import Geometry
from .mathutil import Vec3


@dataclass
class UVSet:
    id: str
    name: str = ""
    label: str = ""
    vertex_count: int = 0
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    #: ``(polygon_index, corner_index) -> uv_index``
    overrides: Dict[Tuple[int, int], int] = field(default_factory=dict)
    source_uri: str = ""

    @classmethod
    def from_dson(cls, data: dict, uri: str = "") -> "UVSet":
        uvs_field = data.get("uvs") or {}
        if isinstance(uvs_field, dict):
            raw_uvs = uvs_field.get("values") or []
        else:  # some files inline the array without the count wrapper
            raw_uvs = uvs_field
        uvs = [(float(uv[0]), float(uv[1])) for uv in raw_uvs if len(uv) >= 2]

        overrides: Dict[Tuple[int, int], int] = {}
        for entry in data.get("polygon_vertex_indices") or []:
            if len(entry) < 3:
                continue
            poly_index, corner_index, uv_index = int(entry[0]), int(entry[1]), int(entry[2])
            overrides[(poly_index, corner_index)] = uv_index

        return cls(
            id=data.get("id") or "default",
            name=data.get("name") or "",
            label=data.get("label") or "",
            vertex_count=int(data.get("vertex_count") or 0),
            uvs=uvs,
            overrides=overrides,
            source_uri=uri,
        )

    # ------------------------------------------------------------ lookups --
    def corner_uv_index(self, polygon_index: int, corner_index: int, vertex_index: int) -> int:
        """Resolve which UV a given face corner uses."""
        override = self.overrides.get((polygon_index, corner_index))
        if override is not None:
            return override
        return vertex_index

    def get_uv(self, uv_index: int) -> Tuple[float, float]:
        if 0 <= uv_index < len(self.uvs):
            return self.uvs[uv_index]
        return (0.0, 0.0)

    # ------------------------------------------------------------ export --
    def corner_uvs(self, geometry: Geometry, flip_v: bool = True) -> List[Tuple[float, float]]:
        """Flatten to one UV per face corner, in polygon/corner order.

        ``flip_v`` converts Daz's top-left UV origin to the bottom-left origin
        used by FBX/OBJ/glTF.
        """
        out: List[Tuple[float, float]] = []
        for poly_index, poly in enumerate(geometry.polygons):
            for corner_index, vertex_index in enumerate(poly.vertices):
                uv_index = self.corner_uv_index(poly_index, corner_index, vertex_index)
                u, v = self.get_uv(uv_index)
                out.append((u, 1.0 - v if flip_v else v))
        return out

    def sanity_report(self, geometry: Geometry) -> List[str]:
        """Cheap consistency checks, surfaced as warnings in the UI."""
        problems: List[str] = []
        if self.vertex_count and self.vertex_count != geometry.vertex_count:
            problems.append(
                f"uv set '{self.id}' declares vertex_count {self.vertex_count} "
                f"but geometry '{geometry.id}' has {geometry.vertex_count}"
            )
        if not self.uvs:
            problems.append(f"uv set '{self.id}' contains no uvs")
        bad = [
            idx
            for idx in self.overrides.values()
            if not 0 <= idx < len(self.uvs)
        ]
        if bad:
            problems.append(
                f"uv set '{self.id}' has {len(bad)} overrides pointing outside its uv array"
            )
        return problems


def placeholder_uvs(geometry: Geometry) -> List[Tuple[float, float]]:
    """Zeroed per-corner UVs, used when an asset ships without a UV set."""
    return [(0.0, 0.0) for _ in range(geometry.corner_count)]


def bounding_box_of(uvs: List[Tuple[float, float]]):
    if not uvs:
        return (0.0, 0.0), (0.0, 0.0)
    us = [u for u, _ in uvs]
    vs = [v for _, v in uvs]
    return (min(us), min(vs)), (max(us), max(vs))
