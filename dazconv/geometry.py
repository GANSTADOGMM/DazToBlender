"""Geometry as stored in a DSON ``geometry_library``.

Verified against real Genesis data, e.g. ``Genesis8_1Female.dsf``:
``vertices.values`` is a float3 array, ``polylist.values`` entries are
``[group_index, material_index, v0, v1, ...]`` and the group name tables are
plain string arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .mathutil import (
    Vec3,
    as_vec3,
    bbox,
    cross,
    normalize,
    sub,
)


@dataclass
class Polygon:
    """One face: indices into :attr:`Geometry.vertices` plus group lookups."""

    vertices: List[int]
    group_index: int = 0
    material_index: int = 0

    @property
    def corner_count(self) -> int:
        return len(self.vertices)

    def is_triangle(self) -> bool:
        return len(self.vertices) == 3


@dataclass
class Geometry:
    id: str
    name: str = ""
    label: str = ""
    type: str = "polygon_mesh"
    vertices: List[Vec3] = field(default_factory=list)
    polygons: List[Polygon] = field(default_factory=list)
    polygon_groups: List[str] = field(default_factory=list)
    polygon_material_groups: List[str] = field(default_factory=list)
    default_uv_set: str = ""
    edge_interpolation_mode: str = ""
    subd_normal_smoothing_mode: str = ""
    current_subdivision_level: int = 0
    source_uri: str = ""
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ loading --
    @classmethod
    def from_dson(cls, data: dict, uri: str = "") -> "Geometry":
        geom = cls(
            id=data.get("id") or data.get("name") or "geometry",
            name=data.get("name") or data.get("id") or "",
            label=data.get("label") or "",
            type=data.get("type") or "polygon_mesh",
            default_uv_set=data.get("default_uv_set") or "",
            edge_interpolation_mode=data.get("edge_interpolation_mode") or "",
            subd_normal_smoothing_mode=data.get("subd_normal_smoothing_mode") or "",
            current_subdivision_level=int(data.get("current_subdivision_level") or 0),
            source_uri=uri,
        )
        verts = data.get("vertices") or {}
        geom.vertices = [as_vec3(v) for v in (verts.get("values") or [])]

        groups = data.get("polygon_groups") or {}
        geom.polygon_groups = list(groups.get("values") or [])

        mat_groups = data.get("polygon_material_groups") or {}
        geom.polygon_material_groups = list(mat_groups.get("values") or [])

        polylist = data.get("polylist") or {}
        for entry in polylist.get("values") or []:
            if not entry:
                continue
            group_index = int(entry[0]) if len(entry) > 0 else 0
            material_index = int(entry[1]) if len(entry) > 1 else 0
            face = [int(v) for v in entry[2:]]
            if len(face) < 3:
                continue
            geom.polygons.append(Polygon(face, group_index, material_index))
        return geom

    # ------------------------------------------------------------ queries --
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def polygon_count(self) -> int:
        return len(self.polygons)

    @property
    def corner_count(self) -> int:
        return sum(p.corner_count for p in self.polygons)

    def group_name(self, index: int) -> str:
        if 0 <= index < len(self.polygon_groups):
            return self.polygon_groups[index]
        return f"Group_{index}"

    def material_group_name(self, index: int) -> str:
        if 0 <= index < len(self.polygon_material_groups):
            return self.polygon_material_groups[index]
        return f"Material_{index}"

    def bounds(self):
        return bbox(self.vertices)

    def is_subdivision(self) -> bool:
        return self.type == "subdivision_surface"

    # ----------------------------------------------------------- topology --
    def triangulated_indices(self) -> List[List[int]]:
        """Fan-triangulate every polygon (needed by formats without n-gons)."""
        out: List[List[int]] = []
        for poly in self.polygons:
            face = poly.vertices
            if len(face) == 3:
                out.append(list(face))
                continue
            for i in range(1, len(face) - 1):
                out.append([face[0], face[i], face[i + 1]])
        return out

    def face_normals(self) -> List[Vec3]:
        normals = []
        verts = self.vertices
        for poly in self.polygons:
            face = poly.vertices
            if len(face) < 3:
                normals.append((0.0, 1.0, 0.0))
                continue
            # Newell's method: robust for non-planar n-gons.
            nx = ny = nz = 0.0
            count = len(face)
            for i in range(count):
                a = verts[face[i]]
                b = verts[face[(i + 1) % count]]
                nx += (a[1] - b[1]) * (a[2] + b[2])
                ny += (a[2] - b[2]) * (a[0] + b[0])
                nz += (a[0] - b[0]) * (a[1] + b[1])
            n = normalize((nx, ny, nz))
            if n == (0.0, 0.0, 0.0):
                n = (0.0, 1.0, 0.0)
            normals.append(n)
        return normals

    def vertex_normals(self, smooth_groups: Sequence[int] | None = None) -> List[Vec3]:
        """Area-weighted averaged normals, one per vertex."""
        accum = [[0.0, 0.0, 0.0] for _ in self.vertices]
        verts = self.vertices
        for poly, normal in zip(self.polygons, self.face_normals()):
            face = poly.vertices
            # Weight by the polygon area so big faces count for more.
            area = 0.5 * _polygon_area(verts, face) or 1.0
            for vi in face:
                if 0 <= vi < len(accum):
                    accum[vi][0] += normal[0] * area
                    accum[vi][1] += normal[1] * area
                    accum[vi][2] += normal[2] * area
        out = []
        for vec in accum:
            n = normalize((vec[0], vec[1], vec[2]))
            out.append(n if n != (0.0, 0.0, 0.0) else (0.0, 1.0, 0.0))
        return out

    def edge_list(self) -> List[tuple]:
        """Unique undirected edges as ``(a, b)`` with ``a < b``."""
        seen = set()
        for poly in self.polygons:
            face = poly.vertices
            count = len(face)
            for i in range(count):
                a, b = face[i], face[(i + 1) % count]
                key = (a, b) if a < b else (b, a)
                seen.add(key)
        return sorted(seen)

    def split_by_material(self) -> dict:
        """Map material-group name -> list of polygons, for split exports."""
        buckets: dict = {}
        for poly in self.polygons:
            buckets.setdefault(self.material_group_name(poly.material_index), []).append(poly)
        return buckets


def _polygon_area(verts: Sequence[Vec3], face: Sequence[int]) -> float:
    total = (0.0, 0.0, 0.0)
    if len(face) < 3:
        return 0.0
    origin = verts[face[0]]
    for i in range(1, len(face) - 1):
        a = sub(verts[face[i]], origin)
        b = sub(verts[face[i + 1]], origin)
        c = cross(a, b)
        total = (total[0] + c[0], total[1] + c[1], total[2] + c[2])
    return math.sqrt(total[0] ** 2 + total[1] ** 2 + total[2] ** 2) * 0.5


def merge_geometries(items: Iterable[Geometry], name: str = "merged") -> Geometry:
    """Concatenate geometries, renumbering face indices."""
    merged = Geometry(id=name, name=name)
    offset = 0
    for geom in items:
        merged.vertices.extend(geom.vertices)
        for poly in geom.polygons:
            merged.polygons.append(
                Polygon([v + offset for v in poly.vertices], poly.group_index, poly.material_index)
            )
        offset += geom.vertex_count
    return merged
