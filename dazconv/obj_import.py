"""Wavefront OBJ reader (the one Daz format that is not DSON).

Supports ``v``/``vt``/``vn``/``f``/``g``/``o``/``usemtl`` plus enough of
``mtllib`` to recover diffuse colours and texture names.  Faces may be
triangles or n-gons and may omit texture/normal indices (``f 1//2``).

Per-corner UV indices are recorded in face order (``geometry.extra["obj_uv"]``)
so the exporter does not have to guess which UV belongs to which corner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import Geometry, Polygon
from .materials import Material
from .uvset import UVSet

OBJ_EXTRA_UVS = "obj_uvs"
OBJ_EXTRA_CORNER_UV = "obj_corner_uv"


@dataclass
class ObjModel:
    """Everything parsed from one ``.obj`` file."""

    name: str
    geometries: List[Geometry] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def geometry(self) -> Optional[Geometry]:
        return self.geometries[0] if self.geometries else None


class _Builder:
    """Per-geometry accumulation state while scanning the file."""

    def __init__(self, name: str) -> None:
        self.geometry = Geometry(id=name, name=name)
        self.corner_uvs: List[int] = []
        self.uv_table: List[Tuple[float, float]] = []
        self.material_names: List[str] = []


def _parse_vec3(tokens: List[str]) -> Tuple[float, float, float]:
    return (
        float(tokens[0]),
        float(tokens[1]),
        float(tokens[2]) if len(tokens) > 2 else 0.0,
    )


def _face_index(token: str, count: int) -> int:
    """OBJ face indices are 1-based and may be negative (relative)."""
    index = int(token)
    return count + index if index < 0 else index - 1


def read_obj(path: str, split_by_object: bool = True) -> ObjModel:
    """Parse an OBJ file into :class:`ObjModel`."""
    model = ObjModel(name=os.path.splitext(os.path.basename(path))[0], source_path=path)
    positions: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float, float]] = []

    builders: List[_Builder] = [_Builder(model.name)]
    active = builders[0]
    active_material = 0
    active_group = 0
    material_index_by_name: Dict[str, int] = {}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as error:
        model.warnings.append(f"could not read {path}: {error}")
        return model

    def start_object(label: str) -> None:
        nonlocal active, active_material, active_group
        active = _Builder(label or f"{model.name}_{len(builders) + 1}")
        builders.append(active)
        active_material = 0
        active_group = 0

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        keyword = parts[0]

        if keyword == "v":
            if len(parts) >= 3:
                positions.append(_parse_vec3(parts[1:]))
        elif keyword == "vt":
            if len(parts) >= 3:
                texcoords.append((float(parts[1]), float(parts[2])))
            elif len(parts) == 2:
                texcoords.append((float(parts[1]), 0.0))
        elif keyword == "vn":
            if len(parts) >= 3:
                normals.append(_parse_vec3(parts[1:]))
        elif keyword == "o":
            if split_by_object:
                start_object(" ".join(parts[1:]))
            else:
                label = " ".join(parts[1:])
                if not active.geometry.polygons:
                    active.geometry.id = label or active.geometry.id
                    active.geometry.name = label or active.geometry.name
        elif keyword == "g":
            label = " ".join(parts[1:]) or "default"
            geom = active.geometry
            if label not in geom.polygon_groups:
                geom.polygon_groups.append(label)
            active_group = geom.polygon_groups.index(label)
            if not geom.name and not geom.polygons:
                geom.id, geom.name = label, label
        elif keyword == "usemtl":
            label = " ".join(parts[1:]) or "default"
            if label not in active.material_names:
                active.material_names.append(label)
            active_material = active.material_names.index(label)
            material_index_by_name.setdefault(label, len(model.materials))
            if label not in material_index_by_name or all(
                m.name != label for m in model.materials
            ):
                material_index_by_name[label] = len(model.materials)
                model.materials.append(Material(id=label, name=label, source_path=path))
        elif keyword == "mtllib":
            library_path = os.path.join(os.path.dirname(path), " ".join(parts[1:]))
            for material in _read_mtl(library_path, model.warnings):
                existing = next((m for m in model.materials if m.name == material.name), None)
                if existing is None:
                    material_index_by_name[material.name] = len(model.materials)
                    model.materials.append(material)
                else:
                    _merge_mtl_into(existing, material)
        elif keyword == "f":
            vertices: List[int] = []
            corner_uv: List[int] = []
            for token in parts[1:]:
                fields = token.split("/")
                try:
                    vertices.append(_face_index(fields[0], len(positions)))
                except ValueError:
                    continue
                uv_index = -1
                if len(fields) > 1 and fields[1]:
                    try:
                        uv_index = _face_index(fields[1], len(texcoords))
                    except ValueError:
                        uv_index = -1
                corner_uv.append(uv_index)
            if len(vertices) < 3:
                model.warnings.append(f"skipping degenerate face: {line[:60]}")
                continue
            geom = active.geometry
            while active_material >= len(geom.polygon_material_groups):
                index = len(geom.polygon_material_groups)
                name = (
                    active.material_names[index]
                    if index < len(active.material_names)
                    else f"material_{index}"
                )
                geom.polygon_material_groups.append(name)
            if not geom.polygon_groups:
                geom.polygon_groups.append("default")
            geom.polygons.append(Polygon(vertices, active_group, active_material))
            active.corner_uvs.extend(corner_uv)

    # Keep only builders that actually received faces.
    used = [b for b in builders if b.geometry.polygons] or builders[:1]
    for builder in used:
        geometry = builder.geometry
        if not geometry.polygon_groups:
            geometry.polygon_groups = ["default"]
        if not geometry.polygon_material_groups:
            geometry.polygon_material_groups = ["default"]
        geometry.extra[OBJ_EXTRA_UVS] = builder.uv_table
        geometry.extra[OBJ_EXTRA_CORNER_UV] = builder.corner_uvs
        # Remap OBJ texture coordinates into a compact per-geometry UV table.
        _compact_uvs(geometry, texcoords, builder)
        model.geometries.append(geometry)

    if normals:
        model.warnings.append(
            f"{len(normals)} OBJ normals ignored (FBX normals are recomputed from faces)"
        )
    if not any(g.polygons for g in model.geometries):
        model.warnings.append("no faces found in OBJ file")
    return model


def _compact_uvs(
    geometry: Geometry,
    texcoords: List[Tuple[float, float]],
    builder: _Builder,
) -> None:
    """Rebuild ``builder.uv_table`` and rewrite corner indices against it."""
    table: List[Tuple[float, float]] = []
    remap: Dict[int, int] = {}
    rewritten: List[int] = []
    for uv_index in builder.corner_uvs:
        if uv_index < 0 or uv_index >= len(texcoords):
            rewritten.append(-1)
            continue
        if uv_index not in remap:
            remap[uv_index] = len(table)
            table.append(texcoords[uv_index])
        rewritten.append(remap[uv_index])
    builder.uv_table = table
    geometry.extra[OBJ_EXTRA_UVS] = table
    geometry.extra[OBJ_EXTRA_CORNER_UV] = rewritten


def build_uv_set(geometry: Geometry, flip_v: bool = True) -> Optional[UVSet]:
    """Convert OBJ corner UVs into a :class:`UVSet`-shaped object."""
    table = geometry.extra.get(OBJ_EXTRA_UVS) or []
    if not table:
        return None
    uv_set = UVSet(
        id="UVMap",
        name="UVMap",
        vertex_count=geometry.vertex_count,
        uvs=[(u, 1.0 - v if flip_v else v) for u, v in table],
    )
    return uv_set


def corner_uvs_from_obj(geometry: Geometry, flip_v: bool = True) -> List[Tuple[float, float]]:
    """Per-corner UVs in polygon/corner order for the FBX exporter."""
    table = geometry.extra.get(OBJ_EXTRA_UVS) or []
    corners = geometry.extra.get(OBJ_EXTRA_CORNER_UV) or []
    out: List[Tuple[float, float]] = []
    for uv_index in corners:
        if uv_index < 0 or uv_index >= len(table):
            out.append((0.0, 0.0))
            continue
        u, v = table[uv_index]
        out.append((u, 1.0 - v if flip_v else v))
    return out


# ------------------------------------------------------------------- MTL --
def _read_mtl(path: str, warnings: List[str]) -> List[Material]:
    """Parse a minimal subset of MTL: ``newmtl``/``Kd``/``d``/``map_Kd``/``Ns``."""
    if not os.path.isfile(path):
        warnings.append(f"material library not found: {os.path.basename(path)}")
        return []
    materials: List[Material] = []
    current: Optional[Material] = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as error:
        warnings.append(f"could not read material library: {error}")
        return []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        keyword = parts[0]
        argument = parts[1].strip() if len(parts) > 1 else ""
        if keyword == "newmtl":
            current = Material(
                id=argument or f"material_{len(materials) + 1}",
                name=argument,
                source_path=path,
            )
            materials.append(current)
            continue
        if current is None:
            continue
        try:
            if keyword == "Kd":
                values = [float(v) for v in argument.split()]
                if len(values) >= 3:
                    current.base_color = (values[0], values[1], values[2])
            elif keyword == "Ks":
                values = [float(v) for v in argument.split()]
                if len(values) >= 3:
                    current.specular_color = (values[0], values[1], values[2])
            elif keyword == "d":
                current.opacity = float(argument)
            elif keyword == "Ns":
                # Phong exponent -> roughness approximation in [0, 1].
                current.roughness = max(0.0, 1.0 - min(1.0, float(argument) / 1000.0))
            elif keyword == "map_Kd":
                current.base_color_map = argument
            elif keyword in ("map_Bump", "bump"):
                current.bump_map = argument
        except ValueError:
            continue
    return materials


def _merge_mtl_into(target: Material, source: Material) -> None:
    target.base_color = source.base_color
    target.specular_color = source.specular_color
    target.opacity = source.opacity
    target.roughness = source.roughness
    if source.base_color_map:
        target.base_color_map = source.base_color_map
    if source.bump_map:
        target.bump_map = source.bump_map
