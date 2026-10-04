"""Intermediate representation: the format-neutral scene the exporters consume.

The DSON side of the converter fills these structures in; every exporter
(FBX / glTF / OBJ) reads only from here, so adding a new output format never
requires touching the Daz parsing code.

Units: all positions are in **centimetres** (Daz native) inside the IR.
Exporters scale as needed (glTF -> metres, FBX keeps cm).
Axis convention: Y up, Z forward-ish, right handed (Daz native).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import math3d


@dataclass
class TextureRef:
    """A bitmap used by a material channel."""
    path: str                       # absolute path on disk (may not exist)
    name: str = ""
    uv_set: int = 0
    tiling: Tuple[float, float] = (1.0, 1.0)
    offset: Tuple[float, float] = (0.0, 0.0)
    gamma: float = 0.0              # 0 = auto, 1 = linear data map, 2.2 = sRGB

    @property
    def exists(self) -> bool:
        import os
        return bool(self.path) and os.path.isfile(self.path)


@dataclass
class Material:
    name: str
    source_type: str = ""           # e.g. studio/material/uber_iray
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    base_color_map: Optional[TextureRef] = None
    metallic: float = 0.0
    metallic_map: Optional[TextureRef] = None
    roughness: float = 0.5
    roughness_map: Optional[TextureRef] = None
    specular: float = 0.5
    normal_map: Optional[TextureRef] = None
    normal_strength: float = 1.0
    bump_map: Optional[TextureRef] = None
    bump_strength: float = 0.0
    opacity: float = 1.0
    opacity_map: Optional[TextureRef] = None
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_map: Optional[TextureRef] = None
    emissive_strength: float = 0.0
    translucency: float = 0.0
    translucency_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translucency_map: Optional[TextureRef] = None
    sss_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    thin_walled: bool = False
    double_sided: bool = True
    alpha_blend: bool = False
    horizontal_tiles: float = 1.0
    vertical_tiles: float = 1.0
    extra: Dict[str, object] = field(default_factory=dict)

    def textures(self) -> List[TextureRef]:
        out = []
        for t in (self.base_color_map, self.metallic_map, self.roughness_map,
                  self.normal_map, self.bump_map, self.opacity_map,
                  self.emissive_map, self.translucency_map):
            if t is not None:
                out.append(t)
        return out


@dataclass
class MorphTarget:
    """A blend shape / shape key.  Deltas are sparse (index -> offset)."""
    name: str
    label: str = ""
    indices: List[int] = field(default_factory=list)
    deltas: List[Tuple[float, float, float]] = field(default_factory=list)
    default_value: float = 0.0

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class SkinWeights:
    """Per-vertex bone influences, already normalised and pruned."""
    bone_names: List[str] = field(default_factory=list)
    # per vertex: list of (bone_index, weight)
    influences: List[List[Tuple[int, float]]] = field(default_factory=list)

    def max_influences(self) -> int:
        return max((len(i) for i in self.influences), default=0)


@dataclass
class Mesh:
    name: str
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    # polygons: list of vertex-index tuples (tri or quad or ngon)
    polygons: List[Tuple[int, ...]] = field(default_factory=list)
    polygon_material: List[int] = field(default_factory=list)   # index into materials
    polygon_group: List[int] = field(default_factory=list)      # index into groups
    groups: List[str] = field(default_factory=list)             # face group names
    materials: List[Material] = field(default_factory=list)
    # Surface names exactly as written in the DSON geometry, in the order the
    # polygon_material indices refer to. Kept separate from ``materials``
    # because a surface can exist with no material assigned to it yet.
    material_groups: List[str] = field(default_factory=list)
    # UVs are per polygon-vertex via uv_indices into the uvs table
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    uv_indices: List[Tuple[int, ...]] = field(default_factory=list)
    normals: Optional[List[Tuple[float, float, float]]] = None  # per vertex
    skin: Optional[SkinWeights] = None
    morphs: List[MorphTarget] = field(default_factory=list)
    smooth: bool = True
    world_matrix: math3d.Mat4 = field(default_factory=math3d.identity)
    source_url: str = ""

    @property
    def num_triangles(self) -> int:
        return sum(max(0, len(p) - 2) for p in self.polygons)

    def triangulate(self) -> List[Tuple[int, int, int]]:
        """Fan triangulation, preserving winding."""
        tris: List[Tuple[int, int, int]] = []
        for poly in self.polygons:
            for i in range(1, len(poly) - 1):
                tris.append((poly[0], poly[i], poly[i + 1]))
        return tris


@dataclass
class Bone:
    name: str
    label: str = ""
    parent: Optional[str] = None
    # Rest pose, in figure/world space (cm)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_point: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # XYZ degrees
    rotation_order: str = "XYZ"
    world_matrix: math3d.Mat4 = field(default_factory=math3d.identity)
    children: List[str] = field(default_factory=list)
    is_root: bool = False

    @property
    def length(self) -> float:
        import math
        d = math3d.sub(self.end_point, self.origin)
        return math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


@dataclass
class Skeleton:
    name: str = "Armature"
    bones: List[Bone] = field(default_factory=list)
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def add(self, bone: Bone) -> Bone:
        self._index[bone.name] = len(self.bones)
        self.bones.append(bone)
        return bone

    def get(self, name: str) -> Optional[Bone]:
        i = self._index.get(name)
        return self.bones[i] if i is not None else None

    def index_of(self, name: str) -> int:
        return self._index.get(name, -1)

    def roots(self) -> List[Bone]:
        return [b for b in self.bones if not b.parent]

    def reindex(self) -> None:
        self._index = {b.name: i for i, b in enumerate(self.bones)}

    def __len__(self) -> int:
        return len(self.bones)


@dataclass
class Figure:
    """A mesh plus (optionally) the skeleton it is bound to."""
    name: str
    label: str = ""
    mesh: Optional[Mesh] = None
    skeleton: Optional[Skeleton] = None
    world_matrix: math3d.Mat4 = field(default_factory=math3d.identity)
    kind: str = "prop"              # figure | prop | geograft
    parent: Optional[str] = None
    source_file: str = ""


@dataclass
class Scene:
    name: str = "DazScene"
    figures: List[Figure] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    up_axis: str = "Y"
    units_per_meter: float = 100.0  # Daz works in centimetres

    def meshes(self) -> List[Mesh]:
        return [f.mesh for f in self.figures if f.mesh is not None]

    def all_materials(self) -> List[Material]:
        seen: Dict[int, Material] = {}
        for m in self.meshes():
            for mat in m.materials:
                seen[id(mat)] = mat
        return list(seen.values())

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def summary(self) -> Dict[str, object]:
        meshes = self.meshes()
        bones = sum(len(f.skeleton) for f in self.figures if f.skeleton)
        morphs = sum(len(m.morphs) for m in meshes)
        return {
            "figures": len(self.figures),
            "meshes": len(meshes),
            "vertices": sum(len(m.vertices) for m in meshes),
            "polygons": sum(len(m.polygons) for m in meshes),
            "triangles": sum(m.num_triangles for m in meshes),
            "materials": len(self.all_materials()),
            "bones": bones,
            "morphs": morphs,
            "warnings": len(self.warnings),
        }
