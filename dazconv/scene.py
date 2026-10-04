"""Scene assembly: turn Daz files into a neutral, export-ready model.

One input file produces a :class:`SceneModel` holding one :class:`MeshAsset`
per geometry.  Each asset carries its own geometry, UV set, skeleton, skin
binding, morphs and materials, which is what the FBX exporter needs.

Supported inputs
----------------
``.duf``/``.daz``  DSON scenes.  Geometry is either inline (``geometry_library``)
                   or referenced out to ``.dsf`` files, which are resolved
                   through :class:`~dazconv.library.ContentLibrary`.
``.dsf``           DSON assets -- figures, props, UV sets, morphs.
``.obj``           Wavefront geometry with an optional ``.mtl``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import dson
from .geometry import Geometry
from .library import ContentLibrary, normalize_url, split_uri
from .materials import Material, sanitize_material_name
from .mathutil import IDENTITY, Mat4
from .morphs import Morph
from .obj_import import (
    build_uv_set as build_obj_uv_set,
    corner_uvs_from_obj as obj_corner_uvs,
    read_obj,
)
from .rig import Skeleton
from .skinning import SkinBinding, VertexInfluences, build_influences
from .uvset import UVSet, placeholder_uvs

DSON_INPUT_EXTENSIONS = (".duf", ".daz", ".dsf", ".dsb", ".dsa", ".dse")


@dataclass
class _OwnerContext:
    """Rig/skin/morph/UV data belonging to one source document."""

    skeleton: Optional[Skeleton] = None
    skins: List[SkinBinding] = field(default_factory=list)
    morphs: List[Morph] = field(default_factory=list)
    uv_sets: Dict[Tuple[str, str], UVSet] = field(default_factory=dict)
    materials: Dict[str, List[Material]] = field(default_factory=dict)


@dataclass
class MeshAsset:
    id: str
    name: str
    geometry: Geometry
    uv_set: Optional[UVSet] = None
    corner_uvs: List[Tuple[float, float]] = field(default_factory=list)
    skeleton: Optional[Skeleton] = None
    skin: Optional[SkinBinding] = None
    influences: Optional[VertexInfluences] = None
    materials: List[Material] = field(default_factory=list)
    morphs: List[Morph] = field(default_factory=list)
    world_matrix: Mat4 = IDENTITY
    source_path: str = ""
    is_figure: bool = False
    subdivision_level: int = 0
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------- helpers --
    @property
    def is_skinned(self) -> bool:
        return bool(self.skin and self.influences and self.skeleton)

    def material_for_face(self, face_index: int) -> Optional[Material]:
        polygon = self.geometry.polygons[face_index]
        if 0 <= polygon.material_index < len(self.materials):
            return self.materials[polygon.material_index]
        return None

    def stats(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "vertices": self.geometry.vertex_count,
            "polygons": self.geometry.polygon_count,
            "corners": self.geometry.corner_count,
            "has_uvs": self.uv_set is not None and bool(self.corner_uvs),
            "bones": self.skeleton.stats()["bone_count"] if self.skeleton else 0,
            "weighted_verts": len(self.influences.joint_weights) if self.influences else 0,
            "morphs": len(self.morphs),
            "materials": len(self.materials),
            "is_figure": self.is_figure,
            "subdivision": self.geometry.is_subdivision(),
        }

    def preview_data(self, max_vertices: int = 6000) -> dict:
        """Downsampled wireframe payload for the browser preview."""
        geometry = self.geometry
        stride = max(1, geometry.vertex_count // max_vertices)
        vertices = geometry.vertices[::stride]
        remap = {
            original: new
            for new, original in enumerate(range(0, geometry.vertex_count, stride))
        }
        edges = []
        for a, b in geometry.edge_list():
            if a in remap and b in remap and remap[a] != remap[b]:
                edges.append((remap[a], remap[b]))
        return {
            "name": self.name or self.id,
            "vertices": [[round(c, 3) for c in v] for v in vertices],
            "edges": edges,
            "bounds": list(geometry.bounds()),
            "stats": self.stats(),
        }


@dataclass
class SceneModel:
    name: str
    source_path: str
    meshes: List[MeshAsset] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    content_roots: List[str] = field(default_factory=list)

    @property
    def vertex_total(self) -> int:
        return sum(m.geometry.vertex_count for m in self.meshes)

    @property
    def polygon_total(self) -> int:
        return sum(m.geometry.polygon_count for m in self.meshes)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "source": os.path.basename(self.source_path) if self.source_path else "",
            "meshes": len(self.meshes),
            "vertices": self.vertex_total,
            "polygons": self.polygon_total,
            "materials": len({m.id for asset in self.meshes for m in asset.materials}),
            "bones": sum(1 for asset in self.meshes if asset.skeleton for _ in asset.skeleton.bones),
            "morphs": sum(len(asset.morphs) for asset in self.meshes),
            "warnings": len(self.warnings),
        }


# --------------------------------------------------------------- loading --
class SceneLoader:
    """Loads Daz files and resolves cross-file references."""

    def __init__(
        self,
        library: Optional[ContentLibrary] = None,
        flip_v: bool = True,
        max_influences: int = 4,
        min_weight: float = 1e-5,
        load_morphs: bool = True,
    ) -> None:
        self.library = library or ContentLibrary()
        self.flip_v = flip_v
        self.max_influences = max_influences
        self.min_weight = min_weight
        self.load_morphs = load_morphs
        self._documents: Dict[str, dict] = {}
        self._owner_cache: Dict[str, "_OwnerContext"] = {}
        self.warnings: List[str] = []

    # ------------------------------------------------------------- public --
    def load(self, path: str) -> SceneModel:
        path = os.path.abspath(path)
        extension = os.path.splitext(path)[1].lower()
        scene = SceneModel(
            name=os.path.splitext(os.path.basename(path))[0],
            source_path=path,
            content_roots=list(self.library.roots),
        )
        if extension == ".obj":
            self._load_obj(path, scene)
        elif extension in DSON_INPUT_EXTENSIONS:
            self._load_dson(path, scene)
        else:
            raise ValueError(
                f"unsupported input '{extension}' - expected .duf, .daz, .dsf or .obj"
            )
        scene.warnings.extend(self.warnings)
        return scene

    # -------------------------------------------------------------- OBJ --
    def _load_obj(self, path: str, scene: SceneModel) -> None:
        model = read_obj(path)
        scene.warnings.extend(model.warnings)
        materials = list(model.materials)
        for geometry in model.geometries:
            asset = MeshAsset(
                id=geometry.id,
                name=geometry.name or geometry.id,
                geometry=geometry,
                source_path=path,
                materials=materials,
            )
            asset.uv_set = build_obj_uv_set(geometry, flip_v=self.flip_v)
            asset.corner_uvs = obj_corner_uvs(geometry, flip_v=self.flip_v)
            if not asset.corner_uvs:
                asset.corner_uvs = placeholder_uvs(geometry)
                asset.warnings.append(f"OBJ object '{geometry.id}' has no texture coordinates")
            scene.meshes.append(asset)

    # ------------------------------------------------------------- DSON --
    def _load_dson(self, path: str, scene: SceneModel) -> None:
        document = self._load_document(path)
        if document is None:
            return

        # geometry_id -> (geometry, owning document, owning path).  A scene file
        # usually only *references* geometry, so the rig/skin/UV data lives in
        # the referenced .dsf rather than in the scene itself.
        owned: Dict[str, tuple] = {}
        for geometry_id, geometry in self._collect_geometries(document, path).items():
            owned[geometry_id] = (geometry, document, path)
        if not owned:
            self._collect_referenced_geometries(document, owned)
        if not owned:
            self.warnings.append(
                f"{os.path.basename(path)} contains no geometry; it may be a pose, "
                "material or UV preset rather than an asset"
            )
            return

        scene_materials = self._collect_materials(document)

        for geometry_id, (geometry, owner_doc, owner_path) in owned.items():
            context = self._owner_context(owner_doc, owner_path)
            asset = MeshAsset(
                id=geometry_id,
                name=geometry.name or geometry_id,
                geometry=geometry,
                source_path=geometry.source_uri or owner_path,
                subdivision_level=geometry.current_subdivision_level,
            )
            asset.uv_set = self._resolve_uv_set(geometry, context.uv_sets)
            if asset.uv_set is not None:
                asset.corner_uvs = asset.uv_set.corner_uvs(geometry, flip_v=self.flip_v)
                asset.warnings.extend(asset.uv_set.sanity_report(geometry))
            if not asset.corner_uvs:
                asset.corner_uvs = placeholder_uvs(geometry)
                asset.warnings.append(f"no UV set resolved for geometry '{geometry_id}'")

            asset.skin = next((s for s in context.skins if s.geometry_id == geometry_id), None)
            asset.skeleton = context.skeleton
            asset.is_figure = asset.skin is not None or any(
                node.get("type") == "figure" for node in owner_doc.get("node_library") or []
            )
            if asset.skin is not None:
                if asset.skeleton is None:
                    asset.warnings.append(
                        f"skin binding for '{geometry_id}' found but the owning file "
                        "has no node_library"
                    )
                else:
                    asset.warnings.extend(asset.skin.validate(geometry.vertex_count))
                    asset.influences = build_influences(
                        asset.skin, asset.skeleton, geometry.vertex_count,
                        self.max_influences, self.min_weight
                    )
            asset.materials = scene_materials.get(geometry_id) or context.materials.get(
                geometry_id
            ) or []
            asset.morphs = [m for m in context.morphs if m.matches_geometry(geometry)]
            scene.meshes.append(asset)

        seen: set = set()
        for asset in scene.meshes:
            for material in asset.materials:
                if material.id not in seen:
                    seen.add(material.id)
                    scene.materials.append(material)

    # ------------------------------------------------------------ helpers --
    def _load_document(self, path_or_url: str) -> Optional[dict]:
        """Load (and cache) a DSON document by path or Daz content URL."""
        path = path_or_url
        if not os.path.isfile(path):
            resolved = self.library.resolve(path_or_url)
            if not resolved:
                self.warnings.append(f"could not resolve Daz reference: {path_or_url}")
                return None
            path = resolved
        cache_key = os.path.normcase(os.path.abspath(path))
        if cache_key in self._documents:
            return self._documents[cache_key]
        try:
            document = dson.load_dson_file(path)
        except dson.DsonError as error:
            self.warnings.append(f"could not parse {os.path.basename(path)}: {error}")
            return None
        if not isinstance(document, dict):
            self.warnings.append(f"unexpected DSON root type in {os.path.basename(path)}")
            return None
        self._documents[cache_key] = document
        return document

    def _collect_geometries(self, document: dict, path: str) -> Dict[str, Geometry]:
        out: Dict[str, Geometry] = {}
        for entry in document.get("geometry_library") or []:
            if not isinstance(entry, dict):
                continue
            geometry = Geometry.from_dson(entry, uri=path)
            if not geometry.vertices:
                continue
            out[geometry.id] = geometry
        return out

    def _collect_referenced_geometries(self, document: dict, owned: Dict[str, tuple]) -> None:
        """Follow ``scene.nodes[].geometries[].url`` into external .dsf files."""
        scene_block = document.get("scene") or {}
        for node in scene_block.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            for reference in node.get("geometries") or []:
                if not isinstance(reference, dict):
                    continue
                url = reference.get("url") or reference.get("source") or ""
                if not url or url.startswith("#"):
                    continue
                self._merge_external_geometry(url, owned)

    def _merge_external_geometry(self, url: str, owned: Dict[str, tuple]) -> None:
        file_part, object_id = split_uri(url)
        if not file_part:
            return
        resolved = self.library.resolve(file_part)
        if not resolved:
            self.warnings.append(
                f"missing asset for reference '{normalize_url(file_part)}' - "
                "add its parent folder to the content directories"
            )
            return
        external = self._load_document(resolved)
        if external is None:
            return
        for entry in external.get("geometry_library") or []:
            if not isinstance(entry, dict):
                continue
            geometry = Geometry.from_dson(entry, uri=resolved)
            if not geometry.vertices:
                continue
            if object_id and geometry.id != object_id:
                continue
            owned[geometry.id] = (geometry, external, resolved)

    # --------------------------------------------------- owner documents --
    def _owner_context(self, document: dict, path: str) -> "_OwnerContext":
        """Collect rig/skin/morph/UV data from the document that owns a geometry."""
        key = os.path.normcase(os.path.abspath(path))
        cached = self._owner_cache.get(key)
        if cached is not None:
            return cached

        modifiers = [m for m in document.get("modifier_library") or [] if isinstance(m, dict)]
        context = _OwnerContext(
            skeleton=(
                Skeleton.from_node_library(document.get("node_library") or [])
                if document.get("node_library")
                else None
            ),
            skins=[s for s in (SkinBinding.from_dson(m) for m in modifiers) if s is not None],
            morphs=(
                [m for m in (Morph.from_dson(mod, source_path=path) for mod in modifiers) if m]
                if self.load_morphs
                else []
            ),
            uv_sets=self._collect_uv_sets(document, path),
            materials=self._collect_materials(document),
        )
        self._owner_cache[key] = context
        return context

    def _collect_uv_sets(self, document: dict, path: str) -> Dict[Tuple[str, str], UVSet]:
        """Index UV sets by ``(normalised file, id)`` and by id alone."""
        index: Dict[Tuple[str, str], UVSet] = {}
        for entry in document.get("uv_set_library") or []:
            if not isinstance(entry, dict):
                continue
            uv_set = UVSet.from_dson(entry, uri=path)
            index[("", uv_set.id)] = uv_set
            index[(normalize_url(path), uv_set.id)] = uv_set
        return index

    def _resolve_uv_set(
        self, geometry: Geometry, index: Dict[Tuple[str, str], UVSet]
    ) -> Optional[UVSet]:
        """Find the geometry's default UV set, following the reference if needed."""
        reference = geometry.default_uv_set
        if reference:
            file_part, object_id = split_uri(reference)
            direct = index.get((normalize_url(file_part), object_id)) or index.get(("", object_id))
            if direct is not None:
                return direct
            resolved = self.library.resolve(file_part) if file_part else None
            if resolved:
                external = self._load_document(resolved)
                if external is not None:
                    for entry in external.get("uv_set_library") or []:
                        if isinstance(entry, dict):
                            candidate = UVSet.from_dson(entry, uri=resolved)
                            if candidate.id == object_id or not object_id:
                                return candidate
        # Fall back to any UV set in this document.
        for (_file, _id), uv_set in index.items():
            return uv_set
        return None

    def _collect_materials(self, document: dict) -> Dict[str, List[Material]]:
        """Build per-geometry material lists in ``polygon_material_groups`` order."""
        base: Dict[str, Material] = {}
        for entry in document.get("material_library") or []:
            if isinstance(entry, dict) and entry.get("id"):
                base[entry["id"]] = Material.from_dson(entry)

        grouped: Dict[str, List[Material]] = {}
        used_names: set = set()
        scene_block = document.get("scene") or {}
        for entry in scene_block.get("materials") or []:
            if not isinstance(entry, dict):
                continue
            _file, geometry_ref = split_uri(entry.get("geometry") or "")
            parent_ref = entry.get("url") or ""
            material = Material.from_dson(entry)
            parent_id = parent_ref.split("#")[-1] if parent_ref else ""
            if parent_id and parent_id in base:
                merged = base[parent_id]
                material = _merge_material(merged, material)
            material.id = sanitize_material_name(material.id or parent_id or "Material", used_names)
            grouped.setdefault(geometry_ref or "", []).append(material)

        # Geometries with no scene materials still get their group names.
        return grouped


def _merge_material(base: Material, override: Material) -> Material:
    """Overlay a scene material's channels on its library definition."""
    merged = Material.from_dson({})
    merged.id = override.id or base.id
    merged.name = override.name or base.name
    merged.label = override.label or base.label
    merged.geometry_ref = override.geometry_ref or base.geometry_ref
    merged.groups = override.groups or base.groups
    merged.channels = {**base.channels, **override.channels}
    for attribute in (
        "base_color",
        "base_color_map",
        "specular_color",
        "normal_map",
        "bump_map",
        "opacity_map",
        "emissive_color",
    ):
        value = getattr(override, attribute, None)
        setattr(merged, attribute, value if value else getattr(base, attribute))
    for attribute in ("roughness", "metallic", "opacity", "emissive_strength"):
        override_value = getattr(override, attribute)
        base_value = getattr(base, attribute)
        setattr(merged, attribute, override_value if override_value != base_value else base_value)
    merged.double_sided = override.double_sided or base.double_sided
    return merged
