"""Walks a DSON document and produces an IR Scene.

Supports two input shapes:

* **Scene files** (``.duf`` saved from File > Save, or a character preset) --
  the ``scene`` block holds node/geometry/material/modifier *instances* that
  reference *definitions* inside library ``.dsf`` files.
* **Asset files** (``.dsf``/``.duf`` containing ``node_library`` and
  ``geometry_library`` directly) -- converted straight from the libraries,
  which is how you grab a bare Genesis figure with no scene around it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import math3d
from ..dson import channels as ch
from ..dson import uri as uri_mod
from ..dson.library import ContentLibrary
from ..ir import Bone, Figure, Material, Mesh, Scene, Skeleton
from . import geometry as geo
from . import morphs as morph_mod
from . import skinning
from .materials import MaterialBuilder, default_material, sanitize


@dataclass
class ConvertOptions:
    """Everything the user can tweak about a conversion."""
    content_dirs: List[str] = field(default_factory=list)
    include_morphs: bool = True
    morph_filter: List[str] = field(default_factory=list)
    include_hd_morphs: bool = False
    include_skeleton: bool = True
    max_influences: int = 8
    pose: str = "rest"                 # rest | scene
    merge_meshes: bool = False
    merge_skeletons: bool = True       # conformed items share the figure rig
    include_hidden: bool = False
    include_geografts: bool = True
    scale: float = 1.0                 # extra multiplier on top of unit conv
    generate_uvs_if_missing: bool = True
    collapse_twist_bones: bool = False
    limit_vertices: int = 0            # 0 = no limit (safety valve for UI)

    def copy(self) -> "ConvertOptions":
        import copy as _c
        return _c.deepcopy(self)


@dataclass
class _NodeInst:
    """A node instance resolved against its library definition."""
    id: str
    name: str
    label: str
    type: str
    parent: Optional[str]
    center_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    translation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    general_scale: float
    rotation_order: str
    source_file: Optional[str]
    definition: Dict[str, Any]
    instance: Dict[str, Any]
    geometries: List[Dict[str, Any]] = field(default_factory=list)
    conform_target: Optional[str] = None
    children: List[str] = field(default_factory=list)
    visible: bool = True

    @property
    def is_bone(self) -> bool:
        return self.type == "bone"


class SceneBuilder:
    def __init__(self, library: ContentLibrary,
                 options: Optional[ConvertOptions] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        self.library = library
        self.options = options or ConvertOptions()
        self.scene = Scene()
        self._progress = progress or (lambda frac, msg: None)
        self._nodes: Dict[str, _NodeInst] = {}
        self._order: List[str] = []
        self._world: Dict[str, math3d.Mat4] = {}
        self._geom_owner: Dict[str, str] = {}      # geometry instance id -> node id
        self._materials_by_geom: Dict[str, List[Dict[str, Any]]] = {}
        self._modifiers_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        self._matbuilder: Optional[MaterialBuilder] = None
        self._missing_files: List[str] = []

    # ------------------------------------------------------------------ API
    def build_from_file(self, path: str) -> Scene:
        path = os.path.abspath(path)
        self.library.add_root_for_file(path)
        for d in self.options.content_dirs:
            self.library.add_root(d)

        self._progress(0.02, "Reading %s" % os.path.basename(path))
        doc = self.library.load_document(path)
        self.scene.name = os.path.splitext(os.path.basename(path))[0]
        self.scene.source_files.append(path)
        self._matbuilder = MaterialBuilder(self.library, path, warn=self.scene.warn)

        if isinstance(doc.get("scene"), dict):
            self._build_scene(doc, path)
        else:
            self._build_asset(doc, path)

        self._finalize()
        return self.scene

    # ----------------------------------------------------------- scene mode
    def _build_scene(self, doc: Dict[str, Any], path: str) -> None:
        scene_block = doc["scene"]
        nodes = scene_block.get("nodes") or []
        self._progress(0.05, "Resolving %d nodes" % len(nodes))

        for inst in nodes:
            if not isinstance(inst, dict):
                continue
            node = self._resolve_node_instance(inst, path)
            if node:
                self._nodes[node.id] = node
                self._order.append(node.id)

        self._link_children()
        self._compute_world_matrices()
        self._index_materials(scene_block, path)
        self._index_modifiers(scene_block, path)
        self._build_figures(path)

    def _resolve_node_instance(self, inst: Dict[str, Any], base: str,
                               local_doc: Optional[Dict[str, Any]] = None
                               ) -> Optional[_NodeInst]:
        url = inst.get("url") or ""
        definition: Dict[str, Any] = {}
        src: Optional[str] = base
        if url:
            found, src_path = self.library.find_asset(url, "node", base_file=base,
                                                      local_doc=local_doc)
            if found:
                definition = found
                src = src_path or base
            else:
                self._note_missing(url)

        merged: Dict[str, Any] = dict(definition)
        merged.update({k: v for k, v in inst.items() if k not in ("extra",)})

        preview = inst.get("preview") or definition.get("preview") or {}
        ntype = str(merged.get("type") or preview.get("type") or "node").lower()

        center = self._triplet(inst, definition, "center_point",
                               preview.get("center_point"))
        end = self._triplet(inst, definition, "end_point",
                            preview.get("end_point"), default=center)
        orient = self._triplet(inst, definition, "orientation", None)
        rot = self._triplet(inst, definition, "rotation", None)
        trans = self._triplet(inst, definition, "translation", None)
        scl = self._triplet(inst, definition, "scale", None, default=(1.0, 1.0, 1.0))
        gscale = ch.as_float(inst.get("general_scale",
                                      definition.get("general_scale")), 1.0)
        rorder = str(merged.get("rotation_order")
                     or preview.get("rotation_order") or "XYZ").upper()

        parent = inst.get("parent")
        parent_id = None
        if isinstance(parent, str) and parent:
            parent_id = uri_mod.parse(parent).asset_id or parent.lstrip("#")

        node = _NodeInst(
            id=str(inst.get("id") or definition.get("id") or "node"),
            name=str(inst.get("name") or definition.get("name") or inst.get("id") or "node"),
            label=str(inst.get("label") or definition.get("label") or ""),
            type=ntype,
            parent=parent_id,
            center_point=center,
            end_point=end,
            orientation=orient,
            rotation=rot,
            translation=trans,
            scale=scl,
            general_scale=gscale,
            rotation_order=rorder,
            source_file=src,
            definition=definition,
            instance=inst,
            geometries=list(inst.get("geometries") or definition.get("geometries") or []),
            visible=_node_visible(inst, definition),
        )
        ct = inst.get("conform_target")
        if isinstance(ct, str):
            node.conform_target = uri_mod.parse(ct).asset_id or ct.lstrip("#")
        for g in node.geometries:
            if isinstance(g, dict) and g.get("id"):
                self._geom_owner[str(g["id"])] = node.id
        return node

    @staticmethod
    def _triplet(inst: Dict[str, Any], definition: Dict[str, Any], key: str,
                 preview_val: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if key in inst:
            base = ch.axis_triplet(definition.get(key), default)
            return ch.axis_triplet(inst.get(key), base)
        if key in definition:
            return ch.axis_triplet(definition.get(key), default)
        if preview_val is not None:
            return ch.axis_triplet(preview_val, default)
        return default

    def _link_children(self) -> None:
        for nid, node in self._nodes.items():
            if node.parent and node.parent in self._nodes:
                self._nodes[node.parent].children.append(nid)
            elif node.parent:
                node.parent = None

    def _compute_world_matrices(self) -> None:
        """DSON node transform:  T(cp+t) * Ro * R * Ro^-1 * S * T(-cp)."""
        use_pose = self.options.pose == "scene"
        for nid in self._topo_order():
            node = self._nodes[nid]
            if use_pose:
                local = _dson_local_matrix(node)
            else:
                local = math3d.identity()
            parent_w = self._world.get(node.parent or "", math3d.identity())
            self._world[nid] = math3d.mat_mul(local, parent_w)

    def _topo_order(self) -> List[str]:
        out: List[str] = []
        seen = set()

        def visit(nid: str, depth: int = 0):
            if nid in seen or depth > 256:
                return
            node = self._nodes.get(nid)
            if node is None:
                return
            if node.parent and node.parent not in seen:
                visit(node.parent, depth + 1)
            if nid not in seen:
                seen.add(nid)
                out.append(nid)

        for nid in self._order:
            visit(nid)
        return out

    def _index_materials(self, scene_block: Dict[str, Any], base: str) -> None:
        for mat in scene_block.get("materials") or []:
            if not isinstance(mat, dict):
                continue
            gref = str(mat.get("geometry") or "")
            gid = uri_mod.parse(gref).asset_id or gref.lstrip("#")
            self._materials_by_geom.setdefault(gid, []).append(mat)

    def _index_modifiers(self, scene_block: Dict[str, Any], base: str) -> None:
        for mod in scene_block.get("modifiers") or []:
            if not isinstance(mod, dict):
                continue
            parent = str(mod.get("parent") or "")
            pid = uri_mod.parse(parent).asset_id or parent.lstrip("#")
            self._modifiers_by_parent.setdefault(pid, []).append(mod)

    # ----------------------------------------------------------- asset mode
    def _build_asset(self, doc: Dict[str, Any], path: str) -> None:
        """Convert a raw library file (figure .dsf, prop .duf, geometry .dsf)."""
        self._progress(0.1, "Reading asset libraries")
        node_lib = doc.get("node_library") or []
        geom_lib = doc.get("geometry_library") or []

        synthetic_nodes: List[Dict[str, Any]] = []
        for nd in node_lib:
            if not isinstance(nd, dict):
                continue
            inst = {
                "id": nd.get("id"),
                "url": "#%s" % nd.get("id"),
                "name": nd.get("name"),
                "label": nd.get("label"),
                "parent": nd.get("parent"),
            }
            if nd.get("geometries"):
                inst["geometries"] = nd["geometries"]
            synthetic_nodes.append(inst)

        if not synthetic_nodes and geom_lib:
            # geometry-only file: fabricate a holder node per geometry
            for g in geom_lib:
                if isinstance(g, dict):
                    synthetic_nodes.append({
                        "id": "%s_node" % g.get("id"),
                        "name": g.get("name") or g.get("id"),
                        "geometries": [{"id": g.get("id"), "url": "#%s" % g.get("id")}],
                    })

        for inst in synthetic_nodes:
            node = self._resolve_node_instance(inst, path, local_doc=doc)
            if node:
                # In asset mode definitions live in this very document.
                if not node.definition:
                    node.definition = _find_by_id(node_lib, node.id) or {}
                    node.type = str(node.definition.get("type") or node.type).lower()
                self._nodes[node.id] = node
                self._order.append(node.id)

        self._link_children()
        self._compute_world_matrices()
        self._build_figures(path, local_doc=doc)

    # ------------------------------------------------------------- figures
    def _build_figures(self, base: str, local_doc: Optional[Dict[str, Any]] = None) -> None:
        carriers = [n for n in (self._nodes[i] for i in self._order) if n.geometries]
        if not carriers:
            self.scene.warn("no geometry found in this file")
            return

        total = len(carriers)
        for i, node in enumerate(carriers):
            self._progress(0.15 + 0.6 * (i / max(1, total)),
                           "Building %s" % (node.label or node.name))
            if not node.visible and not self.options.include_hidden:
                continue
            try:
                fig = self._build_figure(node, base, local_doc)
            except Exception as exc:  # keep converting the rest of the scene
                self.scene.warn("failed to build '%s': %s" % (node.name, exc))
                continue
            if fig is not None:
                self.scene.figures.append(fig)

        if self.options.merge_skeletons:
            self._merge_conformed()

    def _build_figure(self, node: _NodeInst, base: str,
                      local_doc: Optional[Dict[str, Any]]) -> Optional[Figure]:
        geom_inst = node.geometries[0]
        if not isinstance(geom_inst, dict):
            return None
        geom_url = geom_inst.get("url") or ""
        geom_def, geom_src = (None, None)
        if geom_url:
            geom_def, geom_src = self.library.find_asset(
                geom_url, "geometry", base_file=node.source_file or base,
                local_doc=local_doc or _doc_of(self.library, node.source_file))
        if geom_def is None and local_doc is not None:
            geom_def = _find_by_id(local_doc.get("geometry_library") or [],
                                   uri_mod.parse(geom_url).asset_id)
            geom_src = base
        if geom_def is None:
            self._note_missing(geom_url)
            self.scene.warn("geometry for '%s' could not be loaded" % node.name)
            return None

        mesh_name = sanitize(node.label or node.name or str(geom_def.get("id")))
        mesh = geo.parse_geometry(geom_def, mesh_name)
        mesh.source_url = geom_url
        if not mesh.vertices or not mesh.polygons:
            self.scene.warn("geometry '%s' is empty" % mesh_name)
            return None
        if self.options.limit_vertices and len(mesh.vertices) > self.options.limit_vertices:
            self.scene.warn("'%s' exceeds the vertex limit (%d) and was skipped"
                            % (mesh_name, len(mesh.vertices)))
            return None

        geom_id = str(geom_inst.get("id") or geom_def.get("id") or "")
        self._apply_uvs(mesh, geom_def, geom_id, geom_src or base)
        self._apply_materials(mesh, geom_def, geom_id, geom_src or base)

        skeleton = None
        if self.options.include_skeleton:
            skeleton = self._build_skeleton(node)
        if skeleton is not None and len(skeleton):
            self._apply_skin(mesh, node, geom_def, geom_id, geom_src or base, skeleton)

        if self.options.include_morphs:
            self._apply_morphs(mesh, geom_id, node, geom_src or base, local_doc)

        world = self._world.get(node.id, math3d.identity())
        if self.options.pose == "scene" and mesh.skin and skeleton is not None:
            self._bake_pose(mesh, skeleton)
        elif list(world) != math3d.IDENTITY:
            geo.transform_mesh(mesh, world)

        mesh.normals = geo.compute_vertex_normals(mesh)

        kind = "figure" if skeleton and len(skeleton) else "prop"
        if _is_geograft(geom_def):
            kind = "geograft"
            if not self.options.include_geografts:
                return None

        return Figure(name=sanitize(node.name), label=node.label or node.name,
                      mesh=mesh, skeleton=skeleton, kind=kind,
                      parent=node.conform_target or node.parent,
                      source_file=node.source_file or base)

    # ------------------------------------------------------------------ UVs
    def _apply_uvs(self, mesh: Mesh, geom_def: Dict[str, Any], geom_id: str,
                   base: str) -> None:
        uv_url = None
        for mat_inst in self._materials_by_geom.get(geom_id, []):
            if mat_inst.get("uv_set"):
                uv_url = mat_inst["uv_set"]
                break
        uv_url = uv_url or geom_def.get("default_uv_set")
        uv_set = None
        if uv_url:
            uv_set, _src = self.library.find_asset(uv_url, "uv_set", base_file=base)
            if uv_set is None:
                self._note_missing(uv_url)
        if uv_set is None:
            # some geometry embeds its UVs inline
            inline = geom_def.get("uvs") or geom_def.get("uv_set")
            if isinstance(inline, dict):
                uv_set = inline
        if uv_set is not None:
            geo.apply_uv_set(mesh, uv_set)
        if not mesh.uvs and self.options.generate_uvs_if_missing:
            geo.generate_planar_uvs(mesh)
            self.scene.warn("'%s' had no UVs; generated a planar projection"
                            % mesh.name)

    # ------------------------------------------------------------ materials
    def _apply_materials(self, mesh: Mesh, geom_def: Dict[str, Any], geom_id: str,
                         base: str) -> None:
        group_names: List[str] = list(mesh.material_groups)
        instances = self._materials_by_geom.get(geom_id, [])

        by_group: Dict[str, Dict[str, Any]] = {}
        for inst in instances:
            for g in inst.get("groups") or []:
                by_group[str(g)] = inst

        builder = self._matbuilder or MaterialBuilder(self.library, base)
        materials: List[Material] = []
        for gname in group_names:
            inst = by_group.get(gname)
            lib_def = None
            if inst is not None:
                url = inst.get("url") or ""
                if url:
                    lib_def, _src = self.library.find_asset(url, "material",
                                                            base_file=base)
                materials.append(builder.build(lib_def or {}, inst, base_file=base,
                                               name=gname))
            else:
                materials.append(default_material(gname))
        if not materials:
            materials = [default_material(mesh.name)]
            mesh.polygon_material = [0] * len(mesh.polygons)
        mesh.materials = materials
        nmat = len(materials)
        mesh.polygon_material = [m if 0 <= m < nmat else 0
                                 for m in mesh.polygon_material]

    # ------------------------------------------------------------- skeleton
    def _build_skeleton(self, node: _NodeInst) -> Optional[Skeleton]:
        bones: List[_NodeInst] = []
        stack = list(node.children)
        while stack:
            nid = stack.pop(0)
            child = self._nodes.get(nid)
            if child is None:
                continue
            if child.is_bone:
                bones.append(child)
                stack.extend(child.children)
        if not bones:
            return None

        skel = Skeleton(name=sanitize(node.name) + "_rig")
        for b in bones:
            parent_name = None
            if b.parent and b.parent in self._nodes:
                p = self._nodes[b.parent]
                parent_name = p.name if p.is_bone else None
            bone = Bone(name=sanitize(b.name), label=b.label or b.name,
                        parent=sanitize(parent_name) if parent_name else None,
                        origin=b.center_point, end_point=b.end_point,
                        orientation=b.orientation,
                        rotation_order=b.rotation_order,
                        is_root=parent_name is None)
            skel.add(bone)
        skel.reindex()
        _compute_bone_matrices(skel)
        return skel

    def _apply_skin(self, mesh: Mesh, node: _NodeInst, geom_def: Dict[str, Any],
                    geom_id: str, base: str, skeleton: Skeleton) -> None:
        skin_block = self._find_skin_binding(node, geom_def, geom_id, base)
        if skin_block is None:
            return
        bone_names = [b.name for b in skeleton.bones]
        weights, warns = skinning.parse_skin_binding(
            skin_block, len(mesh.vertices), bone_filter=None,
            max_influences=self.options.max_influences)
        # map joint names (raw) onto sanitized skeleton bone names
        mapping: Dict[int, int] = {}
        kept_names: List[str] = []
        for i, raw in enumerate(weights.bone_names):
            s = sanitize(raw)
            idx = skeleton.index_of(s)
            if idx < 0:
                self.scene.warn("skin references unknown bone '%s'" % raw)
                continue
            if s not in kept_names:
                kept_names.append(s)
            mapping[i] = kept_names.index(s)
        skinning.remap_bones(weights, mapping, kept_names)
        skinning.prune_and_normalize(weights, self.options.max_influences)
        root = skeleton.roots()[0].name if skeleton.roots() else bone_names[0]
        if root not in kept_names:
            kept_names.append(root)
            weights.bone_names = kept_names
        skinning.fallback_rigid_bind(weights, kept_names.index(root))
        mesh.skin = weights
        for w in warns:
            self.scene.warn("%s: %s" % (mesh.name, w))

    def _find_skin_binding(self, node: _NodeInst, geom_def: Dict[str, Any],
                           geom_id: str, base: str) -> Optional[Dict[str, Any]]:
        # 1. a scene modifier instance parented to this geometry/node
        for mod in (self._modifiers_by_parent.get(geom_id, []) +
                    self._modifiers_by_parent.get(node.id, [])):
            url = mod.get("url")
            if not url:
                continue
            mdef, _src = self.library.find_asset(url, "modifier", base_file=base)
            if isinstance(mdef, dict) and isinstance(mdef.get("skin"), dict):
                return mdef["skin"]
        # 2. the figure's own .dsf almost always declares SkinBinding inline
        for src in filter(None, [node.source_file, base]):
            try:
                doc = self.library.load_document(src)
            except Exception:
                continue
            for mod in doc.get("modifier_library") or []:
                if not isinstance(mod, dict) or not isinstance(mod.get("skin"), dict):
                    continue
                skin = mod["skin"]
                target = str(skin.get("geometry") or mod.get("parent") or "")
                tid = uri_mod.parse(target).asset_id
                if not tid or tid == geom_id or tid == str(geom_def.get("id")):
                    return skin
        return None

    # ---------------------------------------------------------------- morphs
    def _apply_morphs(self, mesh: Mesh, geom_id: str, node: _NodeInst,
                      base: str, local_doc: Optional[Dict[str, Any]]) -> None:
        found: List = []
        seen_names = set()
        candidates = (self._modifiers_by_parent.get(geom_id, []) +
                      self._modifiers_by_parent.get(node.id, []))
        for mod in candidates:
            url = mod.get("url")
            mdef = None
            if url:
                mdef, _src = self.library.find_asset(url, "modifier", base_file=base)
            if mdef is None and local_doc is not None:
                mdef = _find_by_id(local_doc.get("modifier_library") or [],
                                   str(mod.get("id") or ""))
            if not isinstance(mdef, dict) or "morph" not in mdef:
                continue
            name = str(mod.get("id") or mdef.get("id") or "Morph")
            if name in seen_names:
                continue
            target = morph_mod.parse_morph(mdef, len(mesh.vertices), name=name)
            if target is None:
                continue
            chan = mod.get("channel")
            if isinstance(chan, dict):
                target.default_value = ch.as_float(chan, target.default_value)
            seen_names.add(name)
            found.append(target)

        # Asset mode: morphs can live in the same document.
        if local_doc is not None:
            for t in morph_mod.collect_from_library(local_doc, len(mesh.vertices)):
                if t.name not in seen_names:
                    seen_names.add(t.name)
                    found.append(t)

        mesh.morphs = morph_mod.filter_morphs(
            found, include=self.options.morph_filter,
            exclude_hd=not self.options.include_hd_morphs)

    # ----------------------------------------------------------------- pose
    def _bake_pose(self, mesh: Mesh, skeleton: Skeleton) -> None:
        """Linear-blend-skin the rest mesh into its scene pose."""
        mats: List[math3d.Mat4] = []
        for name in (mesh.skin.bone_names if mesh.skin else []):
            # The DSON node chain is identity at rest, so the posed world
            # matrix *is* the skinning matrix.
            nid = self._node_id_for_bone(name)
            mats.append(self._world.get(nid, math3d.identity()))
        if not mats or mesh.skin is None:
            return
        new_verts = []
        for vi, v in enumerate(mesh.vertices):
            inf = mesh.skin.influences[vi] if vi < len(mesh.skin.influences) else []
            if not inf:
                new_verts.append(v)
                continue
            x = y = z = 0.0
            for bi, w in inf:
                if bi >= len(mats):
                    continue
                p = math3d.transform_point(mats[bi], v)
                x += p[0] * w
                y += p[1] * w
                z += p[2] * w
            new_verts.append((x, y, z))
        mesh.vertices = new_verts
        for bone in skeleton.bones:
            nid = self._node_id_for_bone(bone.name)
            m = self._world.get(nid)
            if m:
                bone.world_matrix = math3d.mat_mul(bone.world_matrix, m)
                bone.origin = (bone.world_matrix[12], bone.world_matrix[13],
                               bone.world_matrix[14])

    def _node_id_for_bone(self, bone_name: str) -> str:
        for nid, node in self._nodes.items():
            if sanitize(node.name) == bone_name:
                return nid
        return bone_name

    # ------------------------------------------------------------ post pass
    def _merge_conformed(self) -> None:
        """Give conformed clothing/hair the skeleton of the figure it follows."""
        by_name = {f.name: f for f in self.scene.figures}
        for fig in self.scene.figures:
            if fig.skeleton and len(fig.skeleton):
                continue
            if not fig.parent:
                continue
            target = by_name.get(sanitize(fig.parent))
            if target is not None and target.skeleton:
                fig.skeleton = target.skeleton
                fig.kind = "conformed"

    def _note_missing(self, dson_uri: str) -> None:
        """Record an unresolvable reference once per *file* (not per asset).

        A missing Genesis .dsf would otherwise produce 300 identical warnings,
        one per bone, and bury the one thing the user needs to know.
        """
        file_path = uri_mod.parse(dson_uri or "").file_path
        if not file_path:
            return
        if file_path not in self._missing_files:
            self._missing_files.append(file_path)

    def _report_missing(self) -> None:
        if not self._missing_files:
            return
        shown = self._missing_files[:6]
        extra = len(self._missing_files) - len(shown)
        self.scene.warn(
            "%d referenced content file(s) could not be found. Add the Daz "
            "content directory that contains them and convert again."
            % len(self._missing_files))
        for f in shown:
            self.scene.warn("  missing: %s" % f)
        if extra > 0:
            self.scene.warn("  ...and %d more" % extra)
        roots = ", ".join(self.library.roots) or "none"
        self.scene.warn("  searched in: %s" % roots)

    def _finalize(self) -> None:
        if self.options.merge_meshes and len(self.scene.figures) > 1:
            parts = [f for f in self.scene.figures if f.mesh]
            merged = geo.merge_meshes([f.mesh for f in parts], name=self.scene.name)
            if not merged.normals:
                merged.normals = geo.compute_vertex_normals(merged)
            # Keep the rig if every skinned part was driven by the same one.
            rigs = {id(f.skeleton): f.skeleton for f in parts if f.skeleton}
            skeleton = list(rigs.values())[0] if len(rigs) == 1 else None
            if len(rigs) > 1:
                self.scene.warn("merged meshes use %d different skeletons; "
                                "the merged mesh is unskinned" % len(rigs))
                merged.skin = None
            keep = Figure(name=self.scene.name, label=self.scene.name,
                          mesh=merged, skeleton=skeleton,
                          kind="figure" if skeleton else "prop")
            self.scene.figures = [keep]

        scale = self.options.scale
        if scale and abs(scale - 1.0) > 1e-9:
            m = math3d.scaling((scale, scale, scale))
            for fig in self.scene.figures:
                if fig.mesh:
                    geo.transform_mesh(fig.mesh, m)
                if fig.skeleton:
                    for b in fig.skeleton.bones:
                        b.origin = tuple(c * scale for c in b.origin)  # type: ignore
                        b.end_point = tuple(c * scale for c in b.end_point)  # type: ignore
                    _compute_bone_matrices(fig.skeleton)

        self._report_missing()
        if self._matbuilder and self._matbuilder.missing_textures:
            self.scene.warn("%d texture file(s) could not be located; add the "
                            "matching content directory and re-run"
                            % len(self._matbuilder.missing_textures))
        self.scene.stats = self.scene.summary()
        self._progress(1.0, "Done")


# --------------------------------------------------------------- utilities
def _compute_bone_matrices(skel: Skeleton) -> None:
    """Rest world matrix per bone: orientation at the joint origin."""
    for bone in skel.bones:
        rot = math3d.euler_to_mat(bone.orientation, "XYZ")
        bone.world_matrix = math3d.compose(bone.origin, rot)
    for bone in skel.bones:
        if bone.parent:
            p = skel.get(bone.parent)
            if p is not None and bone.name not in p.children:
                p.children.append(bone.name)


def _dson_local_matrix(node: _NodeInst) -> math3d.Mat4:
    """T(cp+t) * Ro * R * Ro^-1 * S * T(-cp) in row-vector order."""
    cp = node.center_point
    t = node.translation
    ro = math3d.euler_to_mat(node.orientation, "XYZ")
    r = math3d.euler_to_mat(node.rotation, node.rotation_order)
    gs = node.general_scale
    s = math3d.scaling((node.scale[0] * gs, node.scale[1] * gs, node.scale[2] * gs))
    ro_inv = math3d.mat_inverse(ro)
    # row-vector: first applied is leftmost
    return math3d.mat_mul_many(
        math3d.translation((-cp[0], -cp[1], -cp[2])),
        s,
        ro_inv,
        r,
        ro,
        math3d.translation((cp[0] + t[0], cp[1] + t[1], cp[2] + t[2])),
    )


def _node_visible(inst: Dict[str, Any], definition: Dict[str, Any]) -> bool:
    for src in (inst, definition):
        extras = src.get("extra")
        if not isinstance(extras, list):
            continue
        for block in extras:
            if not isinstance(block, dict):
                continue
            for item in block.get("channels") or []:
                chan = item.get("channel") if isinstance(item, dict) else None
                if isinstance(chan, dict) and str(chan.get("id", "")).lower() == "visible":
                    return ch.as_bool(chan, True)
    return True


def _is_geograft(geom_def: Dict[str, Any]) -> bool:
    return isinstance(geom_def.get("graft"), dict)


def _find_by_id(lib: Sequence[Any], asset_id: str) -> Optional[Dict[str, Any]]:
    if not asset_id:
        return None
    for item in lib:
        if isinstance(item, dict) and str(item.get("id")) == asset_id:
            return item
    return None


def _doc_of(library: ContentLibrary, path: Optional[str]):
    if not path:
        return None
    try:
        return library.load_document(path)
    except Exception:
        return None


def convert_file(path: str, options: Optional[ConvertOptions] = None,
                 progress: Optional[Callable[[float, str], None]] = None) -> Scene:
    """Convenience wrapper: file path -> IR Scene."""
    opts = options or ConvertOptions()
    lib = ContentLibrary(opts.content_dirs)
    return SceneBuilder(lib, opts, progress).build_from_file(path)
