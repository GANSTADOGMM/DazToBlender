"""IR Scene -> binary FBX 7.4 document.

What gets written:

* ``Geometry``/``Model`` pairs with per-polygon-vertex normals, UVs
  (IndexToDirect, so UV seams survive) and per-polygon material indices
* ``Material`` (phong + Maya PBR user properties) with ``Texture``/``Video``
  objects for every map, either referenced or embedded
* ``Model::LimbNode`` bone hierarchy under a null "armature" model, with
  ``NodeAttribute::LimbNode`` size hints so bones draw at the right length
* ``Deformer::Skin`` + one ``Deformer::Cluster`` per bone, plus a
  ``Pose::BindPose`` so importers can restore the bind pose exactly
* ``Deformer::BlendShape`` + channel + ``Geometry::Shape`` per morph
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ... import math3d
from ...ir import Bone, Figure, Material, Mesh, Scene, Skeleton, TextureRef
from ...version import APP_NAME, __version__
from .binary import Node, fbx_name, write_document

FBX_VERSION = 7400
# FBX texture-channel names we connect maps to.
TEX_SLOTS = {
    "base_color_map": "DiffuseColor",
    "roughness_map": "ShininessExponent",
    "metallic_map": "ReflectionFactor",
    "normal_map": "NormalMap",
    "bump_map": "Bump",
    "opacity_map": "TransparencyFactor",
    "emissive_map": "EmissiveColor",
    "translucency_map": "SpecularColor",
}


@dataclass
class FbxOptions:
    embed_textures: bool = False
    copy_textures: bool = True
    texture_subdir: str = "textures"
    export_skin: bool = True
    export_morphs: bool = True
    units_per_cm: float = 1.0      # FBX default unit is centimetres
    axis_up: str = "Y"
    creator: str = "%s %s" % (APP_NAME, __version__)


class _IdGen:
    def __init__(self, start: int = 1000000):
        self._next = start

    def __call__(self) -> int:
        self._next += 1
        return self._next


class FbxWriter:
    def __init__(self, scene: Scene, options: Optional[FbxOptions] = None):
        self.scene = scene
        self.opts = options or FbxOptions()
        self.newid = _IdGen()
        self.objects = Node("Objects")
        self.connections = Node("Connections")
        self.counts: Dict[str, int] = {}
        self._texture_objects: Dict[str, int] = {}
        self._video_objects: Dict[str, int] = {}
        self._bone_models: Dict[int, Dict[str, int]] = {}   # id(skeleton) -> name->model id
        self._pose_entries: List[Tuple[int, math3d.Mat4]] = []
        self._out_dir = ""
        self._copied: Dict[str, str] = {}
        self.written_textures: List[str] = []

    # ------------------------------------------------------------------ API
    def write(self, path: str) -> Dict[str, Any]:
        self._out_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(self._out_dir, exist_ok=True)

        for figure in self.scene.figures:
            self._write_figure(figure)
        self._write_bind_pose()

        roots = [
            self._header(),
            self._global_settings(),
            self._documents(),
            Node("References"),
            self._definitions(),
            self.objects,
            self.connections,
            self._takes(),
        ]
        with open(path, "wb") as fh:
            write_document(fh, roots, FBX_VERSION)
        return {
            "path": path,
            "bytes": os.path.getsize(path),
            "objects": sum(self.counts.values()),
            "textures": list(self.written_textures),
        }

    # -------------------------------------------------------------- preamble
    def _header(self) -> Node:
        n = Node("FBXHeaderExtension")
        n.single("FBXHeaderVersion", "I", 1003)
        n.single("FBXVersion", "I", FBX_VERSION)
        n.single("EncryptionType", "I", 0)
        t = time.localtime()
        ct = n.child("CreationTimeStamp")
        ct.single("Version", "I", 1000)
        ct.single("Year", "I", t.tm_year)
        ct.single("Month", "I", t.tm_mon)
        ct.single("Day", "I", t.tm_mday)
        ct.single("Hour", "I", t.tm_hour)
        ct.single("Minute", "I", t.tm_min)
        ct.single("Second", "I", t.tm_sec)
        ct.single("Millisecond", "I", 0)
        n.single("Creator", "S", self.opts.creator)
        si = n.child("SceneInfo")
        si.add_string("GlobalInfo\x00\x01SceneInfo").add_string("UserData")
        si.single("Type", "S", "UserData")
        si.single("Version", "I", 100)
        meta = si.child("MetaData")
        meta.single("Version", "I", 100)
        meta.single("Title", "S", self.scene.name)
        meta.single("Subject", "S", "Converted from Daz Studio DSON")
        meta.single("Author", "S", APP_NAME)
        meta.single("Keywords", "S", "daz dson fbx")
        meta.single("Revision", "S", __version__)
        meta.single("Comment", "S", "")
        props = si.child("Properties70")
        props.prop70("DocumentUrl", "KString", "Url", "", "")
        props.prop70("SrcDocumentUrl", "KString", "Url", "", "")
        props.prop70("Original|ApplicationName", "KString", "", "", APP_NAME)
        props.prop70("Original|ApplicationVersion", "KString", "", "", __version__)
        props.prop70("LastSaved|ApplicationName", "KString", "", "", APP_NAME)
        return n

    def _global_settings(self) -> Node:
        n = Node("GlobalSettings")
        n.single("Version", "I", 1000)
        p = n.child("Properties70")
        up = 1 if self.opts.axis_up.upper() == "Y" else 2
        p.prop70("UpAxis", "int", "Integer", "", up)
        p.prop70("UpAxisSign", "int", "Integer", "", 1)
        p.prop70("FrontAxis", "int", "Integer", "", 2 if up == 1 else 1)
        p.prop70("FrontAxisSign", "int", "Integer", "", 1)
        p.prop70("CoordAxis", "int", "Integer", "", 0)
        p.prop70("CoordAxisSign", "int", "Integer", "", 1)
        p.prop70("OriginalUpAxis", "int", "Integer", "", up)
        p.prop70("OriginalUpAxisSign", "int", "Integer", "", 1)
        p.prop70("UnitScaleFactor", "double", "Number", "", 1.0)
        p.prop70("OriginalUnitScaleFactor", "double", "Number", "", 1.0)
        p.prop70("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0)
        p.prop70("DefaultCamera", "KString", "", "", "Producer Perspective")
        p.prop70("TimeMode", "enum", "", "", 11)
        p.prop70("TimeSpanStart", "KTime", "Time", "", 0)
        p.prop70("TimeSpanStop", "KTime", "Time", "", 46186158000)
        p.prop70("CustomFrameRate", "double", "Number", "", 30.0)
        return n

    def _documents(self) -> Node:
        n = Node("Documents")
        n.single("Count", "I", 1)
        doc = n.child("Document")
        doc_id = self.newid()
        doc.add_int64(doc_id).add_string("").add_string("Scene")
        p = doc.child("Properties70")
        p.prop70("SourceObject", "object", "", "")
        p.prop70("ActiveAnimStackName", "KString", "", "", "")
        doc.single("RootNode", "L", 0)
        return n

    def _definitions(self) -> Node:
        n = Node("Definitions")
        n.single("Version", "I", 100)
        total = sum(self.counts.values())
        n.single("Count", "I", total)
        gt = n.child("ObjectType")
        gt.add_string("GlobalSettings")
        gt.single("Count", "I", 1)
        for otype, count in self.counts.items():
            ot = n.child("ObjectType")
            ot.add_string(otype)
            ot.single("Count", "I", count)
        return n

    def _takes(self) -> Node:
        n = Node("Takes")
        n.single("Current", "S", "")
        return n

    # ---------------------------------------------------------------- figures
    def _write_figure(self, figure: Figure) -> None:
        mesh = figure.mesh
        if mesh is None or not mesh.polygons:
            return
        geom_id = self._geometry(mesh)
        model_id = self._model(mesh.name, "Mesh", math3d.identity())
        self._connect(geom_id, model_id)
        self._connect(model_id, 0)
        self._pose_entries.append((model_id, math3d.identity()))

        for mat in mesh.materials:
            mat_id = self._material(mat)
            self._connect(mat_id, model_id)

        if figure.skeleton and mesh.skin and self.opts.export_skin:
            bone_ids = self._skeleton(figure.skeleton)
            self._skin(mesh, figure.skeleton, geom_id, model_id, bone_ids)
        if mesh.morphs and self.opts.export_morphs:
            self._blend_shapes(mesh, geom_id)

    # --------------------------------------------------------------- geometry
    def _geometry(self, mesh: Mesh) -> int:
        gid = self.newid()
        n = self.objects.child("Geometry")
        n.add_int64(gid).add_raw_name(fbx_name(mesh.name, "Geometry")).add_string("Mesh")
        self._bump("Geometry")

        verts: List[float] = []
        for v in mesh.vertices:
            verts.extend((v[0], v[1], v[2]))
        n.child("Vertices").add_float64_array(verts)

        indices: List[int] = []
        for poly in mesh.polygons:
            for i, vi in enumerate(poly):
                indices.append(vi if i < len(poly) - 1 else (-vi - 1))
        n.child("PolygonVertexIndex").add_int32_array(indices)

        edges = _edge_list(mesh)
        if edges:
            n.child("Edges").add_int32_array(edges)
        n.single("GeometryVersion", "I", 124)

        self._layer_normals(n, mesh)
        self._layer_uvs(n, mesh)
        self._layer_materials(n, mesh)
        self._layer_smoothing(n, mesh)

        layer = n.child("Layer")
        layer.add_int32(0)
        layer.single("Version", "I", 100)
        for etype in ("LayerElementNormal", "LayerElementSmoothing",
                      "LayerElementUV", "LayerElementMaterial"):
            if not n.has_child(etype):
                continue
            le = layer.child("LayerElement")
            le.single("Type", "S", etype)
            le.single("TypedIndex", "I", 0)
        return gid

    def _layer_normals(self, geom: Node, mesh: Mesh) -> None:
        normals = mesh.normals or []
        if not normals:
            return
        n = geom.child("LayerElementNormal")
        n.add_int32(0)
        n.single("Version", "I", 101)
        n.single("Name", "S", "")
        n.single("MappingInformationType", "S", "ByPolygonVertex")
        n.single("ReferenceInformationType", "S", "Direct")
        flat: List[float] = []
        for poly in mesh.polygons:
            for vi in poly:
                nv = normals[vi] if vi < len(normals) else (0.0, 1.0, 0.0)
                flat.extend((nv[0], nv[1], nv[2]))
        n.child("Normals").add_float64_array(flat)
        n.child("NormalsW").add_float64_array([1.0] * (len(flat) // 3))

    def _layer_uvs(self, geom: Node, mesh: Mesh) -> None:
        if not mesh.uvs:
            return
        n = geom.child("LayerElementUV")
        n.add_int32(0)
        n.single("Version", "I", 101)
        n.single("Name", "S", "UVMap")
        n.single("MappingInformationType", "S", "ByPolygonVertex")
        n.single("ReferenceInformationType", "S", "IndexToDirect")
        flat: List[float] = []
        for uv in mesh.uvs:
            flat.extend((uv[0], uv[1]))
        n.child("UV").add_float64_array(flat)
        idx: List[int] = []
        nuv = len(mesh.uvs)
        for pi, poly in enumerate(mesh.polygons):
            row = mesh.uv_indices[pi] if pi < len(mesh.uv_indices) else poly
            for ui in row:
                idx.append(ui if 0 <= ui < nuv else 0)
        n.child("UVIndex").add_int32_array(idx)

    def _layer_materials(self, geom: Node, mesh: Mesh) -> None:
        n = geom.child("LayerElementMaterial")
        n.add_int32(0)
        n.single("Version", "I", 101)
        n.single("Name", "S", "")
        if len(mesh.materials) <= 1:
            n.single("MappingInformationType", "S", "AllSame")
            n.single("ReferenceInformationType", "S", "IndexToDirect")
            n.child("Materials").add_int32_array([0])
            return
        n.single("MappingInformationType", "S", "ByPolygon")
        n.single("ReferenceInformationType", "S", "IndexToDirect")
        nmat = len(mesh.materials)
        mats = [m if 0 <= m < nmat else 0 for m in
                (mesh.polygon_material or [0] * len(mesh.polygons))]
        if len(mats) < len(mesh.polygons):
            mats += [0] * (len(mesh.polygons) - len(mats))
        n.child("Materials").add_int32_array(mats[:len(mesh.polygons)])

    def _layer_smoothing(self, geom: Node, mesh: Mesh) -> None:
        n = geom.child("LayerElementSmoothing")
        n.add_int32(0)
        n.single("Version", "I", 102)
        n.single("Name", "S", "")
        n.single("MappingInformationType", "S", "ByPolygon")
        n.single("ReferenceInformationType", "S", "Direct")
        n.child("Smoothing").add_int32_array([1 if mesh.smooth else 0] * len(mesh.polygons))

    # ------------------------------------------------------------------ model
    def _model(self, name: str, subtype: str, matrix: Sequence[float],
               extra_props=None) -> int:
        mid = self.newid()
        n = self.objects.child("Model")
        n.add_int64(mid).add_raw_name(fbx_name(name, "Model")).add_string(subtype)
        self._bump("Model")
        n.single("Version", "I", 232)
        p = n.child("Properties70")
        p.prop70("RotationActive", "bool", "", "", 1)
        p.prop70("InheritType", "enum", "", "", 1)
        p.prop70("ScalingMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
        p.prop70("DefaultAttributeIndex", "int", "Integer", "", 0)
        t, r, s = math3d.decompose(matrix)
        p.prop70("Lcl Translation", "Lcl Translation", "", "A", float(t[0]), float(t[1]), float(t[2]))
        p.prop70("Lcl Rotation", "Lcl Rotation", "", "A", float(r[0]), float(r[1]), float(r[2]))
        p.prop70("Lcl Scaling", "Lcl Scaling", "", "A", float(s[0]), float(s[1]), float(s[2]))
        if extra_props:
            for args in extra_props:
                p.prop70(*args)
        n.single("Shading", "C", True)
        n.single("Culling", "S", "CullingOff")
        return mid

    # --------------------------------------------------------------- material
    def _material(self, mat: Material) -> int:
        mid = self.newid()
        n = self.objects.child("Material")
        n.add_int64(mid).add_raw_name(fbx_name(mat.name, "Material")).add_string("")
        self._bump("Material")
        n.single("Version", "I", 102)
        n.single("ShadingModel", "S", "phong")
        n.single("MultiLayer", "I", 0)
        p = n.child("Properties70")
        c = mat.base_color
        p.prop70("ShadingModel", "KString", "", "", "phong")
        p.prop70("DiffuseColor", "Color", "", "A", float(c[0]), float(c[1]), float(c[2]))
        p.prop70("DiffuseFactor", "Number", "", "A", 1.0)
        p.prop70("AmbientColor", "Color", "", "A", 0.0, 0.0, 0.0)
        spec = 0.2 + 0.8 * mat.metallic
        p.prop70("SpecularColor", "Color", "", "A", spec, spec, spec)
        p.prop70("SpecularFactor", "Number", "", "A", float(mat.specular))
        shininess = max(2.0, (1.0 - mat.roughness) ** 2 * 200.0)
        p.prop70("ShininessExponent", "Number", "", "A", float(shininess))
        p.prop70("Shininess", "double", "Number", "", float(shininess))
        p.prop70("ReflectionFactor", "Number", "", "A", float(mat.metallic))
        p.prop70("TransparencyFactor", "Number", "", "A", float(1.0 - mat.opacity))
        p.prop70("TransparentColor", "Color", "", "A", 1.0, 1.0, 1.0)
        e = mat.emissive
        p.prop70("EmissiveColor", "Color", "", "A", float(e[0]), float(e[1]), float(e[2]))
        p.prop70("EmissiveFactor", "Number", "", "A", float(mat.emissive_strength))
        p.prop70("BumpFactor", "double", "Number", "", float(mat.bump_strength))
        # PBR hints understood by Maya / Unity / Unreal importers.
        p.prop70("Maya|base_color", "Vector3D", "Vector", "", float(c[0]), float(c[1]), float(c[2]))
        p.prop70("Maya|metalness", "double", "Number", "", float(mat.metallic))
        p.prop70("Maya|specularRoughness", "double", "Number", "", float(mat.roughness))
        p.prop70("Maya|emission", "double", "Number", "", float(mat.emissive_strength))
        p.prop70("Maya|opacity", "double", "Number", "", float(mat.opacity))
        if mat.source_type:
            p.prop70("Daz|ShaderType", "KString", "", "U", mat.source_type)

        for attr, slot in TEX_SLOTS.items():
            tex = getattr(mat, attr, None)
            if isinstance(tex, TextureRef):
                tid = self._texture(tex, slot)
                self._connect_prop(tid, mid, slot)
        return mid

    def _texture(self, tex: TextureRef, slot: str) -> int:
        key = "%s|%s" % (os.path.normcase(tex.path), slot)
        if key in self._texture_objects:
            return self._texture_objects[key]

        abs_path, rel_path = self._resolve_texture_paths(tex)
        vid = self._video(tex, abs_path, rel_path)

        tid = self.newid()
        n = self.objects.child("Texture")
        n.add_int64(tid).add_raw_name(fbx_name(tex.name or "tex", "Texture")).add_string("")
        self._bump("Texture")
        n.single("Type", "S", "TextureVideoClip")
        n.single("Version", "I", 202)
        n.single("TextureName", "S", fbx_name(tex.name or "tex", "Texture").decode("utf-8", "replace"))
        p = n.child("Properties70")
        p.prop70("UVSet", "KString", "", "", "UVMap")
        p.prop70("UseMaterial", "bool", "", "", 1)
        if tex.tiling != (1.0, 1.0) or tex.offset != (0.0, 0.0):
            p.prop70("Scaling", "Vector", "", "A", float(tex.tiling[0]), float(tex.tiling[1]), 1.0)
            p.prop70("Translation", "Vector", "", "A", float(tex.offset[0]), float(tex.offset[1]), 0.0)
        n.single("Media", "S", fbx_name(tex.name or "tex", "Video").decode("utf-8", "replace"))
        n.single("FileName", "S", abs_path)
        n.single("RelativeFilename", "S", rel_path)
        n.child("ModelUVTranslation").add_float64(float(tex.offset[0])).add_float64(float(tex.offset[1]))
        n.child("ModelUVScaling").add_float64(float(tex.tiling[0])).add_float64(float(tex.tiling[1]))
        n.single("Texture_Alpha_Source", "S", "None")
        cr = n.child("Cropping")
        for _ in range(4):
            cr.add_int32(0)
        self._connect(vid, tid)
        self._texture_objects[key] = tid
        return tid

    def _video(self, tex: TextureRef, abs_path: str, rel_path: str) -> int:
        key = os.path.normcase(abs_path)
        if key in self._video_objects:
            return self._video_objects[key]
        vid = self.newid()
        n = self.objects.child("Video")
        n.add_int64(vid).add_raw_name(fbx_name(tex.name or "tex", "Video")).add_string("Clip")
        self._bump("Video")
        n.single("Type", "S", "Clip")
        p = n.child("Properties70")
        p.prop70("Path", "KString", "XRefUrl", "", abs_path)
        p.prop70("RelPath", "KString", "XRefUrl", "", rel_path)
        n.single("UseMipMap", "I", 0)
        n.single("Filename", "S", abs_path)
        n.single("RelativeFilename", "S", rel_path)
        if self.opts.embed_textures and os.path.isfile(tex.path):
            try:
                with open(tex.path, "rb") as fh:
                    n.single("Content", "R", fh.read())
            except OSError:
                pass
        self._video_objects[key] = vid
        return vid

    def _resolve_texture_paths(self, tex: TextureRef) -> Tuple[str, str]:
        src = tex.path
        if self.opts.copy_textures and os.path.isfile(src):
            dest_dir = os.path.join(self._out_dir, self.opts.texture_subdir)
            dest = self._copied.get(os.path.normcase(src))
            if dest is None:
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, os.path.basename(src))
                base, ext = os.path.splitext(dest)
                counter = 1
                while os.path.exists(dest) and not _same_content(src, dest):
                    dest = "%s_%d%s" % (base, counter, ext)
                    counter += 1
                if not os.path.exists(dest):
                    try:
                        shutil.copy2(src, dest)
                    except OSError:
                        dest = src
                self._copied[os.path.normcase(src)] = dest
                if dest.startswith(self._out_dir):
                    self.written_textures.append(os.path.relpath(dest, self._out_dir))
            rel = os.path.relpath(dest, self._out_dir).replace(os.sep, "/")
            return os.path.abspath(dest), rel
        rel = os.path.basename(src)
        try:
            rel = os.path.relpath(src, self._out_dir).replace(os.sep, "/")
        except ValueError:
            pass
        return os.path.abspath(src) if src else "", rel

    # --------------------------------------------------------------- skeleton
    def _skeleton(self, skel: Skeleton) -> Dict[str, int]:
        cached = self._bone_models.get(id(skel))
        if cached is not None:
            return cached

        root_id = self._model(skel.name, "Null", math3d.identity())
        self._connect(root_id, 0)
        self._pose_entries.append((root_id, math3d.identity()))

        ids: Dict[str, int] = {}
        for bone in skel.bones:
            parent = skel.get(bone.parent) if bone.parent else None
            parent_world = parent.world_matrix if parent else math3d.identity()
            local = math3d.mat_mul(bone.world_matrix, math3d.mat_inverse(parent_world))

            attr_id = self.newid()
            a = self.objects.child("NodeAttribute")
            a.add_int64(attr_id).add_raw_name(fbx_name(bone.name, "NodeAttribute"))
            a.add_string("LimbNode")
            self._bump("NodeAttribute")
            ap = a.child("Properties70")
            ap.prop70("Size", "double", "Number", "", max(0.1, bone.length))
            a.single("TypeFlags", "S", "Skeleton")

            mid = self._model(bone.name, "LimbNode", local)
            self._connect(attr_id, mid)
            ids[bone.name] = mid
            self._pose_entries.append((mid, list(bone.world_matrix)))

        for bone in skel.bones:
            child_id = ids[bone.name]
            parent_id = ids.get(bone.parent) if bone.parent else root_id
            self._connect(child_id, parent_id if parent_id else root_id)

        self._bone_models[id(skel)] = ids
        return ids

    def _skin(self, mesh: Mesh, skel: Skeleton, geom_id: int, model_id: int,
              bone_ids: Dict[str, int]) -> None:
        skin = mesh.skin
        if skin is None or not skin.bone_names:
            return
        skin_id = self.newid()
        n = self.objects.child("Deformer")
        n.add_int64(skin_id).add_raw_name(fbx_name(mesh.name + "_Skin", "Deformer"))
        n.add_string("Skin")
        self._bump("Deformer")
        n.single("Version", "I", 101)
        n.single("Link_DeformAcuracy", "D", 50.0)
        n.single("SkinningType", "S", "Linear")
        self._connect(skin_id, geom_id)

        per_bone: List[Tuple[List[int], List[float]]] = [([], []) for _ in skin.bone_names]
        for vi, inf in enumerate(skin.influences):
            for bi, w in inf:
                if 0 <= bi < len(per_bone):
                    per_bone[bi][0].append(vi)
                    per_bone[bi][1].append(float(w))

        mesh_world = mesh.world_matrix or math3d.identity()
        for bi, bone_name in enumerate(skin.bone_names):
            idxs, wts = per_bone[bi]
            model = bone_ids.get(bone_name)
            if model is None:
                continue
            bone = skel.get(bone_name)
            bone_world = bone.world_matrix if bone else math3d.identity()
            cid = self.newid()
            c = self.objects.child("Deformer")
            c.add_int64(cid).add_raw_name(fbx_name(bone_name, "SubDeformer"))
            c.add_string("Cluster")
            self._bump("Deformer")
            c.single("Version", "I", 100)
            ud = c.child("UserData")
            ud.add_string("").add_string("")
            if idxs:
                c.child("Indexes").add_int32_array(idxs)
                c.child("Weights").add_float64_array(wts)
            transform = math3d.mat_mul(mesh_world, math3d.mat_inverse(bone_world))
            c.child("Transform").add_float64_array([float(x) for x in transform])
            c.child("TransformLink").add_float64_array([float(x) for x in bone_world])
            self._connect(cid, skin_id)
            self._connect(model, cid)

    def _write_bind_pose(self) -> None:
        if not self._pose_entries:
            return
        pid = self.newid()
        n = self.objects.child("Pose")
        n.add_int64(pid).add_raw_name(fbx_name("BIND_POSES", "Pose")).add_string("BindPose")
        self._bump("Pose")
        n.single("Type", "S", "BindPose")
        n.single("Version", "I", 100)
        n.single("NbPoseNodes", "I", len(self._pose_entries))
        for model_id, matrix in self._pose_entries:
            pn = n.child("PoseNode")
            pn.single("Node", "L", model_id)
            pn.child("Matrix").add_float64_array([float(x) for x in matrix])

    # ------------------------------------------------------------ blendshapes
    def _blend_shapes(self, mesh: Mesh, geom_id: int) -> None:
        bs_id = self.newid()
        n = self.objects.child("Deformer")
        n.add_int64(bs_id).add_raw_name(fbx_name(mesh.name + "_BlendShapes", "Deformer"))
        n.add_string("BlendShape")
        self._bump("Deformer")
        n.single("Version", "I", 100)
        self._connect(bs_id, geom_id)

        for morph in mesh.morphs:
            if not morph.indices:
                continue
            ch_id = self.newid()
            c = self.objects.child("Deformer")
            c.add_int64(ch_id).add_raw_name(fbx_name(morph.name, "SubDeformer"))
            c.add_string("BlendShapeChannel")
            self._bump("Deformer")
            c.single("Version", "I", 100)
            c.single("DeformPercent", "D", float(morph.default_value) * 100.0)
            c.child("FullWeights").add_float64_array([100.0])
            self._connect(ch_id, bs_id)

            sh_id = self.newid()
            s = self.objects.child("Geometry")
            s.add_int64(sh_id).add_raw_name(fbx_name(morph.name, "Geometry"))
            s.add_string("Shape")
            self._bump("Geometry")
            s.single("Version", "I", 100)
            s.child("Indexes").add_int32_array(list(morph.indices))
            flat: List[float] = []
            for d in morph.deltas:
                flat.extend((float(d[0]), float(d[1]), float(d[2])))
            s.child("Vertices").add_float64_array(flat)
            s.child("Normals").add_float64_array([0.0] * len(flat))
            self._connect(sh_id, ch_id)

    # ------------------------------------------------------------- utilities
    def _connect(self, child: int, parent: int) -> None:
        c = self.connections.child("C")
        c.add_string("OO").add_int64(child).add_int64(parent)

    def _connect_prop(self, child: int, parent: int, prop: str) -> None:
        c = self.connections.child("C")
        c.add_string("OP").add_int64(child).add_int64(parent).add_string(prop)

    def _bump(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1


def _edge_list(mesh: Mesh) -> List[int]:
    """FBX 'Edges' indexes into PolygonVertexIndex; one entry per unique edge."""
    seen = set()
    edges: List[int] = []
    pvi = 0
    for poly in mesh.polygons:
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                edges.append(pvi + i)
        pvi += n
    return edges


def _same_content(a: str, b: str) -> bool:
    """Cheap "already copied" test: same size wins, otherwise compare bytes."""
    try:
        if os.path.samefile(a, b):
            return True
    except OSError:
        pass
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        import filecmp
        return filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def write_fbx(scene: Scene, path: str,
              options: Optional[FbxOptions] = None) -> Dict[str, Any]:
    return FbxWriter(scene, options).write(path)
