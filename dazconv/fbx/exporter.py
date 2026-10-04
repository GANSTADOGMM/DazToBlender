"""Scene -> ASCII FBX 7.4 exporter.

Writes a complete, self-describing FBX document: header, global settings,
definitions, geometry, models, limb nodes, materials, file textures/videos,
skin deformers with per-vertex clusters, and optional blend shapes.

Matrices are transposed on the way out because FBX stores them in row-vector
layout (``p' = p * M``) while the rest of this package uses column vectors.
"""

from __future__ import annotations

import base64
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..geometry import Geometry
from ..library import ContentLibrary
from ..materials import Material, sanitize_material_name, texture_basename
from ..mathutil import (
    IDENTITY,
    Mat4,
    decompose_trs,
    euler_matrix,
    mat4_mul,
    transpose,
)
from ..morphs import Morph
from ..options import ExportOptions
from ..rig import Bone, Skeleton
from ..scene import MeshAsset, SceneModel
from .writer import FbxNode, escape_string, format_number_list, write_document

MAX_EMBED_BYTES = 64 * 1024 * 1024
DEFAULT_UV_SET_NAME = "UVMap"


class FbxExportError(Exception):
    """Raised when a scene cannot be written to FBX."""


class _IdAllocator:
    def __init__(self, start: int = 4000000) -> None:
        self._next = start

    def take(self) -> int:
        self._next += 1
        return self._next


class FbxExporter:
    """Builds an FBX document from a :class:`SceneModel`."""

    def __init__(
        self,
        options: Optional[ExportOptions] = None,
        library: Optional[ContentLibrary] = None,
    ) -> None:
        self.options = options or ExportOptions()
        self.library = library or ContentLibrary()
        self.ids = _IdAllocator()
        self.report: dict = {}

        self.geometry_nodes: List[FbxNode] = []
        self.model_nodes: List[FbxNode] = []
        self.attribute_nodes: List[FbxNode] = []
        self.material_nodes: List[FbxNode] = []
        self.texture_nodes: List[FbxNode] = []
        self.video_nodes: List[FbxNode] = []
        self.deformer_nodes: List[FbxNode] = []
        self.connections: List[Tuple] = []

        self.material_ids: Dict[str, int] = {}
        self.texture_ids: Dict[str, Tuple[int, int]] = {}
        self.warnings: List[str] = []

    # ---------------------------------------------------------------- API --
    def export(self, scene: SceneModel, output_path: str) -> dict:
        """Write ``scene`` to ``output_path`` and return a summary report."""
        if not scene.meshes:
            raise FbxExportError("scene contains no meshes to export")
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        root_rotation = (-90.0, 0.0, 0.0) if self.options.up_axis == "z" else (0.0, 0.0, 0.0)
        skeleton_models = self._export_skeleton(scene, root_rotation)

        for asset in scene.meshes:
            self._export_mesh(asset, scene, output_path, root_rotation, skeleton_models)

        document = self._build_document(scene)
        text = write_document(
            document,
            header_comment=(
                f"FBX 7.4.0 project file\n"
                f"Created by dazconv (DazToBlender standalone converter)\n"
                f"Source: {os.path.basename(scene.source_path)}\n"
                f"----------------------------------------------------------"
            ),
        )
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

        size = os.path.getsize(output_path)
        self.report = {
            "path": output_path,
            "size_bytes": size,
            "meshes": len(scene.meshes),
            "vertices": scene.vertex_total,
            "polygons": scene.polygon_total,
            "materials": len(self.material_ids),
            "textures": len(self.texture_ids),
            "bones": len(skeleton_models),
            "deformers": sum(1 for n in self.deformer_nodes if n.value is None),
            "blendshapes": self.report.get("blendshapes", 0),
            "warnings": list(self.warnings),
        }
        return self.report

    # ----------------------------------------------------------- skeleton --
    def _export_skeleton(self, scene: SceneModel, root_rotation) -> Dict[int, int]:
        """Create a LimbNode model for every bone. Returns bone.index -> model id."""
        bone_to_model: Dict[int, int] = {}
        if not self.options.export_skeleton:
            return bone_to_model

        exported: Dict[int, Skeleton] = {}
        for asset in scene.meshes:
            skeleton = asset.skeleton
            if skeleton is None or id(skeleton) in exported:
                continue
            exported[id(skeleton)] = skeleton

        for skeleton in exported.values():
            for bone in skeleton.depth_ordered():
                if bone.index in bone_to_model:
                    continue
                model_id = self._add_bone(bone, root_rotation)
                bone_to_model[bone.index] = model_id

        # Parent/child connections now that every bone has a model.
        for skeleton in exported.values():
            for bone in skeleton.bones:
                model_id = bone_to_model.get(bone.index)
                if model_id is None:
                    continue
                parent_model = (
                    bone_to_model.get(bone.parent.index) if bone.parent is not None else None
                )
                self.connections.append(("OO", model_id, parent_model or 0))
        return bone_to_model

    def _add_bone(self, bone: Bone, root_rotation) -> int:
        local = bone.local_matrix
        if bone.parent is None:
            local = mat4_mul(euler_matrix(*root_rotation, order="XYZ"), local)
        translation, rotation, scale = decompose_trs(local)

        model_id = self.ids.take()
        attribute_id = self.ids.take()
        name = sanitize_name(bone.label or bone.id)

        model = FbxNode("Model", compound=(model_id, f"Model::{name}", "LimbNode"))
        model.child("Version", 232)
        props = model.properties70()
        props.prop("Lcl Translation", "Lcl Translation", "", "A", *translation)
        props.prop("Lcl Rotation", "Lcl Rotation", "", "A", *rotation)
        props.prop("Lcl Scaling", "Lcl Scaling", "", "A", *scale)
        props.prop("DefaultAttributeIndex", "int", "Integer", "", 0)
        model.child("Shading", "Y")
        model.child("Culling", "CullingOff")
        self.model_nodes.append(model)

        attribute = FbxNode("NodeAttribute", compound=(attribute_id, f"NodeAttribute::{name}", "LimbNode"))
        attribute_props = attribute.properties70()
        attribute_props.prop("Size", "double", "Number", "", bone.length or 1.0)
        attribute.child("TypeFlags", "Skeleton")
        self.attribute_nodes.append(attribute)

        self.connections.append(("OO", attribute_id, model_id))
        return model_id

    # --------------------------------------------------------------- mesh --
    def _export_mesh(
        self,
        asset: MeshAsset,
        scene: SceneModel,
        output_path: str,
        root_rotation,
        bone_to_model: Dict[int, int],
    ) -> None:
        geometry = asset.geometry
        vertices = list(geometry.vertices)

        # Bake selected morphs straight into the base mesh when requested.
        vertices = self._apply_selected_morphs(asset, vertices)

        if self.options.scale != 1.0:
            scale = self.options.scale
            vertices = [(v[0] * scale, v[1] * scale, v[2] * scale) for v in vertices]

        polygons = geometry.polygons
        if self.options.triangulate:
            polygon_vertex_lists = geometry.triangulated_indices()
            material_indices = _expand_material_indices(geometry, len(polygon_vertex_lists))
        else:
            polygon_vertex_lists = [list(p.vertices) for p in polygons]
            material_indices = [p.material_index for p in polygons]

        geometry_id = self.ids.take()
        node = FbxNode(
            "Geometry",
            compound=(geometry_id, f"Geometry::{sanitize_name(asset.name or asset.id)}", "Mesh"),
        )

        flat_vertices = [component for vertex in vertices for component in vertex]
        node.numbers("Vertices", flat_vertices)

        polygon_indices: List[int] = []
        for face in polygon_vertex_lists:
            polygon_indices.extend(face[:-1])
            polygon_indices.append(-(face[-1] + 1))
        array_node = FbxNode("PolygonVertexIndex", array=("i", polygon_indices))
        node.children.append(array_node)
        node.child("GeometryVersion", 124)

        self._add_normal_layer(node, geometry, vertices)
        self._add_uv_layer(node, asset, geometry)
        self._add_material_layer(node, asset, material_indices)
        self._add_layers(node, asset)
        self.geometry_nodes.append(node)

        # ---- model ------------------------------------------------------
        local = mat4_mul(
            euler_matrix(*root_rotation, order="XYZ"),
            asset.world_matrix,
        )
        translation, rotation, scale = decompose_trs(local)
        if self.options.scale != 1.0:
            scale = tuple(s * self.options.scale for s in scale)

        model_id = self.ids.take()
        model_name = sanitize_name(asset.name or asset.id)
        model = FbxNode("Model", compound=(model_id, f"Model::{model_name}", "Mesh"))
        model.child("Version", 232)
        props = model.properties70()
        props.prop("Lcl Translation", "Lcl Translation", "", "A", *translation)
        props.prop("Lcl Rotation", "Lcl Rotation", "", "A", *rotation)
        props.prop("Lcl Scaling", "Lcl Scaling", "", "A", *scale)
        props.prop("DefaultAttributeIndex", "int", "Integer", "", 0)
        model.child("Shading", "T")
        model.child("Culling", "CullingOff")
        self.model_nodes.append(model)

        self.connections.append(("OO", model_id, 0))
        self.connections.append(("OO", geometry_id, model_id))

        # ---- materials --------------------------------------------------
        if self.options.export_materials:
            self._connect_materials(asset, model_id, output_path)

        # ---- skin -------------------------------------------------------
        if asset.is_skinned and self.options.export_weights:
            self._export_skin(asset, geometry_id, bone_to_model)

        # ---- blend shapes ----------------------------------------------
        if self.options.export_blendshapes:
            self._export_blendshapes(asset, geometry_id, vertices)

    # ------------------------------------------------------------- layers --
    def _add_normal_layer(self, node: FbxNode, geometry: Geometry, vertices) -> None:
        normals = geometry.vertex_normals() if self.options.smooth_normals else []
        if not normals:
            # Face normals expanded per vertex.
            face_normals = geometry.face_normals()
            normals = []
            for polygon, normal in zip(geometry.polygons, face_normals):
                normals.extend([normal] * polygon.corner_count)
            mapping = "ByPolygonVertex"
        else:
            mapping = "ByVertice"
        layer = node.child("LayerElementNormal", 0)
        layer.child("Version", 102)
        layer.child("Name", '""')
        layer.child("MappingInformationType", f'"{mapping}"')
        layer.child("ReferenceInformationType", '"Direct"')
        layer.numbers("Normals", [c for n in normals for c in n])

    def _add_uv_layer(self, node: FbxNode, asset: MeshAsset, geometry: Geometry) -> None:
        corner_uvs = asset.corner_uvs
        if not corner_uvs:
            return
        if self.options.triangulate:
            corner_uvs = _triangulate_corner_uvs(corner_uvs, geometry)
        if len(corner_uvs) != (
            sum(len(f) for f in geometry.triangulated_indices())
            if self.options.triangulate
            else geometry.corner_count
        ):
            self.warnings.append(
                f"{asset.id}: UV count ({len(corner_uvs)}) does not match the "
                "polygon corner count; UVs may be misaligned"
            )
        layer = node.child("LayerElementUV", 0)
        layer.child("Version", 101)
        layer.child("Name", f'"{escape_string(DEFAULT_UV_SET_NAME)}"')
        layer.child("MappingInformationType", '"ByPolygonVertex"')
        layer.child("ReferenceInformationType", '"Direct"')
        layer.numbers("UV", [c for uv in corner_uvs for c in uv])

    def _add_material_layer(self, node: FbxNode, asset: MeshAsset, material_indices) -> None:
        if not asset.materials:
            return
        layer = node.child("LayerElementMaterial", 0)
        layer.child("Version", 101)
        layer.child("Name", '""')
        layer.child("MappingInformationType", '"ByPolygon"')
        layer.child("ReferenceInformationType", '"IndexToDirect"')
        layer.children.append(
            FbxNode("Materials", array=("i", material_indices))
        )

    def _add_layers(self, node: FbxNode, asset: MeshAsset) -> None:
        layer = node.child("Layer", 0)
        layer.child("Version", 100)
        entries = ["LayerElementNormal"]
        if asset.materials:
            entries.append("LayerElementMaterial")
        if asset.corner_uvs:
            entries.append("LayerElementUV")
        for entry in entries:
            element = layer.child("LayerElement")
            element.child("Type", f'"{entry}"')
            element.child("TypedIndex", 0)

    # ---------------------------------------------------------- materials --
    def _connect_materials(self, asset: MeshAsset, model_id: int, output_path: str) -> None:
        used_names: set = set()
        for material in asset.materials:
            material_id = self._add_material(material, used_names, output_path)
            self.connections.append(("OO", material_id, model_id))

    def _add_material(self, material: Material, used_names: set, output_path: str) -> int:
        key = material.id or material.name
        if key in self.material_ids:
            return self.material_ids[key]
        material_id = self.ids.take()
        name = sanitize_material_name(material.label or material.name or material.id, used_names)
        node = FbxNode("Material", compound=(material_id, f"Material::{name}", ""))
        node.child("Version", 102)
        node.child("ShadingModel", '"phong"')
        node.child("MultiLayer", 0)

        props = node.properties70()
        props.prop("ShadingModel", "KString", "", "", "phong")
        props.prop("MultiLayer", "bool", "", "", False)
        props.prop("AmbientColor", "Color", "", "A", 0.0, 0.0, 0.0)
        props.prop("DiffuseColor", "Color", "", "A", *material.base_color)
        props.prop("DiffuseFactor", "Number", "", "A", 1.0)
        props.prop(
            "TransparentColor", "Color", "", "A", 1.0, 1.0, 1.0
        )
        props.prop("TransparencyFactor", "Number", "", "A", 1.0 - material.opacity)
        props.prop("Opacity", "Number", "", "A", material.opacity)
        props.prop("EmissiveColor", "Color", "", "A", *material.emissive_color)
        props.prop("EmissiveFactor", "Number", "", "A", material.emissive_strength)
        props.prop("SpecularColor", "Color", "", "A", *material.specular_color)
        props.prop("SpecularFactor", "Number", "", "A", 1.0)
        shininess = _roughness_to_shininess(material.roughness)
        props.prop("Shininess", "Number", "", "A", shininess)
        props.prop("ShininessExponent", "Number", "", "A", shininess)
        props.prop("ReflectionFactor", "Number", "", "A", material.metallic)
        props.prop("BumpFactor", "double", "Number", "", 1.0)
        if material.normal_map:
            props.prop("NormalMap", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
        self.material_nodes.append(node)
        self.material_ids[key] = material_id

        if self.options.export_textures:
            self._link_textures(material, material_id, output_path)
        return material_id

    def _link_textures(self, material: Material, material_id: int, output_path: str) -> None:
        slots = (
            ("base_color_map", "DiffuseColor"),
            ("normal_map", "NormalMap"),
            ("opacity_map", "TransparencyFactor"),
            ("bump_map", "Bump"),
        )
        for attribute, property_name in slots:
            url = getattr(material, attribute) or ""
            if not url:
                continue
            texture_id, video_id = self._add_texture(url, material, output_path, property_name)
            if texture_id is None:
                continue
            self.connections.append(("OP", texture_id, material_id, property_name))
            if video_id is not None:
                self.connections.append(("OO", video_id, texture_id))

    def _add_texture(
        self,
        url: str,
        material: Material,
        output_path: str,
        slot: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        resolved = material.texture_paths.get(url) or self.library.resolve_texture(url)
        if resolved is None and os.path.isabs(url) and os.path.isfile(url):
            resolved = url
        if resolved is None and material.source_path:
            candidate = os.path.join(os.path.dirname(material.source_path), texture_basename(url))
            if os.path.isfile(candidate):
                resolved = candidate
        key = f"{url}|{slot}"
        if key in self.texture_ids:
            return self.texture_ids[key]
        if resolved is None:
            self.warnings.append(f"texture not found: {url}")
            return (None, None)

        media_name = sanitize_name(f"{material.id or 'mat'}_{slot}")
        texture_id = self.ids.take()
        video_id = self.ids.take()

        if self.options.copy_textures:
            destination = os.path.join(
                os.path.dirname(output_path),
                "textures",
                os.path.basename(resolved),
            )
            try:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if os.path.abspath(destination) != os.path.abspath(resolved):
                    shutil.copy2(resolved, destination)
                resolved_for_file = destination
            except OSError as error:
                self.warnings.append(f"could not copy texture {resolved}: {error}")
                resolved_for_file = resolved
        else:
            resolved_for_file = resolved

        absolute = os.path.abspath(resolved_for_file)
        relative = os.path.relpath(absolute, os.path.dirname(output_path) or ".")
        absolute_fbx = absolute.replace("/", "\\") if os.name == "nt" else absolute
        relative_fbx = relative.replace("/", "\\") if os.name == "nt" else relative

        texture = FbxNode("Texture", compound=(texture_id, f"Texture::{media_name}", ""))
        texture.child("Type", '"TextureVideoClip"')
        texture.child("Version", 202)
        texture.child("TextureName", f'"Texture::{escape_string(media_name)}"')
        texture_props = texture.properties70()
        texture_props.prop("CurrentTextureBlendMode", "enum", "", "", 1)
        texture_props.prop("UVSet", "KString", "", "", DEFAULT_UV_SET_NAME)
        texture_props.prop("UseMaterial", "bool", "", "", True)
        texture_props.prop("UseMipMap", "bool", "", "", False)
        texture.child("Media", f'"Video::{escape_string(media_name)}"')
        texture.child("FileName", f'"{escape_string(absolute_fbx)}"')
        texture.child("RelativeFilename", f'"{escape_string(relative_fbx)}"')
        texture.child("ModelUVTranslation", "0,0")
        texture.child("ModelUVScaling", "1,1")
        texture.child("Texture_Alpha_Source", '"None"')
        texture.child("Cropping", "0,0,0,0")
        self.texture_nodes.append(texture)

        video = FbxNode("Video", compound=(video_id, f"Video::{media_name}", "Clip"))
        video.child("Type", '"Clip"')
        video_props = video.properties70()
        video_props.prop("Path", "KString", "XRefUrl", "", relative_fbx)
        video_props.prop("RelPath", "KString", "XRefUrl", "", relative_fbx)
        video.child("UseMipMap", 0)
        video.child("Filename", f'"{escape_string(absolute_fbx)}"')
        video.child("RelativeFilename", f'"{escape_string(relative_fbx)}"')
        if self.options.embed_textures:
            payload = _embed_image(resolved)
            if payload is None:
                self.warnings.append(
                    f"texture too large to embed, referenced instead: {os.path.basename(resolved)}"
                )
                video.child("Content", '""')
            else:
                video.child("Content", f'"{payload}"')
        else:
            video.child("Content", '""')
        self.video_nodes.append(video)

        self.texture_ids[key] = (texture_id, video_id)
        return (texture_id, video_id)

    # -------------------------------------------------------------- skin --
    def _export_skin(
        self, asset: MeshAsset, geometry_id: int, bone_to_model: Dict[int, int]
    ) -> None:
        influences = asset.influences
        skeleton = asset.skeleton
        assert influences is not None and skeleton is not None

        # Invert the per-vertex influence list once: bone index -> [(vertex, weight)].
        per_bone: Dict[int, List[Tuple[int, float]]] = {}
        for vertex, (indices, weights) in enumerate(
            zip(influences.joint_indices, influences.joint_weights)
        ):
            for bone_index, weight in zip(indices, weights):
                per_bone.setdefault(bone_index, []).append((vertex, weight))

        mesh_bind = asset.world_matrix
        mesh_bind_fbx = list(transpose(mesh_bind))

        skin_id = self.ids.take()
        skin = FbxNode(
            "Deformer",
            compound=(skin_id, f"SubDeformer::Skin {sanitize_name(asset.name)}", "Skin"),
        )
        skin.child("Version", 101)
        skin.child("Link_DeformAcuracy", 50)
        skin.child("SkinningType", '"Linear"')
        self.deformer_nodes.append(skin)
        self.connections.append(("OO", skin_id, geometry_id))

        clusters_written = 0
        for bone in skeleton.bones:
            entries = per_bone.get(bone.index) or []
            if not entries:
                continue
            model_id = bone_to_model.get(bone.index)
            if model_id is None:
                self.warnings.append(f"bone '{bone.id}' has weights but no exported model")
                continue
            cluster_id = self.ids.take()
            cluster = FbxNode(
                "Deformer",
                compound=(
                    cluster_id,
                    f"SubDeformer::Cluster {sanitize_name(asset.name)} {sanitize_name(bone.id)}",
                    "Cluster",
                ),
            )
            cluster.child("Version", 100)
            cluster.child("UserData", '""')
            cluster.children.append(
                FbxNode("Indexes", array=("i", [vertex for vertex, _ in entries]))
            )
            cluster.numbers("Weights", [weight for _, weight in entries])
            cluster.numbers("Transform", mesh_bind_fbx)
            cluster.numbers("TransformLink", list(transpose(bone.world_matrix)))
            self.deformer_nodes.append(cluster)
            self.connections.append(("OO", cluster_id, skin_id))
            self.connections.append(("OO", model_id, cluster_id))
            clusters_written += 1
        self.report["clusters"] = self.report.get("clusters", 0) + clusters_written

    # ------------------------------------------------------- blend shapes --
    def _export_blendshapes(
        self, asset: MeshAsset, geometry_id: int, base_vertices
    ) -> None:
        blend_morphs = [
            morph for morph in asset.morphs if morph.deltas and morph.id not in self._selected_morph_ids()
        ]
        if not blend_morphs:
            return
        blendshape_id = self.ids.take()
        blendshape = FbxNode(
            "Deformer",
            compound=(blendshape_id, f"SubDeformer::BlendShape {sanitize_name(asset.name)}", "BlendShape"),
        )
        blendshape.child("Version", 100)
        self.deformer_nodes.append(blendshape)
        self.connections.append(("OO", blendshape_id, geometry_id))

        count = 0
        for morph in blend_morphs:
            channel_id = self.ids.take()
            shape_id = self.ids.take()
            channel = FbxNode(
                "Deformer",
                compound=(
                    channel_id,
                    f"SubDeformer::BlendShapeChannel {sanitize_name(morph.label or morph.id)}",
                    "BlendShapeChannel",
                ),
            )
            channel.child("Version", 100)
            channel.child("DeformPercent", 0)
            channel_props = channel.properties70()
            channel_props.prop("DeformPercent", "Number", "", "A", 0.0)
            channel.children.append(FbxNode("FullWeights", array=("d", [100.0])))
            self.deformer_nodes.append(channel)

            indices = sorted(morph.deltas)
            deltas = [c for vertex in indices for c in morph.deltas[vertex]]
            shape = FbxNode(
                "Geometry",
                compound=(shape_id, f"Geometry::{sanitize_name(morph.label or morph.id)}", "Shape"),
            )
            shape.child("Version", 100)
            shape.children.append(FbxNode("Indexes", array=("i", indices)))
            shape.numbers("Vertices", deltas)
            shape.numbers("Normals", [0.0] * (len(indices) * 3))
            self.geometry_nodes.append(shape)

            self.connections.append(("OO", channel_id, blendshape_id))
            self.connections.append(("OO", shape_id, channel_id))
            count += 1
        self.report["blendshapes"] = self.report.get("blendshapes", 0) + count

    def _selected_morph_ids(self):
        return {morph_id for morph_id, _ in self.options.morph_selections}

    def _apply_selected_morphs(self, asset: MeshAsset, vertices):
        selections = self.options.morph_selections
        if not selections:
            return vertices
        by_id = {morph.id: morph for morph in asset.morphs}
        for morph_id, weight in selections:
            morph = by_id.get(morph_id)
            if morph is None:
                self.warnings.append(f"morph '{morph_id}' is not available for {asset.id}")
                continue
            vertices = _apply_delta(vertices, morph, weight)
        return vertices

    # ---------------------------------------------------------- document --
    def _build_document(self, scene: SceneModel) -> List[FbxNode]:
        counts = {
            "Geometry": len(self.geometry_nodes),
            "Model": len(self.model_nodes) + len(self.attribute_nodes),
            "NodeAttribute": len(self.attribute_nodes),
            "Material": len(self.material_nodes),
            "Texture": len(self.texture_nodes),
            "Video": len(self.video_nodes),
            "Deformer": len(self.deformer_nodes),
        }

        header = FbxNode("FBXHeaderExtension")
        header.child("FBXHeaderVersion", 1003)
        header.child("FBXVersion", 7400)
        stamp = time.gmtime()
        created = header.child("CreationTimeStamp")
        created.child("Version", 1000)
        created.child("Year", stamp.tm_year)
        created.child("Month", stamp.tm_mon)
        created.child("Day", stamp.tm_mday)
        created.child("Hour", stamp.tm_hour)
        created.child("Minute", stamp.tm_min)
        created.child("Second", stamp.tm_sec)
        created.child("Millisecond", 0)
        header.child("Creator", '"dazconv (DazToBlender standalone converter)"')
        scene_info = header.child("SceneInfo")
        scene_info.compound = (None, "SceneInfo::GlobalInfo", "UserData")
        scene_info.child("Type", '"UserData"')
        scene_info.child("Version", 100)
        meta = scene_info.child("MetaData")
        meta.child("Version", 100)
        meta.child("Title", '""')
        meta.child("Subject", '""')
        meta.child("Author", '""')
        meta.child("Keywords", '""')
        meta.child("Revision", '""')
        meta.child("Comment", '""')
        info_props = scene_info.properties70()
        info_props.prop("DocumentUrl", "KString", "Url", "", scene.source_path)
        info_props.prop("SrcDocumentUrl", "KString", "Url", "", scene.source_path)
        info_props.prop("Original", "Compound", "", "")
        info_props.prop("Original|ApplicationVendor", "KString", "", "", "dazconv")
        info_props.prop("Original|ApplicationName", "KString", "", "", "dazconv")
        info_props.prop("Original|ApplicationVersion", "KString", "", "", "1.0.0")
        info_props.prop("LastSaved", "Compound", "", "")
        info_props.prop("LastSaved|ApplicationVendor", "KString", "", "", "dazconv")
        info_props.prop("LastSaved|ApplicationName", "KString", "", "", "dazconv")
        info_props.prop("LastSaved|ApplicationVersion", "KString", "", "", "1.0.0")

        globals_node = FbxNode("GlobalSettings")
        globals_node.child("Version", 1000)
        props = globals_node.properties70()
        up = 1 if self.options.up_axis == "y" else 2
        front = 2 if self.options.up_axis == "y" else 1
        props.prop("UpAxis", "int", "Integer", "", up)
        props.prop("UpAxisSign", "int", "Integer", "", 1)
        props.prop("FrontAxis", "int", "Integer", "", front)
        props.prop("FrontAxisSign", "int", "Integer", "", 1)
        props.prop("CoordAxis", "int", "Integer", "", 0)
        props.prop("CoordAxisSign", "int", "Integer", "", 1)
        props.prop("OriginalUpAxis", "int", "Integer", "", up)
        props.prop("OriginalUpAxisSign", "int", "Integer", "", 1)
        props.prop("UnitScaleFactor", "double", "Number", "", self.options.unit_scale)
        props.prop("OriginalUnitScaleFactor", "double", "Number", "", self.options.unit_scale)
        props.prop("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0)
        props.prop("DefaultCamera", "KString", "", "", "Producer Perspective")
        props.prop("TimeMode", "enum", "", "", 0)
        props.prop("TimeProtocol", "enum", "", "", 2)
        props.prop("SnapOnFrameMode", "enum", "", "", 0)
        props.prop("TimeSpanStart", "KTime", "Time", "", 0)
        props.prop("TimeSpanStop", "KTime", "Time", "", 46186158000)
        props.prop("CustomFrameRate", "double", "Number", "", -1)
        props.prop("TimeMarker", "Compound", "", "")
        props.prop("CurrentTimeMarker", "int", "Integer", "", -1)

        documents = FbxNode("Documents")
        documents.child("Count", 1)
        document = documents.child("Document", compound=(self.ids.take(), "", "Scene"))
        document_props = document.properties70()
        document_props.prop("SourceObject", "object", "", "")
        document_props.prop("ActiveAnimStackName", "KString", "", "", "")
        document.child("RootNode", 0)

        definitions = FbxNode("Definitions")
        definitions.child("Version", 100)
        block_count = 1 + sum(1 for name, count in counts.items() if count)
        definitions.child("Count", block_count)
        gs = definitions.child("ObjectType", '"GlobalSettings"')
        gs.child("Count", 1)
        templates = {
            "Geometry": _mesh_template,
            "Model": _model_template,
            "NodeAttribute": _attribute_template,
            "Material": _material_template,
            "Texture": _texture_template,
            "Video": _video_template,
        }
        for name, count in counts.items():
            if count == 0:
                continue
            object_type = definitions.child("ObjectType", f'"{name}"')
            object_type.child("Count", count)
            if name in templates:
                templates[name](object_type)

        objects = FbxNode("Objects")
        for group in (
            self.geometry_nodes,
            self.model_nodes,
            self.attribute_nodes,
            self.material_nodes,
            self.texture_nodes,
            self.video_nodes,
            self.deformer_nodes,
        ):
            objects.children.extend(group)

        connections = FbxNode("Connections")
        for entry in self.connections:
            if len(entry) == 3:
                kind, child, parent = entry
                connections.child("C", value=f'"{kind}",{child},{parent}')
            else:
                kind, child, parent, property_name = entry
                connections.child(
                    "C",
                    value=f'"{kind}",{child},{parent}, "{escape_string(property_name)}"',
                )

        takes = FbxNode("Takes")
        takes.child("Current", '""')

        return [header, globals_node, documents, FbxNode("References"), definitions, objects, connections, takes]


# -------------------------------------------------------------- templates --
def _mesh_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxMesh"')
    props = template.properties70()
    props.prop("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8)
    props.prop("BBoxMin", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("BBoxMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("Primary Visibility", "bool", "", "", True)
    props.prop("Casts Shadows", "bool", "", "", True)
    props.prop("Receive Shadows", "bool", "", "", True)


def _model_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxNode"')
    props = template.properties70()
    props.prop("QuaternionInterpolate", "enum", "", "", 0)
    props.prop("RotationOffset", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("RotationPivot", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("ScalingOffset", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("ScalingPivot", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("TranslationActive", "bool", "", "", False)
    props.prop("TranslationMin", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("TranslationMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
    props.prop("RotationActive", "bool", "", "", False)
    props.prop("InheritType", "enum", "", "", 0)
    props.prop("ScalingActive", "bool", "", "", False)
    props.prop("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0)
    props.prop("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0)
    props.prop("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
    props.prop("Visibility", "Visibility", "", "A", True)
    props.prop("Visibility Inheritance", "Visibility Inheritance", "", "", True)


def _attribute_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxNodeAttribute"')
    props = template.properties70()
    props.prop("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8)
    props.prop("Size", "double", "Number", "", 100.0)
    props.prop("LimbLength", "double", "Number", "", 1.0)


def _material_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxSurfacePhong"')
    props = template.properties70()
    props.prop("ShadingModel", "KString", "", "", "Phong")
    props.prop("MultiLayer", "bool", "", "", False)
    props.prop("EmissiveColor", "Color", "", "A", 0.0, 0.0, 0.0)
    props.prop("EmissiveFactor", "Number", "", "A", 1.0)
    props.prop("AmbientColor", "Color", "", "A", 0.2, 0.2, 0.2)
    props.prop("AmbientFactor", "Number", "", "A", 1.0)
    props.prop("DiffuseColor", "Color", "", "A", 0.8, 0.8, 0.8)
    props.prop("DiffuseFactor", "Number", "", "A", 1.0)
    props.prop("TransparentColor", "Color", "", "A", 0.0, 0.0, 0.0)
    props.prop("TransparencyFactor", "Number", "", "A", 0.0)
    props.prop("SpecularColor", "Color", "", "A", 0.2, 0.2, 0.2)
    props.prop("SpecularFactor", "Number", "", "A", 1.0)
    props.prop("ShininessExponent", "Number", "", "A", 20.0)
    props.prop("ReflectionColor", "Color", "", "A", 0.0, 0.0, 0.0)
    props.prop("ReflectionFactor", "Number", "", "A", 1.0)


def _texture_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxFileTexture"')
    props = template.properties70()
    props.prop("TextureTypeUse", "enum", "", "", 0)
    props.prop("CurrentTextureBlendMode", "enum", "", "", 1)
    props.prop("UseMaterial", "bool", "", "", False)
    props.prop("UseMipMap", "bool", "", "", False)
    props.prop("CurrentMappingType", "enum", "", "", 0)
    props.prop("UVSwap", "bool", "", "", False)
    props.prop("WrapModeU", "enum", "", "", 0)
    props.prop("WrapModeV", "enum", "", "", 0)


def _video_template(object_type: FbxNode) -> None:
    template = object_type.child("PropertyTemplate", '"FbxVideo"')
    props = template.properties70()
    props.prop("FrameRate", "double", "Number", "", 0.0)
    props.prop("LastFrame", "int", "Integer", "", 0)
    props.prop("Width", "int", "Integer", "", 0)
    props.prop("Height", "int", "Integer", "", 0)
    props.prop("Path", "KString", "XRefUrl", "", "")
    props.prop("StartFrame", "int", "Integer", "", 0)
    props.prop("StopFrame", "int", "Integer", "", 0)
    props.prop("PlaySpeed", "double", "Number", "", 0.0)
    props.prop("InterlaceMode", "enum", "", "", 0)
    props.prop("AccessMode", "enum", "", "", 0)


# ---------------------------------------------------------------- helpers --
def sanitize_name(name: str) -> str:
    """FBX object names: keep alphanumerics, dots, dashes, underscores, spaces."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in " ._-") else "_" for ch in (name or "Object")
    ).strip(" .")
    return cleaned or "Object"


def _roughness_to_shininess(roughness: float) -> float:
    """Map a 0..1 roughness onto a Phong exponent (2..1000)."""
    roughness = max(0.0, min(1.0, roughness))
    return max(2.0, 1000.0 * (1.0 - roughness) ** 2)


def _triangulate_corner_uvs(corner_uvs, geometry: Geometry):
    """Re-emit per-corner UVs following the same fan triangulation as vertices."""
    out = []
    cursor = 0
    for polygon in geometry.polygons:
        size = polygon.corner_count
        block = corner_uvs[cursor:cursor + size]
        cursor += size
        if len(block) < 3:
            continue
        for i in range(1, size - 1):
            out.extend([block[0], block[i], block[i + 1]])
    return out


def _expand_material_indices(geometry: Geometry, triangle_count: int) -> List[int]:
    indices: List[int] = []
    for polygon in geometry.polygons:
        count = max(1, polygon.corner_count - 2)
        indices.extend([polygon.material_index] * count)
    if len(indices) != triangle_count:
        indices = (indices + [0] * triangle_count)[:triangle_count]
    return indices


def _apply_delta(vertices, morph: Morph, weight: float):
    out = list(vertices)
    for vertex, delta in morph.deltas.items():
        if 0 <= vertex < len(out):
            base = out[vertex]
            out[vertex] = (
                base[0] + delta[0] * weight,
                base[1] + delta[1] * weight,
                base[2] + delta[2] * weight,
            )
    return out


def _embed_image(path: str) -> Optional[str]:
    try:
        size = os.path.getsize(path)
        if size > MAX_EMBED_BYTES:
            return None
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    except OSError:
        return None


def export_fbx(
    scene: SceneModel,
    output_path: str,
    options: Optional[ExportOptions] = None,
    library: Optional[ContentLibrary] = None,
) -> dict:
    """Convenience wrapper: export ``scene`` and return the exporter report."""
    return FbxExporter(options, library).export(scene, output_path)
