"""IR Scene -> glTF 2.0 (.gltf + .bin) or binary glTF (.glb).

glTF is the format used for the in-app 3D preview and the best target for
web/Unity/Godot pipelines.  Everything is written with the standard library:
buffers are packed by hand with `struct`, images are either referenced,
copied or embedded in the GLB binary chunk.

Conventions handled here:
  * centimetres -> metres (glTF is metre-based)
  * polygons -> triangles (glTF has no quads)
  * per-vertex skinning limited to 4 influences per set (JOINTS_0/WEIGHTS_0)
  * sparse morph deltas -> dense POSITION targets
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import math3d
from ..ir import Figure, Material, Mesh, Scene, Skeleton, TextureRef
from ..version import APP_NAME, __version__

COMPONENT_FLOAT = 5126
COMPONENT_USHORT = 5123
COMPONENT_UINT = 5125
COMPONENT_UBYTE = 5121
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@dataclass
class GltfOptions:
    binary: bool = True                 # .glb vs .gltf + .bin
    embed_textures: bool = True         # pack images into the buffer
    copy_textures: bool = False
    texture_subdir: str = "textures"
    scale_to_meters: bool = True
    export_skin: bool = True
    export_morphs: bool = True
    max_morphs: int = 64                # viewers choke beyond this
    y_up: bool = True


class _Buffer:
    def __init__(self):
        self.chunks: List[bytes] = []
        self.length = 0

    def add(self, data: bytes, align: int = 4) -> Tuple[int, int]:
        pad = (align - (self.length % align)) % align
        if pad:
            self.chunks.append(b"\x00" * pad)
            self.length += pad
        offset = self.length
        self.chunks.append(data)
        self.length += len(data)
        return offset, len(data)

    def bytes(self) -> bytes:
        return b"".join(self.chunks)


class GltfWriter:
    def __init__(self, scene: Scene, options: Optional[GltfOptions] = None):
        self.scene = scene
        self.opts = options or GltfOptions()
        self.buf = _Buffer()
        self.gltf: Dict[str, Any] = {
            "asset": {
                "version": "2.0",
                "generator": "%s %s (Daz DSON converter)" % (APP_NAME, __version__),
            },
            "scene": 0,
            "scenes": [{"name": scene.name, "nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }
        self._images: Dict[str, int] = {}
        self._samplers_added = False
        self._out_dir = ""
        self.written_textures: List[str] = []

    # ------------------------------------------------------------------ API
    def write(self, path: str) -> Dict[str, Any]:
        self._out_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(self._out_dir, exist_ok=True)

        for figure in self.scene.figures:
            self._write_figure(figure)

        data = self.buf.bytes()
        if self.opts.binary:
            self.gltf["buffers"] = [{"byteLength": len(data)}]
            self._write_glb(path, data)
        else:
            bin_name = os.path.splitext(os.path.basename(path))[0] + ".bin"
            self.gltf["buffers"] = [{"byteLength": len(data), "uri": bin_name}]
            with open(os.path.join(self._out_dir, bin_name), "wb") as fh:
                fh.write(data)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.gltf, fh, indent=1)
        return {"path": path, "bytes": os.path.getsize(path),
                "textures": list(self.written_textures)}

    def to_bytes(self) -> bytes:
        """Produce a .glb in memory (used by the web preview)."""
        for figure in self.scene.figures:
            self._write_figure(figure)
        data = self.buf.bytes()
        self.gltf["buffers"] = [{"byteLength": len(data)}]
        return self._glb_bytes(data)

    # --------------------------------------------------------------- figures
    def _write_figure(self, figure: Figure) -> None:
        mesh = figure.mesh
        if mesh is None or not mesh.polygons:
            return
        scale = 0.01 if self.opts.scale_to_meters else 1.0

        mesh_index = self._mesh(mesh, scale)
        node: Dict[str, Any] = {"name": figure.name, "mesh": mesh_index}

        skin_index = None
        if (figure.skeleton and mesh.skin and self.opts.export_skin
                and len(figure.skeleton)):
            skin_index, joint_root = self._skin(figure, scale)
            if skin_index is not None:
                node["skin"] = skin_index
                self.gltf["scenes"][0]["nodes"].append(joint_root)

        self.gltf["nodes"].append(node)
        self.gltf["scenes"][0]["nodes"].append(len(self.gltf["nodes"]) - 1)

    # ------------------------------------------------------------------ mesh
    def _mesh(self, mesh: Mesh, scale: float) -> int:
        """Build one glTF mesh with a primitive per material.

        glTF primitives are indexed triangle soups with one material each, so
        the Daz mesh is split by material and vertices are duplicated where a
        position needs more than one UV (seams).
        """
        groups: Dict[int, List[Tuple[int, ...]]] = {}
        for pi, poly in enumerate(mesh.polygons):
            m = mesh.polygon_material[pi] if pi < len(mesh.polygon_material) else 0
            uvrow = mesh.uv_indices[pi] if pi < len(mesh.uv_indices) else poly
            groups.setdefault(m, []).append((pi, poly, uvrow))  # type: ignore

        normals = mesh.normals or []
        primitives: List[Dict[str, Any]] = []
        material_indices = self._materials(mesh)
        active_morphs = (list(mesh.morphs[:self.opts.max_morphs])
                         if self.opts.export_morphs else [])

        for mat_idx in sorted(groups.keys()):
            faces = groups[mat_idx]
            vmap: Dict[Tuple[int, int], int] = {}
            positions: List[float] = []
            norms: List[float] = []
            uvs: List[float] = []
            joints: List[int] = []
            weights: List[float] = []
            indices: List[int] = []
            source_vertex: List[int] = []

            use_skin = (mesh.skin is not None and self.opts.export_skin
                        and bool(mesh.skin.bone_names))

            def emit(vi: int, ui: int) -> int:
                key = (vi, ui)
                got = vmap.get(key)
                if got is not None:
                    return got
                idx = len(source_vertex)
                vmap[key] = idx
                v = mesh.vertices[vi]
                positions.extend((v[0] * scale, v[1] * scale, v[2] * scale))
                if normals:
                    n = normals[vi] if vi < len(normals) else (0.0, 1.0, 0.0)
                    norms.extend(n)
                if mesh.uvs:
                    uv = mesh.uvs[ui] if 0 <= ui < len(mesh.uvs) else (0.0, 0.0)
                    # glTF UV origin is top-left, Daz/OpenGL is bottom-left.
                    uvs.extend((uv[0], 1.0 - uv[1]))
                if use_skin:
                    inf = (mesh.skin.influences[vi]
                           if vi < len(mesh.skin.influences) else [])
                    inf = sorted(inf, key=lambda p: -p[1])[:4]
                    total = sum(w for _, w in inf) or 1.0
                    for k in range(4):
                        if k < len(inf):
                            joints.append(inf[k][0])
                            weights.append(inf[k][1] / total)
                        else:
                            joints.append(0)
                            weights.append(0.0)
                source_vertex.append(vi)
                return idx

            for _pi, poly, uvrow in faces:
                corners = [emit(poly[k], uvrow[k] if k < len(uvrow) else poly[k])
                           for k in range(len(poly))]
                for k in range(1, len(corners) - 1):
                    indices.extend((corners[0], corners[k], corners[k + 1]))

            if not indices:
                continue

            attributes: Dict[str, int] = {
                "POSITION": self._accessor_vec3(positions, minmax=True),
            }
            if norms:
                attributes["NORMAL"] = self._accessor_vec3(norms)
            if uvs:
                attributes["TEXCOORD_0"] = self._accessor_vec2(uvs)
            if use_skin and joints:
                attributes["JOINTS_0"] = self._accessor_joints(joints)
                attributes["WEIGHTS_0"] = self._accessor_vec4(weights)

            prim: Dict[str, Any] = {
                "attributes": attributes,
                "indices": self._accessor_indices(indices),
                "mode": 4,
            }
            if 0 <= mat_idx < len(material_indices):
                prim["material"] = material_indices[mat_idx]

            if active_morphs:
                # Every primitive of a mesh must expose the *same* morph
                # targets in the same order (glTF spec), even where a given
                # primitive is untouched by that morph.
                prim["targets"] = self._morph_targets(active_morphs,
                                                      source_vertex, scale)
                prim.setdefault("extras", {})["targetNames"] = \
                    [m.name for m in active_morphs]
            primitives.append(prim)

        gmesh: Dict[str, Any] = {"name": mesh.name, "primitives": primitives}
        if active_morphs and primitives:
            gmesh["weights"] = [float(m.default_value) for m in active_morphs]
            gmesh.setdefault("extras", {})["targetNames"] = \
                [m.name for m in active_morphs]
        self.gltf["meshes"].append(gmesh)
        return len(self.gltf["meshes"]) - 1

    def _morph_targets(self, morphs, source_vertex: Sequence[int],
                       scale: float) -> List[Dict[str, int]]:
        """Dense POSITION deltas for one primitive, one entry per morph."""
        targets: List[Dict[str, int]] = []
        for morph in morphs:
            lookup = dict(zip(morph.indices, morph.deltas))
            flat: List[float] = []
            for vi in source_vertex:
                d = lookup.get(vi)
                if d is None:
                    flat.extend((0.0, 0.0, 0.0))
                else:
                    flat.extend((d[0] * scale, d[1] * scale, d[2] * scale))
            targets.append({"POSITION": self._accessor_vec3(flat, minmax=True)})
        return targets

    # ------------------------------------------------------------------ skin
    def _skin(self, figure: Figure, scale: float):
        skel = figure.skeleton
        mesh = figure.mesh
        if skel is None or mesh is None or mesh.skin is None:
            return None, 0
        used = mesh.skin.bone_names
        if not used:
            return None, 0

        # Build a node per bone in the full skeleton so the hierarchy is intact.
        node_index: Dict[str, int] = {}
        for bone in skel.bones:
            self.gltf["nodes"].append({"name": bone.name})
            node_index[bone.name] = len(self.gltf["nodes"]) - 1

        roots: List[int] = []
        for bone in skel.bones:
            idx = node_index[bone.name]
            node = self.gltf["nodes"][idx]
            parent = skel.get(bone.parent) if bone.parent else None
            parent_world = parent.world_matrix if parent else math3d.identity()
            local = math3d.mat_mul(bone.world_matrix,
                                   math3d.mat_inverse(parent_world))
            t, r, s = math3d.decompose(local)
            node["translation"] = [t[0] * scale, t[1] * scale, t[2] * scale]
            q = _euler_to_quat(r)
            if any(abs(v) > 1e-9 for v in q[:3]):
                node["rotation"] = list(q)
            if parent is not None:
                p_node = self.gltf["nodes"][node_index[parent.name]]
                p_node.setdefault("children", []).append(idx)
            else:
                roots.append(idx)

        joints = [node_index[n] for n in used if n in node_index]
        if not joints:
            return None, 0

        ibm: List[float] = []
        for name in used:
            bone = skel.get(name)
            world = bone.world_matrix if bone else math3d.identity()
            scaled = list(world)
            scaled[12] *= scale
            scaled[13] *= scale
            scaled[14] *= scale
            ibm.extend(math3d.mat_inverse(scaled))

        acc = self._accessor(struct.pack("<%df" % len(ibm), *ibm),
                             COMPONENT_FLOAT, "MAT4", len(ibm) // 16)
        skin = {"name": skel.name, "joints": joints, "inverseBindMatrices": acc}
        if roots:
            skin["skeleton"] = roots[0]
        self.gltf.setdefault("skins", []).append(skin)
        return len(self.gltf["skins"]) - 1, (roots[0] if roots else joints[0])

    # ------------------------------------------------------------- materials
    def _materials(self, mesh: Mesh) -> List[int]:
        out: List[int] = []
        for mat in mesh.materials:
            out.append(self._material(mat))
        if not out:
            out.append(self._material(Material(name="Default")))
        return out

    def _material(self, mat: Material) -> int:
        pbr: Dict[str, Any] = {
            "baseColorFactor": [mat.base_color[0], mat.base_color[1],
                                mat.base_color[2], mat.opacity],
            "metallicFactor": mat.metallic,
            "roughnessFactor": mat.roughness,
        }
        gmat: Dict[str, Any] = {"name": mat.name, "pbrMetallicRoughness": pbr,
                                "doubleSided": bool(mat.double_sided)}
        tex = self._texture(mat.base_color_map, srgb=True)
        if tex is not None:
            pbr["baseColorTexture"] = {"index": tex}
        tex = self._texture(mat.roughness_map, srgb=False)
        if tex is not None:
            pbr["metallicRoughnessTexture"] = {"index": tex}
        tex = self._texture(mat.normal_map, srgb=False)
        if tex is not None:
            gmat["normalTexture"] = {"index": tex, "scale": mat.normal_strength}
        tex = self._texture(mat.emissive_map, srgb=True)
        if tex is not None:
            gmat["emissiveTexture"] = {"index": tex}
        if any(c > 0 for c in mat.emissive):
            gmat["emissiveFactor"] = list(mat.emissive)
        if mat.alpha_blend or mat.opacity < 0.999 or mat.opacity_map:
            gmat["alphaMode"] = "BLEND"
        self.gltf["materials"].append(gmat)
        return len(self.gltf["materials"]) - 1

    def _texture(self, tex: Optional[TextureRef], srgb: bool) -> Optional[int]:
        if tex is None or not tex.path:
            return None
        key = os.path.normcase(tex.path)
        if key in self._images:
            image_index = self._images[key]
        else:
            image_index = self._image(tex)
            if image_index is None:
                return None
            self._images[key] = image_index
        self._ensure_sampler()
        self.gltf.setdefault("textures", []).append(
            {"sampler": 0, "source": image_index})
        return len(self.gltf["textures"]) - 1

    def _image(self, tex: TextureRef) -> Optional[int]:
        path = tex.path
        ext = os.path.splitext(path)[1].lower()
        mime = MIME_BY_EXT.get(ext)
        exists = os.path.isfile(path)

        if self.opts.embed_textures and exists and mime:
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return None
            offset, length = self.buf.add(data)
            self.gltf["bufferViews"].append(
                {"buffer": 0, "byteOffset": offset, "byteLength": length})
            self.gltf.setdefault("images", []).append(
                {"name": tex.name, "mimeType": mime,
                 "bufferView": len(self.gltf["bufferViews"]) - 1})
            return len(self.gltf["images"]) - 1

        uri = path
        if self.opts.copy_textures and exists:
            dest_dir = os.path.join(self._out_dir, self.opts.texture_subdir)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(path))
            if not os.path.exists(dest):
                try:
                    shutil.copy2(path, dest)
                    self.written_textures.append(
                        os.path.relpath(dest, self._out_dir))
                except OSError:
                    pass
            uri = os.path.relpath(dest, self._out_dir).replace(os.sep, "/")
        else:
            try:
                uri = os.path.relpath(path, self._out_dir).replace(os.sep, "/")
            except ValueError:
                uri = path.replace(os.sep, "/")
        self.gltf.setdefault("images", []).append({"name": tex.name, "uri": uri})
        return len(self.gltf["images"]) - 1

    def _ensure_sampler(self) -> None:
        if not self._samplers_added:
            self.gltf.setdefault("samplers", []).append(
                {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
            self._samplers_added = True

    # -------------------------------------------------------------- accessors
    def _accessor(self, data: bytes, component_type: int, type_str: str,
                  count: int, target: Optional[int] = None,
                  minmax: Optional[Tuple[List[float], List[float]]] = None) -> int:
        offset, length = self.buf.add(data)
        view: Dict[str, Any] = {"buffer": 0, "byteOffset": offset,
                                "byteLength": length}
        if target is not None:
            view["target"] = target
        self.gltf["bufferViews"].append(view)
        acc: Dict[str, Any] = {
            "bufferView": len(self.gltf["bufferViews"]) - 1,
            "componentType": component_type,
            "count": count,
            "type": type_str,
        }
        if minmax:
            acc["min"], acc["max"] = minmax
        self.gltf["accessors"].append(acc)
        return len(self.gltf["accessors"]) - 1

    def _accessor_vec3(self, values: Sequence[float], minmax: bool = False) -> int:
        count = len(values) // 3
        mm = None
        if minmax and count:
            xs = values[0::3]
            ys = values[1::3]
            zs = values[2::3]
            mm = ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])
        return self._accessor(struct.pack("<%df" % len(values), *values),
                              COMPONENT_FLOAT, "VEC3", count,
                              TARGET_ARRAY_BUFFER, mm)

    def _accessor_vec2(self, values: Sequence[float]) -> int:
        return self._accessor(struct.pack("<%df" % len(values), *values),
                              COMPONENT_FLOAT, "VEC2", len(values) // 2,
                              TARGET_ARRAY_BUFFER)

    def _accessor_vec4(self, values: Sequence[float]) -> int:
        return self._accessor(struct.pack("<%df" % len(values), *values),
                              COMPONENT_FLOAT, "VEC4", len(values) // 4,
                              TARGET_ARRAY_BUFFER)

    def _accessor_joints(self, values: Sequence[int]) -> int:
        count = len(values) // 4
        if max(values, default=0) < 256:
            data = struct.pack("<%dB" % len(values), *values)
            ctype = COMPONENT_UBYTE
        else:
            data = struct.pack("<%dH" % len(values), *values)
            ctype = COMPONENT_USHORT
        return self._accessor(data, ctype, "VEC4", count, TARGET_ARRAY_BUFFER)

    def _accessor_indices(self, indices: Sequence[int]) -> int:
        if max(indices, default=0) < 65536:
            data = struct.pack("<%dH" % len(indices), *indices)
            ctype = COMPONENT_USHORT
        else:
            data = struct.pack("<%dI" % len(indices), *indices)
            ctype = COMPONENT_UINT
        return self._accessor(data, ctype, "SCALAR", len(indices),
                              TARGET_ELEMENT_ARRAY_BUFFER)

    # -------------------------------------------------------------------- glb
    def _glb_bytes(self, bin_data: bytes) -> bytes:
        json_bytes = json.dumps(self.gltf, separators=(",", ":")).encode("utf-8")
        json_pad = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * json_pad
        bin_pad = (4 - (len(bin_data) % 4)) % 4
        bin_data = bin_data + b"\x00" * bin_pad

        total = 12 + 8 + len(json_bytes) + (8 + len(bin_data) if bin_data else 0)
        out = bytearray()
        out += struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(json_bytes), 0x4E4F534A)
        out += json_bytes
        if bin_data:
            out += struct.pack("<II", len(bin_data), 0x004E4942)
            out += bin_data
        return bytes(out)

    def _write_glb(self, path: str, bin_data: bytes) -> None:
        with open(path, "wb") as fh:
            fh.write(self._glb_bytes(bin_data))


def _euler_to_quat(angles_deg: Sequence[float]) -> Tuple[float, float, float, float]:
    """XYZ euler (degrees) -> glTF quaternion [x, y, z, w]."""
    import math
    hx = math.radians(angles_deg[0]) * 0.5
    hy = math.radians(angles_deg[1]) * 0.5
    hz = math.radians(angles_deg[2]) * 0.5
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    # R = Rx * Ry * Rz applied to row vectors == Rz*Ry*Rx for column vectors
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return (x, y, z, w)


def write_gltf(scene: Scene, path: str,
               options: Optional[GltfOptions] = None) -> Dict[str, Any]:
    opts = options or GltfOptions()
    opts.binary = path.lower().endswith(".glb")
    return GltfWriter(scene, opts).write(path)


def scene_to_glb(scene: Scene, embed_textures: bool = True) -> bytes:
    opts = GltfOptions(binary=True, embed_textures=embed_textures,
                       copy_textures=False)
    return GltfWriter(scene, opts).to_bytes()
