"""IR Scene -> Wavefront OBJ + MTL.

OBJ has no rigging or blend shapes, but it is the universal fallback and
keeps quads intact (unlike glTF), which matters for anyone re-topologising
or sculpting the result.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO

from ..ir import Material, Mesh, Scene
from ..version import APP_NAME, __version__


@dataclass
class ObjOptions:
    write_mtl: bool = True
    copy_textures: bool = True
    texture_subdir: str = "textures"
    scale: float = 1.0
    triangulate: bool = False
    y_up: bool = True
    group_by_surface: bool = True


class ObjWriter:
    def __init__(self, scene: Scene, options: Optional[ObjOptions] = None):
        self.scene = scene
        self.opts = options or ObjOptions()
        self._out_dir = ""
        self._copied: Dict[str, str] = {}
        self.written_textures: List[str] = []

    def write(self, path: str) -> Dict[str, Any]:
        self._out_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(self._out_dir, exist_ok=True)
        mtl_name = os.path.splitext(os.path.basename(path))[0] + ".mtl"

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# %s %s - converted from Daz Studio DSON\n"
                     % (APP_NAME, __version__))
            fh.write("# source: %s\n" % ", ".join(
                os.path.basename(s) for s in self.scene.source_files))
            if self.opts.write_mtl:
                fh.write("mtllib %s\n" % mtl_name)
            v_off = 1
            vt_off = 1
            vn_off = 1
            for figure in self.scene.figures:
                mesh = figure.mesh
                if mesh is None or not mesh.polygons:
                    continue
                counts = self._write_mesh(fh, mesh, v_off, vt_off, vn_off)
                v_off += counts[0]
                vt_off += counts[1]
                vn_off += counts[2]

        if self.opts.write_mtl:
            self._write_mtl(os.path.join(self._out_dir, mtl_name))
        return {"path": path, "bytes": os.path.getsize(path),
                "textures": list(self.written_textures)}

    def _write_mesh(self, fh: TextIO, mesh: Mesh, v_off: int, vt_off: int,
                    vn_off: int):
        s = self.opts.scale
        fh.write("o %s\n" % _safe(mesh.name))
        for v in mesh.vertices:
            fh.write("v %.6f %.6f %.6f\n" % (v[0] * s, v[1] * s, v[2] * s))
        for uv in mesh.uvs:
            fh.write("vt %.6f %.6f\n" % (uv[0], uv[1]))
        normals = mesh.normals or []
        for n in normals:
            fh.write("vn %.6f %.6f %.6f\n" % (n[0], n[1], n[2]))

        has_uv = bool(mesh.uvs)
        has_n = bool(normals)
        order = sorted(range(len(mesh.polygons)),
                       key=lambda i: (mesh.polygon_material[i]
                                      if i < len(mesh.polygon_material) else 0))
        current_mat = None
        for pi in order:
            poly = mesh.polygons[pi]
            mat_idx = (mesh.polygon_material[pi]
                       if pi < len(mesh.polygon_material) else 0)
            if mat_idx != current_mat:
                current_mat = mat_idx
                if mesh.materials and 0 <= mat_idx < len(mesh.materials):
                    fh.write("usemtl %s\n" % _safe(mesh.materials[mat_idx].name))
            uvrow = mesh.uv_indices[pi] if pi < len(mesh.uv_indices) else poly
            faces = ([poly] if not self.opts.triangulate
                     else [(poly[0], poly[k], poly[k + 1])
                           for k in range(1, len(poly) - 1)])
            for face in faces:
                parts = []
                for k, vi in enumerate(face):
                    ui = uvrow[k] if (has_uv and k < len(uvrow)) else vi
                    if has_uv and has_n:
                        parts.append("%d/%d/%d" % (vi + v_off, ui + vt_off, vi + vn_off))
                    elif has_uv:
                        parts.append("%d/%d" % (vi + v_off, ui + vt_off))
                    elif has_n:
                        parts.append("%d//%d" % (vi + v_off, vi + vn_off))
                    else:
                        parts.append("%d" % (vi + v_off))
                fh.write("f %s\n" % " ".join(parts))
        return len(mesh.vertices), len(mesh.uvs), len(normals)

    def _write_mtl(self, path: str) -> None:
        seen = set()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# %s %s\n" % (APP_NAME, __version__))
            for mesh in self.scene.meshes():
                for mat in mesh.materials:
                    if mat.name in seen:
                        continue
                    seen.add(mat.name)
                    self._write_material(fh, mat)

    def _write_material(self, fh: TextIO, mat: Material) -> None:
        c = mat.base_color
        fh.write("\nnewmtl %s\n" % _safe(mat.name))
        fh.write("Kd %.6f %.6f %.6f\n" % c)
        fh.write("Ka 0.000000 0.000000 0.000000\n")
        spec = 0.2 + 0.8 * mat.metallic
        fh.write("Ks %.6f %.6f %.6f\n" % (spec, spec, spec))
        fh.write("Ns %.4f\n" % max(1.0, (1.0 - mat.roughness) ** 2 * 900.0))
        fh.write("d %.4f\n" % mat.opacity)
        fh.write("illum %d\n" % (2 if mat.opacity >= 0.999 else 4))
        # PBR extensions (understood by Blender, Substance, three.js loaders)
        fh.write("Pr %.4f\n" % mat.roughness)
        fh.write("Pm %.4f\n" % mat.metallic)
        if any(v > 0 for v in mat.emissive):
            fh.write("Ke %.6f %.6f %.6f\n" % mat.emissive)
        for key, tex in (("map_Kd", mat.base_color_map),
                         ("map_Pr", mat.roughness_map),
                         ("map_Pm", mat.metallic_map),
                         ("map_bump", mat.bump_map),
                         ("norm", mat.normal_map),
                         ("map_d", mat.opacity_map),
                         ("map_Ke", mat.emissive_map)):
            if tex is None or not tex.path:
                continue
            fh.write("%s %s\n" % (key, self._texture_path(tex.path)))

    def _texture_path(self, src: str) -> str:
        if self.opts.copy_textures and os.path.isfile(src):
            key = os.path.normcase(src)
            dest = self._copied.get(key)
            if dest is None:
                dest_dir = os.path.join(self._out_dir, self.opts.texture_subdir)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, os.path.basename(src))
                if not os.path.exists(dest):
                    try:
                        shutil.copy2(src, dest)
                        self.written_textures.append(
                            os.path.relpath(dest, self._out_dir))
                    except OSError:
                        dest = src
                self._copied[key] = dest
            try:
                return os.path.relpath(dest, self._out_dir).replace(os.sep, "/")
            except ValueError:
                return dest.replace(os.sep, "/")
        try:
            return os.path.relpath(src, self._out_dir).replace(os.sep, "/")
        except ValueError:
            return src.replace(os.sep, "/")


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in str(name))


def write_obj(scene: Scene, path: str,
              options: Optional[ObjOptions] = None) -> Dict[str, Any]:
    return ObjWriter(scene, options).write(path)
