"""DSON geometry -> IR mesh."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import math3d
from ..ir import Mesh, MorphTarget, SkinWeights


def _values(block: Any) -> List[Any]:
    """Pull the `values` array out of a DSON counted-array block."""
    if isinstance(block, dict):
        v = block.get("values")
        return v if isinstance(v, list) else []
    if isinstance(block, list):
        return block
    return []


def parse_geometry(geom: Dict[str, Any], name: str = "") -> Mesh:
    """Build a Mesh from a DSON ``geometry_library`` entry.

    Handles polygon_mesh and subdivision_surface (base cage) types; the
    polylist encodes ``[group_index, material_index, v0, v1, v2 (, v3)]``.
    """
    mesh = Mesh(name=name or str(geom.get("name") or geom.get("id") or "mesh"))

    for v in _values(geom.get("vertices")):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            mesh.vertices.append((float(v[0]), float(v[1]), float(v[2])))

    mesh.groups = [str(g) for g in _values(geom.get("polygon_groups"))]
    mat_groups = [str(g) for g in _values(geom.get("polygon_material_groups"))]

    polylist = _values(geom.get("polylist"))
    nverts = len(mesh.vertices)
    for face in polylist:
        if not isinstance(face, (list, tuple)) or len(face) < 5:
            continue
        grp = int(face[0])
        mat = int(face[1])
        idx = []
        for vi in face[2:]:
            try:
                i = int(vi)
            except (TypeError, ValueError):
                continue
            # Daz pads triangles inside quad lists by repeating the last index.
            if idx and i == idx[-1]:
                continue
            if 0 <= i < nverts:
                idx.append(i)
        if len(idx) >= 3 and idx[0] == idx[-1]:
            idx.pop()
        if len(idx) < 3:
            continue
        mesh.polygons.append(tuple(idx))
        mesh.polygon_group.append(grp)
        mesh.polygon_material.append(mat)

    mesh.material_groups = list(mat_groups)
    return mesh


def apply_uv_set(mesh: Mesh, uv_set: Dict[str, Any]) -> None:
    """Attach a DSON uv_set asset to a mesh as per polygon-vertex UVs.

    The default UV index for a polygon-vertex is simply the vertex index;
    ``polygon_vertex_indices`` lists the exceptions (UV seams), where a
    vertex gets a different UV depending on which polygon references it.
    """
    uvs_block = uv_set.get("uvs")
    values = _values(uvs_block)
    mesh.uvs = []
    for uv in values:
        if isinstance(uv, (list, tuple)) and len(uv) >= 2:
            mesh.uvs.append((float(uv[0]), float(uv[1])))
    if not mesh.uvs:
        mesh.uv_indices = []
        return

    # (polygon, vertex) -> uv index overrides
    overrides: Dict[Tuple[int, int], int] = {}
    for entry in uv_set.get("polygon_vertex_indices") or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            overrides[(int(entry[0]), int(entry[1]))] = int(entry[2])

    nuv = len(mesh.uvs)
    uv_indices: List[Tuple[int, ...]] = []
    for pi, poly in enumerate(mesh.polygons):
        row = []
        for vi in poly:
            ui = overrides.get((pi, vi), vi)
            if ui < 0 or ui >= nuv:
                ui = vi if vi < nuv else 0
            row.append(ui)
        uv_indices.append(tuple(row))
    mesh.uv_indices = uv_indices


def generate_planar_uvs(mesh: Mesh) -> None:
    """Fallback UVs so exports are never UV-less (box-ish planar projection)."""
    if not mesh.vertices:
        return
    xs = [v[0] for v in mesh.vertices]
    ys = [v[1] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    sx = (maxx - minx) or 1.0
    sy = (maxy - miny) or 1.0
    sz = (maxz - minz) or 1.0
    mesh.uvs = [((v[0] - minx) / sx, (v[1] - miny) / sy) for v in mesh.vertices]
    _ = sz
    mesh.uv_indices = [tuple(poly) for poly in mesh.polygons]


def compute_vertex_normals(mesh: Mesh) -> List[Tuple[float, float, float]]:
    """Area-weighted vertex normals (smooth shading)."""
    acc = [[0.0, 0.0, 0.0] for _ in mesh.vertices]
    verts = mesh.vertices
    for poly in mesh.polygons:
        if len(poly) < 3:
            continue
        a, b, c = verts[poly[0]], verts[poly[1]], verts[poly[2]]
        n = math3d.cross(math3d.sub(b, a), math3d.sub(c, a))
        if len(poly) > 3:
            # Newell's method for ngons/quads is more stable.
            nx = ny = nz = 0.0
            for i in range(len(poly)):
                cur = verts[poly[i]]
                nxt = verts[poly[(i + 1) % len(poly)]]
                nx += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
                ny += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
                nz += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
            n = (nx, ny, nz)
        for vi in poly:
            acc[vi][0] += n[0]
            acc[vi][1] += n[1]
            acc[vi][2] += n[2]
    out: List[Tuple[float, float, float]] = []
    for a in acc:
        ln = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        if ln < 1e-12:
            out.append((0.0, 1.0, 0.0))
        else:
            out.append((a[0] / ln, a[1] / ln, a[2] / ln))
    return out


def transform_mesh(mesh: Mesh, matrix: Sequence[float]) -> None:
    """Bake a world matrix into the vertex positions (and clear it)."""
    if matrix is None or list(matrix) == math3d.IDENTITY:
        return
    mesh.vertices = [math3d.transform_point(matrix, v) for v in mesh.vertices]
    if mesh.normals:
        inv_t = _normal_matrix(matrix)
        mesh.normals = [math3d.normalize(math3d.transform_vector(inv_t, n))
                        for n in mesh.normals]
    for morph in mesh.morphs:
        morph.deltas = [math3d.transform_vector(matrix, d) for d in morph.deltas]
    mesh.world_matrix = math3d.identity()


def _normal_matrix(m: Sequence[float]) -> List[float]:
    inv = math3d.mat_inverse(m)
    return [inv[0], inv[4], inv[8], 0,
            inv[1], inv[5], inv[9], 0,
            inv[2], inv[6], inv[10], 0,
            0, 0, 0, 1]


def remove_polygons(mesh: Mesh, kill: Sequence[int]) -> None:
    """Delete polygons by index (used for geograft hidden faces)."""
    killset = set(int(k) for k in kill)
    if not killset:
        return
    polys, mats, groups, uvi = [], [], [], []
    for i, poly in enumerate(mesh.polygons):
        if i in killset:
            continue
        polys.append(poly)
        mats.append(mesh.polygon_material[i] if i < len(mesh.polygon_material) else 0)
        groups.append(mesh.polygon_group[i] if i < len(mesh.polygon_group) else 0)
        if mesh.uv_indices and i < len(mesh.uv_indices):
            uvi.append(mesh.uv_indices[i])
    mesh.polygons, mesh.polygon_material = polys, mats
    mesh.polygon_group = groups
    if mesh.uv_indices:
        mesh.uv_indices = uvi


def drop_unused_vertices(mesh: Mesh) -> int:
    """Compact the vertex array, remapping polygons/weights/morphs.

    Returns the number of vertices removed.
    """
    used = sorted({vi for poly in mesh.polygons for vi in poly})
    if len(used) == len(mesh.vertices):
        return 0
    remap = {old: new for new, old in enumerate(used)}
    mesh.vertices = [mesh.vertices[i] for i in used]
    mesh.polygons = [tuple(remap[v] for v in poly) for poly in mesh.polygons]
    if mesh.normals:
        mesh.normals = [mesh.normals[i] for i in used]
    if mesh.skin and mesh.skin.influences:
        mesh.skin.influences = [mesh.skin.influences[i] for i in used]
    for morph in mesh.morphs:
        idx, dl = [], []
        for i, d in zip(morph.indices, morph.deltas):
            n = remap.get(i)
            if n is not None:
                idx.append(n)
                dl.append(d)
        morph.indices, morph.deltas = idx, dl
    return 0


def merge_meshes(meshes: Sequence[Mesh], name: str = "Merged") -> Mesh:
    """Combine several meshes into one.

    Material, UV, group and surface tables are merged. Skinning and morphs
    survive the merge: weights are re-indexed onto the combined vertex list and
    morph targets with the same name are concatenated. Bone tables are unioned,
    so a body and its conforming clothing can be welded into a single skinned
    mesh without losing the rig.
    """
    out = Mesh(name=name)
    mat_map: Dict[int, int] = {}
    bone_index: Dict[str, int] = {}
    skin = SkinWeights()
    has_skin = False
    morphs: Dict[str, MorphTarget] = {}
    morph_order: List[str] = []
    has_normals = all(m.normals for m in meshes)
    normals: List[Tuple[float, float, float]] = []

    for mesh in meshes:
        voff = len(out.vertices)
        uvoff = len(out.uvs)
        out.vertices.extend(mesh.vertices)
        out.uvs.extend(mesh.uvs)
        if has_normals and mesh.normals:
            normals.extend(mesh.normals)

        local_mat: Dict[int, int] = {}
        for li, mat in enumerate(mesh.materials):
            key = id(mat)
            if key not in mat_map:
                mat_map[key] = len(out.materials)
                out.materials.append(mat)
                out.material_groups.append(
                    mesh.material_groups[li]
                    if li < len(mesh.material_groups) else mat.name)
            local_mat[li] = mat_map[key]

        goff = len(out.groups)
        out.groups.extend(mesh.groups)
        for pi, poly in enumerate(mesh.polygons):
            out.polygons.append(tuple(v + voff for v in poly))
            m = mesh.polygon_material[pi] if pi < len(mesh.polygon_material) else 0
            out.polygon_material.append(local_mat.get(m, 0))
            g = mesh.polygon_group[pi] if pi < len(mesh.polygon_group) else 0
            out.polygon_group.append(g + goff)
            if mesh.uv_indices and pi < len(mesh.uv_indices):
                out.uv_indices.append(tuple(u + uvoff for u in mesh.uv_indices[pi]))
            else:
                out.uv_indices.append(tuple(v + voff for v in poly))

        # --- skinning -----------------------------------------------------
        while len(skin.influences) < voff:
            skin.influences.append([])
        if mesh.skin:
            has_skin = True
            remap = []
            for bone in mesh.skin.bone_names:
                if bone not in bone_index:
                    bone_index[bone] = len(skin.bone_names)
                    skin.bone_names.append(bone)
                remap.append(bone_index[bone])
            for vi in range(len(mesh.vertices)):
                inf = (mesh.skin.influences[vi]
                       if vi < len(mesh.skin.influences) else [])
                skin.influences.append([(remap[b], w) for b, w in inf
                                        if b < len(remap)])
        else:
            skin.influences.extend([] for _ in mesh.vertices)

        # --- morphs -------------------------------------------------------
        for morph in mesh.morphs:
            target = morphs.get(morph.name)
            if target is None:
                target = MorphTarget(name=morph.name, label=morph.label,
                                     default_value=morph.default_value)
                morphs[morph.name] = target
                morph_order.append(morph.name)
            target.indices.extend(i + voff for i in morph.indices)
            target.deltas.extend(morph.deltas)

    if has_skin:
        out.skin = skin
    if has_normals:
        out.normals = normals
    out.morphs = [morphs[n] for n in morph_order]
    return out
