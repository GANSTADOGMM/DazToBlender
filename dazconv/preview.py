"""Compact geometry payload for the browser preview.

The web UI renders this with its own WebGL code, so the app needs no external
JavaScript library and keeps working offline.  Only what a viewer needs is
sent -- positions, triangle indices, normals, bounds, bone tree and material
colours -- rounded to 4 decimals to keep the JSON small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .mathutil import (
    Mat4,
    mat4_identity,
    mat4_from_scale,
    mat4_mul,
    rot_x,
    transform_point,
    transform_vector,
    vec3,
)
from .options import ExportOptions
from .scene import MeshAsset, SceneModel

ROUND = 4


def _round3(vector) -> List[float]:
    return [
        round(float(vector[0]), ROUND),
        round(float(vector[1]), ROUND),
        round(float(vector[2]), ROUND),
    ]


def _flatten(vectors) -> List[float]:
    """``[[x, y, z], ...]`` -> ``[x, y, z, ...]``; what WebGL buffers want."""
    out: List[float] = []
    for vector in vectors:
        out.append(round(float(vector[0]), ROUND))
        out.append(round(float(vector[1]), ROUND))
        out.append(round(float(vector[2]), ROUND))
    return out


def preview_transform(options: ExportOptions) -> Mat4:
    """The same scale / up-axis transform the exporter bakes into the FBX."""
    matrix = mat4_identity()
    if options.up_axis == "z":
        matrix = mat4_mul(rot_x(-90.0), matrix)
    if options.scale != 1.0:
        matrix = mat4_mul(
            mat4_from_scale(vec3(options.scale, options.scale, options.scale)), matrix
        )
    return matrix


def mesh_preview(asset: MeshAsset, options: Optional[ExportOptions] = None) -> Dict[str, Any]:
    """Build the preview payload for one mesh asset."""
    options = options or ExportOptions()
    geometry = asset.geometry
    matrix = preview_transform(options)
    vertices = _flatten(transform_point(matrix, v) for v in geometry.vertices)

    if options.triangulate:
        faces = geometry.triangulated_indices()
    else:
        faces = [polygon.vertices for polygon in geometry.polygons]

    # Triangle soup for the renderer, plus the original outlines for wireframe.
    triangles: List[int] = []
    wireframe: List[int] = []
    for face in faces:
        for offset in range(1, len(face) - 1):
            triangles.extend((face[0], face[offset], face[offset + 1]))
        wireframe.extend(list(face))
        wireframe.append(-1)  # polyline separator

    normals = _flatten(transform_vector(matrix, n) for n in geometry.vertex_normals())

    colour = [0.72, 0.72, 0.74]
    materials: List[Dict[str, Any]] = []
    for material in asset.materials[:32]:
        base = [round(float(c), 3) for c in material.base_color][:3]
        if material is asset.materials[0]:
            colour = base
        materials.append({
            "name": material.label or material.name or material.id,
            "diffuse": base,
            "opacity": round(float(material.opacity), 3),
            "textures": sorted(material.used_textures().keys()),
        })

    bounds = geometry.bounds()
    influences = asset.influences
    return {
        "id": asset.id,
        "name": asset.name,
        "vertices": vertices,
        "triangles": triangles,
        "wireframe": wireframe,
        "normals": normals,
        "colour": colour,
        "materials": materials,
        "bounds": {"min": _round3(bounds[0]), "max": _round3(bounds[1])},
        "vertex_count": geometry.vertex_count,
        "polygon_count": geometry.polygon_count,
        "has_uvs": bool(asset.corner_uvs),
        "uv_corners": len(asset.corner_uvs),
        "weighted_vertices": len(influences.joint_weights) if influences else 0,
        "bone_count": len(asset.skeleton.bones) if asset.skeleton else 0,
        "morph_count": len(asset.morphs),
        "is_figure": asset.is_figure,
        "subdivision": geometry.is_subdivision(),
    }


def scene_preview(
    scene: SceneModel, options: Optional[ExportOptions] = None
) -> Dict[str, Any]:
    """Build the full preview payload: meshes plus the bone hierarchy."""
    options = options or ExportOptions()
    matrix = preview_transform(options)
    meshes = [mesh_preview(asset, options) for asset in scene.meshes]

    bones: List[Dict[str, Any]] = []
    skeleton = scene.meshes[0].skeleton if scene.meshes else None
    if skeleton is not None and options.export_skeleton:
        for bone in skeleton.depth_ordered():
            entry: Dict[str, Any] = {
                "id": bone.id,
                "name": bone.label or bone.name or bone.id,
                "parent": bone.parent.id if bone.parent is not None else None,
                "position": _round3(transform_point(matrix, bone.world_position)),
                "type": bone.node_type,
            }
            if bone.length > 1e-6:
                tip = transform_point(matrix, transform_point(bone.world_matrix, bone.end_point))
                entry["end"] = _round3(tip)
            bones.append(entry)

    bounds = None
    mins = [mesh["bounds"]["min"] for mesh in meshes]
    maxs = [mesh["bounds"]["max"] for mesh in meshes]
    if mins:
        bounds = {
            "min": [min(m[axis] for m in mins) for axis in range(3)],
            "max": [max(m[axis] for m in maxs) for axis in range(3)],
        }

    return {
        "name": scene.name,
        "source": scene.source_path,
        "meshes": meshes,
        "bones": bones,
        "bounds": bounds,
        "stats": scene.stats(),
        "warnings": list(scene.warnings),
    }
