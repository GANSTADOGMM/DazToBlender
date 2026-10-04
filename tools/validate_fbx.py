#!/usr/bin/env python3
"""Independently validate a dazconv-produced FBX file.

Uses the third-party ``ufbx`` parser -- a completely separate FBX
implementation -- so a green result means the file is genuinely well-formed
rather than merely readable by our own writer.

The check runs standalone (``python tools/validate_fbx.py out.fbx``) and is also
driven by the test suite inside a subprocess: the ``ufbx`` Python bindings have
object-lifetime bugs that can segfault, and a crash there must be reported, not
allowed to take the caller down.

Note: ``ufbx`` crashes if a parent collection proxy is garbage-collected while a
child is still in use, so every list is bound to a name before being indexed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any


def inspect(path: str) -> dict[str, Any]:
    """Load ``path`` with ufbx and return a summary dict.

    Raises on any failure, including the native errors ufbx raises.
    """
    import ufbx

    scene = ufbx.load_file(path)

    meshes = scene.meshes
    nodes = scene.nodes
    materials = scene.materials
    result: dict[str, Any] = {
        "num_meshes": len(meshes),
        "num_nodes": len(nodes),
        "num_materials": len(materials),
        "num_bones": 0,
        "meshes": [],
    }

    bone_nodes = [node for node in nodes if node.bone is not None]
    result["num_bones"] = len(bone_nodes)
    result["bone_names"] = [node.name for node in bone_nodes]

    for mesh in meshes:
        entry: dict[str, Any] = {
            "name": mesh.name,
            "vertices": mesh.num_vertices,
            "faces": mesh.num_faces,
            "indices": mesh.num_indices,
            "uv_sets": [set_.name for set_ in mesh.uv_sets],
            "materials": len(mesh.materials),
            "has_normals": bool(mesh.vertex_normal.exists),
            "vertex_positions": [],
            "uvs": [],
            "uv_count": 0,
            "skin_clusters": 0,
            "skinned_vertices": 0,
            "weight_sum_min": None,
            "weight_sum_max": None,
            "max_influences": 0,
        }

        vertices = mesh.vertices
        for index in range(min(3, mesh.num_vertices)):
            vertex = vertices[index]
            entry["vertex_positions"].append([vertex.x, vertex.y, vertex.z])

        uv_sets = mesh.uv_sets
        if len(uv_sets):
            channel = uv_sets[0]
            stream = channel.vertex_uv
            entry["uv_count"] = len(stream.values)
            values = stream.values
            for index in range(min(3, len(values))):
                value = values[index]
                entry["uvs"].append([value.x, value.y])

        # Keep every intermediate collection bound to a name (see module docstring).
        deformers = mesh.skin_deformers
        if len(deformers):
            deformer = deformers[0]
            clusters = deformer.clusters
            entry["skin_clusters"] = len(clusters)
            totals: dict[int, float] = collections.defaultdict(float)
            per_vertex: collections.Counter = collections.Counter()
            for cluster in clusters:
                indices = cluster.vertices
                weights = cluster.weights
                for vertex, weight in zip(indices, weights):
                    totals[vertex] += weight
                    per_vertex[vertex] += 1
            entry["skinned_vertices"] = len(totals)
            if totals:
                values = list(totals.values())
                entry["weight_sum_min"] = min(values)
                entry["weight_sum_max"] = max(values)
                entry["max_influences"] = max(per_vertex.values())

        result["meshes"].append(entry)

    return result


def compare_vertex_delta(base_path: str, morphed_path: str, deltas: dict) -> dict:
    """Check that ``morphed`` equals ``base`` plus ``deltas``.

    Both files are loaded here rather than in the caller because the ufbx
    bindings segfault when a scene is collected while a mesh is still in use;
    keeping both scenes bound for the lifetime of the process avoids that.
    """
    import ufbx

    base_scene = ufbx.load_file(base_path)
    morphed_scene = ufbx.load_file(morphed_path)
    base_vertices = base_scene.meshes[0].vertices
    morphed_vertices = morphed_scene.meshes[0].vertices

    checked = 0
    moved = 0
    worst = 0.0
    for raw_index, delta in deltas.items():
        index = int(raw_index)
        before = base_vertices[index]
        after = morphed_vertices[index]
        axes = (
            (after.x - before.x, delta[0]),
            (after.y - before.y, delta[1]),
            (after.z - before.z, delta[2]),
        )
        for actual, expected in axes:
            worst = max(worst, abs(actual - expected))
        if max(abs(a) for a, _ in axes) > 1e-6:
            moved += 1
        checked += 1

    return {"ok": True, "checked": checked, "moved": moved, "max_deviation": worst}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fbx", help="FBX file to validate")
    parser.add_argument("--pretty", action="store_true", help="human-readable output")
    parser.add_argument(
        "--compare", metavar="MORPHED",
        help="treat the positional file as the pre-morph base and check it "
             "against this morphed export using --deltas",
    )
    parser.add_argument("--deltas", help="json map of vertex index -> [dx, dy, dz]")
    args = parser.parse_args(argv)

    try:
        if args.compare:
            if not args.deltas:
                parser.error("--compare requires --deltas")
            with open(args.deltas, "r", encoding="utf-8") as handle:
                deltas = json.load(handle)
            summary = compare_vertex_delta(args.fbx, args.compare, deltas)
        else:
            summary = inspect(args.fbx)
    except Exception as error:  # includes ufbx's native error types
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        return 1

    summary["ok"] = True
    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
