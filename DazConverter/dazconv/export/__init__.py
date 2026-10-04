"""Exporters: IR Scene -> FBX / glTF / GLB / OBJ."""

from typing import Any, Callable, Dict

from .fbx.writer import FbxOptions, write_fbx
from .gltf import GltfOptions, scene_to_glb, write_gltf
from .obj import ObjOptions, write_obj

FORMATS = {
    "fbx": {"ext": ".fbx", "label": "FBX (binary 7.4)",
            "rig": True, "morphs": True, "quads": True},
    "glb": {"ext": ".glb", "label": "glTF 2.0 binary (.glb)",
            "rig": True, "morphs": True, "quads": False},
    "gltf": {"ext": ".gltf", "label": "glTF 2.0 (.gltf + .bin)",
             "rig": True, "morphs": True, "quads": False},
    "obj": {"ext": ".obj", "label": "Wavefront OBJ (+ .mtl)",
            "rig": False, "morphs": False, "quads": True},
}


def write(scene, path: str, fmt: str, **kwargs) -> Dict[str, Any]:
    """Dispatch to the right exporter by format key or file extension."""
    fmt = (fmt or "").lower().lstrip(".")
    if fmt not in FORMATS:
        import os
        fmt = os.path.splitext(path)[1].lower().lstrip(".")
    if fmt == "fbx":
        return write_fbx(scene, path, FbxOptions(**kwargs))
    if fmt in ("glb", "gltf"):
        opts = GltfOptions(**kwargs)
        opts.binary = fmt == "glb"
        return write_gltf(scene, path, opts)
    if fmt == "obj":
        return write_obj(scene, path, ObjOptions(**kwargs))
    raise ValueError("unsupported output format: %r" % fmt)


__all__ = ["FORMATS", "FbxOptions", "GltfOptions", "ObjOptions", "scene_to_glb",
           "write", "write_fbx", "write_gltf", "write_obj"]
