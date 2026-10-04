"""High-level conversion API -- the one function most callers need."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import export
from .convert import ConvertOptions, SceneBuilder
from .dson import ContentLibrary, DsonError, autodetect_roots, sniff
from .dson.reader import load as load_dson
from .ir import Scene

ProgressFn = Callable[[float, str], None]


@dataclass
class ConversionResult:
    source: str
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0
    scene: Optional[Scene] = None

    @property
    def ok(self) -> bool:
        return bool(self.outputs)

    @property
    def paths(self) -> List[str]:
        return [o["path"] for o in self.outputs]

    def describe(self) -> str:
        s = self.summary
        lines = ["%s -> %d file(s) in %.2fs"
                 % (os.path.basename(self.source), len(self.outputs), self.duration)]
        if s:
            lines.append("  %s vertices, %s polygons, %s materials, %s bones, "
                         "%s morphs" % (s.get("vertices", 0), s.get("polygons", 0),
                                        s.get("materials", 0), s.get("bones", 0),
                                        s.get("morphs", 0)))
        for o in self.outputs:
            lines.append("  %s (%.1f KB)" % (o["path"], o.get("bytes", 0) / 1024.0))
        for w in self.warnings[:10]:
            lines.append("  ! %s" % w)
        return "\n".join(lines)


def convert(source: str,
            output: str,
            formats: Optional[Sequence[str]] = None,
            options: Optional[ConvertOptions] = None,
            export_options: Optional[Dict[str, Dict[str, Any]]] = None,
            progress: Optional[ProgressFn] = None) -> ConversionResult:
    """Convert a Daz asset to one or more 3D formats.

    Args:
        source: path to a ``.duf`` scene/preset or a ``.dsf`` library asset.
        output: output file path, or a directory when converting to several
            formats at once.
        formats: list of format keys ("fbx", "glb", "gltf", "obj").  Defaults
            to the extension of `output`, or FBX.
        options: DSON-side conversion options (morphs, rigging, pose, ...).
        export_options: per-format keyword overrides, e.g.
            ``{"fbx": {"embed_textures": True}}``.
    """
    start = time.time()
    source = os.path.abspath(os.path.expanduser(source))
    opts = options or ConvertOptions()
    report = progress or (lambda f, m: None)

    if not os.path.isfile(source):
        raise FileNotFoundError(source)

    fmts = _resolve_formats(formats, output)
    out_dir, out_stem = _resolve_output(output, source, len(fmts))

    library = ContentLibrary(opts.content_dirs or autodetect_roots())
    builder = SceneBuilder(library, opts,
                           progress=lambda f, m: report(f * 0.7, m))
    scene = builder.build_from_file(source)

    result = ConversionResult(source=source, scene=scene,
                              warnings=list(scene.warnings),
                              summary=scene.summary())
    if not scene.figures:
        result.duration = time.time() - start
        return result

    exp_opts = export_options or {}
    for i, fmt in enumerate(fmts):
        ext = export.FORMATS[fmt]["ext"]
        path = os.path.join(out_dir, out_stem + ext)
        report(0.7 + 0.3 * (i / max(1, len(fmts))), "Writing %s" % os.path.basename(path))
        info = export.write(scene, path, fmt, **exp_opts.get(fmt, {}))
        info["format"] = fmt
        result.outputs.append(info)

    report(1.0, "Finished")
    result.duration = time.time() - start
    return result


def convert_scene(scene: Scene, output: str, fmt: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
    """Export an already-built IR scene."""
    fmt = fmt or os.path.splitext(output)[1].lstrip(".").lower()
    return export.write(scene, output, fmt, **kwargs)


def scene_dependencies(path: str, content_dirs: Optional[Sequence[str]] = None,
                       limit: int = 400) -> Dict[str, Any]:
    """List the content files a scene references and whether they resolve.

    This is what turns "nothing was exported" into "you are missing the
    Genesis 8 Female base figure, install it or add its content directory".
    """
    from .dson import uri as uri_mod
    path = os.path.abspath(os.path.expanduser(path))
    library = ContentLibrary(list(content_dirs or []) or autodetect_roots())
    library.add_root_for_file(path)
    try:
        doc = load_dson(path)
    except DsonError as exc:
        return {"error": str(exc), "dependencies": []}

    refs: List[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("url", "uv_set", "parent", "conform_target") and isinstance(value, str):
                    f = uri_mod.parse(value).file_path
                    if f and f not in refs:
                        refs.append(f)
                elif key == "image_file" and isinstance(value, str):
                    if value not in refs:
                        refs.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node[:2000]:
                walk(item)

    walk(doc.get("scene") or doc)
    deps = []
    for ref in refs[:limit]:
        resolved = library.resolve_path(ref, path)
        deps.append({"path": ref, "found": bool(resolved), "resolved": resolved})
    missing = [d for d in deps if not d["found"]]
    return {
        "roots": library.roots,
        "dependencies": deps,
        "missing": [d["path"] for d in missing],
        "missingCount": len(missing),
        "total": len(deps),
    }


def inspect_file(path: str, content_dirs: Optional[Sequence[str]] = None
                 ) -> Dict[str, Any]:
    """Cheap metadata probe used by the UI before a full conversion."""
    path = os.path.abspath(os.path.expanduser(path))
    info: Dict[str, Any] = {
        "path": path,
        "name": os.path.basename(path),
        "size": os.path.getsize(path) if os.path.isfile(path) else 0,
        "type": None,
        "figures": [],
        "node_count": 0,
        "has_scene": False,
        "content_root": None,
        "errors": [],
    }
    try:
        doc = load_dson(path)
    except DsonError as exc:
        info["errors"].append(str(exc))
        return info

    asset_info = doc.get("asset_info") or {}
    info["type"] = asset_info.get("type") or sniff(path)
    info["author"] = (asset_info.get("contributor") or {}).get("author")
    info["modified"] = asset_info.get("modified")

    from .dson.library import guess_root_for_file
    info["content_root"] = guess_root_for_file(path)

    scene_block = doc.get("scene")
    if isinstance(scene_block, dict):
        info["has_scene"] = True
        nodes = scene_block.get("nodes") or []
        info["node_count"] = len(nodes)
        info["modifier_count"] = len(scene_block.get("modifiers") or [])
        info["material_count"] = len(scene_block.get("materials") or [])
        for n in nodes:
            if isinstance(n, dict) and n.get("geometries"):
                info["figures"].append({
                    "id": n.get("id"),
                    "label": n.get("label") or n.get("name"),
                    "type": (n.get("preview") or {}).get("type", "node"),
                })
    else:
        info["node_count"] = len(doc.get("node_library") or [])
        info["geometry_count"] = len(doc.get("geometry_library") or [])
        info["modifier_count"] = len(doc.get("modifier_library") or [])
        for g in doc.get("geometry_library") or []:
            if isinstance(g, dict):
                verts = (g.get("vertices") or {}).get("count")
                info["figures"].append({"id": g.get("id"),
                                        "label": g.get("name") or g.get("id"),
                                        "type": g.get("type", "geometry"),
                                        "vertices": verts})
    return info


def _resolve_formats(formats: Optional[Sequence[str]], output: str) -> List[str]:
    if formats:
        out = []
        for f in formats:
            key = str(f).lower().lstrip(".")
            if key not in export.FORMATS:
                raise ValueError("unknown format %r" % f)
            if key not in out:
                out.append(key)
        return out
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    if ext in export.FORMATS:
        return [ext]
    return ["fbx"]


def _resolve_output(output: str, source: str, n_formats: int):
    output = os.path.abspath(os.path.expanduser(output))
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    if ext in export.FORMATS:
        # An explicit filename: keep its stem, write every format beside it.
        return (os.path.dirname(output) or ".",
                os.path.splitext(os.path.basename(output))[0])
    # No known extension: treat it as a directory and name files after the input.
    stem = os.path.splitext(os.path.basename(source))[0]
    return output, stem
