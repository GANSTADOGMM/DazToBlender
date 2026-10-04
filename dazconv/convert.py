"""High level conversion pipeline: Daz/OBJ file in, FBX out.

:func:`convert_file` is the single entry point used by the CLI, the web app and
the unit tests.  It resolves textures, loads the scene, exports FBX and returns
a :class:`ConversionReport` that the UI can render verbatim.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .fbx.exporter import FbxExportError, FbxExporter
from .library import ContentLibrary, guess_roots_from_path
from .morphs import discover_morphs, load_morphs_from_file
from .options import ExportOptions
from .scene import SceneLoader, SceneModel

ProgressFn = Optional[Callable[[str, float, str], None]]


@dataclass
class ConversionReport:
    input_path: str
    output_path: str = ""
    success: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0
    scene_stats: dict = field(default_factory=dict)
    export: dict = field(default_factory=dict)
    mesh_stats: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    content_roots: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "input": self.input_path,
            "input_name": os.path.basename(self.input_path),
            "output": self.output_path,
            "output_name": os.path.basename(self.output_path),
            "success": self.success,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "scene": self.scene_stats,
            "export": self.export,
            "meshes": self.mesh_stats,
            "warnings": self.warnings,
            "content_roots": self.content_roots,
        }


def resolve_textures(scene: SceneModel, library: ContentLibrary) -> List[str]:
    """Fill in ``material.texture_paths`` from Daz content URLs."""
    missing: List[str] = []
    for asset in scene.meshes:
        for material in asset.materials:
            for attribute in ("base_color_map", "normal_map", "opacity_map", "bump_map"):
                url = getattr(material, attribute) or ""
                if not url or url in material.texture_paths:
                    continue
                resolved = library.resolve_texture(url)
                if resolved is None:
                    fallback = _sibling_texture(material.source_path or scene.source_path, url)
                    resolved = fallback
                if resolved is None:
                    if url not in missing:
                        missing.append(url)
                    continue
                material.texture_paths[url] = resolved
    return missing


def _sibling_texture(source_path: str, url: str) -> Optional[str]:
    """Resolve a texture stored next to the asset (common for OBJ + MTL)."""
    if not source_path or not url:
        return None
    candidates = [
        os.path.join(os.path.dirname(source_path), os.path.basename(url.replace("\\", "/"))),
        os.path.join(os.path.dirname(source_path), url.replace("\\", "/").lstrip("/")),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def default_output_path(input_path: str, output_dir: str = "") -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    directory = output_dir or os.path.join(os.path.dirname(os.path.abspath(input_path)), "converted")
    return os.path.join(directory, f"{base}.fbx")


def _attach_extra_morphs(scene, options: ExportOptions) -> List[str]:
    """Load morphs from ``morph_files`` / ``morph_search_dirs`` onto matching meshes.

    A bare geometry ``.dsf`` does not carry its morphs -- Daz keeps them in
    separate files that a ``.duf`` scene references.  This lets the same morphs be
    used when converting a figure on its own.  Morphs are only attached to meshes
    whose vertex count matches, which is how Daz pairs them up too.
    """
    if not (options.morph_files or options.morph_search_dirs):
        return []

    warnings: List[str] = []
    extra = []
    for path in options.morph_files:
        if not os.path.isfile(path):
            warnings.append(f"morph file not found: {path}")
            continue
        loaded = load_morphs_from_file(path)
        if not loaded:
            warnings.append(f"no morphs found in {os.path.basename(path)}")
        extra.extend(loaded)
    if options.morph_search_dirs:
        extra.extend(discover_morphs(options.morph_search_dirs))

    for asset in scene.meshes:
        known = {morph.id for morph in asset.morphs}
        target = asset.geometry.vertex_count
        for morph in extra:
            if morph.id in known:
                continue
            if morph.vertex_count != target:
                continue
            asset.morphs.append(morph)
            known.add(morph.id)
    return warnings


def convert_file(
    input_path: str,
    output_path: str = "",
    options: Optional[ExportOptions] = None,
    library: Optional[ContentLibrary] = None,
    progress: ProgressFn = None,
) -> ConversionReport:
    """Convert one Daz/OBJ file to FBX."""
    started = time.time()
    options = options or ExportOptions()
    input_path = os.path.abspath(input_path)
    report = ConversionReport(input_path=input_path)

    problems = options.validate()
    if problems:
        report.error = "; ".join(problems)
        return report
    if not os.path.isfile(input_path):
        report.error = f"input file not found: {input_path}"
        return report

    library = library or ContentLibrary()
    if not library.roots:
        library.add_roots(guess_roots_from_path(input_path))
        library.add_roots([os.path.dirname(input_path)])
    report.content_roots = list(library.roots)

    def notify(stage: str, fraction: float, message: str = "") -> None:
        if progress:
            try:
                progress(stage, fraction, message)
            except Exception:  # a broken progress sink must not kill a conversion
                pass

    try:
        notify("load", 0.05, f"reading {os.path.basename(input_path)}")
        loader = SceneLoader(
            library=library,
            flip_v=options.flip_uv,
            max_influences=options.max_influences,
            min_weight=options.min_weight,
        )
        scene = loader.load(input_path)
        if not scene.meshes:
            report.error = "no geometry found in the input file"
            report.elapsed_seconds = time.time() - started
            return report

        notify("morphs", 0.42, "loading morphs")
        report.warnings.extend(_attach_extra_morphs(scene, options))

        # Stats are captured here so attached morphs are counted.
        report.scene_stats = scene.stats()
        report.mesh_stats = [asset.stats() for asset in scene.meshes]
        for asset in scene.meshes:
            report.warnings.extend(asset.warnings)
        report.warnings.extend(scene.warnings)

        notify("textures", 0.5, "resolving textures")
        missing = resolve_textures(scene, library)
        if missing and options.export_textures:
            report.warnings.extend(f"texture not found: {url}" for url in missing[:20])
            if len(missing) > 20:
                report.warnings.append(f"...and {len(missing) - 20} more missing textures")

        output_path = os.path.abspath(output_path or default_output_path(input_path))
        report.output_path = output_path
        notify("export", 0.6, f"writing {os.path.basename(output_path)}")
        exporter = FbxExporter(options, library)
        export_report = exporter.export(scene, output_path)
        report.export = export_report
        report.warnings.extend(export_report.get("warnings", []))

        if options.fail_on_warning and report.warnings:
            report.error = "conversion produced warnings and fail_on_warning is set"
            report.elapsed_seconds = time.time() - started
            return report

        report.success = True
        notify("done", 1.0, "finished")
    except (FbxExportError, ValueError, OSError) as error:
        report.error = f"{type(error).__name__}: {error}"
    except Exception as error:  # pragma: no cover - surfaced verbatim in the UI
        report.error = f"unexpected {type(error).__name__}: {error}"

    report.elapsed_seconds = time.time() - started
    return report


def convert_files(
    input_paths: Sequence[str],
    output_dir: str = "",
    options: Optional[ExportOptions] = None,
    library: Optional[ContentLibrary] = None,
    progress: ProgressFn = None,
) -> List[ConversionReport]:
    """Batch convert, reusing one content library across files."""
    library = library or ContentLibrary()
    reports: List[ConversionReport] = []
    total = max(1, len(input_paths))
    for index, path in enumerate(input_paths):
        output = os.path.join(
            output_dir, os.path.splitext(os.path.basename(path))[0] + ".fbx"
        ) if output_dir else ""
        report = convert_file(
            path,
            output,
            options=options,
            library=library,
            progress=progress,
        )
        reports.append(report)
        if progress:
            progress("batch", (index + 1) / total, f"{index + 1}/{total} files")
    return reports
