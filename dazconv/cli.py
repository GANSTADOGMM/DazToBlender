#!/usr/bin/env python3
"""Command-line front end for the Daz to FBX converter.

Examples::

    python -m dazconv.cli scene.duf -o out/
    python -m dazconv.cli *.dsf --library-root "C:/DazContent" --triangulate
    python -m dazconv.cli figure.dsf --morph "FBMHeavy=0.8" --max-influences 4
    python -m dazconv.cli model.obj --up-axis z --scale 0.01

Run ``python -m dazconv.cli --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .convert import convert_file, convert_files
from .library import library_from_cli
from .options import ExportOptions

BAR_WIDTH = 42


def _parse_morphs(values: List[str]) -> list:
    """Parse ``Name=0.5`` / ``Name=50%`` / ``Name`` (meaning 100%).

    Returns ``[(id, weight), ...]`` -- the shape ExportOptions expects.
    """
    morphs: list = []
    for raw in values:
        if "=" not in raw:
            morphs.append((raw.strip(), 1.0))
            continue
        name, _, value = raw.partition("=")
        value = value.strip()
        if value.endswith("%"):
            morphs.append((name.strip(), float(value[:-1]) / 100.0))
        else:
            morphs.append((name.strip(), float(value)))
    return morphs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dazconv",
        description="Convert Daz Studio assets (.duf/.daz/.dsf) and .obj meshes to FBX.",
        epilog="Inputs: .duf/.daz scenes, .dsf geometry/figures, .obj meshes. Output: ASCII FBX 7.4.",
    )
    parser.add_argument("inputs", nargs="+", help="input file(s) to convert")
    parser.add_argument(
        "-o", "--output",
        help="output directory, or a .fbx path when a single input is given "
             "(default: alongside each input)",
    )

    library = parser.add_argument_group("content library")
    library.add_argument(
        "-L", "--library-root", action="append", default=[],
        help="Daz content folder used to resolve referenced assets "
             "(repeatable; defaults to the configured library)",
    )
    library.add_argument(
        "--config", default="",
        help="path to the library config json (default: ~/.dazconv/config.json)",
    )

    content = parser.add_argument_group("what to export")
    content.add_argument("--no-skeleton", action="store_true", help="omit the bone hierarchy")
    content.add_argument("--no-weights", action="store_true", help="omit skin weights")
    content.add_argument("--no-materials", action="store_true", help="omit materials")
    content.add_argument("--no-textures", action="store_true", help="omit texture assignments")
    content.add_argument("--no-embed-textures", action="store_true",
                         help="reference texture files instead of embedding them")

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument("--triangulate", action="store_true", help="convert all polygons to triangles")
    geometry.add_argument("--no-flip-uv", action="store_true",
                          help="do not flip V (Daz UV origin is top-left)")
    geometry.add_argument("--flat-normals", action="store_true", help="per-face instead of smooth normals")
    geometry.add_argument("--max-influences", type=int, default=4,
                          help="maximum bone influences per vertex (default: 4)")
    geometry.add_argument("--min-weight", type=float, default=1e-5,
                          help="drop influences below this weight (default: 1e-5)")

    morphs = parser.add_argument_group("morphs")
    morphs.add_argument(
        "-m", "--morph", action="append", default=[], metavar="NAME=VALUE",
        help="bake a morph into the mesh, e.g. 'FBMHeavy=0.8' or 'FBMHeavy=80%%' (repeatable)",
    )
    morphs.add_argument(
        "--morph-file", action="append", default=[], metavar="PATH",
        help="load morphs from this .dsf/.duf file (repeatable)",
    )
    morphs.add_argument(
        "--morph-dir", action="append", default=[], metavar="DIR",
        help="scan this folder for morph assets, e.g. a Daz 'Morphs' tree (repeatable)",
    )
    morphs.add_argument(
        "--export-blendshapes", action="store_true",
        help="also emit every available morph as an FBX blend shape "
             "(stays editable after import)",
    )

    transform = parser.add_argument_group("transform")
    transform.add_argument("--up-axis", choices=["y", "z"], default="y",
                           help="target up axis (default: y)")
    transform.add_argument("--unit-scale", type=float, default=1.0,
                           help="FBX unit scale factor, cm per unit (default: 1.0)")
    transform.add_argument("--scale", type=float, default=1.0,
                           help="extra uniform scale multiplier (default: 1.0)")

    output = parser.add_argument_group("output")
    output.add_argument("--json-report", help="write the full conversion report as JSON")
    output.add_argument("--fail-on-warning", action="store_true",
                        help="treat warnings as failures")
    output.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    output.add_argument("--version", action="version", version=f"dazconv {__version__}")
    return parser


def _print_progress(stage: str, fraction: float, message: str) -> None:
    if not sys.stdout.isatty():
        return
    filled = int(BAR_WIDTH * min(max(fraction, 0.0), 1.0))
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    sys.stdout.write(f"\r  [{bar}] {fraction * 100:5.1f}%  {stage:9s} {message[:38]:38s}")
    sys.stdout.flush()
    if stage == "done":
        sys.stdout.write("\n")


def _options_from_args(args: argparse.Namespace) -> ExportOptions:
    return ExportOptions(
        triangulate=args.triangulate,
        flip_uv=not args.no_flip_uv,
        smooth_normals=not args.flat_normals,
        max_influences=args.max_influences,
        min_weight=args.min_weight,
        export_skeleton=not args.no_skeleton,
        export_weights=not args.no_weights,
        export_materials=not args.no_materials,
        export_textures=not args.no_textures,
        export_blendshapes=args.export_blendshapes,
        embed_textures=not args.no_embed_textures,
        morph_selections=_parse_morphs(args.morph),
        morph_files=list(args.morph_file),
        morph_search_dirs=list(args.morph_dir),
        up_axis=args.up_axis,
        unit_scale=args.unit_scale,
        scale=args.scale,
        fail_on_warning=args.fail_on_warning,
    )


def _resolve_output(args: argparse.Namespace, source: str) -> Optional[str]:
    """A single input plus an ``-o`` that is not a directory means 'this exact path'."""
    if args.output and len(args.inputs) == 1 and args.output.lower().endswith(".fbx"):
        return args.output
    if args.output:
        return os.path.join(args.output, os.path.splitext(os.path.basename(source))[0] + ".fbx")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    missing = [path for path in args.inputs if not os.path.isfile(path)]
    if missing:
        for path in missing:
            print(f"error: input not found: {path}", file=sys.stderr)
        return 2

    options = _options_from_args(args)
    problem = options.validate()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    library = library_from_cli(args.library_root, args.config)

    if args.output and not args.output.lower().endswith(".fbx"):
        os.makedirs(args.output, exist_ok=True)

    if len(args.inputs) > 1:
        if args.output and args.output.lower().endswith(".fbx"):
            print("error: -o must be a directory when converting multiple inputs",
                  file=sys.stderr)
            return 2
        reports = convert_files(
            args.inputs, args.output or "", options, library, _print_progress
        )
    else:
        reports = [
            convert_file(
                args.inputs[0],
                _resolve_output(args, args.inputs[0]),
                options,
                library,
                _print_progress,
            )
        ]

    failures = 0
    for report in reports:
        if report.success:
            size = os.path.getsize(report.output_path) if report.output_path else 0
            print(f"ok   {report.input_path}")
            print(f"     -> {report.output_path}  ({size / 1024.0:.1f} KB, {report.elapsed_seconds:.2f}s)")
            if not args.quiet:
                stats = report.scene_stats
                print(f"     {stats.get('vertices', 0)} vertices, "
                      f"{stats.get('polygons', 0)} polygons, "
                      f"{stats.get('meshes', 0)} mesh(es), "
                      f"{stats.get('bones', 0)} bones, "
                      f"{stats.get('morphs', 0)} morph(s), "
                      f"{stats.get('materials', 0)} material(s)")
        else:
            failures += 1
            print(f"FAIL {report.input_path}: {report.error}", file=sys.stderr)

        for warning in report.warnings:
            if not args.quiet:
                print(f"  warning: {warning}")
        if report.success and report.warnings and options.fail_on_warning:
            failures += 1
            print(f"  -> failed: {len(report.warnings)} warning(s) with --fail-on-warning",
                  file=sys.stderr)

    if args.json_report:
        payload = {
            "version": __version__,
            "options": options.as_dict(),
            "results": [report.as_dict() for report in reports],
        }
        with open(args.json_report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        if not args.quiet:
            print(f"report: {args.json_report}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
