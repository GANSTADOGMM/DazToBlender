"""Command line interface:  python -m dazconv ..."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from . import config, export
from .api import convert, inspect_file
from .convert import ConvertOptions
from .version import APP_NAME, APP_TAGLINE, __version__


def _add_convert_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default="",
                   help="output file or directory (default: alongside input)")
    p.add_argument("-f", "--format", default="fbx",
                   help="comma separated: fbx,glb,gltf,obj (default fbx)")
    p.add_argument("-c", "--content-dir", action="append", default=[],
                   metavar="DIR",
                   help="Daz content directory root (repeatable)")
    p.add_argument("--no-morphs", action="store_true", help="skip blend shapes")
    p.add_argument("--hd-morphs", action="store_true",
                   help="include *_HD_div* morphs (usually unusable at base res)")
    p.add_argument("--morph", action="append", default=[], metavar="PATTERN",
                   help="only export morphs matching this name/pattern")
    p.add_argument("--no-rig", action="store_true", help="skip skeleton and weights")
    p.add_argument("--pose", choices=("rest", "scene"), default="rest",
                   help="export the rest pose or the pose saved in the scene")
    p.add_argument("--influences", type=int, default=8,
                   help="max bone influences per vertex (default 8)")
    p.add_argument("--merge", action="store_true",
                   help="merge every mesh into one object (keeps the shared rig)")
    p.add_argument("--scale", type=float, default=1.0, help="extra scale factor")
    p.add_argument("--embed-textures", action="store_true",
                   help="embed images in the output file where supported")
    p.add_argument("--no-copy-textures", action="store_true",
                   help="reference textures in place instead of copying them")
    p.add_argument("--hidden", action="store_true", help="include hidden nodes")
    p.add_argument("-q", "--quiet", action="store_true")


def _options_from_args(args) -> ConvertOptions:
    dirs = list(args.content_dir) + [d for d in config.effective_content_dirs()
                                     if d not in args.content_dir]
    return ConvertOptions(
        content_dirs=dirs,
        include_morphs=not args.no_morphs,
        morph_filter=list(args.morph),
        include_hd_morphs=args.hd_morphs,
        include_skeleton=not args.no_rig,
        max_influences=max(1, args.influences),
        pose=args.pose,
        merge_meshes=args.merge,
        include_hidden=args.hidden,
        scale=args.scale,
    )


def _export_options(args) -> dict:
    tex = {"embed_textures": bool(args.embed_textures),
           "copy_textures": not args.no_copy_textures}
    return {
        "fbx": dict(tex),
        "glb": {"embed_textures": bool(args.embed_textures) or True,
                "copy_textures": not args.no_copy_textures},
        "gltf": dict(tex),
        "obj": {"copy_textures": not args.no_copy_textures},
    }


def cmd_convert(args) -> int:
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    for f in formats:
        if f not in export.FORMATS:
            print("unknown format: %s (choose from %s)"
                  % (f, ", ".join(export.FORMATS)), file=sys.stderr)
            return 2

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        print("no input files found", file=sys.stderr)
        return 2

    opts = _options_from_args(args)
    exp = _export_options(args)
    failures = 0

    for src in inputs:
        out = args.output or os.path.dirname(os.path.abspath(src))
        if len(inputs) > 1 and os.path.splitext(out)[1]:
            out = os.path.dirname(out) or "."
        try:
            progress = None if args.quiet else _console_progress(os.path.basename(src))
            result = convert(src, out, formats=formats, options=opts,
                             export_options=exp, progress=progress)
        except Exception as exc:
            failures += 1
            print("\n%s: ERROR %s" % (os.path.basename(src), exc), file=sys.stderr)
            continue
        if not args.quiet:
            sys.stdout.write("\r" + " " * 70 + "\r")
            print(result.describe())
        if not result.ok:
            failures += 1
            if args.quiet:
                # Still say why: a silent non-zero exit is useless in a script.
                print("%s: no files written" % os.path.basename(src),
                      file=sys.stderr)
                for w in result.warnings:
                    print("  %s" % w, file=sys.stderr)
    return 1 if failures else 0


def _console_progress(label: str):
    def report(frac: float, msg: str) -> None:
        width = 28
        filled = int(width * max(0.0, min(1.0, frac)))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write("\r%-22s [%s] %3d%% %-22s" %
                         (label[:22], bar, int(frac * 100), msg[:22]))
        sys.stdout.flush()
    return report


def cmd_inspect(args) -> int:
    for path in _expand_inputs(args.inputs):
        info = inspect_file(path)
        if args.json:
            print(json.dumps(info, indent=2))
            continue
        print("%s" % info["name"])
        print("  type        : %s" % info.get("type"))
        print("  size        : %.1f KB" % (info["size"] / 1024.0))
        print("  content root: %s" % (info.get("content_root") or "not detected"))
        print("  nodes       : %s" % info.get("node_count"))
        if info.get("figures"):
            print("  geometry    :")
            for f in info["figures"][:20]:
                print("    - %s (%s)" % (f.get("label"), f.get("type")))
        for e in info.get("errors", []):
            print("  error: %s" % e)
    return 0


def cmd_demo(args) -> int:
    from . import samples
    root = args.output or samples.default_location()
    manifest = samples.generate(root, overwrite=args.force)
    print("Demo library: %s" % manifest["root"])
    print("Scene file  : %s" % manifest["scene"])
    if manifest.get("created"):
        print("  %d vertices, %d polygons, %d bones, 2 morphs, 3 Iray surfaces"
              % (manifest["vertices"], manifest["polygons"], manifest["bones"]))
    else:
        print("  (already present - use --force to regenerate)")
    return 0


def cmd_serve(args) -> int:
    from .web.server import serve
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_scan(args) -> int:
    from .dson import ContentLibrary
    dirs = args.content_dir or config.effective_content_dirs()
    if not dirs:
        print("no content directories configured; pass --content-dir",
              file=sys.stderr)
        return 2
    lib = ContentLibrary(dirs)
    assets = lib.scan(max_files=args.limit)
    print("%d asset file(s) across %d root(s)" % (len(assets), len(lib.roots)))
    for a in assets[:args.limit]:
        print("  %-50s %s" % (a["name"][:50], a["rel"]))
    return 0


def cmd_formats(args) -> int:
    for key, info in export.FORMATS.items():
        print("%-6s %-28s rig=%-5s morphs=%-5s quads=%s"
              % (key, info["label"], info["rig"], info["morphs"], info["quads"]))
    return 0


def _expand_inputs(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in inputs:
        path = os.path.abspath(os.path.expanduser(item))
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if d.lower() != "data"]
                for fn in sorted(filenames):
                    if fn.lower().endswith((".duf", ".dsf")):
                        out.append(os.path.join(dirpath, fn))
        elif os.path.isfile(path):
            out.append(path)
        else:
            import glob
            out.extend(sorted(glob.glob(item)))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dazconv",
        description="%s %s - %s" % (APP_NAME, __version__, APP_TAGLINE))
    p.add_argument("--version", action="version",
                   version="%s %s" % (APP_NAME, __version__))
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("convert", help="convert Daz assets to 3D formats")
    c.add_argument("inputs", nargs="+", help=".duf/.dsf files or folders")
    _add_convert_options(c)
    c.set_defaults(func=cmd_convert)

    i = sub.add_parser("inspect", help="show what is inside a Daz file")
    i.add_argument("inputs", nargs="+")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    d = sub.add_parser("demo", help="generate a sample Daz-format library")
    d.add_argument("-o", "--output", default="")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="run the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    sc = sub.add_parser("scan", help="index .duf assets in the content library")
    sc.add_argument("-c", "--content-dir", action="append", default=[])
    sc.add_argument("--limit", type=int, default=200)
    sc.set_defaults(func=cmd_scan)

    f = sub.add_parser("formats", help="list output formats")
    f.set_defaults(func=cmd_formats)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130


def main_serve(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the desktop launcher: start the app and open a browser."""
    return main(["serve"] + list(argv or []))


if __name__ == "__main__":
    sys.exit(main())
