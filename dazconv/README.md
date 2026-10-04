# dazconv — Daz Studio assets to FBX, in pure Python

A stand-alone converter that reads Daz Studio assets and writes FBX, with
**no Daz Studio, no Blender, no compiled extensions and no third-party Python
packages**. Everything here is standard library — the DSON reader, the FBX
writer and the web UI are all part of this package.

```
.duf / .daz scene ─┐
.dsf figure/prop  ─┼──▶  dazconv  ──▶  ASCII FBX 7.4  (mesh + rig + weights + UVs + materials)
.obj mesh         ─┘
```

---

## Quick start

```bash
# 1. The web app: drag & drop, live 3D preview, every export option
python -m dazconv.server --library-root "/path/to/My DAZ 3D Library"
#  -> open http://localhost:8000/

# 2. The CLI: single files or batches
python -m dazconv.cli character.duf -o out/
python -m dazconv.cli *.dsf -L "/path/to/My DAZ 3D Library" --triangulate

# 3. The library
python -c "
from dazconv import convert_file, ExportOptions
report = convert_file('character.duf', 'character.fbx', ExportOptions(triangulate=True))
print(report.success, report.export)"
```

Requires Python 3.8 or newer. Nothing to install. (Developed and tested on
3.11; the code avoids 3.9+ constructs but has not been exercised on 3.8 or 3.9.)

---

## Why this exists

The rest of this repository is a Daz Studio plugin: it needs the Daz SDK, the
FBX SDK and a running copy of Daz Studio to do its work. `dazconv` covers the
case where you just have files on disk and want an FBX out of them — no Daz
Studio installed, nothing to compile, no GUI toolkit.

Both paths are useful; they are not alternatives to each other. Use the plugin
when you want a live transfer from a running Daz Studio session, and `dazconv`
for headless, scripted or server-side conversion.

---

## What it reads

| Input | Notes |
| --- | --- |
| `.duf`, `.daz` | DSON scene files. Plain text or gzip — sniffed automatically. |
| `.dsf` | Figures, props, geometry, UV sets, morphs. Plain or gzip. |
| `.obj` (+ `.mtl`) | Per-corner UVs preserved. |

DSON quirks handled: gzip and zlib containers, UTF-8 BOM, `//` and `#`
comments, trailing commas, and spec versions 0.5 through 0.6.

Referenced assets (external geometry, UV sets, morphs, textures) are resolved
through a **content library** — the folders Daz itself uses. Lookups are
case-insensitive, which matters because Daz URLs say `Daz%203D` while the folder
on disk is usually `DAZ 3D`.

Binary `.dsf` files (the legacy `DZSF` container) are detected and reported
clearly rather than parsed. Every `.dsf` Daz 4.5+ writes is DSON.

## What it writes

ASCII FBX 7.4 (`FBXVersion: 7400`, `GeometryVersion: 124`), containing:

- **Mesh** — vertices, polygons (or triangles), normals, one UV set.
- **Skeleton** — the full bone hierarchy with local transforms.
- **Skin** — a `Skin` deformer with one `Cluster` per bone, bind matrices and
  weights, capped and renormalised.
- **Materials** — Phong shading, diffuse/specular/roughness mapped from Daz
  channels, with `Texture` + `Video` pairs and optional base64 embedding.
- **Blend shapes** — optional; Daz morphs emitted as editable targets.

---

## Web app

```bash
python -m dazconv.server --port 8000 --library-root "/path/to/content"
```

- **Drag & drop** anywhere on the page, or click to browse.
- **Live preview** — a WebGL viewer (orbit / pan / zoom, wireframe, skeleton
  overlay, turntable). It is written from scratch, so there is no CDN or
  vendored library to load and it works offline.
- **Content library browser** — scan the configured roots and pick a `.duf`
  without leaving the page.
- **Every export option** the CLI has, plus a JSON report, warnings list and
  one-click download.

Options: `--host` (bind `0.0.0.0` to expose it), `--port`,
`-L/--library-root` (repeatable), `--config`, `--triangulate`,
`--max-influences`, `-v`.

The server binds `127.0.0.1` by default. It has no authentication, so only
expose it on an interface you trust.

### HTTP API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/info` | Version, content roots, default options |
| `GET` | `/api/assets?limit=&ext=` | Scan the content library |
| `POST` | `/api/convert` | multipart upload (`file` + `options`) → `{job_id}` |
| `POST` | `/api/convert-path` | `{path, options}` for a library asset → `{job_id}` |
| `GET` | `/api/job/<id>` | Poll stage, progress, and the report when done |
| `GET` | `/api/preview/<id>` | Geometry payload for the WebGL viewer |
| `GET` | `/api/download/<id>` | The FBX file |

`/api/convert-path` refuses any path outside a configured content root.

---

## CLI

```
python -m dazconv.cli <input> [<input> ...] [options]
```

| Group | Options |
| --- | --- |
| Output | `-o/--output` (directory, or a `.fbx` path for a single input), `--json-report` |
| Library | `-L/--library-root` (repeatable), `--config` |
| Content | `--no-skeleton`, `--no-weights`, `--no-materials`, `--no-textures`, `--no-embed-textures` |
| Geometry | `--triangulate`, `--no-flip-uv`, `--flat-normals`, `--max-influences`, `--min-weight` |
| Morphs | `-m/--morph NAME=0.8`, `--morph-file`, `--morph-dir`, `--export-blendshapes` |
| Transform | `--up-axis y\|z`, `--unit-scale`, `--scale` |
| Behaviour | `-q/--quiet`, `--fail-on-warning` |

Exit codes: `0` success, `1` a conversion failed, `2` bad usage or invalid
options.

```bash
# Bake a morph and keep the rest editable
python -m dazconv.cli g3f.dsf --morph-file FBMPearFigure.dsf -m FBMPearFigure=80% --export-blendshapes

# Batch a whole folder, keep a machine-readable log
python -m dazconv.cli assets/*.duf -L "$DAZ_CONTENT" -o out/ --json-report out/report.json
```

---

## Units and axes

Daz works in centimetres, Y-up, right-handed — which is also the FBX default,
so the default export is a straight pass-through (`UnitScaleFactor: 1`,
`UpAxis: 1`). With `--up-axis z` a −90° rotation about X is premultiplied onto
every top-level model, so bind matrices stay consistent with the geometry.

Daz UVs have their origin at the top-left, so V is flipped (`1 − v`) by default.
Use `--no-flip-uv` to keep Daz's convention.

---

## Library layout

```
dazconv/
├── dson.py        container sniffing (plain/gzip/zlib/DZSF) + tolerant parser
├── geometry.py    polylist decoding, normals, triangulation, material split
├── uvset.py       per-corner UV resolution incl. polygon_vertex_indices
├── rig.py         node_library → bone tree, local and world matrices
├── skinning.py    weighted_joint → per-vertex influences (capped, renormalised)
├── morphs.py      sparse morph deltas, blend-shape targets
├── materials.py   DSON material channels → FBX shading parameters
├── library.py     content roots + case-insensitive asset resolution
├── obj_import.py  OBJ/MTL with per-corner UVs
├── scene.py       assembles a SceneModel from any supported input
├── options.py     ExportOptions (shared by CLI, server and API)
├── convert.py     the convert_file / convert_files entry points
├── preview.py     compact geometry payload for the browser viewer
├── cli.py         command-line front end
├── server.py      stdlib web app (own multipart parser; cgi is gone in 3.13)
├── webui/         index.html, style.css, app.js (WebGL viewer)
└── fbx/
    ├── writer.py    ASCII FBX serialiser and syntax rules
    └── exporter.py  the FBX document itself
```

---

## Tests

```bash
python tools/fetch_samples.py                     # ~5 MB of public sample assets
DAZCONV_SAMPLES=$PWD/../daz-samples python -m unittest discover -s tests -t .
```

Daz content is licensed, so it is **not** committed here; `fetch_samples.py`
pulls four files from the public
[`DRincs-Productions/My-DAZ-3D-Library`](https://github.com/DRincs-Productions/My-DAZ-3D-Library)
repository (a Genesis 8.1 figure, its UV set, a Genesis 3 figure and one morph).
Tests skip cleanly when the samples or `ufbx` are missing.

The suite checks real data, not mock data:

- vertex positions match the source `.dsf` to 1e-5,
- every skin weight sums to exactly 1.0 across all 16,556 vertices,
- UVs resolve to the right corner (Genesis 8.1's first face starts at vertex
  **1594**, not 0, so this is easy to get wrong),
- baked morphs equal `base + delta` to 7e-8 cm,
- and every output is re-parsed by **[ufbx](https://github.com/ufbx/ufbx)**, a
  completely independent FBX implementation.

`tools/validate_fbx.py` is the same check you can run yourself:

```bash
python tools/validate_fbx.py out/character.fbx --pretty
```

It runs in a subprocess from the tests on purpose: ufbx's Python bindings
segfault if a scene is garbage-collected while a mesh is still in use, and a
native crash should fail one test rather than take the whole suite down.

---

## Known limits

- **Subdivision surfaces are not evaluated.** `subdivision_surface` geometry is
  exported at its base level; Daz's `current_subdivision_level` is reported but
  not applied.
- **Binary `.dsf` (`DZSF`) is not decoded** — detected and reported instead.
- **Posed figures**: `apply_pose` exists as an option but bind pose is what gets
  written today.
- **One UV set per mesh** (Daz's default set), and textures are looked up by
  path — missing files produce warnings, not errors.

## License

Same as the rest of this repository: Apache License 2.0.
