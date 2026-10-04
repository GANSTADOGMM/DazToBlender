# DazConv

**Turn Daz Studio assets into FBX, glTF/GLB and OBJ — without Daz Studio.**

DazConv reads Daz's native DSON files (`.duf` / `.dsf`) straight off your disk and
writes finished, rigged, textured 3D files. It is a standalone application: no
Daz Studio install, no bridge plug-in, no account, no cloud, no cost.

```
./DazConv.sh          # macOS / Linux   (DazConv.bat on Windows)
```

Your browser opens on a local page. Drag a `.duf` onto it, press **Convert**,
and get an `.fbx` you can drop into Blender, Unreal, Unity, Houdini, Godot or
Substance Painter.

---

## Why this exists

The official Daz bridges are plug-ins: they run *inside* Daz Studio, which means
Windows or macOS only, a Daz Studio install, and a manual export for every
figure. DazConv does the conversion itself. It parses DSON, resolves the content
library, evaluates the figure, and writes the file. That makes it scriptable,
batchable, headless, and usable on Linux or a build server.

## What it does

| | |
|---|---|
| **Geometry** | Vertices, polygons (quads preserved where the format allows), polygon groups, surface assignment |
| **UVs** | Per-face-vertex UVs with seams intact, plus alternate UV sets referenced by the figure |
| **Materials** | Iray Uber and 3Delight surfaces mapped to PBR: base colour, metallic, roughness (converted from glossiness where needed), normal/bump, emission, cutout opacity, index of refraction |
| **Textures** | Located in the content library, copied next to the output or embedded |
| **Skeleton** | Full joint hierarchy with Daz's `center_point`/`end_point` orientation and rotation order |
| **Skin weights** | General and TriAx weight maps, collapsed, normalised and pruned to a bone-influence budget |
| **Morphs** | Shape keys / blend shapes with their names and default values; HD morphs filtered out by default |
| **Conforming items** | Clothing and hair that conform to a figure inherit that figure's rig instead of getting a duplicate one |
| **Batch** | Whole folders at once, through the queue in the UI or the CLI |

### Output formats

| Format | Rig | Morphs | Quads | Notes |
|---|:--:|:--:|:--:|---|
| **FBX** (binary 7.4) | yes | yes | yes | Written by hand, no FBX SDK. Centimetres, Y-up, `UpAxis = 1` — what Blender/Unreal/Unity expect |
| **GLB / glTF 2.0** | yes | yes | no | Metres, triangulated, textures embedded (GLB) or beside the file |
| **OBJ + MTL** | no | no | yes | Static mesh with materials and textures |

## Requirements

Python 3.8 or newer. That is the entire list.

DazConv has **zero runtime dependencies**. The DSON reader, the 3D maths, the
geometry/skinning/morph pipeline, the FBX, glTF and OBJ writers, the PNG encoder
and the web server are all written against the standard library; three.js (for
the in-browser preview) is vendored into the repository, so the app never talks
to a CDN — or to anything else. Nothing you convert leaves your machine.

## Install

Clone or download, then run the launcher. Optionally install it properly:

```bash
pip install .            # gives you the `dazconv` command
```

## The web app

```bash
dazconv serve                       # or: python3 -m dazconv serve
dazconv serve --port 9000 --no-browser
```

* **Drag and drop** `.duf`/`.dsf` files, or browse your Daz library from the
  page (it auto-detects the usual install locations).
* **Pick formats** — convert to FBX, GLB and OBJ in a single pass.
* **3D preview** of the converted result in the page, with the morph sliders and
  the skeleton overlay, so you can confirm the figure before you download it.
* **Queue** — add a folder, watch each job's progress, download any output
  individually or everything as a zip.
* **Dependency check** — if a scene references content you do not have
  installed, DazConv tells you exactly which files it could not find and where
  it looked, instead of silently producing an empty mesh.

## The command line

```bash
# One figure to FBX
dazconv convert "~/Daz 3D/My Library/People/Genesis 9/Characters/Aiko.duf"

# A whole folder to FBX + GLB, into ./out
dazconv convert ~/Library/People -f fbx,glb -o ./out

# Point at a content root explicitly (repeatable)
dazconv convert scene.duf -c "/Volumes/Daz/My Library" -c ~/DazContent

# Trim the output
dazconv convert scene.duf --no-morphs --influences 4 --scale 0.01
dazconv convert scene.duf --morph "FBM*" --embed-textures

# What's actually in this file?
dazconv inspect scene.duf
dazconv inspect scene.duf --json

# Other bits
dazconv demo -o ~/DemoLibrary     # write a small synthetic Daz library
dazconv scan                      # index the .duf assets it can see
dazconv formats
```

`convert` writes next to the input unless you pass `-o`. Give `-o` a filename to
control the name, or a directory to name outputs after the input file.

## As a library

```python
from dazconv import convert, inspect_file, ConvertOptions

info = inspect_file("scene.duf")
print(info["figures"], info["materials"], info["content_root"])

result = convert(
    "scene.duf", "out/",
    formats=["fbx", "glb"],
    options=ConvertOptions(content_dirs=["~/Daz 3D/My Library"],
                           max_influences=4, include_morphs=True),
    progress=lambda frac, msg: print(f"{frac:5.0%} {msg}"),
)
print(result.describe())
for fig in result.scene.figures:
    print(fig.name, len(fig.mesh.vertices), len(fig.skeleton.bones))
```

## How it is put together

```
dazconv/
  dson/        DSON layer: URI decoding, gzip-aware reader, content library
               resolution (case-insensitive, multi-root), channel helpers
  convert/     DSON -> intermediate representation: geometry, skinning,
               morphs, materials, and the scene builder that drives them
  ir.py        The format-neutral scene: Scene / Figure / Mesh / Skeleton /
               Material / MorphTarget / SkinWeights
  export/      IR -> files.  fbx/binary.py is the FBX container (reader and
               writer); fbx/writer.py, gltf.py and obj.py build the documents
  math3d.py    Matrices, Euler/quaternion conversion, decomposition
  web/         Job queue and the stdlib HTTP server behind the UI
  cli.py       Command line
```

Reading DSON is the easy half. The hard half, and most of what is in
`convert/`, is the stuff Daz does implicitly: a node's transform is
`T(center + translate) · Rorder · R · Rorder⁻¹ · S · T(−center)`, weights arrive
as sparse per-bone lists that may be TriAx triplets, morph deltas are sparse and
indexed against the pre-subdivision mesh, surfaces are matched by name against
`polygon_material_groups`, and almost every URI is percent-encoded and
case-insensitively resolved against several content roots.

## Testing

```bash
cd DazConverter
python3 -m unittest discover -s tests          # 136 tests, no dependencies
```

Those tests cover the DSON layer, the conversion maths, and every exporter —
including parsing our own FBX back out of the file and checking bone matrices,
bind poses and morph deltas.

Self-consistency is not proof, though, so there is a second, optional suite that
feeds the output to **Blender itself** and asks what it sees:

```bash
python3 -m venv /tmp/bpyenv && /tmp/bpyenv/bin/pip install bpy
DAZCONV_BPY_PYTHON=/tmp/bpyenv/bin/python python3 -m unittest discover -s tests
```

It imports the exported FBX with *both* of Blender's importers, plus the GLB and
the OBJ, and asserts the vertex and polygon counts, the bone hierarchy, every
bone's world position in metres, that skin weights sum to 1, that the shape keys
actually displace vertices, that materials resolve to real image files, and that
conforming clothing shares one armature with the body. It skips silently when
`bpy` is not installed.

## Limits

* Geometry is exported at base resolution. HD morphs (`*_HD_div*`) and
  subdivision are not evaluated; turn on `--hd-morphs` only if you know the
  asset stores usable base-res deltas.
* Animation is not exported — figures come out in the rest or scene pose.
* dForce simulation, geografts/geoshells, LIE layered textures, and Daz's
  procedural shaders are not evaluated.
* Converting an asset does not change its licence. Check the EULA of anything
  you convert before you redistribute it.

## Licence

MIT. See `LICENSE`.

DazConv is not affiliated with or endorsed by Daz 3D. "Daz", "Daz Studio" and
"Genesis" are trademarks of Daz Productions, Inc.
