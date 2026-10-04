"""Generates a small but *real* Daz-format content library.

Having a genuine DSON library on hand matters for three reasons:

* the app is usable (and testable) on a machine with no Daz install;
* the test-suite exercises the real code paths -- URI resolution, gzip'd
  ``.dsf`` files, Iray material channels, skin binding, morph deltas -- rather
  than hand-made stubs;
* "Load demo asset" in the web UI gives people something to convert in one
  click before they point it at their own library.

The generated figure is a rigged low-poly humanoid built from tube meshes
along a 21-bone skeleton, with UVs, three Iray surfaces, procedurally painted
PNG textures and two morphs.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

LIBRARY_NAME = "DazConv Demo Library"
FIGURE_ID = "DemoFigure"
GEOM_ID = "DemoFigureGeom"
DATA_DIR = "data/DazConv/DemoFigure"
TEX_DIR = "Runtime/Textures/DazConv"

# name, head (x,y,z), tail, parent, radius_head, radius_tail, surface
SKELETON: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float],
                     Optional[str], float, float, str]] = [
    ("hip",       (0.0, 98.0, 0.0),   (0.0, 110.0, 0.0),  None,      11.0, 10.0, "Torso"),
    ("abdomen",   (0.0, 110.0, 0.0),  (0.0, 121.0, 0.0),  "hip",     10.0, 9.5,  "Torso"),
    ("chest",     (0.0, 121.0, 0.0),  (0.0, 139.0, 0.0),  "abdomen", 9.5, 11.0,  "Torso"),
    ("neck",      (0.0, 139.0, 0.0),  (0.0, 148.0, 0.0),  "chest",   4.2, 3.6,   "Head"),
    ("head",      (0.0, 148.0, 0.0),  (0.0, 170.0, 0.0),  "neck",    7.5, 2.5,   "Head"),

    ("lCollar",   (1.5, 136.0, 0.0),  (14.0, 138.0, 0.0), "chest",   5.0, 4.4,   "Torso"),
    ("lShldr",    (14.0, 138.0, 0.0), (15.0, 117.0, 0.0), "lCollar", 4.4, 3.6,   "Limbs"),
    ("lForearm",  (15.0, 117.0, 0.0), (16.0, 98.0, 0.0),  "lShldr",  3.6, 2.8,   "Limbs"),
    ("lHand",     (16.0, 98.0, 0.0),  (16.5, 89.0, 0.0),  "lForearm", 2.8, 1.6,  "Limbs"),

    ("rCollar",   (-1.5, 136.0, 0.0), (-14.0, 138.0, 0.0), "chest",  5.0, 4.4,   "Torso"),
    ("rShldr",    (-14.0, 138.0, 0.0), (-15.0, 117.0, 0.0), "rCollar", 4.4, 3.6, "Limbs"),
    ("rForearm",  (-15.0, 117.0, 0.0), (-16.0, 98.0, 0.0), "rShldr",  3.6, 2.8,  "Limbs"),
    ("rHand",     (-16.0, 98.0, 0.0), (-16.5, 89.0, 0.0),  "rForearm", 2.8, 1.6, "Limbs"),

    ("lThigh",    (7.0, 97.0, 0.0),   (8.0, 55.0, 0.0),   "hip",     7.5, 5.2,   "Limbs"),
    ("lShin",     (8.0, 55.0, 0.0),   (8.5, 12.0, 0.0),   "lThigh",  5.2, 3.4,   "Limbs"),
    ("lFoot",     (8.5, 12.0, 0.0),   (9.0, 2.0, 10.0),   "lShin",   3.4, 2.6,   "Limbs"),

    ("rThigh",    (-7.0, 97.0, 0.0),  (-8.0, 55.0, 0.0),  "hip",     7.5, 5.2,   "Limbs"),
    ("rShin",     (-8.0, 55.0, 0.0),  (-8.5, 12.0, 0.0),  "rThigh",  5.2, 3.4,   "Limbs"),
    ("rFoot",     (-8.5, 12.0, 0.0),  (-9.0, 2.0, 10.0),  "rShin",   3.4, 2.6,   "Limbs"),
]

CHAINS = [
    ["hip", "abdomen", "chest", "neck", "head"],
    ["lCollar", "lShldr", "lForearm", "lHand"],
    ["rCollar", "rShldr", "rForearm", "rHand"],
    ["lThigh", "lShin", "lFoot"],
    ["rThigh", "rShin", "rFoot"],
]

SURFACES = ["Torso", "Head", "Limbs"]
RADIAL = 12
RINGS_PER_BONE = 3


# --------------------------------------------------------------------- mesh
def _bone_table():
    return {b[0]: b for b in SKELETON}


def _normalize(v):
    ln = math.sqrt(sum(c * c for c in v))
    return tuple(c / ln for c in v) if ln > 1e-9 else (0.0, 1.0, 0.0)


def _basis(direction):
    d = _normalize(direction)
    up = (0.0, 0.0, 1.0) if abs(d[1]) > 0.9 else (0.0, 1.0, 0.0)
    side = _normalize((d[1] * up[2] - d[2] * up[1],
                       d[2] * up[0] - d[0] * up[2],
                       d[0] * up[1] - d[1] * up[0]))
    up2 = (d[1] * side[2] - d[2] * side[1],
           d[2] * side[0] - d[0] * side[2],
           d[0] * side[1] - d[1] * side[0])
    return side, up2


def build_mesh() -> Dict[str, Any]:
    """Generate vertices, quads, UVs, surface assignment and skin weights."""
    bones = _bone_table()
    vertices: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    polys: List[Tuple[int, int, int, int]] = []
    poly_surface: List[int] = []
    poly_group: List[int] = []
    groups: List[str] = []

    for ci, chain in enumerate(CHAINS):
        group_name = "chain_%d" % ci
        groups.append(group_name)
        rings: List[List[int]] = []
        total_rings = 0
        for bi, bname in enumerate(chain):
            _n, head, tail, _p, r0, r1, surf = bones[bname]
            d = (tail[0] - head[0], tail[1] - head[1], tail[2] - head[2])
            side, up = _basis(d)
            steps = RINGS_PER_BONE if bi < len(chain) - 1 else RINGS_PER_BONE + 1
            for s in range(steps):
                t = s / float(RINGS_PER_BONE)
                if bi > 0 and s == 0:
                    continue  # share the ring with the previous bone
                cx = head[0] + d[0] * t
                cy = head[1] + d[1] * t
                cz = head[2] + d[2] * t
                radius = r0 + (r1 - r0) * t
                ring = []
                v_coord = (total_rings % 8) / 8.0
                for a in range(RADIAL):
                    ang = 2.0 * math.pi * a / RADIAL
                    ca, sa = math.cos(ang), math.sin(ang)
                    px = cx + (side[0] * ca + up[0] * sa) * radius
                    py = cy + (side[1] * ca + up[1] * sa) * radius
                    pz = cz + (side[2] * ca + up[2] * sa) * radius
                    ring.append(len(vertices))
                    vertices.append((px, py, pz))
                    uvs.append((a / float(RADIAL), v_coord))
                rings.append(ring)
                total_rings += 1

        # stitch consecutive rings into quads
        ring_bone: List[str] = []
        idx = 0
        for bi, bname in enumerate(chain):
            steps = RINGS_PER_BONE if bi < len(chain) - 1 else RINGS_PER_BONE + 1
            for s in range(steps):
                if bi > 0 and s == 0:
                    continue
                ring_bone.append(bname)
                idx += 1
        for ri in range(len(rings) - 1):
            a_ring, b_ring = rings[ri], rings[ri + 1]
            surf = bones[ring_bone[ri]][6]
            si = SURFACES.index(surf)
            for a in range(RADIAL):
                a2 = (a + 1) % RADIAL
                polys.append((a_ring[a], a_ring[a2], b_ring[a2], b_ring[a]))
                poly_surface.append(si)
                poly_group.append(ci)

        # cap the ends
        for ring, flip in ((rings[0], True), (rings[-1], False)):
            cx = sum(vertices[i][0] for i in ring) / len(ring)
            cy = sum(vertices[i][1] for i in ring) / len(ring)
            cz = sum(vertices[i][2] for i in ring) / len(ring)
            center = len(vertices)
            vertices.append((cx, cy, cz))
            uvs.append((0.5, 0.5))
            surf = bones[ring_bone[0 if flip else -1]][6]
            si = SURFACES.index(surf)
            for a in range(RADIAL):
                a2 = (a + 1) % RADIAL
                if flip:
                    polys.append((ring[a2], ring[a], center, center))
                else:
                    polys.append((ring[a], ring[a2], center, center))
                poly_surface.append(si)
                poly_group.append(ci)

    weights = _compute_weights(vertices)
    return {
        "vertices": vertices,
        "uvs": uvs,
        "polys": polys,
        "poly_surface": poly_surface,
        "poly_group": poly_group,
        "groups": groups,
        "weights": weights,
    }


def _point_segment_distance(p, a, b) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    denom = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / denom))
    cx, cy, cz = a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t
    return math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)


def _compute_weights(vertices, max_influences: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    """Inverse-distance weights against every bone segment."""
    out: Dict[str, List[Tuple[int, float]]] = {b[0]: [] for b in SKELETON}
    for vi, p in enumerate(vertices):
        scored = []
        for name, head, tail, _parent, _r0, _r1, _s in SKELETON:
            d = _point_segment_distance(p, head, tail)
            scored.append((1.0 / ((d + 0.8) ** 4), name))
        scored.sort(reverse=True)
        top = scored[:max_influences]
        total = sum(w for w, _ in top)
        for w, name in top:
            nw = w / total
            if nw > 0.001:
                out[name].append((vi, round(nw, 6)))
    return out


def build_morphs(vertices) -> Dict[str, List[Tuple[int, Tuple[float, float, float]]]]:
    """Two hand-made morphs: a bulk-up and a head scale."""
    muscular = []
    big_head = []
    for vi, p in enumerate(vertices):
        # radial push outwards from the body's vertical axis
        r = math.sqrt(p[0] ** 2 + p[2] ** 2)
        if r > 0.5 and 10.0 < p[1] < 150.0:
            k = 0.22 * min(1.0, r / 12.0)
            muscular.append((vi, (p[0] / r * r * k * 0.25, 0.0, p[2] / r * r * k * 0.25)))
        if p[1] > 146.0:
            t = min(1.0, (p[1] - 146.0) / 24.0)
            big_head.append((vi, (p[0] * 0.35 * t, (p[1] - 148.0) * 0.30 * t,
                                  p[2] * 0.35 * t)))
    return {"Muscular": muscular, "HeadScale": big_head}


# ----------------------------------------------------------------- textures
def _write_png(path: str, width: int, height: int, pixel_fn) -> None:
    """Minimal RGB8 PNG writer (stdlib only)."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            raw.extend((r & 255, g & 255, b & 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
           chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(png)


def _skin_pixel(x, y):
    import math as _m
    checker = ((x // 16) + (y // 16)) % 2
    base = (222, 176, 150) if checker else (208, 160, 136)
    wave = int(12 * _m.sin(x * 0.12) * _m.cos(y * 0.09))
    return (max(0, min(255, base[0] + wave)),
            max(0, min(255, base[1] + wave)),
            max(0, min(255, base[2] + wave)))


def _suit_pixel(x, y):
    stripe = (x // 8) % 2
    base = (40, 62, 110) if stripe else (28, 44, 84)
    return base


def _normal_pixel(x, y):
    import math as _m
    nx = _m.sin(x * 0.25) * 0.25
    ny = _m.cos(y * 0.25) * 0.25
    nz = _m.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
    return (int((nx * 0.5 + 0.5) * 255), int((ny * 0.5 + 0.5) * 255),
            int((nz * 0.5 + 0.5) * 255))


def _rough_pixel(x, y):
    v = 120 + int(60 * (((x // 12) + (y // 12)) % 2))
    return (v, v, v)


# -------------------------------------------------------------- DSON output
def _channel(cid: str, value: float, ctype: str = "float") -> Dict[str, Any]:
    return {"id": cid, "type": ctype, "name": cid.upper(), "value": value,
            "current_value": value}


def _xyz(values: Sequence[float]) -> List[Dict[str, Any]]:
    return [_channel(axis, float(v)) for axis, v in zip("xyz", values)]


def _asset_info(aid: str, atype: str) -> Dict[str, Any]:
    return {
        "id": aid,
        "type": atype,
        "contributor": {"author": "DazConv", "email": "", "website": ""},
        "revision": "1.0",
        "modified": "2024-01-01T00:00:00Z",
    }


def _node_entry(name, head, tail, parent, ntype="bone") -> Dict[str, Any]:
    direction = (tail[0] - head[0], tail[1] - head[1], tail[2] - head[2])
    # Orientation: point the bone's Y axis down its length (Daz convention
    # varies per rig; this is enough to give importers sane bone axes).
    d = _normalize(direction)
    yaw = math.degrees(math.atan2(d[0], d[2] if abs(d[2]) > 1e-6 else 1e-6))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
    entry: Dict[str, Any] = {
        "id": name,
        "name": name,
        "type": ntype,
        "label": name,
        "rotation_order": "YZX",
        "inherits_scale": True,
        "center_point": _xyz(head),
        "end_point": _xyz(tail),
        "orientation": _xyz((0.0, round(yaw, 4), round(pitch, 4))),
        "rotation": _xyz((0.0, 0.0, 0.0)),
        "translation": _xyz((0.0, 0.0, 0.0)),
        "scale": _xyz((1.0, 1.0, 1.0)),
        "general_scale": _channel("general_scale", 1.0),
    }
    if parent:
        entry["parent"] = "#%s" % parent
    return entry


def _iray_material(name: str, diffuse_map: str, normal_map: str,
                   rough_map: str, color, roughness, metallic) -> Dict[str, Any]:
    return {
        "id": name,
        "name": name,
        "type": "studio_material_channels",
        "url": "#%s" % name,
        "diffuse": {"channel": {"id": "diffuse", "type": "float_color",
                                "name": "Diffuse Color", "value": list(color),
                                "current_value": list(color),
                                "image_file": diffuse_map}},
        "extra": [
            {"type": "studio/material/uber_iray", "version": "1.1.0.0"},
            {"type": "studio_material_channels", "channels": [
                {"channel": {"id": "Base Color", "type": "float_color",
                             "current_value": list(color),
                             "image_file": diffuse_map}},
                {"channel": {"id": "Metallic Weight", "type": "float",
                             "current_value": metallic}},
                {"channel": {"id": "Glossy Roughness", "type": "float",
                             "current_value": roughness,
                             "image_file": rough_map}},
                {"channel": {"id": "Normal Map", "type": "float",
                             "current_value": 1.0,
                             "image_file": normal_map}},
                {"channel": {"id": "Cutout Opacity", "type": "float",
                             "current_value": 1.0}},
                {"channel": {"id": "Horizontal Tiles", "type": "float",
                             "current_value": 1.0}},
                {"channel": {"id": "Vertical Tiles", "type": "float",
                             "current_value": 1.0}},
            ]},
        ],
    }


def _dump(path: str, data: Dict[str, Any], compress: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(data, indent=1)
    if compress:
        with gzip.open(path, "wb") as fh:
            fh.write(text.encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)


def generate(root: str, overwrite: bool = False) -> Dict[str, Any]:
    """Write the demo content library under `root`.  Returns a manifest."""
    root = os.path.abspath(os.path.expanduser(root))
    scene_path = os.path.join(root, "People", "DemoFigure.duf")
    if os.path.isfile(scene_path) and not overwrite:
        return {"root": root, "scene": scene_path, "created": False}

    mesh = build_mesh()
    morphs = build_morphs(mesh["vertices"])

    # ---- textures -----------------------------------------------------
    tex_root = os.path.join(root, TEX_DIR)
    _write_png(os.path.join(tex_root, "demo_skin_d.png"), 256, 256, _skin_pixel)
    _write_png(os.path.join(tex_root, "demo_suit_d.png"), 256, 256, _suit_pixel)
    _write_png(os.path.join(tex_root, "demo_normal.png"), 256, 256, _normal_pixel)
    _write_png(os.path.join(tex_root, "demo_rough.png"), 256, 256, _rough_pixel)

    # ---- figure .dsf (geometry + nodes + skin + uvs + materials) -------
    verts = mesh["vertices"]
    polylist = []
    for pi, poly in enumerate(mesh["polys"]):
        polylist.append([mesh["poly_group"][pi], mesh["poly_surface"][pi]] + list(poly))

    geometry = {
        "id": GEOM_ID,
        "name": FIGURE_ID,
        "label": "Demo Figure Mesh",
        "type": "polygon_mesh",
        "edge_interpolation_mode": "edges_and_corners",
        "vertices": {"count": len(verts),
                     "values": [[round(c, 5) for c in v] for v in verts]},
        "polygon_groups": {"count": len(mesh["groups"]), "values": mesh["groups"]},
        "polygon_material_groups": {"count": len(SURFACES), "values": SURFACES},
        "polylist": {"count": len(polylist), "values": polylist},
        "default_uv_set": "/%s/DemoFigure.dsf#default" % DATA_DIR,
        "extra": [{"type": "studio_geometry_channels", "version": 1}],
    }

    node_lib = [{
        "id": FIGURE_ID,
        "name": FIGURE_ID,
        "type": "figure",
        "label": "Demo Figure",
        "rotation_order": "YXZ",
        "center_point": _xyz((0, 0, 0)),
        "end_point": _xyz((0, 100, 0)),
        "orientation": _xyz((0, 0, 0)),
        "rotation": _xyz((0, 0, 0)),
        "translation": _xyz((0, 0, 0)),
        "scale": _xyz((1, 1, 1)),
        "general_scale": _channel("general_scale", 1.0),
        "geometries": [{"id": GEOM_ID, "url": "#%s" % GEOM_ID,
                        "name": FIGURE_ID, "type": "polygon_mesh"}],
    }]
    for name, head, tail, parent, _r0, _r1, _s in SKELETON:
        node_lib.append(_node_entry(name, head, tail, parent or FIGURE_ID))

    joints = []
    for name, _h, _t, _p, _r0, _r1, _s in SKELETON:
        pairs = mesh["weights"].get(name) or []
        if not pairs:
            continue
        joints.append({
            "id": name,
            "node": "#%s" % name,
            "node_weights": {"count": len(pairs),
                             "values": [[vi, w] for vi, w in pairs]},
        })

    skin_modifier = {
        "id": "SkinBinding",
        "name": "SkinBinding",
        "label": "Skin Binding",
        "parent": "#%s" % GEOM_ID,
        "skin": {
            "node": "#%s" % FIGURE_ID,
            "geometry": "#%s" % GEOM_ID,
            "vertex_count": len(verts),
            "joints": joints,
        },
    }

    uv_set = {
        "id": "default",
        "name": "default",
        "label": "Default UVs",
        "vertex_count": len(verts),
        "uvs": {"count": len(mesh["uvs"]),
                "values": [[round(u, 6), round(v, 6)] for u, v in mesh["uvs"]]},
    }

    tex = lambda n: "/%s/%s" % (TEX_DIR, n)
    material_lib = [
        _iray_material("Torso", tex("demo_suit_d.png"), tex("demo_normal.png"),
                       tex("demo_rough.png"), [0.35, 0.42, 0.68], 0.45, 0.05),
        _iray_material("Head", tex("demo_skin_d.png"), tex("demo_normal.png"),
                       tex("demo_rough.png"), [0.86, 0.70, 0.60], 0.55, 0.0),
        _iray_material("Limbs", tex("demo_skin_d.png"), tex("demo_normal.png"),
                       tex("demo_rough.png"), [0.84, 0.68, 0.58], 0.52, 0.0),
    ]

    figure_doc = {
        "file_version": "0.6.0.0",
        "asset_info": _asset_info("/%s/DemoFigure.dsf" % DATA_DIR, "figure"),
        "geometry_library": [geometry],
        "node_library": node_lib,
        "uv_set_library": [uv_set],
        "modifier_library": [skin_modifier],
        "material_library": material_lib,
    }
    figure_path = os.path.join(root, DATA_DIR, "DemoFigure.dsf")
    # real Daz library files are gzip'd -- keep it realistic
    _dump(figure_path, figure_doc, compress=True)

    # ---- morph .dsf files ----------------------------------------------
    morph_paths = {}
    for mname, deltas in morphs.items():
        doc = {
            "file_version": "0.6.0.0",
            "asset_info": _asset_info("/%s/Morphs/%s.dsf" % (DATA_DIR, mname),
                                      "modifier"),
            "modifier_library": [{
                "id": mname,
                "name": mname,
                "label": mname,
                "parent": "/%s/DemoFigure.dsf#%s" % (DATA_DIR, GEOM_ID),
                "presentation": {"type": "Modifier/Shape", "label": mname},
                "channel": {"id": "value", "type": "float", "name": "Value",
                            "label": mname, "value": 0, "min": 0, "max": 1,
                            "current_value": 0},
                "region": "Body",
                "group": "/Demo",
                "morph": {
                    "vertex_count": len(verts),
                    "deltas": {"count": len(deltas),
                               "values": [[vi, round(d[0], 5), round(d[1], 5),
                                           round(d[2], 5)] for vi, d in deltas]},
                },
            }],
        }
        p = os.path.join(root, DATA_DIR, "Morphs", "%s.dsf" % mname)
        _dump(p, doc, compress=True)
        morph_paths[mname] = p

    # ---- scene .duf ------------------------------------------------------
    figure_url = "/%s/DemoFigure.dsf" % DATA_DIR
    nodes = [{
        "id": FIGURE_ID,
        "url": "%s#%s" % (figure_url, FIGURE_ID),
        "name": FIGURE_ID,
        "label": "Demo Figure",
        "geometries": [{"id": GEOM_ID, "url": "%s#%s" % (figure_url, GEOM_ID),
                        "name": FIGURE_ID, "label": "shape",
                        "type": "polygon_mesh"}],
        "preview": {"type": "figure", "center_point": [0, 0, 0],
                    "end_point": [0, 100, 0], "rotation_order": "YXZ"},
    }]
    for name, head, tail, parent, _r0, _r1, _s in SKELETON:
        nodes.append({
            "id": name,
            "url": "%s#%s" % (figure_url, name),
            "name": name,
            "label": name,
            "parent": "#%s" % (parent or FIGURE_ID),
            "preview": {"type": "bone", "center_point": list(head),
                        "end_point": list(tail), "rotation_order": "YZX"},
        })

    scene_materials = []
    for surf in SURFACES:
        scene_materials.append({
            "id": "%s-1" % surf,
            "url": "%s#%s" % (figure_url, surf),
            "geometry": "#%s" % GEOM_ID,
            "groups": [surf],
            "uv_set": "%s#default" % figure_url,
        })

    scene_modifiers = []
    for mname in morphs:
        scene_modifiers.append({
            "id": mname,
            "url": "/%s/Morphs/%s.dsf#%s" % (DATA_DIR, mname, mname),
            "parent": "#%s" % GEOM_ID,
            "channel": {"id": "value", "type": "float", "name": "Value",
                        "value": 0, "current_value": 0},
        })

    scene_doc = {
        "file_version": "0.6.0.0",
        "asset_info": _asset_info("/People/DemoFigure.duf", "scene"),
        "scene": {
            "nodes": nodes,
            "materials": scene_materials,
            "modifiers": scene_modifiers,
            "extra": [{"type": "studio_scene_data", "version": 1}],
        },
    }
    _dump(scene_path, scene_doc, compress=False)

    return {
        "root": root,
        "scene": scene_path,
        "figure": figure_path,
        "morphs": morph_paths,
        "vertices": len(verts),
        "polygons": len(mesh["polys"]),
        "bones": len(SKELETON),
        "created": True,
    }


def default_location() -> str:
    from .config import data_dir
    return os.path.join(data_dir(), "DemoLibrary")


def ensure(overwrite: bool = False) -> Dict[str, Any]:
    """Create (once) and return the demo library manifest."""
    return generate(default_location(), overwrite=overwrite)
