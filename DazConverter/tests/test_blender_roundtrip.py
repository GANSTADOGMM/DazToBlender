"""Round-trip the exported files through a real Blender importer.

These tests are *optional*: they only run when a Python interpreter that can
``import bpy`` is available. Point ``DAZCONV_BPY_PYTHON`` at it, e.g.::

    python3 -m venv /tmp/bpyenv && /tmp/bpyenv/bin/pip install bpy
    DAZCONV_BPY_PYTHON=/tmp/bpyenv/bin/python python3 -m unittest discover -s tests

Everything else in the suite validates our own files with our own parser, which
can only prove self-consistency. This module proves that the industry's most
widely used FBX/glTF/OBJ importer agrees with us.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from dazconv import samples
from dazconv.api import convert
from dazconv.convert import ConvertOptions

BPY_PYTHON = os.environ.get("DAZCONV_BPY_PYTHON", "")


def _bpy_available():
    if not BPY_PYTHON or not os.path.isfile(BPY_PYTHON):
        return False
    try:
        out = subprocess.run([BPY_PYTHON, "-c", "import bpy"],
                             capture_output=True, timeout=180)
        return out.returncode == 0
    except Exception:
        return False


HAVE_BPY = _bpy_available()

# Script executed inside Blender. Prints one JSON blob prefixed with a marker
# so Blender's own chatter on stdout can be filtered out.
PROBE = textwrap.dedent('''
    import json, sys, os
    import bpy

    path, fmt, importer = sys.argv[-3], sys.argv[-2], sys.argv[-1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if fmt == "fbx":
        if importer == "cpp" and hasattr(bpy.ops.wm, "fbx_import"):
            bpy.ops.wm.fbx_import(filepath=path)
        else:
            bpy.ops.import_scene.fbx(filepath=path)
    elif fmt in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif fmt == "obj":
        bpy.ops.wm.obj_import(filepath=path)

    # Blender's glTF importer spawns a hidden "Icosphere" as a bone display
    # shape; it lives in a special collection and is not part of the asset.
    special = "glTF_not_exported"
    ignore = set()
    if special in bpy.data.collections:
        ignore = {o.name for o in bpy.data.collections[special].objects}

    rep = {"meshes": [], "armatures": [], "materials": [], "images": []}
    for ob in bpy.context.scene.objects:
        if ob.name in ignore:
            continue
        if ob.type == "MESH":
            me = ob.data
            pts = [ob.matrix_world @ v.co for v in me.vertices]
            keys = me.shape_keys.key_blocks if me.shape_keys else []
            rep["meshes"].append({
                "name": ob.name,
                "verts": len(me.vertices),
                "polys": len(me.polygons),
                "tris_only": all(len(p.vertices) == 3 for p in me.polygons),
                "uv_layers": len(me.uv_layers),
                "has_uvs": bool(me.uv_layers)
                           and any(uv.vector.length > 0 for uv in me.uv_layers[0].uv),
                "materials": [m.name for m in me.materials if m],
                "shape_keys": [k.name for k in keys][1:],
                "shape_deltas": {
                    k.name: max((k.data[i].co - me.vertices[i].co).length
                                for i in range(len(me.vertices)))
                    for k in list(keys)[1:]},
                "vgroups": len(ob.vertex_groups),
                "weight_sums": [sum(g.weight for g in v.groups)
                                for v in list(me.vertices)[:40]],
                "armature_modifiers": [m.object.name for m in ob.modifiers
                                       if m.type == "ARMATURE" and m.object],
                "bbox_min": [min(p[i] for p in pts) for i in range(3)],
                "bbox_max": [max(p[i] for p in pts) for i in range(3)],
                "normals_unit": all(abs(p.normal.length - 1.0) < 1e-3
                                    for p in list(me.polygons)[:200]),
            })
        elif ob.type == "ARMATURE":
            bones = {b.name: (b.parent.name if b.parent else None)
                     for b in ob.data.bones}
            rep["armatures"].append({
                "name": ob.name,
                "bones": bones,
                "roots": sorted(n for n, p in bones.items() if p is None),
                "heads_world": {b.name: list(ob.matrix_world @ b.head_local)
                                for b in ob.data.bones},
            })
    for m in bpy.data.materials:
        entry = {"name": m.name, "textures": []}
        if m.use_nodes:
            bsdf = next((n for n in m.node_tree.nodes
                         if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf:
                entry["base_color"] = list(bsdf.inputs["Base Color"].default_value)
                entry["roughness"] = bsdf.inputs["Roughness"].default_value
                entry["metallic"] = bsdf.inputs["Metallic"].default_value
                entry["linked_inputs"] = [i.name for i in bsdf.inputs if i.is_linked]
            entry["textures"] = [n.image.name for n in m.node_tree.nodes
                                 if n.type == "TEX_IMAGE" and n.image]
        rep["materials"].append(entry)
    rep["images"] = [{"name": i.name, "size": list(i.size), "pixels": i.has_data}
                     for i in bpy.data.images if i.name != "Render Result"]
    sys.stdout.write("@@DAZCONV@@" + json.dumps(rep) + "\\n")
''')


def run_blender(path, fmt, importer="python"):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(PROBE)
        script = fh.name
    try:
        proc = subprocess.run([BPY_PYTHON, script, path, fmt, importer],
                              capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(script)
    for line in proc.stdout.splitlines():
        if line.startswith("@@DAZCONV@@"):
            return json.loads(line[len("@@DAZCONV@@"):])
    raise AssertionError("Blender probe failed (%s, %s):\n%s\n%s"
                         % (fmt, importer, proc.stdout[-3000:], proc.stderr[-3000:]))


@unittest.skipUnless(HAVE_BPY, "set DAZCONV_BPY_PYTHON to a python with bpy")
class TestBlenderImport(unittest.TestCase):
    """Convert the demo figure, then ask Blender what it sees."""

    HEIGHT_M = 1.70

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        lib = os.path.join(cls.dir, "lib")
        cls.manifest = samples.generate(lib, overwrite=True)
        cls.out = os.path.join(cls.dir, "out")
        convert(cls.manifest["scene"], cls.out, formats=["fbx", "glb", "obj"],
                options=ConvertOptions(content_dirs=[lib]))
        cls.files = {f: os.path.join(cls.out, "DemoFigure." + f)
                     for f in ("fbx", "glb", "obj")}
        cls.reports = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def report(self, fmt, importer="python"):
        key = (fmt, importer)
        if key not in self.reports:
            self.reports[key] = run_blender(self.files[fmt], fmt, importer)
        return self.reports[key]

    # ---------------------------------------------------------------- FBX

    def test_fbx_mesh_matches_source(self):
        rep = self.report("fbx")
        self.assertEqual(len(rep["meshes"]), 1)
        mesh = rep["meshes"][0]
        self.assertEqual(mesh["verts"], 586)
        self.assertEqual(mesh["polys"], 636)
        self.assertEqual(mesh["uv_layers"], 1)
        self.assertTrue(mesh["has_uvs"])
        self.assertEqual(sorted(mesh["materials"]), ["Head", "Limbs", "Torso"])

    def test_fbx_skeleton(self):
        rep = self.report("fbx")
        self.assertEqual(len(rep["armatures"]), 1)
        arm = rep["armatures"][0]
        self.assertEqual(len(arm["bones"]), len(samples.SKELETON))
        self.assertEqual(arm["roots"], ["hip"])
        self.assertEqual(arm["bones"]["head"], "neck")
        self.assertEqual(arm["bones"]["lForearm"], "lShldr")

    def test_fbx_bind_pose_is_in_metres_and_z_up(self):
        """Blender converts cm/Y-up on import; the rig must land upright."""
        arm = self.report("fbx")["armatures"][0]
        origins = {b[0]: b[1] for b in samples.SKELETON}
        for name, world in arm["heads_world"].items():
            cm = origins[name]
            # Daz is cm / Y-up; Blender is m / Z-up.
            self.assertAlmostEqual(world[0], cm[0] / 100.0, places=4, msg=name)
            self.assertAlmostEqual(world[1], -cm[2] / 100.0, places=4, msg=name)
            self.assertAlmostEqual(world[2], cm[1] / 100.0, places=4, msg=name)
        self.assertGreater(arm["heads_world"]["head"][2],
                           arm["heads_world"]["hip"][2],
                           "figure must be the right way up")

    def test_fbx_mesh_is_bound_and_weighted(self):
        mesh = self.report("fbx")["meshes"][0]
        self.assertEqual(mesh["vgroups"], len(samples.SKELETON))
        self.assertEqual(mesh["armature_modifiers"], ["DemoFigure_rig"])
        for total in mesh["weight_sums"]:
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_fbx_blend_shapes_move_vertices(self):
        mesh = self.report("fbx")["meshes"][0]
        self.assertEqual(sorted(mesh["shape_keys"]), ["HeadScale", "Muscular"])
        for name, delta in mesh["shape_deltas"].items():
            self.assertGreater(delta, 1e-4, "%s has no deltas" % name)

    def test_fbx_scale_and_orientation(self):
        mesh = self.report("fbx")["meshes"][0]
        self.assertAlmostEqual(mesh["bbox_max"][2], self.HEIGHT_M, places=2)
        self.assertLess(mesh["bbox_min"][2], 0.05)
        self.assertLess(mesh["bbox_max"][0], 0.5, "figure is not metres-scaled")

    def test_fbx_materials_and_textures(self):
        rep = self.report("fbx")
        by_name = {m["name"]: m for m in rep["materials"]}
        self.assertEqual(sorted(by_name), ["Head", "Limbs", "Torso"])
        self.assertTrue(by_name["Torso"]["textures"], "no texture bound")
        self.assertIn("Base Color", by_name["Torso"]["linked_inputs"])
        self.assertTrue(all(i["pixels"] for i in rep["images"]),
                        "textures did not resolve to real image files")

    def test_fbx_cpp_importer_agrees(self):
        """Blender 4.5+ ships a second, independent FBX importer."""
        rep = self.report("fbx", "cpp")
        self.assertEqual(len(rep["meshes"]), 1)
        self.assertEqual(rep["meshes"][0]["verts"], 586)
        self.assertEqual(rep["meshes"][0]["polys"], 636)
        self.assertEqual(len(rep["armatures"][0]["bones"]), len(samples.SKELETON))

    # --------------------------------------------------------------- glTF

    def test_glb_matches_fbx(self):
        glb = self.report("glb")["meshes"][0]
        fbx = self.report("fbx")["meshes"][0]
        self.assertTrue(glb["tris_only"])
        self.assertEqual(glb["polys"], 1152)
        self.assertEqual(sorted(glb["shape_keys"]), sorted(fbx["shape_keys"]))
        self.assertEqual(glb["vgroups"], fbx["vgroups"])
        for axis in range(3):
            self.assertAlmostEqual(glb["bbox_min"][axis], fbx["bbox_min"][axis],
                                   places=3)
            self.assertAlmostEqual(glb["bbox_max"][axis], fbx["bbox_max"][axis],
                                   places=3)

    def test_glb_skeleton_matches_fbx(self):
        glb = self.report("glb")["armatures"][0]
        fbx = self.report("fbx")["armatures"][0]
        self.assertEqual(glb["bones"], fbx["bones"])
        for name, head in glb["heads_world"].items():
            for axis in range(3):
                self.assertAlmostEqual(head[axis], fbx["heads_world"][name][axis],
                                       places=3, msg="bone %s" % name)

    def test_glb_weights_normalised(self):
        for total in self.report("glb")["meshes"][0]["weight_sums"]:
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_glb_materials_have_textures(self):
        rep = self.report("glb")
        self.assertEqual(len(rep["materials"]), 3)
        self.assertTrue(any(m["textures"] for m in rep["materials"]))
        self.assertTrue(all(i["pixels"] for i in rep["images"]))

    # ---------------------------------------------------------------- OBJ

    def test_obj_geometry_and_materials(self):
        rep = self.report("obj")
        mesh = rep["meshes"][0]
        self.assertEqual(mesh["verts"], 586)
        self.assertEqual(mesh["polys"], 636)
        self.assertTrue(mesh["has_uvs"])
        self.assertEqual(sorted(mesh["materials"]), ["Head", "Limbs", "Torso"])
        self.assertTrue(all(i["pixels"] for i in rep["images"]))


@unittest.skipUnless(HAVE_BPY, "set DAZCONV_BPY_PYTHON to a python with bpy")
class TestBlenderConformed(unittest.TestCase):
    """Two figures, one shared rig: the classic clothing case."""

    @classmethod
    def setUpClass(cls):
        from tests.test_app import TestConformedFigures as T
        cls.dir = tempfile.mkdtemp()
        cls.manifest = samples.generate(cls.dir, overwrite=True)
        cls.scene_path = os.path.join(cls.dir, "People", "WithJacket.duf")
        T.dir = cls.dir
        T.manifest = cls.manifest
        T.scene_path = cls.scene_path
        T._write_outfit()
        cls.fbx = os.path.join(cls.dir, "out", "both.fbx")
        convert(cls.scene_path, cls.fbx, formats=["fbx"],
                options=ConvertOptions(content_dirs=[cls.dir]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_merged_export_keeps_one_rigged_mesh(self):
        merged = os.path.join(self.dir, "out", "merged.fbx")
        convert(self.scene_path, merged, formats=["fbx"],
                options=ConvertOptions(content_dirs=[self.dir],
                                       merge_meshes=True))
        rep = run_blender(merged, "fbx")
        self.assertEqual(len(rep["meshes"]), 1)
        mesh = rep["meshes"][0]
        self.assertEqual(mesh["verts"], 586 + 8)
        self.assertEqual(len(rep["armatures"]), 1)
        self.assertTrue(mesh["armature_modifiers"])
        self.assertEqual(sorted(mesh["shape_keys"]), ["HeadScale", "Muscular"])
        for total in mesh["weight_sums"]:
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_two_meshes_one_armature(self):
        rep = run_blender(self.fbx, "fbx")
        self.assertEqual(len(rep["meshes"]), 2)
        self.assertEqual(len(rep["armatures"]), 1)
        rigs = {m["armature_modifiers"][0] for m in rep["meshes"]
                if m["armature_modifiers"]}
        self.assertEqual(len(rigs), 1, "clothing must share the body's rig")


if __name__ == "__main__":
    unittest.main()
