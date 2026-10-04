"""Exporters: binary FBX container, FBX document structure, glTF, OBJ."""

import json
import math
import os
import shutil
import struct
import tempfile
import unittest

from dazconv import math3d, samples
from dazconv.api import convert
from dazconv.convert import ConvertOptions
from dazconv.export import FORMATS, write
from dazconv.export.fbx import binary
from dazconv.export.fbx.writer import FbxOptions, write_fbx
from dazconv.export.gltf import GltfOptions, write_gltf
from dazconv.export.obj import ObjOptions, write_obj


class TestFbxContainer(unittest.TestCase):
    """The low-level binary format, independent of any 3D content."""

    def _roundtrip(self, node):
        import io
        buf = io.BytesIO()
        binary.write_document(buf, [node])
        return binary.parse(buf.getvalue())

    def test_header_and_footer(self):
        import io
        buf = io.BytesIO()
        binary.write_document(buf, [binary.Node("Empty")])
        data = buf.getvalue()
        self.assertTrue(data.startswith(binary.HEADER_MAGIC))
        self.assertEqual(struct.unpack_from("<I", data, 23)[0], 7400)
        self.assertTrue(data.endswith(binary.FOOTER_MAGIC))

    def test_scalar_properties(self):
        n = binary.Node("P")
        n.add_bool(True).add_int16(-7).add_int32(123456).add_int64(2 ** 40)
        n.add_float32(1.5).add_float64(2.25).add_string("hello")
        parsed = self._roundtrip(n).children[0]
        self.assertEqual(parsed.name, b"P")
        self.assertEqual(parsed.props[0], True)
        self.assertEqual(parsed.props[1], -7)
        self.assertEqual(parsed.props[2], 123456)
        self.assertEqual(parsed.props[3], 2 ** 40)
        self.assertAlmostEqual(parsed.props[4], 1.5)
        self.assertAlmostEqual(parsed.props[5], 2.25)
        self.assertEqual(parsed.props[6], b"hello")

    def test_small_array_uncompressed(self):
        n = binary.Node("A")
        n.add_int32_array([1, 2, 3])
        parsed = self._roundtrip(n).children[0]
        self.assertEqual(parsed.props[0], [1, 2, 3])

    def test_large_array_compressed_round_trip(self):
        values = [float(i) * 0.5 for i in range(5000)]
        n = binary.Node("A")
        n.add_float64_array(values)
        parsed = self._roundtrip(n).children[0]
        self.assertEqual(len(parsed.props[0]), 5000)
        self.assertAlmostEqual(parsed.props[0][4999], 2499.5)

    def test_compression_actually_shrinks(self):
        import io
        values = [0.0] * 20000
        n = binary.Node("A")
        n.add_float64_array(values)
        buf = io.BytesIO()
        binary.write_document(buf, [n])
        self.assertLess(len(buf.getvalue()), 20000 * 8 / 10)

    def test_nested_nodes(self):
        root = binary.Node("Objects")
        child = root.child("Model")
        child.add_int64(99)
        grand = child.child("Properties70")
        grand.prop70("Lcl Translation", "Lcl Translation", "", "A", 1.0, 2.0, 3.0)
        parsed = self._roundtrip(root).children[0]
        self.assertEqual(parsed.find("Model").props[0], 99)
        p = parsed.find("Model").find("Properties70").find("P")
        self.assertEqual(p.props[0], b"Lcl Translation")
        self.assertAlmostEqual(p.props[6], 3.0)

    def test_name_class_encoding(self):
        self.assertEqual(binary.fbx_name("Body", "Model"), b"Body\x00\x01Model")

    def test_ktime_is_int64(self):
        n = binary.Node("P")
        n.prop70("TimeSpanStop", "KTime", "Time", "", 46186158000)
        parsed = self._roundtrip(n).children[0].find("P")
        self.assertEqual(parsed.props[4], 46186158000)


class TestExportedScene(unittest.TestCase):
    """End-to-end: demo library -> all formats, then read the files back."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.lib = os.path.join(cls.dir, "lib")
        cls.out = os.path.join(cls.dir, "out")
        cls.manifest = samples.generate(cls.lib, overwrite=True)
        cls.result = convert(cls.manifest["scene"], os.path.join(cls.out, "demo.fbx"),
                             formats=["fbx", "glb", "gltf", "obj"],
                             options=ConvertOptions(content_dirs=[cls.lib]))
        cls.scene = cls.result.scene
        cls.fbx = binary.parse(os.path.join(cls.out, "demo.fbx"))
        cls.objects = cls.fbx.find("Objects")
        cls.connections = cls.fbx.find("Connections")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    # ------------------------------------------------------------ general
    def test_all_formats_written(self):
        self.assertEqual(len(self.result.outputs), 4)
        for out in self.result.outputs:
            self.assertTrue(os.path.isfile(out["path"]))
            self.assertGreater(out["bytes"], 500)

    def test_no_warnings(self):
        self.assertEqual(self.result.warnings, [])

    # ---------------------------------------------------------------- FBX
    def test_fbx_top_level_sections(self):
        names = [c.name.decode() for c in self.fbx.children]
        for required in ("FBXHeaderExtension", "GlobalSettings", "Documents",
                         "Definitions", "Objects", "Connections", "Takes"):
            self.assertIn(required, names)

    def test_fbx_version_7400(self):
        self.assertEqual(self.fbx.props[0], 7400)
        hdr = self.fbx.find("FBXHeaderExtension")
        self.assertEqual(hdr.find("FBXVersion").props[0], 7400)

    def test_fbx_units_are_centimetres_y_up(self):
        props = self.fbx.find("GlobalSettings").find("Properties70")
        values = {p.props[0]: p.props[4] for p in props.find_all("P")}
        self.assertEqual(values[b"UpAxis"], 1)
        self.assertAlmostEqual(values[b"UnitScaleFactor"], 1.0)

    def test_fbx_geometry_indices(self):
        geom = next(c for c in self.objects.children
                    if c.name == b"Geometry" and c.props[2] == b"Mesh")
        verts = geom.find("Vertices").props[0]
        pvi = geom.find("PolygonVertexIndex").props[0]
        mesh = self.scene.figures[0].mesh
        self.assertEqual(len(verts) // 3, len(mesh.vertices))
        # one negative (~index) terminator per polygon
        self.assertEqual(sum(1 for i in pvi if i < 0), len(mesh.polygons))
        decoded = [(-i - 1) if i < 0 else i for i in pvi]
        self.assertTrue(all(0 <= i < len(mesh.vertices) for i in decoded))

    def test_fbx_uv_layer_is_index_to_direct(self):
        geom = next(c for c in self.objects.children
                    if c.name == b"Geometry" and c.props[2] == b"Mesh")
        uv = geom.find("LayerElementUV")
        self.assertEqual(uv.find("MappingInformationType").props[0], b"ByPolygonVertex")
        self.assertEqual(uv.find("ReferenceInformationType").props[0], b"IndexToDirect")
        n_corners = len(geom.find("PolygonVertexIndex").props[0])
        self.assertEqual(len(uv.find("UVIndex").props[0]), n_corners)

    def test_fbx_normals_per_polygon_vertex(self):
        geom = next(c for c in self.objects.children
                    if c.name == b"Geometry" and c.props[2] == b"Mesh")
        nrm = geom.find("LayerElementNormal")
        n_corners = len(geom.find("PolygonVertexIndex").props[0])
        self.assertEqual(len(nrm.find("Normals").props[0]) // 3, n_corners)

    def test_fbx_material_layer_matches_polygon_count(self):
        geom = next(c for c in self.objects.children
                    if c.name == b"Geometry" and c.props[2] == b"Mesh")
        mat_layer = geom.find("LayerElementMaterial")
        mats = mat_layer.find("Materials").props[0]
        self.assertEqual(len(mats), len(self.scene.figures[0].mesh.polygons))
        self.assertTrue(all(0 <= m < 3 for m in mats))

    def test_fbx_models_and_bones(self):
        models = [c for c in self.objects.children if c.name == b"Model"]
        kinds = [m.props[2] for m in models]
        self.assertEqual(kinds.count(b"Mesh"), 1)
        self.assertEqual(kinds.count(b"LimbNode"), self.manifest["bones"])
        self.assertEqual(kinds.count(b"Null"), 1)

    def test_fbx_bone_local_transforms_reconstruct_world(self):
        """Walking Lcl Translation/Rotation up the chain must rebuild the rig."""
        models = {m.props[1].split(b"\x00\x01")[0].decode(): m
                  for m in self.objects.children if m.name == b"Model"}
        skel = self.scene.figures[0].skeleton
        # rebuild world matrices from the exported local transforms
        world = {}
        for bone in skel.bones:
            model = models[bone.name]
            props = {p.props[0]: p.props[4:7]
                     for p in model.find("Properties70").find_all("P")}
            t = props[b"Lcl Translation"]
            r = props[b"Lcl Rotation"]
            local = math3d.compose(t, math3d.euler_to_mat(r, "XYZ"))
            parent_world = world.get(bone.parent, math3d.identity())
            world[bone.name] = math3d.mat_mul(local, parent_world)
            for a, b in zip(world[bone.name], bone.world_matrix):
                self.assertAlmostEqual(a, b, places=6,
                                       msg="bone %s world mismatch" % bone.name)

    def test_fbx_cluster_transforms_are_inverse_bind(self):
        clusters = [c for c in self.objects.children
                    if c.name == b"Deformer" and c.props[2] == b"Cluster"]
        self.assertEqual(len(clusters), self.manifest["bones"])
        for cl in clusters:
            transform = cl.find("Transform").props[0]
            link = cl.find("TransformLink").props[0]
            product = math3d.mat_mul(link, transform)
            for a, b in zip(product, math3d.IDENTITY):
                self.assertAlmostEqual(a, b, places=6)

    def test_fbx_cluster_weights_match_scene(self):
        mesh = self.scene.figures[0].mesh
        total_links = sum(len(inf) for inf in mesh.skin.influences)
        clusters = [c for c in self.objects.children
                    if c.name == b"Deformer" and c.props[2] == b"Cluster"]
        exported = 0
        for cl in clusters:
            idx = cl.find("Indexes")
            if idx is None:
                continue
            weights = cl.find("Weights").props[0]
            self.assertEqual(len(idx.props[0]), len(weights))
            exported += len(weights)
            self.assertTrue(all(0.0 < w <= 1.0 for w in weights))
        self.assertEqual(exported, total_links)

    def test_fbx_bind_pose_present(self):
        pose = next(c for c in self.objects.children if c.name == b"Pose")
        self.assertEqual(pose.find("Type").props[0], b"BindPose")
        n = pose.find("NbPoseNodes").props[0]
        self.assertEqual(n, len(pose.find_all("PoseNode")))
        for pn in pose.find_all("PoseNode"):
            self.assertEqual(len(pn.find("Matrix").props[0]), 16)

    def test_fbx_blend_shapes(self):
        shapes = [c for c in self.objects.children
                  if c.name == b"Geometry" and c.props[2] == b"Shape"]
        self.assertEqual(len(shapes), 2)
        by_name = {s.props[1].split(b"\x00\x01")[0].decode(): s for s in shapes}
        self.assertIn("Muscular", by_name)
        morph = next(m for m in self.scene.figures[0].mesh.morphs
                     if m.name == "Muscular")
        shape = by_name["Muscular"]
        self.assertEqual(len(shape.find("Indexes").props[0]), len(morph.indices))
        verts = shape.find("Vertices").props[0]
        self.assertEqual(len(verts) // 3, len(morph.deltas))
        # FBX shape vertices are deltas, not absolute positions
        self.assertAlmostEqual(verts[0], morph.deltas[0][0], places=5)

    def test_fbx_textures_and_videos_connected(self):
        textures = [c for c in self.objects.children if c.name == b"Texture"]
        videos = [c for c in self.objects.children if c.name == b"Video"]
        self.assertTrue(textures)
        self.assertEqual(len(videos), len(textures))
        op = [c for c in self.connections.children if c.props[0] == b"OP"]
        slots = {c.props[3] for c in op}
        self.assertIn(b"DiffuseColor", slots)
        self.assertIn(b"NormalMap", slots)

    def test_fbx_connections_resolve(self):
        ids = {c.props[0] for c in self.objects.children} | {0}
        for conn in self.connections.children:
            self.assertIn(conn.props[1], ids)
            self.assertIn(conn.props[2], ids)

    def test_fbx_definitions_count_matches_objects(self):
        defs = self.fbx.find("Definitions")
        counts = {}
        for ot in defs.find_all("ObjectType"):
            counts[ot.props[0]] = ot.find("Count").props[0]
        actual = {}
        for obj in self.objects.children:
            actual[obj.name] = actual.get(obj.name, 0) + 1
        for name, n in actual.items():
            self.assertEqual(counts.get(name), n,
                             "Definitions count wrong for %s" % name)

    def test_fbx_textures_copied_next_to_model(self):
        tex_dir = os.path.join(self.out, "textures")
        self.assertTrue(os.path.isdir(tex_dir))
        files = os.listdir(tex_dir)
        self.assertTrue(files)
        # no duplicate "_1" copies
        self.assertFalse([f for f in files if "_1." in f], files)

    # --------------------------------------------------------------- glTF
    def _glb_doc(self):
        path = os.path.join(self.out, "demo.glb")
        with open(path, "rb") as fh:
            data = fh.read()
        magic, version, length = struct.unpack_from("<III", data, 0)
        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(length, len(data))
        json_len, json_type = struct.unpack_from("<II", data, 12)
        self.assertEqual(json_type, 0x4E4F534A)
        doc = json.loads(data[20:20 + json_len].decode("utf-8"))
        self.assertEqual(doc["asset"]["version"], "2.0")
        return doc

    def test_glb_structure(self):
        self.assertIsNotNone(self._glb_doc())

    def test_gltf_accessors_within_buffer(self):
        with open(os.path.join(self.out, "demo.gltf"), "r") as fh:
            doc = json.load(fh)
        blen = doc["buffers"][0]["byteLength"]
        for view in doc["bufferViews"]:
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], blen)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "demo.bin")))

    def test_gltf_morph_targets_consistent(self):
        doc = self._glb_doc()
        for mesh in doc["meshes"]:
            counts = {len(p.get("targets", [])) for p in mesh["primitives"]}
            self.assertEqual(len(counts), 1,
                             "all primitives need the same target count")
            if mesh.get("weights"):
                self.assertEqual(len(mesh["weights"]), counts.pop())

    def test_gltf_skin_inverse_bind_matrices(self):
        """IBM * bone world must be identity (in metres)."""
        doc = self._glb_doc()
        skin = doc["skins"][0]
        acc = doc["accessors"][skin["inverseBindMatrices"]]
        view = doc["bufferViews"][acc["bufferView"]]
        with open(os.path.join(self.out, "demo.glb"), "rb") as fh:
            data = fh.read()
        json_len = struct.unpack_from("<I", data, 12)[0]
        bin_start = 20 + json_len + 8
        off = bin_start + view.get("byteOffset", 0)
        skel = self.scene.figures[0].skeleton
        mesh = self.scene.figures[0].mesh
        for i, name in enumerate(mesh.skin.bone_names):
            ibm = list(struct.unpack_from("<16f", data, off + i * 64))
            world = list(skel.get(name).world_matrix)
            world[12] *= 0.01
            world[13] *= 0.01
            world[14] *= 0.01
            product = math3d.mat_mul(world, ibm)
            for a, b in zip(product, math3d.IDENTITY):
                self.assertAlmostEqual(a, b, places=4)

    def test_gltf_joint_indices_in_range(self):
        doc = self._glb_doc()
        n_joints = len(doc["skins"][0]["joints"])
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                acc = doc["accessors"][prim["attributes"]["JOINTS_0"]]
                self.assertEqual(acc["type"], "VEC4")
                self.assertIn(acc["componentType"], (5121, 5123))
        self.assertEqual(n_joints, self.manifest["bones"])

    def test_gltf_scaled_to_metres(self):
        doc = self._glb_doc()
        acc = doc["accessors"][doc["meshes"][0]["primitives"][0]
                               ["attributes"]["POSITION"]]
        self.assertLess(max(acc["max"]), 3.0, "model should be ~1.7 m, not 170")

    def test_gltf_materials(self):
        doc = self._glb_doc()
        self.assertEqual(len(doc["materials"]), 3)
        for mat in doc["materials"]:
            pbr = mat["pbrMetallicRoughness"]
            self.assertIn("baseColorTexture", pbr)
            self.assertIn("normalTexture", mat)
        self.assertTrue(all("bufferView" in img for img in doc["images"]),
                        "GLB images should be embedded")

    # ---------------------------------------------------------------- OBJ
    def test_obj_contents(self):
        path = os.path.join(self.out, "demo.obj")
        v = vt = vn = f = 0
        materials = set()
        with open(path) as fh:
            for line in fh:
                if line.startswith("v "):
                    v += 1
                elif line.startswith("vt "):
                    vt += 1
                elif line.startswith("vn "):
                    vn += 1
                elif line.startswith("f "):
                    f += 1
                    for part in line.split()[1:]:
                        idx = int(part.split("/")[0])
                        self.assertGreaterEqual(idx, 1)
                elif line.startswith("usemtl "):
                    materials.add(line.split(None, 1)[1].strip())
        mesh = self.scene.figures[0].mesh
        self.assertEqual(v, len(mesh.vertices))
        self.assertEqual(vn, len(mesh.vertices))
        self.assertEqual(f, len(mesh.polygons))
        self.assertEqual(len(materials), 3)

    def test_mtl_written(self):
        path = os.path.join(self.out, "demo.mtl")
        self.assertTrue(os.path.isfile(path))
        with open(path) as fh:
            text = fh.read()
        self.assertIn("newmtl", text)
        self.assertIn("map_Kd", text)
        self.assertIn("Pr ", text)   # PBR roughness extension


class TestExportOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.lib = os.path.join(cls.dir, "lib")
        cls.manifest = samples.generate(cls.lib, overwrite=True)
        from dazconv.convert import SceneBuilder
        from dazconv.dson import ContentLibrary
        opts = ConvertOptions(content_dirs=[cls.lib])
        cls.scene = SceneBuilder(ContentLibrary([cls.lib]), opts) \
            .build_from_file(cls.manifest["scene"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_embed_textures_in_fbx(self):
        out = os.path.join(self.dir, "embed", "x.fbx")
        write_fbx(self.scene, out, FbxOptions(embed_textures=True,
                                              copy_textures=False))
        doc = binary.parse(out)
        videos = [c for c in doc.find("Objects").children if c.name == b"Video"]
        self.assertTrue(any(v.find("Content") is not None for v in videos))

    def test_no_skin_option(self):
        out = os.path.join(self.dir, "norig", "x.fbx")
        write_fbx(self.scene, out, FbxOptions(export_skin=False,
                                              copy_textures=False))
        doc = binary.parse(out)
        clusters = [c for c in doc.find("Objects").children
                    if c.name == b"Deformer" and c.props[2] == b"Cluster"]
        self.assertEqual(clusters, [])

    def test_gltf_without_morphs(self):
        out = os.path.join(self.dir, "nomorph", "x.gltf")
        write_gltf(self.scene, out, GltfOptions(binary=False,
                                                export_morphs=False,
                                                embed_textures=False))
        with open(out) as fh:
            doc = json.load(fh)
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                self.assertNotIn("targets", prim)

    def test_obj_triangulate(self):
        out = os.path.join(self.dir, "tri", "x.obj")
        write_obj(self.scene, out, ObjOptions(triangulate=True,
                                              copy_textures=False))
        with open(out) as fh:
            faces = [l for l in fh if l.startswith("f ")]
        self.assertTrue(all(len(l.split()) == 4 for l in faces))

    def test_format_dispatch_by_extension(self):
        out = os.path.join(self.dir, "disp", "x.obj")
        info = write(self.scene, out, "")
        self.assertTrue(os.path.isfile(info["path"]))

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            write(self.scene, "/tmp/x.xyz", "xyz")

    def test_format_registry(self):
        self.assertEqual(set(FORMATS), {"fbx", "glb", "gltf", "obj"})


if __name__ == "__main__":
    unittest.main()
