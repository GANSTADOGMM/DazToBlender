"""Conversion layer: math, geometry, skinning, morphs, materials, builder."""

import math
import os
import random
import shutil
import tempfile
import unittest

from dazconv import math3d, samples
from dazconv.convert import ConvertOptions, builder, geometry, morphs, skinning
from dazconv.convert.materials import MaterialBuilder, merge_material
from dazconv.dson import ContentLibrary
from dazconv.ir import Mesh


class TestMath(unittest.TestCase):
    def test_euler_round_trip(self):
        random.seed(7)
        for _ in range(500):
            angles = [random.uniform(-179, 179) for _ in range(3)]
            m = math3d.euler_to_mat(angles, "XYZ")
            back = math3d.mat_to_euler_xyz(m)
            m2 = math3d.euler_to_mat(back, "XYZ")
            for a, b in zip(m, m2):
                self.assertAlmostEqual(a, b, places=9)

    def test_inverse(self):
        m = math3d.mat_mul_many(math3d.scaling((2, 3, 4)),
                                math3d.euler_to_mat((10, 20, 30)),
                                math3d.translation((5, 6, 7)))
        ident = math3d.mat_mul(m, math3d.mat_inverse(m))
        for a, b in zip(ident, math3d.IDENTITY):
            self.assertAlmostEqual(a, b, places=9)

    def test_decompose(self):
        m = math3d.mat_mul_many(math3d.scaling((2, 2, 2)),
                                math3d.euler_to_mat((0, 45, 0)),
                                math3d.translation((1, 2, 3)))
        t, r, s = math3d.decompose(m)
        self.assertAlmostEqual(t[0], 1.0)
        self.assertAlmostEqual(r[1], 45.0, places=6)
        self.assertAlmostEqual(s[0], 2.0, places=9)

    def test_row_vector_order(self):
        """p * (A*B) must equal (p*A)*B."""
        a = math3d.euler_to_mat((15, 0, 0))
        b = math3d.translation((1, 0, 0))
        p = (1.0, 2.0, 3.0)
        direct = math3d.transform_point(math3d.mat_mul(a, b), p)
        staged = math3d.transform_point(b, math3d.transform_point(a, p))
        for x, y in zip(direct, staged):
            self.assertAlmostEqual(x, y, places=12)


class TestGeometry(unittest.TestCase):
    def _geom(self):
        return {
            "id": "g",
            "vertices": {"count": 6, "values": [[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                                [0, 1, 0], [2, 0, 0], [2, 1, 0]]},
            "polygon_groups": {"count": 1, "values": ["body"]},
            "polygon_material_groups": {"count": 2, "values": ["Skin", "Nails"]},
            "polylist": {"count": 2, "values": [[0, 0, 0, 1, 2, 3],
                                                [0, 1, 1, 4, 5]]},
        }

    def test_parse_quads_and_tris(self):
        mesh = geometry.parse_geometry(self._geom())
        self.assertEqual(len(mesh.vertices), 6)
        self.assertEqual(len(mesh.polygons), 2)
        self.assertEqual(len(mesh.polygons[0]), 4)
        self.assertEqual(len(mesh.polygons[1]), 3)
        self.assertEqual(mesh.polygon_material, [0, 1])

    def test_degenerate_quad_becomes_triangle(self):
        g = self._geom()
        g["polylist"]["values"] = [[0, 0, 0, 1, 2, 2]]
        mesh = geometry.parse_geometry(g)
        self.assertEqual(len(mesh.polygons[0]), 3)

    def test_uv_seams(self):
        mesh = geometry.parse_geometry(self._geom())
        uv_set = {
            "uvs": {"count": 7, "values": [[0, 0], [1, 0], [1, 1], [0, 1],
                                           [2, 0], [2, 1], [9, 9]]},
            "polygon_vertex_indices": [[1, 4, 6]],
        }
        geometry.apply_uv_set(mesh, uv_set)
        self.assertEqual(mesh.uv_indices[0], (0, 1, 2, 3))
        # vertex 4 in polygon 1 uses the seam UV (index 6)
        self.assertEqual(mesh.uv_indices[1][1], 6)

    def test_vertex_normals(self):
        mesh = geometry.parse_geometry(self._geom())
        normals = geometry.compute_vertex_normals(mesh)
        self.assertEqual(len(normals), 6)
        for n in normals:
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0, places=6)
            self.assertAlmostEqual(abs(n[2]), 1.0, places=6)  # flat in XY plane

    def test_transform_mesh(self):
        mesh = geometry.parse_geometry(self._geom())
        geometry.transform_mesh(mesh, math3d.translation((10, 0, 0)))
        self.assertAlmostEqual(mesh.vertices[0][0], 10.0)

    def test_triangulate(self):
        mesh = geometry.parse_geometry(self._geom())
        self.assertEqual(len(mesh.triangulate()), 3)  # quad -> 2, tri -> 1
        self.assertEqual(mesh.num_triangles, 3)

    def test_merge_meshes(self):
        a = geometry.parse_geometry(self._geom(), "a")
        b = geometry.parse_geometry(self._geom(), "b")
        merged = geometry.merge_meshes([a, b], "m")
        self.assertEqual(len(merged.vertices), 12)
        self.assertEqual(len(merged.polygons), 4)
        self.assertTrue(max(max(p) for p in merged.polygons) < 12)


class TestSkinning(unittest.TestCase):
    def _skin(self):
        return {"joints": [
            {"id": "hip", "node": "#hip",
             "node_weights": {"count": 2, "values": [[0, 1.0], [1, 0.5]]}},
            {"id": "abdomen", "node": "#abdomen",
             "node_weights": {"count": 2, "values": [[1, 0.5], [2, 1.0]]}},
        ]}

    def test_parse_and_normalize(self):
        weights, warns = skinning.parse_skin_binding(self._skin(), 3)
        self.assertEqual(weights.bone_names, ["hip", "abdomen"])
        self.assertAlmostEqual(sum(w for _, w in weights.influences[1]), 1.0)
        self.assertEqual(weights.influences[0], [(0, 1.0)])

    def test_local_weights_collapse(self):
        skin = {"joints": [{"id": "b", "node": "#b", "local_weights": {
            "x": {"values": [[0, 0.3]]},
            "y": {"values": [[0, 0.9]]},
            "z": {"values": [[0, 0.1]]}}}]}
        weights, _ = skinning.parse_skin_binding(skin, 1)
        self.assertEqual(len(weights.influences[0]), 1)
        self.assertAlmostEqual(weights.influences[0][0][1], 1.0)

    def test_prune_to_max_influences(self):
        joints = [{"id": "b%d" % i, "node": "#b%d" % i,
                   "node_weights": {"values": [[0, 1.0 / (i + 1)]]}}
                  for i in range(10)]
        weights, _ = skinning.parse_skin_binding({"joints": joints}, 1,
                                                 max_influences=4)
        self.assertEqual(len(weights.influences[0]), 4)
        self.assertAlmostEqual(sum(w for _, w in weights.influences[0]), 1.0)

    def test_unweighted_vertices_reported(self):
        _w, warns = skinning.parse_skin_binding(self._skin(), 5)
        self.assertTrue(any("no skin weights" in w for w in warns))

    def test_fallback_rigid_bind(self):
        weights, _ = skinning.parse_skin_binding(self._skin(), 5)
        skinning.fallback_rigid_bind(weights, 0)
        self.assertTrue(all(inf for inf in weights.influences))


class TestMorphs(unittest.TestCase):
    def _mod(self, name="Smile", count=3):
        return {"id": name, "name": name,
                "channel": {"id": "value", "current_value": 0.4},
                "morph": {"vertex_count": 10,
                          "deltas": {"count": count,
                                     "values": [[i, 0.1 * i, 0.0, 0.0]
                                                for i in range(count)]}}}

    def test_parse(self):
        m = morphs.parse_morph(self._mod(), 10)
        self.assertIsNotNone(m)
        # the first delta is all-zero and gets dropped
        self.assertEqual(len(m.indices), 2)
        self.assertAlmostEqual(m.default_value, 0.4)

    def test_out_of_range_deltas_dropped(self):
        mod = self._mod(count=3)
        mod["morph"]["deltas"]["values"].append([999, 1, 1, 1])
        m = morphs.parse_morph(mod, 10)
        self.assertTrue(all(i < 10 for i in m.indices))

    def test_hd_detection_and_filter(self):
        self.assertTrue(morphs.is_hd_morph("PHMMouth_HD_div4"))
        self.assertFalse(morphs.is_hd_morph("PHMMouth"))
        a = morphs.parse_morph(self._mod("PHMMouth_HD_div4"), 10)
        b = morphs.parse_morph(self._mod("Smile"), 10)
        kept = morphs.filter_morphs([a, b], exclude_hd=True)
        self.assertEqual([m.name for m in kept], ["Smile"])

    def test_name_pattern_filter(self):
        a = morphs.parse_morph(self._mod("eCTRLSmile"), 10)
        b = morphs.parse_morph(self._mod("Heavy"), 10)
        kept = morphs.filter_morphs([a, b], include=["ectrl*"])
        self.assertEqual([m.name for m in kept], ["eCTRLSmile"])

    def test_apply_morph(self):
        m = morphs.parse_morph(self._mod(), 10)
        verts = [(0.0, 0.0, 0.0)] * 10
        out = morphs.apply_morph(verts, m, 1.0)
        self.assertAlmostEqual(out[1][0], 0.1)
        self.assertAlmostEqual(out[2][0], 0.2)

    def test_sanitize_name(self):
        self.assertEqual(morphs.sanitize_name("head/eyes right.dsf"),
                         "head_eyes_right")


class TestMaterials(unittest.TestCase):
    def test_iray_uber(self):
        mat_def = {
            "id": "Face", "name": "Face",
            "extra": [{"type": "studio/material/uber_iray"},
                      {"type": "studio_material_channels", "channels": [
                          {"channel": {"id": "Base Color", "current_value": [0.8, 0.6, 0.5],
                                       "image_file": "/Runtime/Textures/x.png"}},
                          {"channel": {"id": "Metallic Weight", "current_value": 0.25}},
                          {"channel": {"id": "Glossy Roughness", "current_value": 0.3}},
                          {"channel": {"id": "Cutout Opacity", "current_value": 0.5}},
                      ]}]}
        mat = MaterialBuilder(None).build(mat_def)
        self.assertAlmostEqual(mat.metallic, 0.25)
        self.assertAlmostEqual(mat.roughness, 0.3)
        self.assertAlmostEqual(mat.opacity, 0.5)
        self.assertTrue(mat.alpha_blend)
        self.assertIsNotNone(mat.base_color_map)

    def test_3delight_glossiness_to_roughness(self):
        mat_def = {"id": "Legacy",
                   "diffuse": {"channel": {"id": "diffuse", "current_value": [1, 1, 1]}},
                   "glossiness": {"channel": {"id": "glossiness", "current_value": 0.8}}}
        mat = MaterialBuilder(None).build(mat_def)
        self.assertAlmostEqual(mat.roughness, 0.2, places=6)

    def test_transparency_is_inverted(self):
        mat_def = {"id": "Glass",
                   "transparency": {"channel": {"id": "transparency",
                                                "current_value": 0.75}}}
        mat = MaterialBuilder(None).build(mat_def)
        self.assertAlmostEqual(mat.opacity, 0.25)

    def test_instance_overrides_library(self):
        lib = {"id": "Skin", "extra": [{"type": "studio_material_channels", "channels": [
            {"channel": {"id": "Metallic Weight", "current_value": 0.0}}]}]}
        inst = {"url": "#Skin", "extra": [{"type": "studio_material_channels", "channels": [
            {"channel": {"id": "Metallic Weight", "current_value": 1.0}}]}]}
        merged = merge_material(lib, inst)
        mat = MaterialBuilder(None).build(merged)
        self.assertAlmostEqual(mat.metallic, 1.0)


class TestBuilderOnDemoLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.manifest = samples.generate(cls.dir, overwrite=True)
        opts = ConvertOptions(content_dirs=[cls.dir])
        lib = ContentLibrary([cls.dir])
        cls.scene = builder.SceneBuilder(lib, opts).build_from_file(cls.manifest["scene"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_scene_built(self):
        self.assertEqual(len(self.scene.figures), 1)
        self.assertFalse(self.scene.warnings, self.scene.warnings)

    def test_mesh(self):
        mesh = self.scene.figures[0].mesh
        self.assertEqual(len(mesh.vertices), self.manifest["vertices"])
        self.assertEqual(len(mesh.polygons), self.manifest["polygons"])
        self.assertTrue(mesh.uvs)
        self.assertEqual(len(mesh.uv_indices), len(mesh.polygons))
        self.assertEqual(len(mesh.normals), len(mesh.vertices))

    def test_materials_resolved_with_textures(self):
        mats = self.scene.figures[0].mesh.materials
        self.assertEqual(len(mats), 3)
        for mat in mats:
            self.assertIsNotNone(mat.base_color_map)
            self.assertTrue(os.path.isfile(mat.base_color_map.path),
                            "texture should resolve on disk: %s" % mat.base_color_map.path)
            self.assertIsNotNone(mat.normal_map)

    def test_skeleton(self):
        skel = self.scene.figures[0].skeleton
        self.assertEqual(len(skel), self.manifest["bones"])
        roots = skel.roots()
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].name, "hip")
        head = skel.get("head")
        self.assertEqual(head.parent, "neck")
        self.assertGreater(head.length, 0.0)

    def test_weights_normalised(self):
        mesh = self.scene.figures[0].mesh
        self.assertIsNotNone(mesh.skin)
        self.assertEqual(len(mesh.skin.influences), len(mesh.vertices))
        for inf in mesh.skin.influences:
            self.assertTrue(inf)
            self.assertAlmostEqual(sum(w for _, w in inf), 1.0, places=5)
            self.assertLessEqual(len(inf), 8)

    def test_morphs(self):
        names = {m.name for m in self.scene.figures[0].mesh.morphs}
        self.assertEqual(names, {"Muscular", "HeadScale"})

    def test_summary(self):
        s = self.scene.summary()
        self.assertEqual(s["figures"], 1)
        self.assertEqual(s["bones"], self.manifest["bones"])
        self.assertEqual(s["morphs"], 2)

    def test_options_disable_features(self):
        opts = ConvertOptions(content_dirs=[self.dir], include_skeleton=False,
                              include_morphs=False)
        scene = builder.SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.manifest["scene"])
        fig = scene.figures[0]
        self.assertIsNone(fig.skeleton)
        self.assertIsNone(fig.mesh.skin)
        self.assertEqual(fig.mesh.morphs, [])

    def test_asset_mode_on_dsf(self):
        """Converting the raw figure .dsf (no scene wrapper) also works."""
        opts = ConvertOptions(content_dirs=[self.dir])
        scene = builder.SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.manifest["figure"])
        self.assertEqual(len(scene.figures), 1)
        self.assertTrue(scene.figures[0].mesh.vertices)
        self.assertIsNotNone(scene.figures[0].skeleton)


if __name__ == "__main__":
    unittest.main()
