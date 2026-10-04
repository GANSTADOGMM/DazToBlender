"""Integration tests against real Daz data, validated with an independent reader.

The Daz sample assets are *not* committed to this repository (they are licensed
content).  Run ``python tools/fetch_samples.py`` to fetch them, or point
``DAZCONV_SAMPLES`` at an existing content folder.  Tests skip cleanly when the
samples or the ``ufbx`` package are unavailable.
"""

from __future__ import annotations

import collections
import json
import os
import tempfile
import unittest

from . import (
    compare_vertex_delta,
    SAMPLE_FIGURE,
    SAMPLE_G3F,
    SAMPLE_MORPH,
    SAMPLE_SCENE,
    SAMPLE_UVSET,
    have_samples,
    SAMPLES_DIR,
    have_scene,
    have_ufbx,
    run_ufbx_validate,
    samples_library,
)

from dazconv import ContentLibrary, ExportOptions, SceneLoader, convert_file
from dazconv.geometry import Geometry
from dazconv.dson import load_dson_file
from dazconv.morphs import load_morphs_from_file
from dazconv.rig import Skeleton
from dazconv.skinning import SkinBinding, build_influences
from dazconv.uvset import UVSet


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class GeometryParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_dson_file(SAMPLE_FIGURE)
        cls.geometry = Geometry.from_dson(cls.document["geometry_library"][0])

    def test_vertex_and_polygon_counts(self):
        self.assertEqual(self.geometry.vertex_count, 16556)
        self.assertEqual(self.geometry.polygon_count, 16368)

    def test_corner_count_is_quads(self):
        self.assertEqual(self.geometry.corner_count, 16368 * 4)

    def test_group_tables(self):
        self.assertEqual(len(self.geometry.polygon_groups), 61)
        self.assertEqual(len(self.geometry.polygon_material_groups), 17)

    def test_polylist_indices_in_range(self):
        for polygon in self.geometry.polygons:
            for vertex in polygon.vertices:
                self.assertGreaterEqual(vertex, 0)
                self.assertLess(vertex, self.geometry.vertex_count)

    def test_subdivision_surface_type(self):
        self.assertTrue(self.geometry.is_subdivision())
        self.assertEqual(self.geometry.edge_interpolation_mode, "edges_only")

    def test_triangulation_is_consistent(self):
        triangles = self.geometry.triangulated_indices()
        self.assertEqual(len(triangles), 16368 * 2)
        self.assertTrue(all(len(t) == 3 for t in triangles))

    def test_normals_are_unit_length(self):
        for normal in self.geometry.vertex_normals()[:200]:
            length = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
            self.assertAlmostEqual(length, 1.0, places=5)


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class RigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document = load_dson_file(SAMPLE_FIGURE)
        cls.skeleton = Skeleton.from_node_library(document["node_library"])

    def test_bone_count(self):
        # 171 node_library entries: 1 figure node + 170 bones.
        self.assertEqual(len(self.skeleton.bones), 171)

    def test_single_root(self):
        self.assertEqual(len(self.skeleton.roots), 1)
        self.assertEqual(self.skeleton.roots[0].id, "Genesis8_1Female")

    def test_parent_before_child_ordering(self):
        seen = set()
        for bone in self.skeleton.depth_ordered():
            if bone.parent is not None:
                self.assertIn(bone.parent.id, seen, f"{bone.id} listed before its parent")
            seen.add(bone.id)

    def test_hip_origin_matches_source(self):
        hip = self.skeleton.find("hip")
        self.assertIsNotNone(hip)
        self.assertAlmostEqual(hip.origin[1], 105.1781, places=4)

    def test_child_world_position_accumulates(self):
        """A child's world position is its parent's plus its own centre point."""
        hip = self.skeleton.find("hip")
        pelvis = self.skeleton.find("pelvis")
        self.assertIsNotNone(pelvis)
        for axis in range(3):
            self.assertAlmostEqual(
                pelvis.world_position[axis],
                hip.world_position[axis] + pelvis.origin[axis],
                places=4,
            )
        # Sanity: in Genesis 8.1 pelvis sits just above hip (107.04 vs 105.18).
        self.assertGreater(pelvis.world_position[1], hip.world_position[1])


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class SkinningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_dson_file(SAMPLE_FIGURE)
        cls.geometry = Geometry.from_dson(cls.document["geometry_library"][0])
        cls.skeleton = Skeleton.from_node_library(cls.document["node_library"])
        binding = None
        for modifier in cls.document["modifier_library"]:
            binding = SkinBinding.from_dson(modifier) or binding
        cls.binding = binding

    def test_binding_found(self):
        self.assertIsNotNone(self.binding)
        self.assertEqual(self.binding.geometry_id, self.geometry.id)
        self.assertEqual(self.binding.vertex_count, self.geometry.vertex_count)

    def test_source_weights_sum_to_one(self):
        totals = self.binding.weight_sums()
        self.assertEqual(len(totals), self.geometry.vertex_count)
        for vertex, total in totals.items():
            self.assertAlmostEqual(total, 1.0, places=4, msg=f"vertex {vertex}")

    def test_influences_capped_and_renormalised(self):
        influences = build_influences(
            self.binding, self.skeleton, self.geometry.vertex_count, max_influences=4
        )
        self.assertEqual(len(influences.joint_weights), self.geometry.vertex_count)
        self.assertLessEqual(influences.max_influences, 4)
        for weights in influences.joint_weights:
            if weights:
                self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_validation_reports_no_problems(self):
        self.assertEqual(self.binding.validate(self.geometry.vertex_count), [])


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class UVSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_dson_file(SAMPLE_UVSET)
        cls.uv_set = UVSet.from_dson(cls.document["uv_set_library"][0])
        cls.geometry = Geometry.from_dson(load_dson_file(SAMPLE_FIGURE)["geometry_library"][0])

    def test_counts(self):
        self.assertEqual(self.uv_set.vertex_count, 16556)
        self.assertEqual(len(self.uv_set.uvs), 18293)
        self.assertEqual(len(self.uv_set.overrides), 3308)

    def test_override_lookup(self):
        self.assertEqual(self.uv_set.corner_uv_index(6410, 2, 999999), 17165)

    def test_default_corner_uses_vertex_index(self):
        self.assertEqual(self.uv_set.corner_uv_index(0, 0, 42), 42)

    def test_corner_uvs_cover_every_corner(self):
        corner_uvs = self.uv_set.corner_uvs(self.geometry)
        self.assertEqual(len(corner_uvs), self.geometry.corner_count)

    def test_v_is_flipped_for_fbx(self):
        raw = self.uv_set.get_uv(self.geometry.polygons[0].vertices[0])[1]
        flipped = self.uv_set.corner_uvs(self.geometry, flip_v=True)[0][1]
        self.assertAlmostEqual(flipped, 1.0 - raw, places=6)

    def test_sanity_report_clean(self):
        self.assertEqual(self.uv_set.sanity_report(self.geometry), [])


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class SceneAssemblyTests(unittest.TestCase):
    def test_direct_dsf_asset(self):
        scene = SceneLoader(library=samples_library()).load(SAMPLE_FIGURE)
        self.assertEqual(len(scene.meshes), 1)
        asset = scene.meshes[0]
        self.assertEqual(asset.geometry.vertex_count, 16556)
        self.assertTrue(asset.is_skinned)
        self.assertIsNotNone(asset.uv_set)
        self.assertEqual(len(asset.corner_uvs), asset.geometry.corner_count)
        self.assertEqual(asset.warnings, [])

    @unittest.skipUnless(have_scene(), "repo scene not available")
    def test_scene_resolves_external_geometry(self):
        """The repo .duf only references geometry; it must be pulled from the .dsf."""
        scene = SceneLoader(library=samples_library()).load(SAMPLE_SCENE)
        self.assertEqual(len(scene.meshes), 1)
        asset = scene.meshes[0]
        self.assertEqual(asset.geometry.vertex_count, 16556)
        self.assertTrue(asset.is_skinned, "skeleton/skin must come from the owning .dsf")
        self.assertTrue(asset.is_figure)
        self.assertIsNotNone(asset.uv_set, "uv set must resolve via the content library")
        self.assertEqual(len(asset.materials), 17)

    @unittest.skipUnless(have_scene(), "repo scene not available")
    def test_case_insensitive_content_lookup(self):
        """Daz URLs say 'Daz 3D' while the library folder is 'DAZ 3D'."""
        library = samples_library()
        resolved = library.resolve(
            "/data/Daz%203D/Genesis%208/Female%208_1/Genesis8_1Female.dsf"
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.isfile(resolved))


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class MorphTests(unittest.TestCase):
    def test_load_gzipped_morph(self):
        """Morph files are gzipped DSON; the base figure is plain text."""
        if not os.path.isfile(SAMPLE_MORPH):
            self.skipTest("morph sample not present")
        with open(SAMPLE_MORPH, "rb") as handle:
            self.assertEqual(handle.read(2), b"\x1f\x8b", "expected a gzip container")
        morphs = load_morphs_from_file(SAMPLE_MORPH)
        self.assertTrue(morphs)
        morph = morphs[0]
        self.assertEqual(morph.id, "FBMPearFigure")
        self.assertEqual(morph.vertex_count, 17418)
        self.assertGreater(morph.affected_count(), 1000)
        self.assertTrue(morph.deltas)


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
class ConversionTests(unittest.TestCase):
    def test_convert_produces_fbx(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "figure.fbx")
            report = convert_file(
                SAMPLE_FIGURE, output, ExportOptions(), samples_library()
            )
            self.assertTrue(report.success, report.error)
            self.assertTrue(os.path.isfile(output))
            self.assertGreater(os.path.getsize(output), 100_000)
            self.assertEqual(report.scene_stats["vertices"], 16556)
            self.assertEqual(report.export["vertices"], 16556)

    def test_progress_callback_fires(self):
        stages = []
        with tempfile.TemporaryDirectory() as tmp:
            convert_file(
                SAMPLE_FIGURE,
                os.path.join(tmp, "f.fbx"),
                ExportOptions(),
                samples_library(),
                progress=lambda stage, fraction, message: stages.append(stage),
            )
        for expected in ("load", "textures", "export", "done"):
            self.assertIn(expected, stages)

    def test_options_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = convert_file(
                SAMPLE_FIGURE,
                os.path.join(tmp, "f.fbx"),
                ExportOptions(scale=0.0),
                samples_library(),
            )
            self.assertFalse(report.success)
            self.assertIn("scale", report.error)


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
@unittest.skipUnless(have_ufbx(), "ufbx package not available")
class FbxRoundTripTests(unittest.TestCase):
    """The real test: an independent FBX parser must agree with our source data.

    ``ufbx`` is run in a subprocess because its Python bindings can segfault on
    object-lifetime bugs; a crash must fail the test, not kill the suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "figure.fbx")
        cls.report = convert_file(
            SAMPLE_FIGURE, cls.path, ExportOptions(), samples_library()
        )
        assert cls.report.success, cls.report.error
        cls.info = run_ufbx_validate(cls.path)
        cls.source = load_dson_file(SAMPLE_FIGURE)
        cls.src_geometry = cls.source["geometry_library"][0]
        cls.mesh = cls.info["meshes"][0]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_file_loads_with_independent_parser(self):
        self.assertTrue(self.info["ok"], self.info.get("error"))
        self.assertEqual(self.info["num_meshes"], 1)
        self.assertEqual(self.mesh["vertices"], 16556)
        self.assertEqual(self.mesh["faces"], 16368)

    def test_uv_channel_present_and_sized(self):
        self.assertEqual(self.mesh["uv_sets"], ["UVMap"])
        # One UV per polygon corner, and the geometry is all quads.
        self.assertEqual(self.mesh["uv_count"], 16368 * 4)

    def test_uv_values_match_source(self):
        """Corner UVs must come from the uv_set, V flipped for FBX."""
        uv_document = load_dson_file(SAMPLE_UVSET)["uv_set_library"][0]
        uvs = uv_document["uvs"]["values"]
        # Resolve the first face's first corner exactly as the exporter does:
        # a polygon_vertex_indices override if present, else the vertex index.
        first_face = self.src_geometry["polylist"]["values"][0]
        first_vertex = first_face[2]  # 1594 in Genesis 8.1 -- not 0
        raw_overrides = uv_document.get("polygon_vertex_indices", [])
        if isinstance(raw_overrides, dict):
            raw_overrides = raw_overrides.get("values", [])
        overrides = {(int(p), int(c)): int(u) for p, c, u in raw_overrides}
        uv_index = overrides.get((0, 0), first_vertex)
        expected = uvs[uv_index]
        got = self.mesh["uvs"][0]
        self.assertAlmostEqual(got[0], expected[0], places=5)
        self.assertAlmostEqual(got[1], 1.0 - expected[1], places=5)

    def test_vertex_positions_match_source(self):
        """Vertex order is preserved 1:1 with the DSON vertex array."""
        source = self.src_geometry["vertices"]["values"]
        for index in range(3):
            got = self.mesh["vertex_positions"][index]
            for axis in range(3):
                # FBX stores doubles; our ASCII carries 7 decimals.
                self.assertAlmostEqual(got[axis], source[index][axis], places=4)

    def test_normals_present(self):
        self.assertTrue(self.mesh["has_normals"])

    def test_skeleton_exported(self):
        # 1 figure node + 170 bones, all emitted as skeleton models.
        self.assertEqual(self.info["num_bones"], 171)
        self.assertIn("Hip", self.info["bone_names"])
        self.assertIn("Head", self.info["bone_names"])
        self.assertIn("Left Hand", self.info["bone_names"])

    def test_bone_local_transforms_match_daz_center_points(self):
        """Checked with ufbx directly, using only lifetime-safe accessors."""
        import ufbx

        scene = ufbx.load_file(self.path)
        nodes = scene.nodes
        by_name = {node.name: node for node in nodes}
        source_nodes = {n["id"]: n for n in self.source["node_library"]}
        for source_id, fbx_name in (("hip", "Hip"), ("head", "Head")):
            channels = {
                c["id"]: c.get("value", 0.0)
                for c in source_nodes[source_id].get("center_point", [])
                if isinstance(c, dict)
            }
            expected = (channels.get("x", 0.0), channels.get("y", 0.0), channels.get("z", 0.0))
            translation = by_name[fbx_name].local_transform.translation
            for axis, attribute in enumerate(("x", "y", "z")):
                self.assertAlmostEqual(
                    getattr(translation, attribute),
                    expected[axis],
                    places=3,
                    msg=f"{fbx_name} axis {axis}",
                )

    def test_skin_weights_sum_to_one_for_every_vertex(self):
        self.assertEqual(self.mesh["skin_clusters"], 167)
        self.assertEqual(self.mesh["skinned_vertices"], 16556)
        self.assertAlmostEqual(self.mesh["weight_sum_min"], 1.0, places=3)
        self.assertAlmostEqual(self.mesh["weight_sum_max"], 1.0, places=3)
        self.assertLessEqual(self.mesh["max_influences"], 4)

    def test_bare_figure_has_no_materials(self):
        """A geometry .dsf defines no materials -- those come from a .duf scene."""
        self.assertEqual(self.info["num_materials"], 0)
        self.assertEqual(self.mesh["materials"], 0)


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
@unittest.skipUnless(have_ufbx(), "ufbx package not available")
class FbxVariantTests(unittest.TestCase):
    """Every export variant must still parse with the independent reader."""

    VARIANTS = {
        "geometry only": dict(export_skeleton=False, export_weights=False, export_materials=False),
        "triangulated": dict(triangulate=True),
        "no uv flip": dict(flip_uv=False),
        "z-up": dict(up_axis="z"),
        "scaled": dict(scale=0.01),
        "one influence": dict(max_influences=1),
        "materials only": dict(export_skeleton=False, export_weights=False),
        "flat normals": dict(smooth_normals=False),
    }

    def test_variants_load(self):
        for name, kwargs in self.VARIANTS.items():
            with self.subTest(variant=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "v.fbx")
                    report = convert_file(
                        SAMPLE_FIGURE, path, ExportOptions(**kwargs), samples_library()
                    )
                    self.assertTrue(report.success, report.error)
                    info = run_ufbx_validate(path)
                    self.assertTrue(info["ok"], info.get("error"))
                    self.assertEqual(info["meshes"][0]["vertices"], 16556)
                    self.assertEqual(info["meshes"][0]["uv_sets"], ["UVMap"])

    def test_triangulated_face_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.fbx")
            report = convert_file(
                SAMPLE_FIGURE, path, ExportOptions(triangulate=True), samples_library()
            )
            self.assertTrue(report.success, report.error)
            info = run_ufbx_validate(path)
            self.assertTrue(info["ok"], info.get("error"))
            self.assertEqual(info["meshes"][0]["faces"], 16368 * 2)

    def test_z_up_still_weighed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "z.fbx")
            report = convert_file(
                SAMPLE_FIGURE,
                path,
                ExportOptions(up_axis="z"),
                samples_library(),
            )
            self.assertTrue(report.success, report.error)
            info = run_ufbx_validate(path)
            self.assertTrue(info["ok"], info.get("error"))
            self.assertAlmostEqual(info["meshes"][0]["weight_sum_max"], 1.0, places=3)


@unittest.skipUnless(have_samples(), "Daz sample assets not available")
@unittest.skipUnless(have_ufbx(), "ufbx package not available")
class MorphExportTests(unittest.TestCase):
    """Baked morphs must land on the mesh exactly as the source deltas specify."""

    @classmethod
    def setUpClass(cls):
        cls.figure = SAMPLE_G3F
        cls.morph = SAMPLE_MORPH
        if not (os.path.isfile(cls.figure) and os.path.isfile(cls.morph)):
            raise unittest.SkipTest("Genesis 3 figure + morph sample not available")
        cls.library = samples_library()

    def test_morph_file_is_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = convert_file(
                self.figure,
                os.path.join(tmp, "g3f.fbx"),
                ExportOptions(morph_files=[self.morph]),
                self.library,
            )
            self.assertTrue(report.success, report.error)
            self.assertEqual(report.scene_stats["morphs"], 1)

    def test_unknown_morph_is_reported(self):
        """Without a morph source there is nothing to apply -- say so."""
        with tempfile.TemporaryDirectory() as tmp:
            report = convert_file(
                self.figure,
                os.path.join(tmp, "g3f.fbx"),
                ExportOptions(morph_selections=[("FBMPearFigure", 1.0)]),
                self.library,
            )
            self.assertTrue(report.success, report.error)
            self.assertTrue(
                any("FBMPearFigure" in w for w in report.warnings),
                f"expected a morph-not-found warning, got {report.warnings}",
            )

    def test_morph_file_not_found_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = convert_file(
                self.figure,
                os.path.join(tmp, "g3f.fbx"),
                ExportOptions(morph_files=["/nonexistent/morph.dsf"]),
                self.library,
            )
            self.assertTrue(report.success, report.error)
            self.assertTrue(
                any("morph file not found" in w for w in report.warnings),
                report.warnings,
            )

    def test_baked_morph_matches_source_deltas(self):
        """Compare a morphed export against a plain one, via ufbx in a subprocess."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, weight in (("base", None), ("morphed", 1.0)):
                path = os.path.join(tmp, f"{name}.fbx")
                report = convert_file(
                    self.figure,
                    path,
                    ExportOptions(
                        morph_files=[self.morph],
                        morph_selections=[("FBMPearFigure", weight)] if weight else [],
                    ),
                    self.library,
                )
                self.assertTrue(report.success, report.error)
                paths[name] = path

            morphs = load_morphs_from_file(self.morph)
            self.assertTrue(morphs)
            deltas = {int(k): tuple(v) for k, v in morphs[0].deltas.items()}
            self.assertEqual(morphs[0].id, "FBMPearFigure")

            summary = compare_vertex_delta(paths["base"], paths["morphed"], deltas)
            self.assertTrue(summary["ok"], summary.get("error"))
            self.assertEqual(summary["checked"], len(deltas))
            self.assertGreater(summary["moved"], 10000)
            # ASCII FBX carries 7 decimals, so allow for float32 rounding.
            self.assertLess(summary["max_deviation"], 1e-4, summary)

    def test_blend_shapes_are_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g3f.fbx")
            report = convert_file(
                self.figure,
                path,
                ExportOptions(morph_files=[self.morph], export_blendshapes=True),
                self.library,
            )
            self.assertTrue(report.success, report.error)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn('"BlendShape"', text)
            self.assertIn("BlendShapeChannel", text)
            info = run_ufbx_validate(path)
            self.assertTrue(info["ok"], info.get("error"))


if __name__ == "__main__":
    unittest.main()
