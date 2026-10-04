"""High-level API, conformed figures, CLI and the local web server."""

import io
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from dazconv import samples
from dazconv.api import convert, inspect_file, scene_dependencies
from dazconv.convert import ConvertOptions, SceneBuilder
from dazconv.dson import ContentLibrary
from dazconv.export.fbx import binary


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.lib = os.path.join(cls.dir, "lib")
        cls.manifest = samples.generate(cls.lib, overwrite=True)
        cls.opts = ConvertOptions(content_dirs=[cls.lib])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_convert_single_format(self):
        out = os.path.join(self.dir, "single", "a.fbx")
        res = convert(self.manifest["scene"], out, options=self.opts)
        self.assertTrue(res.ok)
        self.assertEqual(len(res.outputs), 1)
        self.assertEqual(res.outputs[0]["path"], out)
        self.assertGreater(res.summary["vertices"], 0)

    def test_convert_to_directory(self):
        out = os.path.join(self.dir, "multi")
        res = convert(self.manifest["scene"], out, formats=["fbx", "obj"],
                      options=self.opts)
        names = sorted(os.path.basename(p) for p in res.paths)
        self.assertEqual(names, ["DemoFigure.fbx", "DemoFigure.obj"])

    def test_format_inferred_from_extension(self):
        out = os.path.join(self.dir, "infer", "b.glb")
        res = convert(self.manifest["scene"], out, options=self.opts)
        self.assertEqual(res.outputs[0]["format"], "glb")

    def test_progress_callback(self):
        seen = []
        convert(self.manifest["scene"], os.path.join(self.dir, "prog", "c.obj"),
                options=self.opts, progress=lambda f, m: seen.append((f, m)))
        self.assertTrue(seen)
        self.assertAlmostEqual(seen[-1][0], 1.0)
        self.assertTrue(all(0.0 <= f <= 1.0 for f, _ in seen))

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            convert("/nope/nope.duf", "/tmp/x.fbx")

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            convert(self.manifest["scene"], "/tmp/x", formats=["stl"])

    def test_describe(self):
        res = convert(self.manifest["scene"],
                      os.path.join(self.dir, "desc", "d.obj"), options=self.opts)
        text = res.describe()
        self.assertIn("vertices", text)
        self.assertIn(".obj", text)

    def test_inspect_scene(self):
        info = inspect_file(self.manifest["scene"])
        self.assertEqual(info["type"], "scene")
        self.assertTrue(info["has_scene"])
        self.assertEqual(len(info["figures"]), 1)
        self.assertEqual(info["content_root"], os.path.abspath(self.lib))

    def test_inspect_asset_file(self):
        info = inspect_file(self.manifest["figure"])
        self.assertEqual(info["type"], "figure")
        self.assertFalse(info["has_scene"])
        self.assertGreater(info["geometry_count"], 0)

    def test_dependencies_all_found(self):
        deps = scene_dependencies(self.manifest["scene"], [self.lib])
        self.assertGreater(deps["total"], 0)
        self.assertEqual(deps["missingCount"], 0, deps["missing"])

    def test_dependencies_reports_missing(self):
        """A scene copied away from its library reports what it needs."""
        orphan_dir = os.path.join(self.dir, "orphan")
        os.makedirs(orphan_dir, exist_ok=True)
        orphan = os.path.join(orphan_dir, "DemoFigure.duf")
        shutil.copy2(self.manifest["scene"], orphan)
        deps = scene_dependencies(orphan, ["/definitely/not/here"])
        self.assertGreater(deps["missingCount"], 0)
        self.assertTrue(any("DemoFigure.dsf" in m for m in deps["missing"]))

    def test_conversion_without_library_warns_clearly(self):
        orphan_dir = os.path.join(self.dir, "orphan2")
        os.makedirs(orphan_dir, exist_ok=True)
        orphan = os.path.join(orphan_dir, "DemoFigure.duf")
        shutil.copy2(self.manifest["scene"], orphan)
        res = convert(orphan, os.path.join(self.dir, "orphan2", "x.fbx"),
                      options=ConvertOptions(content_dirs=[]))
        self.assertFalse(res.ok)
        joined = " ".join(res.warnings)
        self.assertIn("content file(s) could not be found", joined)
        self.assertIn("DemoFigure.dsf", joined)

    def test_scale_option(self):
        opts = ConvertOptions(content_dirs=[self.lib], scale=2.0)
        res = convert(self.manifest["scene"],
                      os.path.join(self.dir, "scaled", "e.obj"), options=opts)
        mesh = res.scene.figures[0].mesh
        self.assertGreater(max(v[1] for v in mesh.vertices), 300)  # ~170cm * 2

    def test_merge_meshes_option(self):
        opts = ConvertOptions(content_dirs=[self.lib], merge_meshes=True)
        scene = SceneBuilder(ContentLibrary([self.lib]), opts) \
            .build_from_file(self.manifest["scene"])
        self.assertEqual(len(scene.figures), 1)


class TestConformedFigures(unittest.TestCase):
    """A second figure that conforms to the first must inherit its rig."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.manifest = samples.generate(cls.dir, overwrite=True)
        cls.scene_path = os.path.join(cls.dir, "People", "WithJacket.duf")
        cls._write_outfit()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def _write_outfit(cls):
        data_rel = "data/DazConv/Jacket"
        jacket = {
            "file_version": "0.6.0.0",
            "asset_info": {"id": "/%s/Jacket.dsf" % data_rel, "type": "figure"},
            "geometry_library": [{
                "id": "JacketGeom", "name": "Jacket", "type": "polygon_mesh",
                "vertices": {"count": 8, "values": [
                    [-10, 110, -6], [10, 110, -6], [10, 140, -6], [-10, 140, -6],
                    [-10, 110, 6], [10, 110, 6], [10, 140, 6], [-10, 140, 6]]},
                "polygon_groups": {"count": 1, "values": ["torso"]},
                "polygon_material_groups": {"count": 1, "values": ["Cloth"]},
                "polylist": {"count": 6, "values": [
                    [0, 0, 0, 1, 2, 3], [0, 0, 5, 4, 7, 6], [0, 0, 4, 5, 1, 0],
                    [0, 0, 3, 2, 6, 7], [0, 0, 4, 0, 3, 7], [0, 0, 1, 5, 6, 2]]},
            }],
            "node_library": [{
                "id": "Jacket", "name": "Jacket", "type": "figure", "label": "Jacket",
                "center_point": [{"id": "x", "value": 0}, {"id": "y", "value": 0},
                                 {"id": "z", "value": 0}],
                "geometries": [{"id": "JacketGeom", "url": "#JacketGeom"}],
            }],
        }
        path = os.path.join(cls.dir, data_rel, "Jacket.dsf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(jacket, fh)

        with open(cls.manifest["scene"]) as fh:
            scene = json.load(fh)
        scene["scene"]["nodes"].append({
            "id": "Jacket",
            "url": "/%s/Jacket.dsf#Jacket" % data_rel,
            "name": "Jacket", "label": "Jacket",
            "parent": "#DemoFigure",
            "conform_target": "#DemoFigure",
            "geometries": [{"id": "JacketGeom",
                            "url": "/%s/Jacket.dsf#JacketGeom" % data_rel}],
            "preview": {"type": "figure"},
        })
        with open(cls.scene_path, "w") as fh:
            json.dump(scene, fh)

    def test_both_figures_converted(self):
        opts = ConvertOptions(content_dirs=[self.dir])
        scene = SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.scene_path)
        names = sorted(f.name for f in scene.figures)
        self.assertEqual(names, ["DemoFigure", "Jacket"])

    def test_jacket_inherits_skeleton(self):
        opts = ConvertOptions(content_dirs=[self.dir], merge_skeletons=True)
        scene = SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.scene_path)
        jacket = next(f for f in scene.figures if f.name == "Jacket")
        figure = next(f for f in scene.figures if f.name == "DemoFigure")
        self.assertIsNotNone(jacket.skeleton)
        self.assertIs(jacket.skeleton, figure.skeleton)
        self.assertEqual(jacket.kind, "conformed")

    def test_fbx_shares_one_armature(self):
        out = os.path.join(self.dir, "out", "both.fbx")
        convert(self.scene_path, out, formats=["fbx"],
                options=ConvertOptions(content_dirs=[self.dir]))
        doc = binary.parse(out)
        models = [c for c in doc.find("Objects").children if c.name == b"Model"]
        self.assertEqual(sum(1 for m in models if m.props[2] == b"Mesh"), 2)
        # bones must not be duplicated for the conformed item
        self.assertEqual(sum(1 for m in models if m.props[2] == b"LimbNode"),
                         len(samples.SKELETON))
        self.assertEqual(sum(1 for m in models if m.props[2] == b"Null"), 1)

    def test_merge_keeps_rig_weights_and_morphs(self):
        opts = ConvertOptions(content_dirs=[self.dir], merge_meshes=True)
        scene = SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.scene_path)
        self.assertEqual(len(scene.figures), 1)
        fig = scene.figures[0]
        self.assertIsNotNone(fig.skeleton)
        self.assertIsNotNone(fig.mesh.skin)
        self.assertEqual(len(fig.mesh.skin.influences), len(fig.mesh.vertices))
        self.assertEqual(sorted(m.name for m in fig.mesh.morphs),
                         ["HeadScale", "Muscular"])
        self.assertEqual(len(fig.mesh.materials), 4)  # 3 body + jacket
        body_verts = 586
        self.assertEqual(len(fig.mesh.vertices), body_verts + 8)
        # body vertices stay weighted, the unskinned jacket does not invent any
        self.assertTrue(fig.mesh.skin.influences[0])
        self.assertEqual(fig.mesh.skin.influences[-1], [])
        for inf in fig.mesh.skin.influences[:body_verts]:
            self.assertAlmostEqual(sum(w for _, w in inf), 1.0, places=5)
        for morph in fig.mesh.morphs:
            self.assertTrue(all(i < len(fig.mesh.vertices) for i in morph.indices))

    def test_default_material_for_unpainted_surface(self):
        opts = ConvertOptions(content_dirs=[self.dir])
        scene = SceneBuilder(ContentLibrary([self.dir]), opts) \
            .build_from_file(self.scene_path)
        jacket = next(f for f in scene.figures if f.name == "Jacket")
        self.assertEqual(len(jacket.mesh.materials), 1)
        self.assertEqual(jacket.mesh.materials[0].name, "Cloth")


class TestCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, args):
        import contextlib
        from dazconv.cli import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(args)
        return code, out.getvalue()

    def test_formats_command(self):
        code, text = self._run(["formats"])
        self.assertEqual(code, 0)
        self.assertIn("fbx", text)
        self.assertIn("glb", text)

    def test_demo_and_convert(self):
        lib = os.path.join(self.dir, "lib")
        code, text = self._run(["demo", "-o", lib, "--force"])
        self.assertEqual(code, 0)
        scene = os.path.join(lib, "People", "DemoFigure.duf")
        self.assertTrue(os.path.isfile(scene))

        out = os.path.join(self.dir, "out")
        code, text = self._run(["convert", scene, "-o", out, "-f", "fbx,obj",
                                "-c", lib, "--quiet"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(out, "DemoFigure.fbx")))
        self.assertTrue(os.path.isfile(os.path.join(out, "DemoFigure.obj")))

    def test_convert_rejects_unknown_format(self):
        code, _ = self._run(["convert", "x.duf", "-f", "blend"])
        self.assertEqual(code, 2)

    def test_inspect_json(self):
        lib = os.path.join(self.dir, "lib2")
        self._run(["demo", "-o", lib, "--force"])
        scene = os.path.join(lib, "People", "DemoFigure.duf")
        code, text = self._run(["inspect", scene, "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["type"], "scene")

    def test_no_args_prints_help(self):
        code, text = self._run([])
        self.assertEqual(code, 0)
        self.assertIn("usage", text.lower())


class TestWebServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dazconv.web import server as srv
        cls.srv = srv
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def _get(self, path):
        with urllib.request.urlopen(self._url(path), timeout=20) as r:
            return r.status, r.read()

    def _post(self, path, payload):
        req = urllib.request.Request(
            self._url(path), data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def test_index_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"DazConv", body)

    def test_static_assets(self):
        for path in ("/app.css", "/js/app.js", "/js/viewer.js",
                     "/vendor/three.module.min.js",
                     "/vendor/jsm/loaders/GLTFLoader.js"):
            status, body = self._get(path)
            self.assertEqual(status, 200, path)
            self.assertTrue(body, path)

    def test_state_endpoint(self):
        status, body = self._get("/api/state")
        data = json.loads(body)
        self.assertEqual(data["app"]["name"], "DazConv")
        self.assertEqual(len(data["formats"]), 4)
        self.assertIn("config", data)

    def test_directory_traversal_blocked(self):
        try:
            self._get("/../../../../etc/passwd")
        except urllib.error.HTTPError as exc:
            self.assertIn(exc.code, (403, 404))
        else:
            self.fail("traversal should not succeed")

    def test_browse(self):
        data = self._post("/api/browse", {"path": os.path.expanduser("~"),
                                          "files": False})
        self.assertIn("entries", data)
        self.assertTrue(all(e["dir"] for e in data["entries"]))

    def test_full_conversion_through_api(self):
        demo = self._post("/api/demo", {"force": True})
        scene = demo["manifest"]["scene"]
        res = self._post("/api/jobs", {"paths": [scene],
                                       "formats": ["fbx", "glb"],
                                       "options": {}})
        self.assertTrue(res["jobs"])
        job_id = res["jobs"][0]["id"]

        import time
        for _ in range(100):
            status, body = self._get("/api/jobs/%s" % job_id)
            job = json.loads(body)
            if job["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.1)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(len(job["outputs"]), 2)
        self.assertTrue(job["hasPreview"])
        self.assertEqual(job["summary"]["bones"], len(samples.SKELETON))

        status, glb = self._get("/api/preview/%s" % job_id)
        self.assertEqual(status, 200)
        self.assertEqual(glb[:4], b"glTF")

        status, zip_bytes = self._get("/api/download/%s" % job_id)
        self.assertEqual(status, 200)
        import zipfile
        names = zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()
        self.assertTrue(any(n.endswith(".fbx") for n in names))
        self.assertEqual(len(names), len(set(names)), "no duplicate zip entries")

    def test_bad_job_returns_error(self):
        res = self._post("/api/jobs", {"paths": ["/nope/nope.duf"],
                                       "formats": ["fbx"]})
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
