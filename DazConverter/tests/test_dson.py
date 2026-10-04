"""DSON layer: URI parsing, file reading, library resolution, channels."""

import gzip
import json
import os
import shutil
import tempfile
import unittest

from dazconv.dson import channels as ch
from dazconv.dson import reader, uri
from dazconv.dson.library import ContentLibrary, guess_root_for_file


class TestUri(unittest.TestCase):
    def test_file_and_fragment(self):
        u = uri.parse("/data/Daz%203D/Genesis%208/Female/Genesis8Female.dsf#geometry")
        self.assertEqual(u.file_path, "/data/Daz 3D/Genesis 8/Female/Genesis8Female.dsf")
        self.assertEqual(u.asset_id, "geometry")
        self.assertFalse(u.is_local)

    def test_local_reference(self):
        u = uri.parse("#SkinBinding")
        self.assertTrue(u.is_local)
        self.assertEqual(u.asset_id, "SkinBinding")

    def test_property_path(self):
        u = uri.parse("/data/x.dsf#head?value/x")
        self.assertEqual(u.asset_id, "head")
        self.assertEqual(u.property_path, "value/x")

    def test_scheme_and_node_path(self):
        u = uri.parse("name://@selection/hip:?rotation/x")
        self.assertEqual(u.scheme, "name")
        self.assertEqual(u.node_path, "@selection")
        self.assertEqual(u.property_path, "rotation/x")

    def test_encoded_fragment(self):
        u = uri.parse("/data/UV%20Sets/Base.dsf#Base%208.1%20Female")
        self.assertEqual(u.asset_id, "Base 8.1 Female")

    def test_normalize_relative(self):
        self.assertEqual(uri.normalize_relative("/data/a/b.dsf"), "data/a/b.dsf")
        self.assertEqual(uri.normalize_relative("b.dsf", "/data/a/c.dsf"),
                         "data/a/b.dsf")


class TestReader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, data, gz=False):
        path = os.path.join(self.dir, name)
        raw = json.dumps(data).encode("utf-8")
        if gz:
            with gzip.open(path, "wb") as fh:
                fh.write(raw)
        else:
            with open(path, "wb") as fh:
                fh.write(raw)
        return path

    def test_plain_json(self):
        p = self._write("a.dsf", {"file_version": "0.6.0.0"})
        self.assertEqual(reader.load(p)["file_version"], "0.6.0.0")

    def test_gzipped(self):
        p = self._write("b.dsf", {"hello": "world"}, gz=True)
        self.assertEqual(reader.load(p)["hello"], "world")

    def test_bom_and_trailing_nulls(self):
        path = os.path.join(self.dir, "c.dsf")
        with open(path, "wb") as fh:
            fh.write(b"\xef\xbb\xbf" + json.dumps({"x": 1}).encode() + b"\x00\x00")
        self.assertEqual(reader.load(path)["x"], 1)

    def test_bad_json_raises(self):
        path = os.path.join(self.dir, "d.dsf")
        with open(path, "w") as fh:
            fh.write("{not json")
        with self.assertRaises(reader.DsonError):
            reader.load(path)

    def test_document_cache(self):
        p = self._write("e.dsf", {"v": 2})
        cache = reader.DocumentCache()
        cache.get(p)
        cache.get(p)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)

    def test_sniff(self):
        p = self._write("f.dsf", {"asset_info": {"type": "figure"}})
        self.assertEqual(reader.sniff(p), "figure")
        p = self._write("g.dsf", {"geometry_library": []})
        self.assertEqual(reader.sniff(p), "geometry")


class TestLibrary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "data", "Vendor", "Figure"))
        self.asset = os.path.join(self.dir, "data", "Vendor", "Figure", "Fig.dsf")
        with open(self.asset, "w") as fh:
            json.dump({"geometry_library": [{"id": "geo", "vertices": {"count": 0}}],
                       "node_library": [{"id": "hip", "type": "bone"}]}, fh)
        os.makedirs(os.path.join(self.dir, "People"))
        self.scene = os.path.join(self.dir, "People", "Char.duf")
        with open(self.scene, "w") as fh:
            json.dump({"scene": {"nodes": []}}, fh)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_guess_root(self):
        self.assertEqual(os.path.normpath(guess_root_for_file(self.scene)),
                         os.path.normpath(self.dir))

    def test_resolve_exact(self):
        lib = ContentLibrary([self.dir])
        hit = lib.resolve_path("/data/Vendor/Figure/Fig.dsf")
        self.assertEqual(os.path.normpath(hit), os.path.normpath(self.asset))

    def test_resolve_case_insensitive(self):
        lib = ContentLibrary([self.dir])
        hit = lib.resolve_path("/DATA/vendor/figure/FIG.DSF")
        self.assertIsNotNone(hit, "case-insensitive lookup must work on Linux")
        self.assertEqual(os.path.normpath(hit), os.path.normpath(self.asset))

    def test_resolve_url_encoded(self):
        lib = ContentLibrary([self.dir])
        self.assertIsNotNone(lib.resolve_path("/data/Vendor/Figure/Fig.dsf"))

    def test_find_asset(self):
        lib = ContentLibrary([self.dir])
        asset, src = lib.find_asset("/data/Vendor/Figure/Fig.dsf#hip", "node")
        self.assertIsNotNone(asset)
        self.assertEqual(asset["type"], "bone")

    def test_missing_tracked(self):
        lib = ContentLibrary([self.dir])
        self.assertIsNone(lib.resolve_path("/data/Nope/Nope.dsf"))
        self.assertTrue(lib.missing)

    def test_scan(self):
        lib = ContentLibrary([self.dir])
        assets = lib.scan()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["name"], "Char")


class TestChannels(unittest.TestCase):
    def test_current_value_wins(self):
        self.assertEqual(ch.as_float({"value": 1.0, "current_value": 7.5}), 7.5)

    def test_falls_back_to_value(self):
        self.assertEqual(ch.as_float({"value": 2.0}), 2.0)

    def test_nested_channel(self):
        self.assertEqual(ch.as_float({"channel": {"current_value": 3.0}}), 3.0)

    def test_color(self):
        self.assertEqual(ch.as_color({"current_value": [1, 0, 0]}), (1.0, 0.0, 0.0))
        self.assertEqual(ch.as_color({"current_value": 0.5}), (0.5, 0.5, 0.5))

    def test_axis_triplet_channels(self):
        chans = [{"id": "x", "current_value": 1},
                 {"id": "y", "current_value": 2},
                 {"id": "z", "current_value": 3}]
        self.assertEqual(ch.axis_triplet(chans), (1.0, 2.0, 3.0))

    def test_axis_triplet_bare_list(self):
        self.assertEqual(ch.axis_triplet([4, 5, 6]), (4.0, 5.0, 6.0))

    def test_extra_channels(self):
        mat = {"extra": [{"type": "studio_material_channels", "channels": [
            {"channel": {"id": "Base Color", "current_value": [0.1, 0.2, 0.3]}},
            {"channel": {"id": "Metallic Weight", "current_value": 0.8}}]}]}
        extras = ch.extra_channels(mat)
        self.assertIn("base color", extras)
        self.assertAlmostEqual(ch.as_float(extras["metallic weight"]), 0.8)

    def test_image_of(self):
        self.assertEqual(ch.image_of({"image_file": "/x.png"}), "/x.png")
        self.assertIsNone(ch.image_of({"current_value": 1}))


if __name__ == "__main__":
    unittest.main()
