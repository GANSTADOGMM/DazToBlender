"""Unit tests for DSON container sniffing and the tolerant text parser."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
import zlib

from . import ROOT  # noqa: F401  (import side effect: path setup)

from dazconv import dson


class SniffTests(unittest.TestCase):
    def test_detects_gzip(self):
        payload = gzip.compress(b'{"a": 1}')
        self.assertEqual(dson.sniff_container(payload), "gzip")

    def test_detects_zlib(self):
        payload = zlib.compress(b'{"a": 1}')
        self.assertEqual(dson.sniff_container(payload), "zlib")

    def test_detects_plain_json(self):
        self.assertEqual(dson.sniff_container(b'{\n\t"a": 1\n}'), "plain")

    def test_detects_binary_dzsf(self):
        self.assertEqual(dson.sniff_container(b"DZSF\x01\x00\x00\x00"), "binary")

    def test_binary_raises_helpful_error(self):
        with self.assertRaises(dson.BinaryDsonError) as ctx:
            dson.load_dson(b"DZSF\x01\x00\x00\x00")
        self.assertIn("DZSF", str(ctx.exception))

    def test_empty_input_rejected(self):
        with self.assertRaises(dson.DsonError):
            dson.sniff_container(b"")


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(dson.load_dson(b'{"file_version": "0.6.0.0"}')["file_version"], "0.6.0.0")

    def test_gzipped_json(self):
        payload = gzip.compress(json.dumps({"geometry_library": [{"id": "g"}]}).encode())
        parsed = dson.load_dson(payload)
        self.assertEqual(parsed["geometry_library"][0]["id"], "g")

    def test_utf8_bom_tolerated(self):
        payload = b"\xef\xbb\xbf" + b'{"ok": true}'
        self.assertTrue(dson.load_dson(payload)["ok"])

    def test_comments_stripped(self):
        text = b'{\n // a line comment\n "a": 1, # trailing comment\n "b": "keep # hash"\n}'
        parsed = dson.load_dson(text)
        self.assertEqual(parsed, {"a": 1, "b": "keep # hash"})

    def test_trailing_comma_repaired(self):
        parsed = dson.load_dson(b'{"a": [1, 2, 3,], "b": {"c": 4,},}')
        self.assertEqual(parsed["a"], [1, 2, 3])
        self.assertEqual(parsed["b"], {"c": 4})

    def test_garbage_raises_dson_error(self):
        with self.assertRaises(dson.DsonError):
            dson.load_dson(b"this is not dson at all")

    def test_missing_file_raises(self):
        with self.assertRaises(dson.DsonError):
            dson.load_dson_file("/nonexistent/path.duf")


class RepoSceneTests(unittest.TestCase):
    """The repo ships a real gzipped .duf -- parse it for real."""

    def setUp(self):
        self.path = os.path.join(ROOT, "Test", "QA-Test-Scene-01.duf")
        if not os.path.isfile(self.path):
            self.skipTest("QA-Test-Scene-01.duf not present")

    def test_repo_scene_is_gzipped_dson(self):
        with open(self.path, "rb") as handle:
            self.assertEqual(dson.sniff_container(handle.read(4)), "gzip")

    def test_repo_scene_parses(self):
        document = dson.load_dson_file(self.path)
        self.assertEqual(document["file_version"], "0.6.0.0")
        self.assertIn("scene", document)
        self.assertTrue(document["scene"]["nodes"])


if __name__ == "__main__":
    unittest.main()
