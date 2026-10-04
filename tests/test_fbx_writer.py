"""Unit tests for the ASCII FBX serialiser.

These pin down the exact syntax that real FBX readers require, each case
derived from Autodesk FBX SDK output.
"""

from __future__ import annotations

import unittest

from . import ROOT  # noqa: F401

from dazconv.fbx.writer import FbxNode, escape_string, format_float, write_document


def payload_lines(text: str) -> list[str]:
    """The array payload lines: the ``a:`` line plus bare continuation lines."""
    lines = []
    seen_a = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("a: "):
            lines.append(stripped[3:])
            seen_a = True
        elif seen_a and stripped != "}":
            lines.append(stripped)
    return lines


def _array_payload(text: str) -> str:
    return "".join(payload_lines(text))


class ScalarTests(unittest.TestCase):
    def test_floats_never_use_exponent_notation(self):
        for value in (1e-5, 1e-9, 12345.6789, -0.000001, 1e8):
            text = format_float(value)
            self.assertNotIn("e", text.lower(), f"{value} formatted as {text}")

    def test_float_precision_retained(self):
        self.assertEqual(format_float(16.5252), "16.5252")
        self.assertEqual(format_float(0.0000153), "0.0000153")

    def test_zero_and_junk(self):
        self.assertEqual(format_float(0.0), "0")
        self.assertEqual(format_float(float("nan")), "0")
        self.assertEqual(format_float(float("inf")), "0")

    def test_escapes(self):
        self.assertEqual(escape_string('a"b'), 'a\\"b')
        self.assertEqual(escape_string("a\\b"), "a\\\\b")
        self.assertEqual(escape_string("a\nb"), "a b")


class NodeSyntaxTests(unittest.TestCase):
    def test_scalar_node(self):
        node = FbxNode("GeometryVersion", 124)
        self.assertEqual(node.serialize(), "GeometryVersion: 124")

    def test_string_value_is_written_verbatim(self):
        node = FbxNode("Name", '""')
        self.assertEqual(node.serialize(), 'Name: ""')

    def test_container_without_value(self):
        node = FbxNode("Objects")
        node.child("Inner", 1)
        self.assertEqual(node.serialize(), "Objects:  {\n\tInner: 1\n}")

    def test_object_header_uses_colon_space(self):
        """Real ASCII FBX: ``Geometry: <id>, "Geometry::x", "Mesh" {``."""
        node = FbxNode("Geometry", compound=(2000, "Geometry::cube", "Mesh"))
        node.child("GeometryVersion", 124)
        text = node.serialize()
        self.assertTrue(text.startswith('Geometry: 2000, "Geometry::cube", "Mesh" {'), text)

    def test_object_header_without_id(self):
        node = FbxNode("SceneInfo", compound=(None, "SceneInfo::GlobalInfo", "UserData"))
        node.child("Type", '"UserData"')
        self.assertTrue(node.serialize().startswith('SceneInfo: "SceneInfo::GlobalInfo", "UserData" {'))

    def test_property_lines_are_flat(self):
        props = FbxNode("Properties70")
        props.prop("UpAxis", "int", "Integer", "", 1)
        props.prop("DocumentUrl", "KString", "Url", "", "/tmp/a.duf")
        props.prop("Original", "Compound", "", "")
        props.prop("Original|ApplicationName", "KString", "", "", "dazconv")
        text = props.serialize()
        self.assertIn('P: "UpAxis", "int", "Integer", "",1', text)
        self.assertIn('P: "DocumentUrl", "KString", "Url", "", "/tmp/a.duf"', text)
        self.assertIn('P: "Original", "Compound", "", ""', text)
        # Compound members are siblings, never nested inside the parent P: line.
        self.assertNotIn('"" {', text)

    def test_array_single_line(self):
        node = FbxNode("Root")
        node.numbers("Vertices", [1.0, 2.0, 3.0])
        self.assertEqual(node.serialize(), "Root:  {\n\tVertices: *3 {\n\t\ta: 1,2,3\n\t}\n}")

    def test_int_array_has_no_decimals(self):
        node = FbxNode("Root")
        node.children.append(FbxNode("Indexes", array=("i", [0, 1, 2])))
        self.assertIn("a: 0,1,2", node.serialize())


class ArrayWrappingTests(unittest.TestCase):
    """Long arrays wrap -- but never inside a number, and never repeating a:."""

    def test_wrapping_keeps_numbers_intact(self):
        node = FbxNode("Root")
        values = [round(i * 0.123456, 6) for i in range(3000)]
        node.numbers("Vertices", values)
        text = node.serialize()
        self.assertEqual(text.count("a: "), 1, "the 'a:' prefix must appear exactly once")
        payload = _array_payload(text)
        self.assertGreater(len(payload_lines(text)), 1, "expected the payload to wrap")
        self.assertEqual(payload.split(","), [format_float(v) for v in values])

    def test_no_number_is_split_across_lines(self):
        node = FbxNode("Root")
        node.numbers("Weights", [16.5252 + i for i in range(600)])
        text = node.serialize()
        body_lines = [
            line.strip().replace("a: ", "", 1)
            for line in text.splitlines()
            if line.strip().startswith(("a:", "-", "0", "1", "2"))
        ]
        for chunk in body_lines:
            if chunk == "":
                continue
            # Every wrapped chunk must be a whole comma-separated token list.
            self.assertRegex(chunk, r"^-?[\d.]+(,-?[\d.]+)*,?$", f"broken chunk: {chunk!r}")

    def test_document_round_trip_shape(self):
        root = FbxNode("Takes")
        root.child("Current", '""')
        text = write_document([root], header_comment="hello")
        self.assertTrue(text.startswith("; hello\n"))
        self.assertIn('Takes:  {\n\tCurrent: ""\n}', text)


if __name__ == "__main__":
    unittest.main()
