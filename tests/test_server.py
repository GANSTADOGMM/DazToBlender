"""Tests for the local web app: multipart parsing, routing and safety."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request

from . import ROOT, SAMPLES_DIR, SAMPLE_G3F, have_samples  # noqa: F401

from dazconv import ExportOptions
from dazconv.library import ContentLibrary
from dazconv.server import (
    ConversionApp,
    JobStore,
    extract_boundary,
    make_server,
    parse_multipart,
)


def build_body(boundary: str, parts) -> bytes:
    """Assemble a multipart body from ``[(name, filename, content_type, data)]``."""
    body = bytearray()
    for name, filename, content_type, data in parts:
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        body += f"--{boundary}\r\n{disposition}\r\n".encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += data if isinstance(data, bytes) else data.encode()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body)


class MultipartTests(unittest.TestCase):
    BOUNDARY = "XyZbOuNdArY"

    def test_extracts_boundary(self):
        self.assertEqual(
            extract_boundary("multipart/form-data; boundary=XyZbOuNdArY"), b"XyZbOuNdArY"
        )
        self.assertEqual(extract_boundary("multipart/form-data"), b"")

    def test_extracts_boundary_with_quotes(self):
        self.assertEqual(
            extract_boundary('multipart/form-data; boundary="AbC"'), b"AbC"
        )

    def test_parses_text_and_binary_parts(self):
        body = build_body(self.BOUNDARY, [
            ("options", None, "application/json", b'{"triangulate": true}'),
            ("file", "model.duf", "application/octet-stream", b"\x1f\x8b\x00\xffbinary"),
        ])
        parts = parse_multipart(body, b"XyZbOuNdArY")
        self.assertEqual([p.name for p in parts], ["options", "file"])
        self.assertEqual(parts[0].filename, "")
        self.assertEqual(json.loads(parts[0].data), {"triangulate": True})
        self.assertEqual(parts[1].filename, "model.duf")
        self.assertEqual(parts[1].data, b"\x1f\x8b\x00\xffbinary")

    def test_binary_data_containing_crlf_is_preserved(self):
        payload = b"\x00\r\n\r\n\x01\x02\r\n\x03"
        body = build_body(self.BOUNDARY, [
            ("file", "a.dsf", "application/octet-stream", payload),
        ])
        parts = parse_multipart(body, b"XyZbOuNdArY")
        self.assertEqual(parts[0].data, payload)

    def test_quoted_filename_with_spaces_and_escapes(self):
        body = (
            b"--B\r\nContent-Disposition: form-data; name=\"f\"; "
            b'filename="my \\"figure\\".duf"\r\n\r\ndata\r\n--B--\r\n'
        )
        parts = parse_multipart(body, b"B")
        self.assertEqual(parts[0].filename, 'my "figure".duf')

    def test_part_without_name_is_dropped(self):
        body = b"--B\r\nContent-Disposition: form-data\r\n\r\nx\r\n--B--\r\n"
        self.assertEqual(parse_multipart(body, b"B"), [])

    def test_missing_boundary_returns_empty(self):
        self.assertEqual(parse_multipart(b"anything", b""), [])


class JobStoreTests(unittest.TestCase):
    def test_create_and_get(self):
        store = JobStore()
        job = store.create("figure.dsf")
        self.assertTrue(job.job_id)
        self.assertIs(store.get(job.job_id), job)
        self.assertIsNone(store.get("nope"))

    def test_status_shape(self):
        store = JobStore()
        job = store.create("figure.dsf")
        status = job.status()
        for key in ("job_id", "source", "stage", "fraction", "message", "done", "error"):
            self.assertIn(key, status)
        self.assertFalse(status["done"])
        self.assertNotIn("report", status)  # only present once finished


class ServerTests(unittest.TestCase):
    """Drive a real server on an ephemeral port."""

    @classmethod
    def setUpClass(cls):
        roots = [SAMPLES_DIR] if os.path.isdir(SAMPLES_DIR) else []
        library = ContentLibrary(roots=roots)
        cls.app = ConversionApp(library, ExportOptions())
        cls.httpd = make_server("127.0.0.1", 0, cls.app)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    # ------------------------------------------------------------- helpers --
    def get(self, path):
        return urllib.request.urlopen(self.base + path)

    def post_multipart(self, parts, boundary="B0und"):
        body = build_body(boundary, parts)
        request = urllib.request.Request(
            self.base + "/api/convert",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return urllib.request.urlopen(request)

    def post_json(self, path, payload):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request)

    def wait_for_job(self, job_id, timeout=180):
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.get(f"/api/job/{job_id}") as response:
                status = json.load(response)
            if status["done"]:
                return status
            time.sleep(0.15)
        self.fail(f"job {job_id} did not finish in {timeout}s")

    # --------------------------------------------------------------- routes --
    def test_index_is_served(self):
        with self.get("/") as response:
            body = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("dazconv", body)
        self.assertIn('id="dropzone"', body)

    def test_static_assets(self):
        for name, needle in (("app.js", "dazconv"), ("style.css", "dropzone")):
            with self.get(f"/static/{name}") as response:
                self.assertEqual(response.status, 200, name)
                self.assertIn(needle, response.read().decode())

    def test_static_traversal_is_refused(self):
        # urllib normalises "..", so send the encoded form the browser would not.
        for attempt in ("/static/%2e%2e/server.py", "/static/../../etc/passwd"):
            with self.assertRaises(urllib.error.HTTPError):
                self.get(attempt)

    def test_info(self):
        with self.get("/api/info") as response:
            info = json.load(response)
        self.assertTrue(info["ok"])
        self.assertEqual(info["outputs"], ["fbx"])
        self.assertIn(".duf", info["inputs"])
        self.assertIn("max_influences", info["default_options"])

    def test_unknown_route_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/does-not-exist")
        self.assertEqual(caught.exception.code, 404)

    def test_upload_requires_a_file(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_multipart([("options", None, "application/json", b"{}")])
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("no file", json.load(caught.exception)["error"])

    def test_invalid_options_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_multipart([
                ("options", None, "application/json", b'{"scale": 0}'),
                ("file", "x.dsf", "application/octet-stream", b"{}"),
            ])
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("scale", json.load(caught.exception)["error"])

    def test_bad_json_options_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_multipart([
                ("options", None, "application/json", b"{not json"),
                ("file", "x.dsf", "application/octet-stream", b"{}"),
            ])
        self.assertEqual(caught.exception.code, 400)

    def test_convert_path_requires_a_path(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_json("/api/convert-path", {})
        self.assertEqual(caught.exception.code, 400)

    def test_convert_path_missing_file_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_json("/api/convert-path", {"path": "/nonexistent/x.duf"})
        self.assertEqual(caught.exception.code, 404)

    def test_convert_path_outside_library_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post_json("/api/convert-path", {"path": "/etc/hostname"})
        self.assertEqual(caught.exception.code, 403)
        self.assertIn("outside the content library", json.load(caught.exception)["error"])

    def test_preview_before_completion_is_409(self):
        job = self.app.jobs.create("pending.duf")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get(f"/api/preview/{job.job_id}")
        self.assertEqual(caught.exception.code, 409)

    def test_unknown_job_is_404(self):
        for path in ("/api/job/", "/api/preview/", "/api/download/"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path + "deadbeef")
            self.assertEqual(caught.exception.code, 404, path)

    @unittest.skipUnless(have_samples(), "Daz sample assets not available")
    def test_full_upload_conversion_and_download(self):
        source = os.path.join(
            SAMPLES_DIR, "data", "DAZ 3D", "Genesis 8", "Female 8_1", "Genesis8_1Female.dsf"
        )
        with open(source, "rb") as handle:
            payload = handle.read()
        with self.post_multipart([
            ("options", None, "application/json", json.dumps({"max_influences": 4}).encode()),
            ("file", "Genesis8_1Female.dsf", "application/octet-stream", payload),
        ]) as response:
            job_id = json.load(response)["job_id"]

        status = self.wait_for_job(job_id)
        self.assertEqual(status["stage"], "done", status.get("error"))
        self.assertEqual(status["report"]["scene"]["vertices"], 16556)
        self.assertEqual(status["report"]["scene"]["bones"], 171)
        self.assertEqual(status["report"]["export"]["deformers"], 168)

        with self.get(f"/api/preview/{job_id}") as response:
            preview = json.load(response)
        mesh = preview["meshes"][0]
        self.assertEqual(len(mesh["vertices"]) // 3, 16556)
        self.assertEqual(len(mesh["triangles"]) // 3, 16368 * 2)
        self.assertEqual(len(preview["bones"]), 171)

        with self.get(f"/api/download/{job_id}") as response:
            fbx = response.read()
            self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
        self.assertTrue(fbx.startswith(b"; FBX 7.4.0"), fbx[:40])
        self.assertEqual(len(fbx), status["report"]["export"]["size_bytes"])

    @unittest.skipUnless(have_samples(), "Daz sample assets not available")
    def test_convert_path_inside_library(self):
        with self.post_json("/api/convert-path", {"path": SAMPLE_G3F, "options": {}}) as response:
            job_id = json.load(response)["job_id"]
        status = self.wait_for_job(job_id)
        self.assertEqual(status["stage"], "done", status.get("error"))
        self.assertEqual(status["report"]["scene"]["vertices"], 17418)


if __name__ == "__main__":
    unittest.main()
