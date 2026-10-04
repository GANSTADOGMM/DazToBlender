#!/usr/bin/env python3
"""Local web app for the Daz to FBX converter.

Pure standard library -- no Flask, no cgi (removed in Python 3.13), no build
step.  Start it with::

    python -m dazconv.server --port 8000 --library-root "/path/to/DAZ 3D"

then open the printed URL.  The browser UI supports drag & drop, live 3D
preview and every export option the CLI exposes.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .convert import convert_file
from .library import ContentLibrary, library_from_cli
from .options import ExportOptions
from .preview import scene_preview
from .scene import SceneLoader

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MB
JOB_TTL_SECONDS = 60 * 60  # jobs and their outputs live for an hour


# ------------------------------------------------------------- multipart --
@dataclass
class UploadPart:
    name: str
    filename: str
    content_type: str
    data: bytes


def _unquote_token(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace("\\\\", "\\").replace('\\"', '"')


def parse_multipart(body: bytes, boundary: bytes) -> List[UploadPart]:
    """Split a ``multipart/form-data`` body into parts.

    The stdlib ``cgi`` module used to do this but was removed in Python 3.13, so
    the format is handled directly.  Parts are returned as raw bytes, which keeps
    binary uploads (and JSON option blobs) intact.
    """
    if not boundary:
        return []
    delimiter = b"--" + boundary
    parts: List[UploadPart] = []
    for chunk in body.split(delimiter):
        if not chunk or chunk.startswith(b"--"):
            continue
        chunk = chunk.lstrip(b"\r\n")
        head, separator, data = chunk.partition(b"\r\n\r\n")
        if not separator:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]

        name = ""
        filename = ""
        content_type = "application/octet-stream"
        for line in head.split(b"\r\n"):
            lowered = line.lower()
            if lowered.startswith(b"content-disposition:"):
                for token in line.split(b";")[1:]:
                    token = token.strip()
                    if token.lower().startswith(b"name="):
                        name = _unquote_token(token[5:])
                    elif token.lower().startswith(b"filename="):
                        filename = _unquote_token(token[9:])
            elif lowered.startswith(b"content-type:"):
                content_type = line.split(b":", 1)[1].decode("utf-8", "replace").strip()
        if not name:
            continue
        parts.append(UploadPart(name, filename, content_type, data))
    return parts


def extract_boundary(content_type: str) -> bytes:
    for token in content_type.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            return token[9:].strip().strip('"').encode("ascii", "replace")
    return b""


# ------------------------------------------------------------------ jobs --
@dataclass
class Job:
    job_id: str
    source_name: str
    stage: str = "queued"
    fraction: float = 0.0
    message: str = ""
    done: bool = False
    report: Optional[dict] = None
    preview: Optional[dict] = None
    error: str = ""
    output_path: str = ""
    created: float = field(default_factory=time.time)

    def status(self) -> dict:
        payload = {
            "job_id": self.job_id,
            "source": self.source_name,
            "stage": self.stage,
            "fraction": round(self.fraction, 4),
            "message": self.message,
            "done": self.done,
            "error": self.error,
            "has_output": bool(self.output_path and os.path.isfile(self.output_path)),
        }
        if self.done:
            payload["report"] = self.report
        return payload


class JobStore:
    """In-memory job registry with a temp workspace per conversion."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.workspace = tempfile.mkdtemp(prefix="dazconv-")

    def create(self, source_name: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], source_name=source_name)
        with self._lock:
            self._jobs[job.job_id] = job
            self._expire_locked()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _expire_locked(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        for job_id in [key for key, job in self._jobs.items() if job.created < cutoff]:
            job = self._jobs.pop(job_id)
            if job.output_path and os.path.isfile(job.output_path):
                try:
                    os.unlink(job.output_path)
                except OSError:
                    pass

    def job_dir(self, job: Job) -> str:
        path = os.path.join(self.workspace, job.job_id)
        os.makedirs(path, exist_ok=True)
        return path


# --------------------------------------------------------------- the app --
class ConversionApp:
    """Owns the library and runs conversions in worker threads."""

    def __init__(self, library: ContentLibrary, options: ExportOptions) -> None:
        self.library = library
        self.default_options = options
        self.jobs = JobStore()

    def info(self) -> dict:
        return {
            "version": __version__,
            "python": sys.version.split()[0],
            "content_roots": list(self.library.roots),
            "inputs": [".duf", ".daz", ".dsf", ".obj"],
            "outputs": ["fbx"],
            "default_options": self.default_options.as_dict(),
        }

    def assets(self, limit: int, extension: str) -> List[dict]:
        extensions = [extension] if extension else (".duf", ".daz", ".dsf", ".obj")
        try:
            found = self.library.scan_assets(extensions, limit=limit)
        except OSError:
            return []
        return found

    # ------------------------------------------------------------ running --
    def start(
        self,
        options: ExportOptions,
        source_path: str,
        source_name: str,
        job: Optional[Job] = None,
    ) -> Job:
        job = job or self.jobs.create(source_name)

        def run() -> None:
            try:
                self._run(job, options, source_path)
            except Exception as error:  # surfaced verbatim in the browser
                job.error = f"{type(error).__name__}: {error}"
                job.done = True
                job.stage = "failed"

        threading.Thread(target=run, daemon=True, name=f"convert-{job.job_id}").start()
        return job

    def _run(self, job: Job, options: ExportOptions, source_path: str) -> None:
        def progress(stage: str, fraction: float, message: str = "") -> None:
            job.stage = stage
            job.fraction = fraction
            job.message = message

        progress("load", 0.05, os.path.basename(source_path))
        report = convert_file(source_path, "", options, self.library, progress)

        job.report = report.as_dict()
        if not report.success:
            job.error = report.error or "conversion failed"
            job.stage = "failed"
            job.done = True
            return

        job.output_path = report.output_path
        job.stage = "preview"
        job.fraction = 0.95
        job.message = "building preview"
        try:
            scene = SceneLoader(
                library=self.library,
                flip_v=options.flip_uv,
                max_influences=options.max_influences,
                min_weight=options.min_weight,
            ).load(source_path)
            # Morphs attached during conversion are not in a fresh load, so the
            # preview reflects the exported geometry only when none were baked.
            job.preview = scene_preview(scene, options)
        except Exception as error:
            # A preview failure must not fail an otherwise good conversion.
            job.preview = {"error": f"{type(error).__name__}: {error}", "meshes": []}

        job.stage = "done"
        job.fraction = 1.0
        job.message = "finished"
        job.done = True

    def save_upload(self, job: Job, filename: str, data: bytes) -> str:
        safe = os.path.basename(filename or "input.duf") or "input.duf"
        path = os.path.join(self.jobs.job_dir(job), safe)
        with open(path, "wb") as handle:
            handle.write(data)
        return path


# --------------------------------------------------------------- routing --
def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


class RequestHandler(BaseHTTPRequestHandler):
    server_version = f"dazconv/{__version__}"
    protocol_version = "HTTP/1.1"
    app: ConversionApp  # injected by make_server

    # ------------------------------------------------------------- plumbing --
    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        if getattr(self.server, "verbose", False):
            sys.stderr.write(f"[dazconv] {self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str,
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status)

    # ------------------------------------------------------------------ GET --
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")

        if path == "/api/info":
            return self._send_json({"ok": True, **self.app.info()})

        if path == "/api/assets":
            try:
                limit = min(int(query.get("limit", ["500"])[0]), 5000)
            except ValueError:
                limit = 500
            extension = query.get("ext", [""])[0].lower()
            if extension and not extension.startswith("."):
                extension = "." + extension
            return self._send_json({"ok": True, "assets": self.app.assets(limit, extension)})

        if path.startswith("/api/job/"):
            job = self.app.jobs.get(path.rsplit("/", 1)[-1])
            if job is None:
                return self._send_error_json(404, "unknown job")
            return self._send_json({"ok": True, **job.status()})

        if path.startswith("/api/preview/"):
            job = self.app.jobs.get(path.rsplit("/", 1)[-1])
            if job is None:
                return self._send_error_json(404, "unknown job")
            if not job.done:
                return self._send_error_json(409, "conversion is still running")
            return self._send_json(job.preview or {"meshes": [], "bones": []})

        if path.startswith("/api/download/"):
            job = self.app.jobs.get(path.rsplit("/", 1)[-1])
            if job is None:
                return self._send_error_json(404, "unknown job")
            if not job.output_path or not os.path.isfile(job.output_path):
                return self._send_error_json(404, "no output for this job")
            return self._stream_file(job.output_path)

        return self._send_error_json(404, f"no route for {path}")

    def _serve_static(self, name: str) -> None:
        # Reject traversal and anything outside the webui folder.
        safe = os.path.normpath(name).lstrip("/\\")
        if safe.startswith("..") or os.path.isabs(safe):
            return self._send_error_json(400, "bad path")
        full = os.path.join(STATIC_DIR, safe)
        if not os.path.isfile(full):
            return self._send_error_json(404, f"no such file: {name}")
        content_type, _ = mimetypes.guess_type(full)
        with open(full, "rb") as handle:
            body = handle.read()
        self._send(200, body, content_type or "application/octet-stream")

    def _stream_file(self, path: str) -> None:
        name = os.path.basename(path)
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ----------------------------------------------------------------- POST --
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._send_error_json(400, "empty request body")
        if length > MAX_UPLOAD_BYTES:
            return self._send_error_json(
                413, f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )

        body = self.rfile.read(length)

        if path == "/api/convert":
            return self._handle_upload(body)
        if path == "/api/convert-path":
            return self._handle_path(body)
        return self._send_error_json(404, f"no route for {path}")

    def _options_from(self, raw: bytes) -> Tuple[ExportOptions, str]:
        """Parse the client's options blob, falling back to the server defaults."""
        if not raw or not raw.strip():
            return ExportOptions.from_dict(self.app.default_options.as_dict()), ""
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError) as error:
            return ExportOptions(), f"options are not valid json: {error}"
        options = ExportOptions.from_dict(data if isinstance(data, dict) else {})
        problems = options.validate()
        if problems:
            return options, "; ".join(problems)
        return options, ""

    def _handle_upload(self, body: bytes) -> None:
        boundary = extract_boundary(self.headers.get("Content-Type", ""))
        if not boundary:
            return self._send_error_json(400, "expected multipart/form-data")
        parts = parse_multipart(body, boundary)
        uploads = [part for part in parts if part.filename]
        if not uploads:
            return self._send_error_json(400, "no file uploaded")
        option_parts = [part for part in parts if part.name == "options"]
        options, problem = self._options_from(option_parts[0].data if option_parts else b"")
        if problem:
            return self._send_error_json(400, problem)

        first = uploads[0]
        job = self.app.jobs.create(first.filename)
        source = self.app.save_upload(job, first.filename, first.data)
        self.app.start(options, source, first.filename, job)
        self._send_json({"ok": True, "job_id": job.job_id})

    def _handle_path(self, body: bytes) -> None:
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError) as error:
            return self._send_error_json(400, f"invalid json: {error}")
        if not isinstance(data, dict):
            return self._send_error_json(400, "expected a json object")
        source = str(data.get("path", "")).strip()
        if not source:
            return self._send_error_json(400, "path is required")
        source = os.path.abspath(os.path.expanduser(source))
        if not os.path.isfile(source):
            return self._send_error_json(404, f"file not found: {source}")
        # Only files inside a configured content root may be opened this way.
        roots = [os.path.abspath(root) for root in self.app.library.roots]
        if roots and not any(
            source == root or source.startswith(root + os.sep) for root in roots
        ):
            return self._send_error_json(403, "path is outside the content library")
        options, problem = self._options_from(_json_bytes(data.get("options", {})))
        if problem:
            return self._send_error_json(400, problem)
        job = self.app.start(options, source, os.path.basename(source))
        self._send_json({"ok": True, "job_id": job.job_id})


# ----------------------------------------------------------------- startup --
def make_server(host: str, port: int, app: ConversionApp, verbose: bool = False):
    handler = type(
        "BoundRequestHandler", (RequestHandler,), {"app": app}
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.verbose = verbose  # type: ignore[attr-defined]
    return httpd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dazconv.server",
        description="Run the dazconv web app (drag & drop, live preview, FBX export).",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose it)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    parser.add_argument("-L", "--library-root", action="append", default=[],
                        help="Daz content folder (repeatable)")
    parser.add_argument("--config", default="", help="library config json path")
    parser.add_argument("--triangulate", action="store_true", help="default to triangles")
    parser.add_argument("--max-influences", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true", help="log every request")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    library = library_from_cli(args.library_root, args.config)
    options = ExportOptions(
        triangulate=args.triangulate, max_influences=args.max_influences
    )
    app = ConversionApp(library, options)

    try:
        httpd = make_server(args.host, args.port, app, args.verbose)
    except OSError as error:
        print(f"error: cannot bind {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1

    shown = "localhost" if args.host in ("0.0.0.0", "::", "") else args.host
    print(f"dazconv {__version__} web app")
    print(f"  url:            http://{shown}:{args.port}/")
    print(f"  python:         {sys.version.split()[0]}")
    if library.roots:
        for root in library.roots:
            print(f"  content root:   {root}")
    else:
        print("  content root:   (none - drop files in the browser to convert them)")
    print(f"  workspace:      {app.jobs.workspace}")
    print("  press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
