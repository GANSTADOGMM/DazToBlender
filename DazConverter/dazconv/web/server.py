"""The local web app: a zero-dependency HTTP server + JSON API.

Everything runs on the user's machine; nothing is uploaded anywhere.  The
browser is only used as the UI toolkit (and as a 3D viewport via three.js,
which is vendored into ``static/vendor`` so the app works offline).
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import posixpath
import re
import shutil
import socket
import threading
import time
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .. import config, samples
from ..api import inspect_file
from ..dson import ContentLibrary, autodetect_roots
from ..export import FORMATS
from ..version import APP_NAME, APP_TAGLINE, __version__
from .jobs import JobQueue

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB

QUEUE = JobQueue(workers=1)


class Router:
    def __init__(self):
        self.routes: List[Tuple[str, str, Any, Callable]] = []

    def add(self, method: str, pattern: str, handler: Callable) -> None:
        regex = re.compile("^" + re.sub(r"<(\w+)>", r"(?P<\1>[^/]+)", pattern) + "$")
        self.routes.append((method, pattern, regex, handler))

    def match(self, method: str, path: str):
        for m, _pattern, regex, handler in self.routes:
            if m != method:
                continue
            found = regex.match(path)
            if found:
                return handler, found.groupdict()
        return None, {}


ROUTER = Router()


def route(method: str, pattern: str):
    def deco(fn):
        ROUTER.add(method, pattern, fn)
        return fn
    return deco


# ------------------------------------------------------------------ handlers
@route("GET", "/api/state")
def api_state(req, params):
    cfg = config.load()
    detected = autodetect_roots()
    demo_root = samples.default_location()
    return {
        "app": {"name": APP_NAME, "version": __version__, "tagline": APP_TAGLINE},
        "config": cfg,
        "detectedContentDirs": detected,
        "formats": [{"key": k, **v} for k, v in FORMATS.items()],
        "demo": {"root": demo_root,
                 "scene": os.path.join(demo_root, "People", "DemoFigure.duf"),
                 "exists": os.path.isfile(os.path.join(demo_root, "People",
                                                       "DemoFigure.duf"))},
        "home": os.path.expanduser("~"),
        "separator": os.sep,
        "jobs": [j.to_json() for j in QUEUE.list()],
    }


@route("POST", "/api/settings")
def api_settings(req, params):
    body = req.json_body()
    updates = {k: v for k, v in body.items() if k in config.DEFAULTS}
    if "content_dirs" in updates:
        updates["content_dirs"] = [os.path.abspath(os.path.expanduser(d))
                                   for d in updates["content_dirs"]
                                   if isinstance(d, str) and d.strip()]
    return {"config": config.save(updates)}


@route("POST", "/api/browse")
def api_browse(req, params):
    body = req.json_body()
    path = body.get("path") or os.path.expanduser("~")
    path = os.path.abspath(os.path.expanduser(path))
    show_files = bool(body.get("files", True))
    if not os.path.isdir(path):
        path = os.path.dirname(path) or os.path.expanduser("~")
    entries = []
    try:
        for name in sorted(os.listdir(path), key=lambda s: s.lower()):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            is_dir = os.path.isdir(full)
            if not is_dir:
                if not show_files:
                    continue
                if not name.lower().endswith((".duf", ".dsf")):
                    continue
            entries.append({
                "name": name, "path": full, "dir": is_dir,
                "size": (os.path.getsize(full) if not is_dir else 0),
            })
    except OSError as exc:
        return {"error": str(exc), "path": path, "entries": []}
    parent = os.path.dirname(path)
    return {"path": path, "parent": parent if parent != path else None,
            "entries": entries,
            "isContentRoot": _looks_like_content_root(path)}


def _looks_like_content_root(path: str) -> bool:
    try:
        names = {n.lower() for n in os.listdir(path)}
    except OSError:
        return False
    return "data" in names or "runtime" in names


@route("POST", "/api/inspect")
def api_inspect(req, params):
    body = req.json_body()
    path = body.get("path") or ""
    if not os.path.isfile(path):
        return {"error": "file not found: %s" % path}
    return inspect_file(path)


@route("POST", "/api/demo")
def api_demo(req, params):
    body = req.json_body()
    manifest = samples.generate(samples.default_location(),
                                overwrite=bool(body.get("force")))
    cfg = config.load()
    dirs = list(cfg.get("content_dirs") or [])
    if manifest["root"] not in dirs:
        dirs.append(manifest["root"])
        config.save({"content_dirs": dirs})
    return {"manifest": manifest, "config": config.load()}


@route("POST", "/api/jobs")
def api_jobs_create(req, params):
    body = req.json_body()
    paths = body.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    formats = body.get("formats") or ["fbx"]
    options = body.get("options") or {}
    output_dir = body.get("outputDir") or config.load().get("output_dir")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    content_dirs = list(options.get("content_dirs") or [])
    for d in config.effective_content_dirs():
        if d not in content_dirs:
            content_dirs.append(d)
    options["content_dirs"] = content_dirs

    embed = bool(options.pop("embed_textures", False))
    copy_tex = bool(options.pop("copy_textures", True))
    options["export_options"] = {
        "fbx": {"embed_textures": embed, "copy_textures": copy_tex},
        "glb": {"embed_textures": True, "copy_textures": False},
        "gltf": {"embed_textures": embed, "copy_textures": copy_tex},
        "obj": {"copy_textures": copy_tex},
    }

    created = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(str(p)))
        if not os.path.isfile(p):
            continue
        job = QUEUE.submit(p, list(formats), dict(options), output_dir)
        config.add_recent(p)
        created.append(job.to_json())
    if not created:
        return {"error": "no valid input files", "jobs": []}
    return {"jobs": created}


@route("GET", "/api/jobs")
def api_jobs_list(req, params):
    return {"jobs": [j.to_json() for j in QUEUE.list()]}


@route("GET", "/api/jobs/<job_id>")
def api_job_get(req, params):
    job = QUEUE.get(params["job_id"])
    if job is None:
        return {"error": "unknown job"}
    return job.to_json()


@route("POST", "/api/jobs/<job_id>/cancel")
def api_job_cancel(req, params):
    return {"cancelled": QUEUE.cancel(params["job_id"])}


@route("POST", "/api/jobs/clear")
def api_jobs_clear(req, params):
    return {"removed": QUEUE.clear_finished()}


@route("GET", "/api/preview/<job_id>")
def api_preview(req, params):
    job = QUEUE.get(params["job_id"])
    if job is None or not job.preview or not os.path.isfile(job.preview):
        req.send_error(404, "no preview")
        return None
    req.send_file(job.preview, "model/gltf-binary")
    return None


@route("GET", "/api/download/<job_id>")
def api_download(req, params):
    job = QUEUE.get(params["job_id"])
    if job is None:
        req.send_error(404, "unknown job")
        return None
    wanted = req.query.get("file", [None])[0]
    if wanted:
        for out in job.outputs:
            if os.path.basename(out["path"]) == wanted and os.path.isfile(out["path"]):
                req.send_file(out["path"], "application/octet-stream",
                              download_name=wanted)
                return None
        req.send_error(404, "file not in job")
        return None

    # no file specified: zip everything (model files + textures folder)
    buf = io.BytesIO()
    base_dir = job.output_dir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen = set()

        def add(path: str) -> None:
            key = os.path.normcase(path)
            if key in seen or not os.path.isfile(path):
                return
            seen.add(key)
            zf.write(path, os.path.basename(path))

        for out in job.outputs:
            add(out["path"])
            stem = os.path.splitext(out["path"])[0]
            add(stem + ".mtl")
            add(stem + ".bin")
        tex_dir = os.path.join(base_dir, "textures")
        if os.path.isdir(tex_dir):
            for name in sorted(os.listdir(tex_dir)):
                full = os.path.join(tex_dir, name)
                if os.path.isfile(full):
                    zf.write(full, "textures/" + name)
    data = buf.getvalue()
    req.send_bytes(data, "application/zip",
                   download_name="%s.zip" % os.path.splitext(job.name)[0])
    return None


@route("POST", "/api/upload")
def api_upload(req, params):
    files = req.multipart_files()
    if not files:
        return {"error": "no files received", "paths": []}
    session = os.path.join(config.data_dir(), "uploads", time.strftime("%Y%m%d-%H%M%S"))
    saved: List[str] = []
    for rel_name, data in files:
        safe = _safe_relpath(rel_name)
        dest = os.path.join(session, safe)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        if dest.lower().endswith((".duf", ".dsf")):
            saved.append(dest)
    # make the upload folder usable as a content root
    cfg = config.load()
    dirs = list(cfg.get("content_dirs") or [])
    root = _upload_content_root(session)
    if root and root not in dirs:
        dirs.append(root)
        config.save({"content_dirs": dirs})
    return {"paths": saved, "root": session, "count": len(files)}


def _upload_content_root(session: str) -> Optional[str]:
    """If the user dropped a whole library folder, find its root."""
    for dirpath, dirnames, _files in os.walk(session):
        lowered = {d.lower() for d in dirnames}
        if "data" in lowered or "runtime" in lowered:
            return dirpath
    return session


def _safe_relpath(name: str) -> str:
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/")
             if p not in ("", ".", "..") and not p.startswith("/")]
    return os.path.join(*parts) if parts else "upload.bin"


@route("POST", "/api/reveal")
def api_reveal(req, params):
    """Open the output folder in the OS file manager (best effort)."""
    body = req.json_body()
    path = body.get("path") or config.load().get("output_dir")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    try:
        import subprocess
        import sys as _sys
        if _sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"opened": True, "path": path}
    except Exception as exc:
        return {"opened": False, "path": path, "error": str(exc)}


@route("GET", "/api/scan")
def api_scan(req, params):
    dirs = config.effective_content_dirs()
    limit = int(req.query.get("limit", ["300"])[0])
    lib = ContentLibrary(dirs)
    assets = lib.scan(max_files=limit)
    return {"roots": lib.roots, "assets": assets, "count": len(assets)}


# -------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "%s/%s" % (APP_NAME, __version__)
    protocol_version = "HTTP/1.1"

    # --------------------------------------------------------- request utils
    def log_message(self, fmt, *args):  # quieter console
        if os.environ.get("DAZCONV_VERBOSE"):
            super().log_message(fmt, *args)

    @property
    def query(self) -> Dict[str, List[str]]:
        return parse_qs(urlparse(self.path).query)

    def body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_UPLOAD:
            raise ValueError("upload too large")
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def json_body(self) -> Dict[str, Any]:
        raw = self.body_bytes()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def multipart_files(self) -> List[Tuple[str, bytes]]:
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return []
        match = re.search(r'boundary="?([^";]+)"?', ctype)
        if not match:
            return []
        boundary = ("--" + match.group(1)).encode("ascii")
        body = self.body_bytes()
        out: List[Tuple[str, bytes]] = []
        for part in body.split(boundary):
            if not part or part in (b"--\r\n", b"--"):
                continue
            part = part.lstrip(b"\r\n")
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            headers = part[:header_end].decode("utf-8", "replace")
            data = part[header_end + 4:]
            if data.endswith(b"\r\n"):
                data = data[:-2]
            name_match = re.search(r'filename="([^"]*)"', headers)
            if not name_match or not name_match.group(1):
                continue
            path_match = re.search(r'name="([^"]*)"', headers)
            rel = name_match.group(1)
            if path_match and "/" in path_match.group(1):
                rel = path_match.group(1)
            out.append((rel, data))
        return out

    # ------------------------------------------------------------ responders
    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, ctype: str,
                   download_name: Optional[str] = None,
                   cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % download_name)
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: str, ctype: Optional[str] = None,
                  download_name: Optional[str] = None) -> None:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            self.send_error(404, "not found")
            return
        if ctype is None:
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_bytes(data, ctype, download_name)

    # ---------------------------------------------------------------- verbs
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        handler, params = ROUTER.match("GET", path)
        if handler:
            self._dispatch(handler, params)
            return
        self._serve_static(path)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        handler, params = ROUTER.match("POST", path)
        if handler:
            self._dispatch(handler, params)
            return
        self.send_error(404, "no such endpoint")

    def _dispatch(self, handler, params):
        try:
            result = handler(self, params)
        except Exception as exc:  # never kill the server on a bad request
            import traceback
            traceback.print_exc()
            self.send_json({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)
            return
        if result is not None:
            self.send_json(result)

    def _serve_static(self, path: str):
        if path in ("/", "", "/index.html"):
            rel = "index.html"
        else:
            rel = posixpath.normpath(unquote(path)).lstrip("/")
        full = os.path.join(STATIC_DIR, rel)
        if not os.path.abspath(full).startswith(os.path.abspath(STATIC_DIR)):
            self.send_error(403, "forbidden")
            return
        if not os.path.isfile(full):
            self.send_error(404, "not found")
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif full.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif full.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        try:
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError:
            self.send_error(404, "not found")
            return
        cache = "/vendor/" in path
        self.send_bytes(data, ctype, cache=cache)


def _free_port(host: str, port: int) -> int:
    for candidate in [port] + [port + i for i in range(1, 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return port


def serve(host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    mimetypes.add_type("model/gltf-binary", ".glb")
    mimetypes.add_type("model/gltf+json", ".gltf")
    port = _free_port(host, port)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("0.0.0.0", "127.0.0.1") else host, port)
    print("%s %s" % (APP_NAME, __version__))
    print(APP_TAGLINE)
    print("\n  -> %s\n" % url)
    print("Output folder : %s" % config.load().get("output_dir"))
    dirs = config.effective_content_dirs()
    print("Content dirs  : %s" % (", ".join(dirs) if dirs else "none configured yet"))
    print("\nPress Ctrl+C to stop.")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
