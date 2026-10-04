"""Background conversion queue for the web app."""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..api import convert
from ..convert import ConvertOptions
from ..export import FORMATS


@dataclass
class Job:
    id: str
    source: str
    name: str
    formats: List[str]
    options: Dict[str, Any]
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "Waiting"
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    output_dir: str = ""
    preview: str = ""               # path to a .glb used by the 3D viewport
    morphs: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "name": self.name,
            "formats": self.formats,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "outputs": [{"path": o["path"],
                         "name": os.path.basename(o["path"]),
                         "format": o.get("format", ""),
                         "bytes": o.get("bytes", 0)} for o in self.outputs],
            "warnings": self.warnings,
            "summary": self.summary,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
            "outputDir": self.output_dir,
            "hasPreview": bool(self.preview),
            "morphs": self.morphs,
            "duration": round((self.finished or time.time()) - self.created, 2),
        }


class JobQueue:
    """A tiny single-worker queue -- conversions are CPU and memory heavy,
    so running them one at a time keeps the machine responsive."""

    def __init__(self, workers: int = 1):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._pending: List[str] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cancelled: set = set()
        self._threads = [threading.Thread(target=self._run, daemon=True,
                                          name="dazconv-worker-%d" % i)
                         for i in range(max(1, workers))]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ API
    def submit(self, source: str, formats: List[str], options: Dict[str, Any],
               output_dir: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], source=source,
                  name=os.path.basename(source),
                  formats=[f for f in formats if f in FORMATS] or ["fbx"],
                  options=options, output_dir=output_dir)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._pending.append(job.id)
        self._wake.set()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> List[Job]:
        with self._lock:
            ids = self._order[-limit:]
        return [self._jobs[i] for i in ids if i in self._jobs][::-1]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in ("done", "error", "cancelled"):
            return False
        self._cancelled.add(job_id)
        if job.status == "queued":
            job.status = "cancelled"
            job.message = "Cancelled"
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
        return True

    def clear_finished(self) -> int:
        with self._lock:
            keep = []
            removed = 0
            for jid in self._order:
                job = self._jobs.get(jid)
                if job and job.status in ("done", "error", "cancelled"):
                    self._jobs.pop(jid, None)
                    removed += 1
                else:
                    keep.append(jid)
            self._order = keep
        return removed

    # --------------------------------------------------------------- worker
    def _next(self) -> Optional[Job]:
        with self._lock:
            while self._pending:
                jid = self._pending.pop(0)
                job = self._jobs.get(jid)
                if job and job.status == "queued":
                    return job
        return None

    def _run(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            self._execute(job)

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.progress = 0.01
        job.message = "Reading DSON"

        def report(frac: float, msg: str) -> None:
            if job.id in self._cancelled:
                raise _Cancelled()
            job.progress = max(job.progress, min(0.98, frac))
            job.message = msg

        try:
            opts = ConvertOptions(**{k: v for k, v in job.options.items()
                                     if k in ConvertOptions.__dataclass_fields__})
            exp = job.options.get("export_options") or {}
            result = convert(job.source, job.output_dir, formats=job.formats,
                             options=opts, export_options=exp, progress=report)
            job.outputs = result.outputs
            job.warnings = result.warnings
            job.summary = result.summary
            job.morphs = _morph_names(result.scene)
            job.preview = self._make_preview(job, result)
            job.status = "done"
            job.progress = 1.0
            job.message = "Converted in %.1fs" % result.duration
            if not result.outputs:
                job.status = "error"
                job.error = "nothing to export (no geometry found)"
                job.message = job.error
        except _Cancelled:
            job.status = "cancelled"
            job.message = "Cancelled"
        except Exception as exc:
            job.status = "error"
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.message = job.error
            job.warnings.append(traceback.format_exc(limit=3))
        finally:
            job.finished = time.time()
            self._cancelled.discard(job.id)

    def _make_preview(self, job: Job, result) -> str:
        """Reuse an exported .glb, or generate one just for the viewport."""
        for out in result.outputs:
            if out.get("format") == "glb":
                return out["path"]
        try:
            from ..export.gltf import GltfOptions, GltfWriter
            from .. import config as cfg
            preview_dir = os.path.join(cfg.data_dir(), "previews")
            os.makedirs(preview_dir, exist_ok=True)
            path = os.path.join(preview_dir, "%s.glb" % job.id)
            opts = GltfOptions(binary=True, embed_textures=True,
                               copy_textures=False)
            GltfWriter(result.scene, opts).write(path)
            return path
        except Exception:
            return ""


def _morph_names(scene) -> List[str]:
    if scene is None:
        return []
    names: List[str] = []
    for mesh in scene.meshes():
        for m in mesh.morphs:
            if m.name not in names:
                names.append(m.name)
    return names


class _Cancelled(Exception):
    pass
