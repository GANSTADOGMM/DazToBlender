"""Loading and caching of DSON documents (.duf / .dsf / .dsa / .dse).

Daz files are JSON, optionally gzip-compressed (most library .dsf files are).
Some older files are latin-1 or have a UTF-8 BOM, and a few ship with trailing
NUL padding, so the loader is deliberately forgiving.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, Optional

DSON_EXTENSIONS = (".duf", ".dsf", ".dson", ".daz", ".dsa", ".dse", ".dsb")


class DsonError(Exception):
    pass


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(2)
        fh.seek(0)
        data = fh.read()
    if head[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data)
        except (OSError, EOFError) as exc:
            # Truncated gzip: salvage what we can, some Daz files are padded.
            try:
                d = gzip.GzipFile(fileobj=_Buffer(data))
                return d.read()
            except Exception:
                raise DsonError("corrupt gzip in %s (%s)" % (path, exc))
    return data


class _Buffer:
    def __init__(self, data: bytes):
        import io
        self._b = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._b.read(n)

    def seek(self, *a):
        return self._b.seek(*a)

    def tell(self):
        return self._b.tell()


def decode(data: bytes, path: str = "") -> Dict[str, Any]:
    text = data.rstrip(b"\x00")
    for codec in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            s = text.decode(codec)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        raise DsonError("cannot decode %s" % path)
    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        # A handful of shipped assets contain NaN / trailing commas.
        cleaned = s.replace("-1.#IND", "0").replace("1.#INF", "0").replace("nan", "0")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise DsonError("invalid JSON in %s: %s" % (path or "<memory>", exc))


def load(path: str) -> Dict[str, Any]:
    """Load a single DSON document from disk (no caching)."""
    if not os.path.isfile(path):
        raise DsonError("file not found: %s" % path)
    return decode(read_bytes(path), path)


def is_dson_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in DSON_EXTENSIONS


def sniff(path: str) -> Optional[str]:
    """Return the DSON asset type ('scene', 'figure', 'modifier', ...).

    Reads only as much as needed to identify the file.
    """
    try:
        doc = load(path)
    except DsonError:
        return None
    info = doc.get("asset_info") or {}
    t = info.get("type")
    if t:
        return str(t)
    if "scene" in doc:
        return "scene"
    for key, name in (("geometry_library", "geometry"),
                      ("node_library", "node"),
                      ("modifier_library", "modifier"),
                      ("material_library", "material"),
                      ("uv_set_library", "uv_set")):
        if key in doc:
            return name
    return None


class DocumentCache:
    """Keeps parsed documents in memory for the duration of a conversion.

    Daz figures reference the same .dsf dozens of times (one per bone, per
    morph, ...), so caching turns an O(n) disk storm into a single read.
    """

    def __init__(self, max_documents: int = 512):
        self._docs: "Dict[str, Dict[str, Any]]" = {}
        self._order: list = []
        self.max_documents = max_documents
        self.hits = 0
        self.misses = 0

    def get(self, path: str) -> Dict[str, Any]:
        key = os.path.normcase(os.path.abspath(path))
        doc = self._docs.get(key)
        if doc is not None:
            self.hits += 1
            return doc
        self.misses += 1
        doc = load(path)
        self._docs[key] = doc
        self._order.append(key)
        if len(self._order) > self.max_documents:
            old = self._order.pop(0)
            self._docs.pop(old, None)
        return doc

    def clear(self) -> None:
        self._docs.clear()
        self._order.clear()
