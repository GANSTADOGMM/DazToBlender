"""Content library handling: mapping DSON paths to real files on disk.

Daz content is addressed by paths that are relative to one of several
"content directory" roots (e.g. ``/data/Daz 3D/Genesis 8/...``).  A scene file
may pull geometry, morphs, UV sets and textures from different roots, so the
resolver searches every known root and caches the result.

The resolver is also case-insensitive: Daz authors on Windows and the casing
inside DSON files frequently disagrees with the casing on disk, which breaks
every naive port to macOS/Linux.
"""

from __future__ import annotations

import os
import posixpath
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import reader, uri as uri_mod

# Marker directories that identify a Daz content-directory root.
ROOT_MARKERS = ("data", "runtime", "people", "props", "environments", "scenes")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".tga", ".bmp",
                    ".exr", ".hdr", ".webp")


def _candidate_user_roots() -> List[str]:
    home = os.path.expanduser("~")
    out = [
        os.path.join(home, "Documents", "DAZ 3D", "Studio", "My Library"),
        os.path.join(home, "Documents", "My DAZ 3D Library"),
        os.path.join(home, "DAZ 3D", "Studio", "My Library"),
        os.path.join(home, "My DAZ 3D Library"),
    ]
    if sys.platform.startswith("win"):
        for drive in ("C:", "D:", "E:"):
            out.append(drive + os.sep + os.path.join("Users", "Public",
                                                     "Documents", "My DAZ 3D Library"))
            out.append(drive + os.sep + "My DAZ 3D Library")
            out.append(drive + os.sep + os.path.join("Program Files", "DAZ 3D",
                                                     "DAZStudio4", "content"))
    elif sys.platform == "darwin":
        out.append("/Users/Shared/My DAZ 3D Library")
        out.append("/Applications/DAZ 3D/DAZStudio4/content")
    return out


def autodetect_roots() -> List[str]:
    """Best-effort discovery of installed Daz content directories."""
    found: List[str] = []
    for p in _candidate_user_roots():
        if os.path.isdir(p) and p not in found:
            found.append(os.path.abspath(p))
    found.extend(r for r in _read_daz_prefs() if r not in found)
    return found


def _read_daz_prefs() -> List[str]:
    """Read content dirs out of Daz Studio's own preferences, when present."""
    roots: List[str] = []
    candidates = []
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        candidates.append(os.path.join(appdata, "DAZ 3D", "Studio4", "dsx",
                                       "ContentDirectoryManager.dsx"))
    elif sys.platform == "darwin":
        candidates.append(os.path.join(home, "Library", "Application Support",
                                       "DAZ 3D", "Studio4", "dsx",
                                       "ContentDirectoryManager.dsx"))
    else:
        candidates.append(os.path.join(home, ".daz3d", "Studio4", "dsx",
                                       "ContentDirectoryManager.dsx"))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
            for el in tree.iter():
                p = el.get("PATH") or el.get("path")
                if p and os.path.isdir(p):
                    roots.append(os.path.abspath(p))
        except Exception:
            continue
    return roots


def guess_root_for_file(path: str) -> Optional[str]:
    """Walk up from a .duf/.dsf file looking for the content-directory root.

    A root is a directory that contains a ``data`` or ``Runtime`` subfolder.
    """
    cur = os.path.dirname(os.path.abspath(path))
    last = None
    for _ in range(12):
        if cur == last:
            break
        try:
            entries = {e.lower() for e in os.listdir(cur)}
        except OSError:
            break
        if "data" in entries or "runtime" in entries:
            return cur
        last, cur = cur, os.path.dirname(cur)
    return None


class ContentLibrary:
    """Resolves DSON paths/URIs against a set of content directory roots."""

    def __init__(self, roots: Optional[Iterable[str]] = None,
                 cache: Optional[reader.DocumentCache] = None):
        self.roots: List[str] = []
        self.cache = cache or reader.DocumentCache()
        self._path_cache: Dict[str, Optional[str]] = {}
        self._dir_listings: Dict[str, Dict[str, str]] = {}
        self.missing: List[str] = []
        for r in roots or []:
            self.add_root(r)

    # ------------------------------------------------------------------ roots
    def add_root(self, root: str) -> bool:
        if not root:
            return False
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            return False
        if root not in self.roots:
            self.roots.append(root)
            self._path_cache.clear()
        return True

    def add_root_for_file(self, path: str) -> Optional[str]:
        root = guess_root_for_file(path)
        if root:
            self.add_root(root)
        # Also allow sibling-relative lookups from the file's own folder.
        self.add_root(os.path.dirname(os.path.abspath(path)))
        return root

    # ------------------------------------------------------------- resolution
    def _listing(self, directory: str) -> Dict[str, str]:
        """Case-folded name -> real name map for a directory."""
        key = os.path.normpath(directory)
        got = self._dir_listings.get(key)
        if got is None:
            try:
                got = {e.lower(): e for e in os.listdir(key)}
            except OSError:
                got = {}
            self._dir_listings[key] = got
        return got

    def _resolve_ci(self, root: str, rel: str) -> Optional[str]:
        """Case-insensitive join of `rel` onto `root`."""
        cur = root
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        for i, part in enumerate(parts):
            direct = os.path.join(cur, part)
            if os.path.exists(direct):
                cur = direct
                continue
            real = self._listing(cur).get(part.lower())
            if real is None:
                return None
            cur = os.path.join(cur, real)
        return cur if os.path.exists(cur) else None

    def resolve_path(self, dson_path: str, base_file: Optional[str] = None) -> Optional[str]:
        """Map a content-relative DSON path to an absolute file on disk."""
        if not dson_path:
            return None
        raw = dson_path.replace("\\", "/")
        # Already absolute and present (scene files sometimes embed abs paths).
        if os.path.isabs(dson_path) and os.path.isfile(dson_path):
            return os.path.abspath(dson_path)
        key = raw + "|" + (base_file or "")
        if key in self._path_cache:
            return self._path_cache[key]

        rel = uri_mod.normalize_relative(raw, base_file)
        result = None
        for root in self.roots:
            hit = self._resolve_ci(root, rel)
            if hit and os.path.isfile(hit):
                result = hit
                break
        if result is None and base_file:
            # relative to the referring document
            sibling = os.path.normpath(os.path.join(os.path.dirname(base_file),
                                                    uri_mod.unquote(raw).lstrip("/")))
            if os.path.isfile(sibling):
                result = sibling
        self._path_cache[key] = result
        if result is None and rel not in self.missing:
            self.missing.append(rel)
        return result

    def resolve_image(self, dson_path: str, base_file: Optional[str] = None) -> Optional[str]:
        """Like resolve_path but tolerant of swapped image extensions."""
        hit = self.resolve_path(dson_path, base_file)
        if hit:
            return hit
        rel = uri_mod.normalize_relative(dson_path.replace("\\", "/"), base_file)
        stem, _ext = posixpath.splitext(rel)
        for ext in IMAGE_EXTENSIONS:
            hit = self.resolve_path(stem + ext, base_file)
            if hit:
                return hit
        return None

    # ----------------------------------------------------------- doc / assets
    def load_document(self, path: str) -> Dict[str, Any]:
        return self.cache.get(path)

    def resolve_document(self, dson_uri: str,
                         base_file: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return (document, absolute_path) for the file part of a URI."""
        u = uri_mod.parse(dson_uri)
        if not u.file_path:
            return None, None
        abspath = self.resolve_path(u.file_path, base_file)
        if not abspath:
            return None, None
        try:
            return self.load_document(abspath), abspath
        except reader.DsonError:
            return None, abspath

    LIBRARY_KEYS = {
        "geometry": "geometry_library",
        "node": "node_library",
        "modifier": "modifier_library",
        "material": "material_library",
        "image": "image_library",
        "uv_set": "uv_set_library",
    }

    def find_asset(self, dson_uri: str, kind: str,
                   base_file: Optional[str] = None,
                   local_doc: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Look up an asset definition by URI.

        Returns (asset_dict, defining_file_path).  `kind` is one of the keys of
        LIBRARY_KEYS.  `local_doc` is used when the URI has no file part.
        """
        u = uri_mod.parse(dson_uri)
        lib_key = self.LIBRARY_KEYS.get(kind, kind)
        if u.file_path:
            doc, abspath = self.resolve_document(dson_uri, base_file)
        else:
            doc, abspath = local_doc, base_file
        if not doc:
            return None, abspath
        asset = _find_in_library(doc, lib_key, u.asset_id)
        if asset is None:
            # Some files declare assets in a differently-named library.
            for alt in self.LIBRARY_KEYS.values():
                asset = _find_in_library(doc, alt, u.asset_id)
                if asset is not None:
                    break
        return asset, abspath

    # ------------------------------------------------------------------ index
    def scan(self, max_files: int = 100000) -> List[Dict[str, Any]]:
        """Index user-facing asset files (.duf) under every root."""
        out: List[Dict[str, Any]] = []
        for root in self.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if d.lower() not in ("data", "runtime", "api")
                               and not d.startswith(".")]
                for fn in filenames:
                    if fn.lower().endswith(".duf"):
                        full = os.path.join(dirpath, fn)
                        out.append({
                            "path": full,
                            "name": os.path.splitext(fn)[0],
                            "root": root,
                            "rel": os.path.relpath(full, root).replace(os.sep, "/"),
                            "size": _safe_size(full),
                        })
                        if len(out) >= max_files:
                            return out
        return out


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _find_in_library(doc: Dict[str, Any], lib_key: str, asset_id: str) -> Optional[Dict[str, Any]]:
    lib = doc.get(lib_key)
    if not isinstance(lib, list):
        return None
    if not asset_id:
        return lib[0] if lib else None
    for entry in lib:
        if isinstance(entry, dict) and entry.get("id") == asset_id:
            return entry
    # ids are sometimes URL-encoded in one place and not the other
    from urllib.parse import unquote
    target = unquote(asset_id)
    for entry in lib:
        if isinstance(entry, dict) and unquote(str(entry.get("id", ""))) == target:
            return entry
    return None
