"""Daz content-directory resolution.

Daz references assets by *content-relative URL* rather than by disk path::

    /data/Daz%203D/Genesis%208/Female%208_1/Genesis8_1Female.dsf#Genesis81FemaleGeom
    /Runtime/Textures/DAZ/Characters/Genesis8_1/FemaleBase/G8_1FBaseBodyMapD.jpg

Those URLs are relative to one or more *content roots* (the folders listed in
Daz Studio's Content Directory Manager).  This module maps a URL to a real
file given a set of roots, with a case-insensitive fallback index for
libraries that were authored on Windows.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

CONFIG_FILENAME = "config.json"
TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".tga", ".exr", ".hdr", ".psd")
GEOMETRY_EXTENSIONS = (".dsf", ".duf", ".obj")

#: Locations Daz Install Manager uses by default, per platform.
DEFAULT_ROOT_CANDIDATES = [
    os.path.join("C:", os.sep, "Users", "Public", "Documents", "My DAZ 3D Library"),
    os.path.join("/Users/Shared/My DAZ 3D Library"),
    os.path.expanduser("~/Documents/DAZ 3D/Studio/My Library"),
    os.path.expanduser("~/My DAZ 3D Library"),
    os.path.expanduser("~/Documents/My DAZ 3D Library"),
]


def normalize_url(url: str) -> str:
    """Turn a Daz content URL into a clean relative path.

    * drops the ``#fragment`` used to address an object inside a file
    * percent-decodes (``%20`` -> space)
    * converts Windows separators and strips the leading slash
    """
    if not url:
        return ""
    path = url.split("#", 1)[0].split("?", 1)[0]
    path = urllib.parse.unquote(path)
    path = path.replace("\\", "/")
    return path.lstrip("/")


def split_uri(uri: str) -> Tuple[str, str]:
    """Split ``file#ObjectId`` into ``(file, object_id)``.

    The fragment is percent-decoded: Daz writes object ids escaped, so
    ``...Female.dsf#Base%208.1%20Female`` addresses the object whose id is the
    literal string ``Base 8.1 Female``.
    """
    if not uri:
        return "", ""
    file_part, _, fragment = uri.partition("#")
    return file_part, urllib.parse.unquote(fragment)


@dataclass
class ContentLibrary:
    """A set of content roots plus a lazy case-insensitive lookup index."""

    roots: List[str] = field(default_factory=list)
    #: lowercase relative path -> absolute path, per root
    _index: Dict[str, Dict[str, str]] = field(default_factory=dict, repr=False)
    config_path: str = ""

    # -------------------------------------------------------------- setup --
    @classmethod
    def load(cls, config_path: str = "", roots: Optional[Iterable[str]] = None) -> "ContentLibrary":
        config_path = config_path or default_config_path()
        library = cls(roots=[], config_path=config_path)
        saved = _read_config(config_path)
        library.roots = [r for r in saved.get("content_roots", []) if os.path.isdir(r)]
        if roots:
            library.add_roots(roots)
        return library

    def add_roots(self, roots: Iterable[str]) -> List[str]:
        """Register roots, keeping only directories that exist."""
        added = []
        for root in roots:
            root = os.path.abspath(os.path.expanduser(root)) if root else ""
            if not root or not os.path.isdir(root):
                continue
            if root not in self.roots:
                self.roots.append(root)
                added.append(root)
        return added

    def detect_default_roots(self) -> List[str]:
        """Add any well-known Daz library folders that exist on this machine."""
        return self.add_roots(DEFAULT_ROOT_CANDIDATES)

    def save(self) -> bool:
        if not self.config_path:
            return False
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as handle:
                json.dump({"content_roots": self.roots}, handle, indent=2)
            return True
        except OSError:
            return False

    # ---------------------------------------------------------- resolving --
    def resolve(self, url: str) -> Optional[str]:
        """Map a Daz URL to an absolute file path, or ``None``."""
        if not url:
            return None
        candidate = os.path.expanduser(url.replace("/", os.sep))
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate

        relative = normalize_url(url)
        if not relative:
            return None
        # Fast path: direct join against each root.
        for root in self.roots:
            direct = os.path.join(root, *relative.split("/"))
            if os.path.exists(direct):
                return direct
        # Slow path: case-insensitive index (libraries authored on Windows).
        lowered = relative.lower()
        for root in self.roots:
            index = self._build_index(root)
            hit = index.get(lowered)
            if hit and os.path.exists(hit):
                return hit
        return None

    def resolve_texture(self, url: str) -> Optional[str]:
        """Resolve a texture URL, trying sibling extensions when needed."""
        direct = self.resolve(url)
        if direct:
            return direct
        base, ext = os.path.splitext(normalize_url(url))
        if ext:
            for alternative in TEXTURE_EXTENSIONS:
                hit = self.resolve(base + alternative)
                if hit:
                    return hit
        return None

    def exists(self, url: str) -> bool:
        return self.resolve(url) is not None

    # ------------------------------------------------------------ scanning --
    def _build_index(self, root: str) -> Dict[str, str]:
        cached = self._index.get(root)
        if cached is not None:
            return cached
        index: Dict[str, str] = {}
        for current, _dirs, files in os.walk(root):
            relative = os.path.relpath(current, root)
            for name in files:
                key = (
                    f"{relative}{os.sep}{name}".lower()
                    if relative != "."
                    else name.lower()
                )
                index[key.replace(os.sep, "/")] = os.path.join(current, name)
        self._index[root] = index
        return index

    def scan_assets(
        self,
        extensions: Iterable[str] = GEOMETRY_EXTENSIONS,
        progress=None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """List convertible assets across all roots."""
        wanted = tuple(ext.lower() for ext in extensions)
        assets: List[dict] = []
        for root in self.roots:
            for current, _dirs, files in os.walk(root):
                for name in files:
                    if not name.lower().endswith(wanted):
                        continue
                    path = os.path.join(current, name)
                    assets.append(
                        {
                            "name": os.path.splitext(name)[0],
                            "path": path,
                            "root": root,
                            "relative": os.path.relpath(path, root),
                            "extension": os.path.splitext(name)[1].lower(),
                            "size": _safe_size(path),
                        }
                    )
                    if limit and len(assets) >= limit:
                        return assets
                if progress:
                    progress(len(assets))
        assets.sort(key=lambda item: item["relative"].lower())
        return assets


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def default_config_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".dazconv")
    return os.path.join(base, CONFIG_FILENAME)


def _read_config(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def library_from_cli(roots: Optional[Iterable[str]] = None, config: str = "") -> ContentLibrary:
    """Build a library from CLI roots, config file, then auto-detection."""
    library = ContentLibrary.load(config, roots)
    if not library.roots:
        library.detect_default_roots()
    # A folder passed on the command line is often the asset's own folder.
    return library


def guess_roots_from_path(path: str) -> List[str]:
    """Infer a content root from an asset path (walk up to ``data``/``Runtime``)."""
    roots = []
    current = os.path.abspath(path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(12):
        name = os.path.basename(current).lower()
        if name in ("data", "runtime", "people"):
            parent = os.path.dirname(current)
            if parent != current:
                roots.append(parent)
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return roots
