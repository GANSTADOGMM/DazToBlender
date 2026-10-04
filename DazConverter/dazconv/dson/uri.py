"""DSON URI parsing.

A DSON URI looks like::

    scheme://node_path/file_path#asset_id?property_path

In practice the forms that appear in real content are:

    /data/Daz%203D/Genesis%208/Female/Genesis8Female.dsf#geometry
    #SkinBinding
    name://@selection/hip:?rotation/x
    /data/.../Morphs/foo.dsf#head?value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote


@dataclass(frozen=True)
class DsonUri:
    scheme: str = ""
    node_path: str = ""
    file_path: str = ""
    asset_id: str = ""
    property_path: str = ""
    raw: str = ""

    @property
    def is_local(self) -> bool:
        """True when the URI refers to an asset inside the current file."""
        return not self.file_path

    def __str__(self) -> str:
        return self.raw


def parse(uri: str) -> DsonUri:
    """Parse a DSON URI into its components, URL-decoding each part."""
    if uri is None:
        return DsonUri()
    raw = uri.strip()
    rest = raw

    scheme = ""
    if "://" in rest:
        scheme, rest = rest.split("://", 1)
        # node path runs up to the first '/' (or the whole thing)
        if rest.startswith("@") or (rest and not rest.startswith("/")):
            if "/" in rest:
                node_path, rest = rest.split("/", 1)
                rest = "/" + rest
            else:
                node_path, rest = rest, ""
        else:
            node_path = ""
    else:
        node_path = ""

    prop = ""
    if "?" in rest:
        rest, prop = rest.split("?", 1)

    asset_id = ""
    if "#" in rest:
        rest, asset_id = rest.split("#", 1)

    file_path = rest
    # A trailing ':' marks "the current file" in node:// style URIs.
    if file_path.endswith(":"):
        file_path = file_path[:-1]

    return DsonUri(
        scheme=scheme,
        node_path=unquote(node_path),
        file_path=unquote(file_path),
        asset_id=unquote(asset_id),
        property_path=unquote(prop),
        raw=raw,
    )


def asset_id_of(uri: str) -> str:
    """Shortcut for the fragment part of a URI."""
    return parse(uri).asset_id


def file_of(uri: str) -> str:
    return parse(uri).file_path


def join(file_path: str, asset_id: str) -> str:
    from urllib.parse import quote
    if not file_path:
        return "#" + quote(asset_id)
    return "%s#%s" % (quote(file_path), quote(asset_id))


def normalize_relative(path: str, base_file: Optional[str] = None) -> str:
    """Return a content-relative POSIX path with no leading slash.

    DSON paths are always relative to a content directory root.  Some files
    use paths relative to the referring file instead; `base_file` lets us
    resolve that case.
    """
    import posixpath
    p = unquote(path or "").replace("\\", "/")
    if p.startswith("/"):
        return p.lstrip("/")
    if base_file:
        base_dir = posixpath.dirname(base_file.replace("\\", "/").lstrip("/"))
        if base_dir:
            return posixpath.normpath(posixpath.join(base_dir, p))
    return p
