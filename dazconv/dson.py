"""DSON (DAZ Scene Object Notation) file loading.

DSON is a JSON superset.  In the wild the files show up as:

* plain UTF-8 JSON text (``{`` at offset 0) -- most ``.dsf`` data files
* gzip-compressed JSON (``1f 8b``) -- most ``.duf`` scenes, some ``.dsf`` morphs
* raw zlib/deflate streams (``78 xx``)
* the legacy binary interchange stream (``DZSF`` magic), which Daz Studio 4.x
  no longer writes for regular assets

:meth:`load_dson` sniffs the container, then falls back to a small "repair"
pass that tolerates the comment syntax DSON permits but strict JSON rejects.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from typing import Any

MAGIC_GZIP = b"\x1f\x8b"
MAGIC_ZLIB_LOW = (0x78,)
MAGIC_BINARY = b"DZSF"

#: File extensions that Daz uses for DSON documents.
DSON_EXTENSIONS = (".duf", ".dsf", ".daz", ".dsb", ".dsa", ".dse", ".dsp", ".dui")


class DsonError(Exception):
    """Raised when a file cannot be understood as DSON."""


class BinaryDsonError(DsonError):
    """Raised for legacy binary ``DZSF`` streams, which are not decoded here."""


def sniff_container(data: bytes) -> str:
    """Identify the container of a DSON payload.

    Returns one of ``"plain"``, ``"gzip"``, ``"zlib"`` or ``"binary"``.
    """
    if not data:
        raise DsonError("file is empty")
    if data.startswith(MAGIC_GZIP):
        return "gzip"
    if data.startswith(MAGIC_BINARY):
        return "binary"
    first = data[0]
    # zlib headers: CMF byte with low nibble 8 (deflate) and a valid check sum.
    if first in MAGIC_ZLIB_LOW and len(data) > 1:
        if ((first << 8) | data[1]) % 31 == 0:
            return "zlib"
    if first in (0x7B, 0x5B, ord("\t"), ord(" "), ord("\n"), ord("\r"), 0xEF):
        return "plain"
    # Last resort: try both text and deflate before giving up.
    try:
        data.decode("utf-8")
        return "plain"
    except UnicodeDecodeError:
        return "zlib"


def decompress_payload(data: bytes, container: str | None = None) -> str:
    """Decompress if needed and return the DSON text."""
    container = container or sniff_container(data)
    if container == "gzip":
        data = gzip.decompress(data)
    elif container == "zlib":
        try:
            data = zlib.decompress(data)
        except zlib.error:
            # Some tools emit a bare deflate stream without the zlib header.
            data = zlib.decompress(data, -zlib.MAX_WBITS)
    elif container == "binary":
        raise BinaryDsonError(
            "legacy binary DSON stream (DZSF). Save the asset from Daz Studio "
            "with 'Compressed' turned off, or re-save it as a .duf/.dsf."
        )
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


# ---------------------------------------------------------------- repair --
def _strip_comments(text: str) -> str:
    """Remove ``//`` and ``#`` comments that DSON allows but JSON does not."""
    out = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    out = []
    in_string = False
    escaped = False
    last_significant = None
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            last_significant = ch
            continue
        if ch == "," and out:
            # Look ahead: if the next non-space char closes the container, drop it.
            out.append(ch)
            last_significant = ch
            continue
        if ch in "}]" and last_significant == ",":
            for idx in range(len(out) - 1, -1, -1):
                if out[idx] == ",":
                    out[idx] = " "
                    break
        if not ch.isspace():
            last_significant = ch
        out.append(ch)
    return "".join(out)


def parse_dson_text(text: str) -> Any:
    """Parse DSON text, repairing common non-JSON dialect quirks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        repaired = _strip_trailing_commas(_strip_comments(text))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise DsonError(
                f"could not parse DSON text: {first_error}"
            ) from first_error


def load_dson(data: bytes) -> Any:
    """Parse an in-memory DSON payload."""
    if isinstance(data, str):
        return parse_dson_text(data)
    return parse_dson_text(decompress_payload(data))


def load_dson_file(path: str) -> Any:
    """Load a ``.duf``/``.dsf`` (or any DSON) file from disk."""
    if not os.path.isfile(path):
        raise DsonError(f"not a file: {path}")
    with open(path, "rb") as handle:
        return load_dson(handle.read())


def looks_like_dson(path: str) -> bool:
    """Cheap extension+magic check used by the file browser."""
    ext = os.path.splitext(path)[1].lower()
    if ext in DSON_EXTENSIONS:
        return True
    if ext != ".json":
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    return sniff_container(head) in ("plain", "gzip", "zlib")
