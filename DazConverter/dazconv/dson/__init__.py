"""DSON (Daz Scene Object Notation) parsing layer."""

from .library import ContentLibrary, autodetect_roots, guess_root_for_file
from .reader import DocumentCache, DsonError, load, sniff
from . import channels, uri

__all__ = [
    "ContentLibrary",
    "DocumentCache",
    "DsonError",
    "autodetect_roots",
    "channels",
    "guess_root_for_file",
    "load",
    "sniff",
    "uri",
]
