"""Persistent user settings (content directories, output folder, defaults)."""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, List

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}

DEFAULTS: Dict[str, Any] = {
    "content_dirs": [],
    "output_dir": "",
    "formats": ["fbx"],
    "include_morphs": True,
    "include_skeleton": True,
    "include_hd_morphs": False,
    "max_influences": 8,
    "pose": "rest",
    "merge_meshes": False,
    "copy_textures": True,
    "embed_textures": False,
    "scale": 1.0,
    "recent": [],
}


def data_dir() -> str:
    """Per-user application data directory."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(base, "DazConv")
    elif sys.platform == "darwin":
        path = os.path.expanduser("~/Library/Application Support/DazConv")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        path = os.path.join(base, "dazconv")
    os.makedirs(path, exist_ok=True)
    return path


def config_path() -> str:
    return os.path.join(data_dir(), "config.json")


def default_output_dir() -> str:
    home = os.path.expanduser("~")
    for candidate in (os.path.join(home, "Documents"), home):
        if os.path.isdir(candidate):
            return os.path.join(candidate, "DazConv Output")
    return os.path.join(data_dir(), "output")


def load() -> Dict[str, Any]:
    with _LOCK:
        if _CACHE:
            return dict(_CACHE)
        cfg = dict(DEFAULTS)
        try:
            with open(config_path(), "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                cfg.update(stored)
        except (OSError, ValueError):
            pass
        if not cfg.get("output_dir"):
            cfg["output_dir"] = default_output_dir()
        cfg["content_dirs"] = [d for d in cfg.get("content_dirs", [])
                               if isinstance(d, str)]
        _CACHE.update(cfg)
        return dict(cfg)


def save(updates: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load()
    cfg.update({k: v for k, v in updates.items() if k in DEFAULTS})
    with _LOCK:
        _CACHE.clear()
        _CACHE.update(cfg)
        try:
            with open(config_path(), "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        except OSError:
            pass
    return cfg


def add_recent(path: str, limit: int = 15) -> None:
    cfg = load()
    recent: List[str] = [p for p in cfg.get("recent", []) if p != path]
    recent.insert(0, path)
    save({"recent": recent[:limit]})


def effective_content_dirs() -> List[str]:
    from .dson import autodetect_roots
    cfg = load()
    dirs = list(cfg.get("content_dirs") or [])
    for d in autodetect_roots():
        if d not in dirs:
            dirs.append(d)
    return dirs
