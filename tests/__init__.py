"""Shared helpers for the dazconv test suite."""

from __future__ import annotations

import os
import sys

# Make the package importable when the suite is run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Where ``tools/fetch_samples.py`` puts the Daz sample assets.  Overridable so
#: CI can point at a different location; tests skip when it is missing.
SAMPLES_DIR = os.environ.get(
    "DAZCONV_SAMPLES", os.path.join(os.path.dirname(ROOT), "daz-samples")
)

def _first_existing(candidates):
    """Pick the first path that exists.

    Daz URLs and on-disk folders disagree on case ("DAZ 3D" vs "Daz 3D"), and
    hand-made sample trees may use either, so samples are resolved against a
    list of spellings instead of one hard-coded path.
    """
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


_G81_DIR = os.path.join(SAMPLES_DIR, "data", "DAZ 3D", "Genesis 8", "Female 8_1")
_G3_DIR = os.path.join(SAMPLES_DIR, "data", "DAZ 3D", "Genesis 3", "Female")

SAMPLE_FIGURE = _first_existing([os.path.join(_G81_DIR, "Genesis8_1Female.dsf")])

SAMPLE_UVSET = _first_existing([
    os.path.join(_G81_DIR, "UV Sets", vendor, "Base", "Base 8.1 Female.dsf")
    for vendor in ("DAZ 3D", "Daz 3D", "daz 3d")
])

SAMPLE_G3F = _first_existing([
    os.path.join(_G3_DIR, "Genesis3Female.dsf"),
    os.path.join(SAMPLES_DIR, "data", "g3f.dsf"),
])

SAMPLE_MORPH = _first_existing([
    os.path.join(_G3_DIR, "Morphs", vendor, "Body", "FBMPearFigure.dsf")
    for vendor in ("DAZ 3D", "Daz 3D")
] + [os.path.join(SAMPLES_DIR, "data", "g3f_pear.dsf")])

SAMPLE_SCENE = os.path.join(ROOT, "Test", "QA-Test-Scene-01.duf")


def have_samples() -> bool:
    return os.path.isfile(SAMPLE_FIGURE)


def have_scene() -> bool:
    """True when both the repo scene and the referenced figure are available."""
    return os.path.isfile(SAMPLE_SCENE) and have_samples()


def have_ufbx() -> bool:
    try:
        import ufbx  # noqa: F401
    except ImportError:
        return False
    return True


def samples_library():
    from dazconv.library import ContentLibrary

    return ContentLibrary(roots=[SAMPLES_DIR])


def run_ufbx_validate(fbx_path: str) -> dict:
    """Validate ``fbx_path`` with tools/validate_fbx.py in a subprocess.

    ``ufbx``'s Python bindings crash the interpreter on some access patterns, so
    the native library is never imported into the test process.  A segfault shows
    up here as a return code we can report instead of killing the whole suite.
    """
    import json
    import subprocess

    script = os.path.join(ROOT, "tools", "validate_fbx.py")
    completed = subprocess.run(
        [sys.executable, script, fbx_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 and not completed.stdout.strip():
        return {
            "ok": False,
            "error": (
                f"validate_fbx.py exited with code {completed.returncode} "
                f"(segfault = 139); stderr: {completed.stderr.strip()[-400:]}"
            ),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": f"unparsable validator output: {completed.stdout[:400]!r}",
        }


def compare_vertex_delta(base_fbx: str, morphed_fbx: str, deltas: dict) -> dict:
    """Verify ``morphed_fbx`` == ``base_fbx`` + ``deltas`` via a subprocess.

    Two ufbx scenes cannot safely coexist in one process (see
    tools/validate_fbx.py), so the comparison runs in a child.
    """
    import json
    import subprocess
    import tempfile

    script = os.path.join(ROOT, "tools", "validate_fbx.py")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump({str(k): list(v) for k, v in deltas.items()}, handle)
        deltas_path = handle.name
    try:
        completed = subprocess.run(
            [
                sys.executable, script, base_fbx,
                "--compare", morphed_fbx, "--deltas", deltas_path,
            ],
            capture_output=True, text=True, timeout=600,
        )
        if completed.returncode != 0 and not completed.stdout.strip():
            return {
                "ok": False,
                "error": f"comparator exited {completed.returncode}: "
                         f"{completed.stderr.strip()[-300:]}",
            }
        return json.loads(completed.stdout)
    finally:
        os.unlink(deltas_path)
