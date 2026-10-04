#!/usr/bin/env python3
"""Download the Daz sample assets the test suite uses.

Daz content is licensed, so it is deliberately *not* committed to this
repository.  These four files come from the public community library
``DRincs-Productions/My-DAZ-3D-Library`` and are enough to exercise geometry,
UV sets, a figure rig, skin weights and a morph.

    python tools/fetch_samples.py                 # -> ../daz-samples
    python tools/fetch_samples.py -o ~/daz-test   # somewhere else
    DAZCONV_SAMPLES=~/daz-test python -m unittest discover -s tests -t .

Downloads go through the GitHub API (``gh`` if available, otherwise plain
HTTPS) because raw.githubusercontent.com is blocked on some networks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

REPO = "DRincs-Productions/My-DAZ-3D-Library"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/HEAD/"

#: Upstream path -> relative path inside the samples folder.
SAMPLES = [
    "data/DAZ 3D/Genesis 8/Female 8_1/Genesis8_1Female.dsf",
    "data/DAZ 3D/Genesis 8/Female 8_1/UV Sets/DAZ 3D/Base/Base 8.1 Female.dsf",
    "data/DAZ 3D/Genesis 3/Female/Genesis3Female.dsf",
    "data/DAZ 3D/Genesis 3/Female/Morphs/DAZ 3D/Body/FBMPearFigure.dsf",
]


def fetch_via_gh(relative_path: str) -> bytes | None:
    """Use the authenticated ``gh`` CLI, which reaches the API through proxies."""
    if not shutil.which("gh"):
        return None
    encoded = urllib.parse.quote(relative_path)
    try:
        completed = subprocess.run(
            [
                "gh", "api", f"repos/{REPO}/contents/{encoded}",
                "-H", "Accept: application/vnd.github.raw",
            ],
            capture_output=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


def fetch_via_https(relative_path: str) -> bytes | None:
    url = RAW_BASE + urllib.parse.quote(relative_path)
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            return response.read()
    except Exception:
        return None


def fetch(relative_path: str) -> bytes:
    for method in (fetch_via_gh, fetch_via_https):
        data = method(relative_path)
        if data:
            return data
    raise RuntimeError(
        f"could not download {relative_path}\n"
        "  Tried the GitHub API via `gh` and raw.githubusercontent.com.\n"
        "  Either install/authenticate the `gh` CLI or fetch the file manually."
    )


def main(argv=None) -> int:
    default_target = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "daz-samples",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", default=os.environ.get("DAZCONV_SAMPLES", default_target),
        help="destination folder (default: %(default)s)",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)
    print(f"fetching {len(SAMPLES)} sample assets into {output}")

    failures = 0
    for relative in SAMPLES:
        destination = os.path.join(output, relative)
        if os.path.isfile(destination) and not args.force:
            print(f"  skip  {relative} (already present)")
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            data = fetch(relative)
        except RuntimeError as error:
            print(f"  FAIL  {relative}\n{error}", file=sys.stderr)
            failures += 1
            continue
        with open(destination, "wb") as handle:
            handle.write(data)
        print(f"  ok    {relative}  ({len(data) / 1024.0:.1f} KB)")

    if failures:
        print(f"\n{failures} file(s) failed to download.", file=sys.stderr)
        return 1

    print("\ndone.  Run the suite with:")
    print(f"  DAZCONV_SAMPLES='{output}' python -m unittest discover -s tests -t .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
