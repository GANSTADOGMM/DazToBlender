#!/usr/bin/env sh
# Start DazConv and open it in your browser.  macOS / Linux.
#
# Nothing to install: DazConv only needs Python 3.8 or newer, which ships with
# macOS and every Linux distribution.  Double-click this file, or run ./DazConv.sh
set -e

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$here"

for py in python3 python py; do
    if command -v "$py" >/dev/null 2>&1; then
        if "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
            exec "$py" -m dazconv serve "$@"
        fi
    fi
done

echo "DazConv needs Python 3.8 or newer." >&2
echo "Install it from https://www.python.org/downloads/ and run this again." >&2
exit 1
