"""Allow ``python -m dazconv`` to run the CLI, and ``python -m dazconv.web`` the UI."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
