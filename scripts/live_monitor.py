#!/usr/bin/env python3
"""Compatibility shim. The monitor now lives in the package as ``iudbms-monitor``
(``iudbms.live.cli``), so an installed deployment does not depend on a checkout
of this repository. Run ``iudbms-monitor --help`` for options."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iudbms.live.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
