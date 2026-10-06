"""iudbms-api -- the operator API and dashboard.

    iudbms-api                      # http://127.0.0.1:8000
    iudbms-api --host 0.0.0.0       # only if you mean to expose it

Binds to loopback by default. The API has state-changing endpoints and no
authentication, so it should not be reachable from the network unless something
in front of it provides that.
"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="iudbms-api", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("IUDBMS_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("IUDBMS_API_PORT", 8000)))
    ap.add_argument("--db", help="overrides $IUDBMS_DB")
    ap.add_argument("--models", help="overrides $IUDBMS_MODELS")
    args = ap.parse_args(argv)
    if args.db:
        os.environ["IUDBMS_DB"] = args.db
    if args.models:
        os.environ["IUDBMS_MODELS"] = args.models

    import uvicorn
    uvicorn.run("iudbms.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
