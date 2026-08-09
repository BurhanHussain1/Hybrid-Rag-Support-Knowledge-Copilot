#!/usr/bin/env python
"""Run the API.

    python serve.py                    # http://localhost:8000, docs at /docs
    python serve.py --port 9000
    python serve.py --reload           # auto-restart on code changes

--reload is for editing code, not for using the service: every restart reloads
the models, which is the ~30 second startup cost all over again.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="serve.py", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "worker processes. Keep at 1 unless you have RAM to spare: each worker "
            "loads its own copy of every model."
        ),
    )
    args = parser.parse_args()

    import uvicorn

    print(f"\n  API   http://{args.host}:{args.port}")
    print(f"  docs  http://{args.host}:{args.port}/docs")
    print("  loading models before accepting traffic...\n")

    uvicorn.run(
        "copilot.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
