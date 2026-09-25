"""Launch the TactiVision dashboard API + static UI."""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve TactiVision web dashboard")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port (defaults to $PORT or 8000; Render sets PORT)",
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install web extras: pip install -e \".[web]\"") from exc
    uvicorn.run("tactivision.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
