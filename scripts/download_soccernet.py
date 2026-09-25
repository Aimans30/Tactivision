"""Download one SoccerNet game into data/raw. Never the full dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME = (
    "europe_uefa-champions-league/2016-2017/"
    "2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich"
)


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a single SoccerNet game")
    parser.add_argument("--game", default=DEFAULT_GAME, help="SoccerNet game path")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["1_720p.mkv", "video.ini"],
        help="Files for that game only. First half by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "soccernet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(ROOT / ".env")
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        print("error: set SOCCERNET_PASSWORD in .env", file=sys.stderr)
        return 2

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("error: pip install SoccerNet", file=sys.stderr)
        return 2

    downloader = SoccerNetDownloader(LocalDirectory=str(args.output))
    downloader.password = password
    print(f"Downloading {args.files} for one game into {args.output}")
    # The emailed password unlocks the OwnCloud path, not the Hugging Face gate.
    downloader.downloadGame(files=args.files, game=args.game, source="EXRCSDrive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
