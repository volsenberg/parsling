"""
Convert a DoclingDocument JSON file to metadata-rich Markdown.

Usage:
    python examples/json_to_rich_md.py output/01_aceh_fast_1/01_aceh_fast_1.json
    python examples/json_to_rich_md.py output/01_aceh_fast_1/01_aceh_fast_1.json --out my_output.md
    python examples/json_to_rich_md.py output/01_aceh_fast_1/01_aceh_fast_1.json --no-merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the implementation from the parsling package so logic stays in one place.
try:
    from parsling.exporters import _build_rich_markdown
except ImportError:
    print(
        "ERROR: parsling package not found. Run from the repo root with:\n"
        "  pip install -e .",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DoclingDocument JSON → metadata-rich Markdown"
    )
    parser.add_argument("input", type=Path, help="Path to .json DoclingDocument file")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .md path (default: <input-stem>_rich.md in same directory)",
    )
    parser.add_argument(
        "--no-pictures", action="store_true",
        help="Omit figure placeholders from output",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Keep page-split tables as separate fragments instead of merging",
    )
    parser.add_argument(
        "--caption-window", type=int, default=4, metavar="N",
        help="Body positions to search around a table for an adjacent caption (default 4)",
    )
    parser.add_argument(
        "--min-text", type=int, default=5, metavar="N",
        help="Minimum text length to emit; filters short OCR noise (default 5)",
    )
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {src} …", file=sys.stderr)
    raw = json.loads(src.read_text(encoding="utf-8"))

    md = _build_rich_markdown(
        raw,
        include_pictures=not args.no_pictures,
        caption_window=args.caption_window,
        min_text_len=args.min_text,
        merge_split_tables=not args.no_merge,
    )

    out = args.out or src.with_name(src.stem + "_rich.md")
    out.write_text(md, encoding="utf-8")
    print(f"Written {len(md):,} chars → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
