#!/usr/bin/env python
"""Single entrypoint for xarm-teleop. Thin CLI that dispatches into the src package.

Usage:
  python scripts/teleop.py wilor-image [--image PATH] [--out PATH] [--device cuda] [--dtype float16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import commands  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teleop", description="xArm7 hand teleoperation")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("wilor-image", help="Phase 0: run WiLoR on a still image")
    p.add_argument("--image", default=None, help="input image path (default: bundled sample)")
    p.add_argument("--out", default=None, help="output overlay path (default: outputs/<name>_wilor.jpg)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default="float16", help="model dtype (float16|float32)")
    p.set_defaults(func=commands.cmd_wilor_image)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
