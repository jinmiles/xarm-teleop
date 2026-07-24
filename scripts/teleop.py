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

    p = sub.add_parser("live", help="Phase 1: real-time hand tracking (webcam/video)")
    p.add_argument("--source", default=None,
                   help="webcam index (e.g. 0) or video path (default: bundled sample video)")
    p.add_argument("--record", default=None, help="write annotated mp4 to this path")
    p.add_argument("--display", action="store_true", help="show a live window (needs a display)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--primary", default="auto", choices=["auto", "left", "right"],
                   help="which hand controls (default: auto = largest)")
    p.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff")
    p.add_argument("--beta", type=float, default=0.02, help="One-Euro speed coefficient")
    p.add_argument("--proc-max-side", type=int, default=640,
                   help="downscale frames so longest side <= this before inference (0=off)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default="float16", help="model dtype (float16|float32)")
    p.set_defaults(func=commands.cmd_live)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
