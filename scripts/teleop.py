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


def _add_hand_flags(p: argparse.ArgumentParser) -> None:
    """Flags for an Inspire RH56 dexterous hand on its own serial link (opt-in)."""
    p.add_argument("--hand-port", default=None,
                   help="serial port of the Inspire RH56 hand, e.g. /dev/ttyUSB0 (enables it)")
    p.add_argument("--hand-baud", type=int, default=115200, help="hand serial baud (default 115200)")
    p.add_argument("--hand-id", type=int, default=1, help="hand RS485 id (default 1)")
    p.add_argument("--hand-speed", type=int, default=500, help="hand SPEED_SET, 0-1000")
    p.add_argument("--hand-force", type=int, default=300, help="hand FORCE_SET grip threshold, 0-1000 g")
    p.add_argument("--hand-speed-reg", type=int, default=None,
                   help="modbus register of SPEED_SET; --hand-speed is skipped without it")
    p.add_argument("--hand-force-reg", type=int, default=None,
                   help="modbus register of FORCE_SET; --hand-force is skipped without it")
    p.add_argument("--hand-calib", default=None,
                   help="finger calibration JSON (default: data/hand_calib.json if present)")
    p.add_argument("--hand-dry-run", action="store_true",
                   help="build hand frames without opening the port (no finger motion)")


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
                   help="'realsense' (D435), a webcam index (0), or a video path (default: sample)")
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

    p = sub.add_parser("sim", help="Phase 2: drive MuJoCo xArm7 from hand teleop")
    p.add_argument("--source", default=None,
                   help="'realsense' (D435), a webcam index (0), or a video path (default: sample)")
    p.add_argument("--record", default=None, help="side-by-side mp4 path (default: outputs/<name>_sim.mp4)")
    p.add_argument("--display", action="store_true",
                   help="show the live side-by-side window (needs a display; Esc/q to stop)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--scale", type=float, default=3.0, help="hand->robot position scale")
    p.add_argument("--depth-scale", type=float, default=0.4, help="scale for the (noisy) camera depth axis")
    p.add_argument("--pos-only", action="store_true", help="ignore hand orientation (position teleop)")
    p.add_argument("--primary", default="auto", choices=["auto", "left", "right"], help="controlling hand")
    p.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff")
    p.add_argument("--beta", type=float, default=0.02, help="One-Euro speed coefficient")
    p.add_argument("--proc-max-side", type=int, default=640, help="downscale longest side before inference (0=off)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default="float16", help="model dtype (float16|float32)")
    _add_hand_flags(p)
    p.set_defaults(func=commands.cmd_sim)

    p = sub.add_parser("teleop", help="Phase 3: drive the real xArm7 (dry-run unless --execute)")
    p.add_argument("--ip", default=None, help="xArm controller IP (required with --execute)")
    p.add_argument("--execute", action="store_true",
                   help="actually command the robot (default: dry-run, no connection)")
    p.add_argument("--tcp-speed", type=float, default=100.0, help="servo TCP speed cap (mm/s)")
    p.add_argument("--max-step-m", type=float, default=0.02, help="max TCP move per tick (m)")
    p.add_argument("--source", default=None,
                   help="'realsense' (D435), a webcam index (0), or a video path (default: sample)")
    p.add_argument("--record", default=None, help="side-by-side mp4 path")
    p.add_argument("--display", action="store_true",
                   help="show the live side-by-side window (needs a display; Esc/q to stop)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--scale", type=float, default=3.0, help="hand->robot position scale")
    p.add_argument("--depth-scale", type=float, default=0.4, help="scale for the (noisy) camera depth axis")
    p.add_argument("--pos-only", action="store_true", help="ignore hand orientation")
    p.add_argument("--primary", default="auto", choices=["auto", "left", "right"], help="controlling hand")
    p.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff")
    p.add_argument("--beta", type=float, default=0.02, help="One-Euro speed coefficient")
    p.add_argument("--proc-max-side", type=int, default=640, help="downscale longest side before inference (0=off)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default="float16", help="model dtype (float16|float32)")
    _add_hand_flags(p)
    p.set_defaults(func=commands.cmd_teleop)

    p = sub.add_parser("hand-test", help="Bring-up: sweep each DOF of the Inspire RH56 hand")
    p.add_argument("--port", default="/dev/ttyUSB0", help="serial port (default /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200, help="serial baud (default 115200)")
    p.add_argument("--id", type=int, default=1, help="hand RS485 id (default 1)")
    p.add_argument("--speed", type=int, default=300, help="SPEED_SET for the sweep, 0-1000")
    p.add_argument("--force", type=int, default=300, help="FORCE_SET during the sweep, 0-1000 g")
    p.add_argument("--speed-reg", type=int, default=None,
                   help="modbus register of SPEED_SET; --speed is skipped without it")
    p.add_argument("--force-reg", type=int, default=None,
                   help="modbus register of FORCE_SET; --force is skipped without it")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to hold each sweep step")
    p.add_argument("--dry-run", action="store_true", help="log frames without opening the port")
    p.set_defaults(func=commands.cmd_hand_test)

    p = sub.add_parser("hand-calib", help="Record open/fist finger angles for dex-hand retargeting")
    p.add_argument("--source", default=None,
                   help="'realsense' (D435), a webcam index (0), or a video path (default: sample)")
    p.add_argument("--out", default=None, help="output JSON (default: data/hand_calib.json)")
    p.add_argument("--frames", type=int, default=30, help="frames to average per pose")
    p.add_argument("--countdown", type=int, default=3, help="seconds of countdown before each pose")
    p.add_argument("--primary", default="auto", choices=["auto", "left", "right"], help="controlling hand")
    p.add_argument("--proc-max-side", type=int, default=640, help="downscale longest side before inference (0=off)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--dtype", default="float16", help="model dtype (float16|float32)")
    p.set_defaults(func=commands.cmd_hand_calib)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
