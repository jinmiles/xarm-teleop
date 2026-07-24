# xarm-teleop

Real-time markerless teleoperation of a **UFACTORY xArm7** from a monocular hand camera.

**Pipeline:** camera → **WiLoR** hand pose (wrist 6DOF + MANO) → retargeting (wrist → TCP,
pinch → gripper) → xArm7 control. See [`docs/PLAN.md`](docs/PLAN.md) for the full plan.

> Status: **Phase 0** — scaffolding + WiLoR still-image inference. The RealSense D435 is not
> yet available; Phases 0–3 are developed with a webcam and run entirely without it.

## Environment

Dedicated conda env `xarm-teleop` (Python 3.10, torch 2.1 + cu118), cloned from the machine's
`dynhamr` env and extended with [`wilor-mini`](https://github.com/warmshao/WiLoR-mini).

```bash
conda activate xarm-teleop
```

Model weights (WiLoR + MANO) are downloaded automatically by wilor-mini into the shared
HuggingFace cache on first run. External assets (MANO, sample images, the WiLoR detector) are
reused in place from other projects and are never copied into or committed to this repo — see
`src/paths.py`.

## Usage

```bash
# Phase 0: run WiLoR on a still image and save an annotated overlay to outputs/
python scripts/teleop.py wilor-image --image /path/to/hand.jpg
python scripts/teleop.py wilor-image            # uses a bundled sample image
```

## Layout

```
scripts/teleop.py        single CLI entrypoint (thin; dispatches into the package)
src/
  paths.py               single source of repo paths + external asset locations + shims
  log.py                 shared logger
  commands.py            thin CLI command handlers
  perception/            WiLoR wrapper + overlay visualization
third_party/             vendored upstream (kept clean; not modified for project behavior)
docs/PLAN.md             phased implementation plan (local-only)
data/  outputs/          git-ignored local workspace
```
