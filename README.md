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
# Phase 0: WiLoR on a still image -> annotated overlay in outputs/
python scripts/teleop.py wilor-image [--image hand.jpg]

# Phase 1: real-time hand tracking over a webcam/video (single controlling hand)
python scripts/teleop.py live --source 0            # webcam
python scripts/teleop.py live                       # bundled sample video

# Phase 2: drive the MuJoCo xArm7 sim from hand teleop (side-by-side mp4)
python scripts/teleop.py sim --source 0

# Phase 3: real xArm7 -- dry-run by default (no connection); needs hardware to --execute
python scripts/teleop.py teleop --source 0                     # dry-run
python scripts/teleop.py teleop --execute --ip 192.168.1.xxx   # real robot (hardware only)
```

Outputs are H.264 mp4 (playable in VSCode/browser). `sim`/`teleop` share one backend-agnostic
loop, so retargeting and safety behave identically in simulation and on hardware.

## Layout

```
scripts/teleop.py        single CLI entrypoint (thin; dispatches into the package)
src/
  paths.py               single source of repo paths + external asset locations + shims
  log.py  video.py       shared logger; H.264 mp4 writer (ffmpeg)
  commands.py            thin CLI command handlers
  loop.py                backend-agnostic teleop loop (perception->retarget->safety->robot)
  camera/                CameraSource: webcam / video / realsense(stub) + factory
  perception/            WiLoR wrapper (single-hand) + real-time tracker + overlay
  retarget/              wrist 6DOF -> TCP (clutch/relative) + pinch -> gripper
  control/               RobotBackend, SafetyLimiter, real xArm7 controller (SDK)
  sim/                   MuJoCo xArm7 backend (differential IK + gripper)
third_party/             vendored upstream (mujoco_menagerie; git-ignored, fetched on demand)
docs/PLAN.md             phased implementation plan (local-only)
data/  outputs/          git-ignored local workspace
```
