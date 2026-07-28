# xarm-teleop

Real-time markerless **teleoperation of a UFACTORY xArm7** from a hand camera. Point an Intel
RealSense **D435** at your hand: a hand-pose model (**WiLoR**) tracks your wrist and pinch, and
the robot's end-effector follows your hand while a pinch opens/closes the gripper.

```
D435 (color + depth) ──► WiLoR (wrist 6DOF + pinch) ──► retarget (wrist→TCP, pinch→gripper)
                                                    ──► safety limiter ──► xArm7
```

The same code drives a **MuJoCo simulation** and the **real robot** through one backend
interface, so you can validate everything in sim before the arm ever moves.

> **Status:** perception, retargeting, simulation, safety, and the control path are implemented
> and validated. The **D435 streaming** and **real-robot motion** paths follow the standard
> librealsense / xArm-SDK APIs but must be verified on your hardware during first bring-up (see
> [Verified vs. needs-hardware](#verified-vs-needs-hardware)).

---

## 1. Requirements

**Hardware**
- NVIDIA GPU (tested on RTX 3090; ~4 GB VRAM used) with a CUDA 11.8-capable driver
- Intel RealSense **D435** (USB 3.0)
- UFACTORY **xArm7** + control box on your network, with a gripper (UFACTORY 2-finger by default)

**Software**
- Linux, Python **3.10**, conda (or venv)
- For MuJoCo offscreen rendering on a headless host: an EGL-capable GL stack (NVIDIA EGL works)

## 2. Installation

**One command** (creates a conda env, installs everything in the right order, verifies imports):

```bash
bash scripts/install_env.sh          # env name: xarm-teleop (pass a name to override)
conda activate xarm-teleop
```

This recipe is validated on a clean Python 3.10 machine. If you prefer to run the steps
yourself, they are:

```bash
conda create -n xarm-teleop python=3.10 -y && conda activate xarm-teleop
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==1.26.4"
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"
```

The order and the two odd-looking steps matter on a fresh machine:
- **torch first** (from the cu118 index) so nothing pulls a CPU build.
- **`numpy==1.26.4`** — numpy 2.x breaks chumpy/opencv.
- **chumpy with `--no-build-isolation`** — its ancient `setup.py` imports `pip`, which is
  absent inside pip's isolated build env, so an ordinary install fails.
- **WiLoR-mini with `--no-deps`** — otherwise it rebuilds chumpy from a git URL under isolation
  and fails again; its dependencies are already installed above.

WiLoR + MANO model weights download automatically to the HuggingFace cache on first run
(~2 GB). No manual weight setup is needed. Verify with `python scripts/teleop.py wilor-image`.

For the simulation backend (optional but recommended for validation), the xArm7 MuJoCo model is
fetched once into `third_party/`:

```bash
git clone --depth 1 --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie.git \
    third_party/mujoco_menagerie
```

Quick check that the install works (no hardware needed):

```bash
python scripts/teleop.py wilor-image        # runs WiLoR on a test image -> outputs/*.jpg
```

## 3. Hardware setup

**D435** — plug into USB 3.0 (blue port). Verify the OS sees it:
```bash
rs-enumerate-devices | head        # from librealsense; or just run the perception test below
```

**xArm7** — connect the control box to your network and note its IP (default often `192.168.1.xxx`).
```bash
ping <YOUR_XARM_IP>                # must succeed
```
Enable the controller for remote motion (UFACTORY Studio → in remote/idle state, no active errors).

## 4. Run it on your D435 + xArm7

Go through these **in order** — each step de-risks the next.

**Step 1 — Perception only (no robot).** Confirm the D435 tracks your hand and reports *metric*
wrist depth (~0.3–0.8 m), not the monocular fallback:
```bash
python scripts/teleop.py live --source realsense --display     # live window if you have a display
python scripts/teleop.py live --source realsense               # else writes outputs/realsense_live.mp4
```
You should see a hand skeleton, a yellow box on the controlling hand, and `wrist(m)` with a
realistic depth. Pinch your thumb+index — `pinch` should drop toward ~10 mm.

**Step 2 — Retargeting in simulation (no robot risk).** With metric depth, use `--scale 1`:
```bash
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 --display
```
Move your hand — the simulated xArm7 should follow; pinch to close the gripper. Tune
`--scale`, `--min-cutoff`, `--beta` until motion feels right. Output: `outputs/realsense_sim.mp4`.
`--display` opens the live side-by-side window (camera overlay | robot) — Esc or `q` stops the run
cleanly. Drop it when running headless/over SSH; recording is unaffected.

**Step 3 — Real robot, DRY-RUN (no motion).** Validates the exact commands without connecting:
```bash
python scripts/teleop.py teleop --source realsense --scale 1.0 --depth-scale 1.0 --display
```
The right panel shows the **commanded** TCP. Confirm it stays inside your workspace box and moves
sensibly with your hand.

**Step 4 — Real robot, EXECUTE (low speed, E-stop in hand).** Only after Steps 1–3 look right and
you've read [Safety](#5-safety):
```bash
python scripts/teleop.py teleop --execute --ip <YOUR_XARM_IP> \
    --source realsense --scale 1.0 --depth-scale 1.0 --tcp-speed 80
```

## 5. Safety

Read before Step 4. Teleop moves a real arm from your hand motion — treat it like a live robot.

- **Physical E-stop within reach.** The camera is *not* a safety system.
- **Set your workspace box.** Edit `DEFAULT_WORKSPACE` in `src/control/safety.py` to a reachable,
  collision-free box for your cell. Targets are hard-clamped to it.
- **Start slow.** Keep `--tcp-speed` low (≤ 80 mm/s) and `max_step_m` small (default 0.02 m/tick,
  in `safety.py`) for first runs — this bounds TCP speed.
- **Verify gripper direction** on the bench first (Step 4 with the arm parked): if open/close is
  reversed, flip the gripper mapping in `XArm7Controller.servo_to` (uses `GRIPPER_MAX`; the code
  assumes 0 = closed, `GRIPPER_MAX` = open).
- **Clear the area** around the robot and keep the deadman/clutch in mind: control engages on the
  first tracked frame and re-indexes after a tracking loss.

## 6. Configuration & tuning

| What | Where | Notes |
|---|---|---|
| Reachable workspace box | `src/control/safety.py` → `DEFAULT_WORKSPACE` | **edit for your cell** |
| Max TCP speed / step | `src/control/safety.py` → `max_step_m`, or `--tcp-speed` | keep low at first |
| Camera→robot axis map | `src/retarget/wrist_to_tcp.py` → `default_align()` | if motion is mirrored/rotated |
| Position scale | `--scale` (use ~1.0 with D435 metric depth) | larger = arm moves more |
| Home pose | `src/control/xarm_controller.py` → `HOME_Q` | joint angles (rad) |
| Gripper range/direction | `src/control/xarm_controller.py` → `GRIPPER_MAX`, `servo_to` | verify on hardware |
| Smoothing | `--min-cutoff`, `--beta` (One-Euro) | lower cutoff = smoother, more lag |

## 7. Command reference

```bash
python scripts/teleop.py wilor-image [--image IMG]        # Phase 0: model check on a still image
python scripts/teleop.py live   --source realsense|0|VID  # Phase 1: perception + overlay
python scripts/teleop.py sim    --source realsense|0|VID  # Phase 2: drive MuJoCo xArm7
python scripts/teleop.py teleop --source realsense|0|VID  # Phase 3: real xArm7 (dry-run)
python scripts/teleop.py teleop --execute --ip IP ...     # Phase 3: real xArm7 (moves!)
```
Common flags: `--scale`, `--depth-scale`, `--pos-only`, `--primary auto|left|right`,
`--min-cutoff`, `--beta`, `--proc-max-side` (downscale before inference), `--device`, `--record`,
`--display` (live window on `live`/`sim`/`teleop`; Esc or `q` ends the run cleanly).
Outputs are H.264 mp4 (playable in VSCode/browser). The H.264 encoder is probed from the local
ffmpeg build at runtime (`libx264` -> `h264_nvenc` -> `libopenh264`), so LGPL builds without x264
work too; if none is usable the writer falls back to OpenCV mp4v and logs a warning.

## 8. Troubleshooting

- **D435 not found** — use a USB 3.0 port; check `rs-enumerate-devices`; replug.
- **Wrist depth looks wrong / huge** — depth fusion only runs with `--source realsense`; on a
  webcam/video the wrist is monocular (up-to-scale), which is why sim/video use `--scale 3`.
- **xArm won't connect** — `ping` the IP; clear errors in UFACTORY Studio; check firmware; the
  controller must not be in an error/estop state.
- **Gripper opens when it should close** — flip the gripper mapping in `XArm7Controller.servo_to`.
- **Arm overshoots / lags** — lower `--tcp-speed` and `max_step_m`, or raise One-Euro smoothing.
- **Orientation off** — the controller uses the axis-angle servo (`set_servo_cartesian_aa`);
  verify your firmware's `get_position_aa` convention during bring-up. Use `--pos-only` to ignore
  orientation while you debug position.
- **`Unrecognized option 'preset'` / recording dies with `BrokenPipeError`** — an ffmpeg build
  without libx264 (common with conda-forge LGPL packages). Handled automatically now: the encoder
  is probed at startup and libopenh264 is used instead. To get x264 back:
  `conda install -c conda-forge 'ffmpeg=*=*gpl*'`.
- **`--display` shows nothing** — over SSH you need `ssh -X`/`-Y` (or run on the robot PC's own
  session). Without a usable display the window disables itself with a warning and the run keeps
  going; the mp4 is still written. The MuJoCo offscreen render is separate and unaffected.
- **Slow inference (<20 fps)** — expected on eager PyTorch (~50 ms/frame single hand on a 3090);
  the control loop is decoupled from perception. See `docs/PLAN.md` for the optimization path.

## 9. Verified vs. needs-hardware

**Verified without hardware:** WiLoR perception, retargeting, MuJoCo sim teleop, safety limits,
the full control code path (dry-run), and the depth back-projection math.

**Needs on-hardware verification (first bring-up):** D435 live streaming; xArm7 motion, gripper
direction, and axis-angle pose convention; end-to-end latency.

## 10. Project layout

```
scripts/teleop.py        single CLI entrypoint (thin; dispatches into src/)
src/
  loop.py                backend-agnostic teleop loop (perception→retarget→safety→robot)
  camera/                CameraSource: realsense (D435) / webcam / video + factory
  perception/            WiLoR wrapper (single hand) + real-time tracker + depth lift + overlay
  retarget/              wrist 6DOF → TCP (clutch/relative) + pinch → gripper
  control/               RobotBackend, SafetyLimiter, real xArm7 controller (SDK)
  sim/                   MuJoCo xArm7 backend (differential IK + gripper)
  paths.py log.py video.py commands.py
third_party/             mujoco_menagerie (fetched on demand; git-ignored)
outputs/                 H.264 mp4 / images (git-ignored)
docs/PLAN.md             phased implementation plan
```
