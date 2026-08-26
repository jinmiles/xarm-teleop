# xarm-teleop

Real-time markerless **teleoperation of a UFACTORY xArm7** from a hand camera. Point an Intel
RealSense **D435** at your hand: a hand-pose model (**WiLoR**) tracks your wrist and pinch, and
the robot's end-effector follows your hand while a pinch opens/closes the gripper. With an
Inspire **RH56** 5-finger hand mounted, each finger is teleoperated individually instead.

```
D435 (color + depth) ──► WiLoR (wrist 6DOF + 21 keypoints) ──► retarget (wrist→TCP, pinch→gripper,
                                                               fingers→RH56 6 DOF)
                                                           ──► safety limiter ──► xArm7 (+ hand)
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
- *Optional:* Inspire **RH56** 5-finger hand on RS485 (e.g. `/dev/ttyUSB0`) for per-finger teleop

**Software**
- Linux, Python **3.13**, conda (Python 3.9 or older will not work: several dependencies,
  including mujoco, huggingface_hub and scikit-image, require 3.10+)
- For MuJoCo offscreen rendering on a headless host: an EGL-capable GL stack (NVIDIA EGL works)

## 2. Installation

**One command** (creates a conda env, installs everything in the right order, verifies imports):

```bash
bash scripts/install_env.sh          # env name: xarm-teleop (pass a name to override)
conda activate xarm-teleop
```

This recipe is validated on a clean Python 3.13 machine. If you prefer to run the steps
yourself, they are:

```bash
conda create -n xarm-teleop python=3.13 pip setuptools -y && conda activate xarm-teleop
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==2.2.6"
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"
```

The order and the two odd-looking steps matter on a fresh machine:
- **torch first** (from the cu118 index) so nothing pulls a CPU build.
- **numpy pinned** — install it before chumpy and opencv so everything is built against one ABI.
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

**Inspire RH56 dexterous hand (optional)** — RS485 on its own USB adapter, 8N1 115200, hand id 1.
```bash
ls -l /dev/ttyUSB0                 # must exist; add yourself to the dialout group for access
python scripts/teleop.py hand-test --port /dev/ttyUSB0     # sweeps each DOF, one at a time
```
The hand speaks **Modbus RTU** (slave id 1), *not* the `EB 90` framing in the RH56 manual: the
6 DOF are written as big-endian int16 to holding registers `1040..1045` in the order
`[little, ring, middle, index, thumb_bend, thumb_rot]`. Commands are raw device units, not the
manual's 0–1000 scale — the driver interpolates between the two hardware-verified poses in
`src/control/inspire_hand.py` (`CMD_OPEN` / `CMD_CLOSED`) and clamps to the envelope they span, so
the four fingers and the thumb bend close by *decreasing* while thumb rotation opposes by
*increasing*. `CMD_CLOSED` is a deliberate light grip; widen it only after checking the real end
stops. Watch the sweep and confirm each named DOF moves the finger it claims before going further.
Without `--hand-port`, everything below runs the 2-finger gripper path exactly as before.

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

**Step 2b — Dexterous hand (only if you run the RH56).** Calibrate your own hand once, then
teleop the fingers against the simulated arm — the hand is real, the arm is not, which is the
safest way to validate finger retargeting:
```bash
python scripts/teleop.py hand-calib --source realsense          # hold open hand, then a fist
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 \
    --hand-port /dev/ttyUSB0 --display
```
`hand-calib` records your open/fist finger angles to `data/hand_calib.json` (picked up
automatically; override with `--hand-calib`). The live window shows one bar per DOF so you can see
what is being commanded. Curl each finger in turn and check the right one moves.

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
- **Leave the gripper alone in software too.** With the RH56 on the flange there is no UFACTORY
  gripper to talk to; `--hand-port` therefore implies `--no-gripper`. Any gripper call latches
  controller error 19 and the arm then rejects every servo command. Use `--gripper` only if a
  2-finger gripper really is mounted alongside.
- **Verify gripper direction** on the bench first (Step 4 with the arm parked): if open/close is
  reversed, flip the gripper mapping in `XArm7Controller.servo_to` (uses `GRIPPER_MAX`; the code
  assumes 0 = closed, `GRIPPER_MAX` = open).
- **Clear the area** around the robot and keep the deadman/clutch in mind: control engages on the
  first tracked frame and re-indexes after a tracking loss.
- **Dexterous hand**: `--hand-port` moves real fingers even when the arm is in dry-run. Bench-test
  with `hand-test` first. Grip force is **not** limited by default — `--hand-force` is only written
  if you also pass `--hand-force-reg`, since FORCE_SET's register is unconfirmed on this hand — so
  the travel limit is `CMD_CLOSED` in `src/control/inspire_hand.py`, which stops at a light grip.
  Remember the hand *holds its last position* when tracking is lost — it will not drop a grasped
  object, but it will not open either.

## 6. Configuration & tuning

| What | Where | Notes |
|---|---|---|
| Reachable workspace box | `src/control/safety.py` → `DEFAULT_WORKSPACE` | **edit for your cell** |
| Max TCP speed / step | `src/control/safety.py` → `max_step_m`, or `--tcp-speed` | keep low at first |
| Camera→robot axis map | `src/retarget/wrist_to_tcp.py` → `default_align()` | if motion is mirrored/rotated |
| Position scale | `--scale` (use ~1.0 with D435 metric depth) | larger = arm moves more |
| Home pose | `src/control/xarm_controller.py` → `HOME_Q` | joint angles (rad) |
| Gripper range/direction | `src/control/xarm_controller.py` → `GRIPPER_MAX`, `servo_to` | verify on hardware |
| 2-finger gripper on/off | `--no-gripper` / `--gripper` | `--hand-port` implies `--no-gripper` |
| Smoothing | `--min-cutoff`, `--beta` (One-Euro) | lower cutoff = smoother, more lag |
| Finger open/closed range | `data/hand_calib.json` via `hand-calib` | per operator; defaults are rough |
| Finger open/closed commands | `src/control/inspire_hand.py` → `CMD_OPEN`, `CMD_CLOSED` | raw device units, verified on hardware |
| Hand speed / grip force | `--hand-speed`, `--hand-force` + `--hand-speed-reg`, `--hand-force-reg` | skipped unless you supply the registers |
| Finger command deadband | `src/control/inspire_hand.py` → `min_delta` | anti-jitter, in raw device units |

## 7. Command reference

```bash
python scripts/teleop.py wilor-image [--image IMG]        # Phase 0: model check on a still image
python scripts/teleop.py live   --source realsense|0|VID  # Phase 1: perception + overlay
python scripts/teleop.py sim    --source realsense|0|VID  # Phase 2: drive MuJoCo xArm7
python scripts/teleop.py teleop --source realsense|0|VID  # Phase 3: real xArm7 (dry-run)
python scripts/teleop.py teleop --execute --ip IP ...     # Phase 3: real xArm7 (moves!)
python scripts/teleop.py hand-test  --port /dev/ttyUSB0   # RH56 bring-up: sweep each DOF
python scripts/teleop.py hand-calib --source realsense    # record open/fist finger angles
```
Common flags: `--scale`, `--depth-scale`, `--pos-only`, `--primary auto|left|right`,
`--min-cutoff`, `--beta`, `--proc-max-side` (downscale before inference), `--device`, `--record`,
`--display` (live window on `live`/`sim`/`teleop`; Esc or `q` ends the run cleanly).
Dexterous-hand flags on `sim`/`teleop`: `--hand-port` (enables the RH56), `--hand-baud`,
`--hand-id`, `--hand-speed`, `--hand-force`, `--hand-speed-reg`, `--hand-force-reg`,
`--hand-calib`, `--hand-dry-run` (build frames without opening the port). On `teleop`,
`--hand-port` also implies `--no-gripper`; pass `--gripper` to drive both.
Outputs are H.264 mp4 (playable in VSCode/browser). The H.264 encoder is probed from the local
ffmpeg build at runtime (`libx264` -> `h264_nvenc` -> `libopenh264`), so LGPL builds without x264
work too; if none is usable the writer falls back to OpenCV mp4v and logs a warning.

## 8. Troubleshooting

- **D435 not found** — use a USB 3.0 port; check `rs-enumerate-devices`; replug.
- **Wrist depth looks wrong / huge** — depth fusion only runs with `--source realsense`; on a
  webcam/video the wrist is monocular (up-to-scale), which is why sim/video use `--scale 3`.
- **xArm won't connect** — `ping` the IP; clear errors in UFACTORY Studio; check firmware; the
  controller must not be in an error/estop state.
- **`ControllerError, code: 19` / `set_servo_cartesian_aa -> code=1`** — *End Effector
  Communication Error*: a gripper call put traffic on the tool RS485 bus with no gripper on the
  flange to answer. Run with `--hand-port` (implies `--no-gripper`). Note the SDK's `set_gripper_*`
  calls first write the gripper baud rate onto that bus via `checkset_modbus_baud`, so skipping the
  gripper commands alone is not enough — `baud_checkset` is disabled too when no gripper is
  configured. The controller does not poll the tool bus by itself, so nothing needs changing in
  UFACTORY Studio; a leftover latched error is cleared by `connect()`. Once an error is latched the
  arm ignores every servo command while the hand keeps moving, which looks like an arm-only
  failure — `connect()` now aborts instead of limping into that state.
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
- **`pip check` complains about torch / ultralytics** — WiLoR-mini declares stale bounds
  (`torch<=2.5`, `ultralytics==8.1.34`). The pinned versions are newer and validated; the warning
  is cosmetic. Downgrade to `torch 2.5.1+cu118` / `ultralytics 8.1.34` if you want it silent.
- **`Permission denied: /dev/ttyUSB0`** — add yourself to the `dialout` group
  (`sudo usermod -aG dialout $USER`, then log out and back in).
- **Hand does not respond / `no valid ack from hand`** — check the RS485 A/B polarity, the hand id
  (`--hand-id`, default 1) and the baud (`--hand-baud`, default 115200). The warning carries the
  reason (CRC, wrong id, Modbus exception, or no bytes at all). `hand-test --dry-run` prints the
  frames without a port so you can confirm the CLI side independently.
- **Wrong finger moves** — the DOF order is fixed by the vendor as
  `[little, ring, middle, index, thumb_bend, thumb_rot]`; if your unit differs, remap in
  `InspireHand.apply`. Run `hand-test` to see which physical finger each index drives.
- **Fingers barely move / slam shut** — first recalibrate (`hand-calib`): a small open/closed span
  in `data/hand_calib.json` means the captured poses were too similar, and the command warns per
  DOF when the span is under 0.15 rad. If the ratios look right but the travel does not, widen
  `CMD_OPEN` / `CMD_CLOSED` in `src/control/inspire_hand.py` — the defaults stop at a light grip.
- **Slow inference (<20 fps)** — expected on eager PyTorch (~50 ms/frame single hand on a 3090);
  the control loop is decoupled from perception. See `docs/PLAN.md` for the optimization path.

## 9. Verified vs. needs-hardware

**Verified without hardware:** WiLoR perception, retargeting, MuJoCo sim teleop, safety limits,
the full control code path (dry-run), and the depth back-projection math.

**Verified without hardware (environment):** the full Python 3.13 / CUDA 11.8 stack above —
fresh conda env, WiLoR inference on a test image (~50 ms/frame on a 3090), a MuJoCo sim teleop run
with H.264 recording, and the CPU-only test scripts.

**Verified without hardware (dex hand):** RH56 Modbus RTU frame construction byte-for-byte against
a bench script confirmed on an RH56F1, reply/exception validation and a register round-trip against
a fake-serial emulator, and the MANO -> 6-DOF finger retargeting on synthetic open/half/fist
skeletons.

**Needs on-hardware verification (first bring-up):** D435 live streaming; xArm7 motion, gripper
direction, and axis-angle pose convention; end-to-end latency; RH56 serial link, DOF order,
angle direction, thumb-rotation sense, the usable end stops beyond the light-grip default, and the
register addresses for SPEED_SET / FORCE_SET / actual angle (only ANGLE_SET is confirmed).

## 10. Project layout

```
scripts/teleop.py        single CLI entrypoint (thin; dispatches into src/)
src/
  loop.py                backend-agnostic teleop loop (perception→retarget→safety→robot)
  camera/                CameraSource: realsense (D435) / webcam / video + factory
  perception/            WiLoR wrapper (single hand) + real-time tracker + depth lift + overlay
  retarget/              wrist 6DOF → TCP (clutch/relative), pinch → gripper, fingers → dex hand
  control/               RobotBackend, EndEffector, SafetyLimiter, xArm7 controller, RH56 hand
  sim/                   MuJoCo xArm7 backend (differential IK + gripper)
  paths.py log.py video.py display.py commands.py
third_party/             mujoco_menagerie (fetched on demand; git-ignored)
outputs/                 H.264 mp4 / images (git-ignored)
docs/PLAN.md             phased implementation plan
```
