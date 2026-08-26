# xarm-teleop

Real-time markerless **teleoperation of a UFACTORY xArm7 + Inspire RH56 5-finger hand** from a
camera. Point an Intel RealSense **D435** at your hand: a hand-pose model (**WiLoR**) tracks your
wrist and all 21 keypoints, the arm's end-effector follows your wrist, and each finger of the RH56
follows the matching finger of your hand.

```
D435 (color + depth) ──► WiLoR (wrist 6DOF + 21 keypoints) ──┬─► wrist → TCP ──► safety ──► xArm7
                                                             └─► fingers → RH56 6 DOF (RS485)
```

## 1. How it works

**Wrist → arm.** WiLoR gives an up-to-scale wrist pose; the D435 depth at the wrist pixel replaces
its unreliable camera-Z, so deltas are metric (run with `--scale 1 --depth-scale 1`). Control is
*relative*: the arm engages on the first tracked frame and follows your motion from there, so you
can re-index by letting tracking drop. Every target is clamped to a workspace box and a max step
per tick before it reaches the arm.

**Fingers → hand.** Per-finger curl is the summed flexion along each finger chain, and thumb
rotation is the thumb metacarpal's angle out of the palm plane. These are *angles*, so they are
independent of your hand size — but not of your range of motion, which is what the calibration in
§3 captures. The result is 6 closed-ratios in `[0,1]` that map onto the hand's 6 DOF.

**The RH56 link.** The hand speaks **Modbus RTU** (slave id 1, 8N1 115200), *not* the `EB 90`
framing in the vendor manual: 6 big-endian int16 registers at `1040..1045`, in the order
`[little, ring, middle, index, thumb_bend, thumb_rot]`. Values are raw device units, not the
manual's 0–1000 scale. The driver interpolates between two hardware-verified poses and clamps to
the envelope they span, so per-DOF direction is handled automatically — the four fingers and the
thumb bend close by *decreasing*, thumb rotation opposes by *increasing*:

```python
# src/control/inspire_hand.py
CMD_OPEN   = [1740, 1740, 1740, 1740, 1350, 1500]
CMD_CLOSED = [1400, 1400, 1400, 1400, 1250, 1650]   # a light grip, not a full fist
```

`CMD_CLOSED` is deliberately short of the real end stops. Widen it once you have checked them with
`hand-test`.

The same code drives a **MuJoCo simulation** and the real robot through one backend interface, so
finger retargeting can be validated against a simulated arm before the real one moves.

## 2. Setup

**Requirements** — Linux, Python **3.13**, conda, an NVIDIA GPU with a CUDA 11.8-capable driver
(~4 GB VRAM), a RealSense **D435** on USB 3.0, an **xArm7** on your network, and an Inspire
**RH56** on RS485 (its own USB adapter, e.g. `/dev/ttyUSB0`).

```bash
bash scripts/install_env.sh          # env name: xarm-teleop (pass a name to override)
conda activate xarm-teleop
python scripts/teleop.py wilor-image # model check; weights download to the HF cache (~2 GB)
```

The script installs torch from the cu118 index first, pins numpy before chumpy and opencv, builds
chumpy with `--no-build-isolation`, and installs WiLoR-mini with `--no-deps`. That order matters
on a fresh machine — see the script if you want to run the steps yourself. For the simulation
backend, fetch the xArm7 MuJoCo model once:

```bash
git clone --depth 1 --filter=blob:none \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
```

**Hardware checks**

```bash
rs-enumerate-devices | head        # D435 on a USB 3.0 (blue) port
ping <YOUR_XARM_IP>                # xArm7; enable remote motion in UFACTORY Studio, no errors
ls -l /dev/ttyUSB0                 # RH56; add yourself to the dialout group for access
```

Then bring the hand up on its own — this is the fastest way to prove the RS485 link before any
camera or arm is involved:

```bash
python scripts/teleop.py hand-test --port /dev/ttyUSB0
```

It refuses to start unless the hand answers a Modbus read, then sweeps each DOF open→bent→open.
Confirm each named DOF moves the finger it claims.

## 3. Calibration

Run once per operator. It records your own open-hand and fist finger angles, which is what turns
raw curl into a usable 0–1 ratio:

```bash
python scripts/teleop.py hand-calib --source realsense    # hold open hand, then a fist
```

Written to `data/hand_calib.json` and picked up automatically by `sim`/`teleop`, which take
`--hand-calib <path>` to point elsewhere and `--hand-calib none` to force the built-in defaults.
**Be in the pose before the countdown ends** —
an "open" capture that is really a half-fist leaves no usable span, and every DOF then saturates
to a constant. Startup logs the span and warns per DOF when it is under 0.15 rad.

## 4. Run the 5-finger teleop

Validate against the simulated arm first — the hand is real, the arm is not:

```bash
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 \
    --hand-port /dev/ttyUSB0 --display
```

Curl each finger in turn and check the right one moves. The window shows one bar per DOF, so you
can see what is being commanded. Then the real arm, at low speed with the E-stop in reach:

```bash
python scripts/teleop.py teleop --execute --ip <YOUR_XARM_IP> \
    --source realsense --scale 1.0 --depth-scale 1.0 --tcp-speed 80 \
    --hand-port /dev/ttyUSB0 --display
```

Drop `--execute` for a dry run that builds every command without connecting to the arm. Drop
`--display` when headless; recording to `outputs/` is unaffected.

`--hand-port` implies `--no-gripper`: the RH56 occupies the tool flange, and any UFACTORY gripper
call would latch controller error 19. Pass `--gripper` only if a 2-finger gripper really is
mounted alongside.

| Flag | Default | Notes |
|---|---|---|
| `--scale`, `--depth-scale` | 3.0, 0.4 | use `1.0`/`1.0` with D435 metric depth |
| `--tcp-speed` | 100 mm/s | keep ≤ 80 for first runs |
| `--primary` | `right` | also overrides the detector's handedness, which sets thumb-rotation direction |
| `--min-cutoff`, `--beta` | 1.0, 0.02 | One-Euro smoothing; lower cutoff = smoother, more lag |
| `--hand-rate` | 20 Hz | cap on finger command rate (RS485 needs quiet between frames) |
| `--hand-id`, `--hand-baud` | 1, 115200 | RH56 Modbus address / baud |

Other tunables live in the code: the workspace box and max step per tick in
`src/control/safety.py` (`DEFAULT_WORKSPACE`, `max_step_m`), the home pose in
`src/control/xarm_controller.py` (`HOME_Q`), and the finger command envelope and anti-jitter
deadband in `src/control/inspire_hand.py` (`CMD_OPEN`/`CMD_CLOSED`, `min_delta`).

## 5. Safety

- **Physical E-stop within reach.** The camera is not a safety system.
- **Set `DEFAULT_WORKSPACE` in `src/control/safety.py`** to a reachable, collision-free box for
  your cell. Targets are hard-clamped to it.
- **Start slow** — `--tcp-speed 80` or lower, and a small `max_step_m` (0.02 m/tick default).
- **`--hand-port` moves real fingers even when the arm is in dry-run.** Bench-test with
  `hand-test` first.
- **The hand holds its last position when tracking is lost.** It will not drop a grasped object,
  but it will not open either.

## 6. Troubleshooting

- **`no reply from the hand ... after 3 probes`** — check the power first. A powered-down hand
  rests in the open pose and is indistinguishable, from the camera side, from teleop that tracks
  perfectly and never grips. Then RS485 A/B polarity, `--hand-id`, `--hand-baud`, the port.
- **`ControllerError, code: 19` / `set_servo_cartesian_aa -> code=1`** — a gripper call put traffic
  on the tool RS485 bus with no gripper to answer. Run with `--hand-port` (implies `--no-gripper`).
  Nothing needs changing in UFACTORY Studio; `connect()` clears a leftover latched error.
- **`Permission denied: /dev/ttyUSB0`** — `sudo usermod -aG dialout $USER`, then re-login.
- **Fingers do not move but the hand opens at startup** — the link is fine, the targets are not.
  The run logs `dex ratio range:` every 150 tracked frames and at exit; a range stuck at one value
  means the calibration saturates. Confirm with `--hand-calib none`, then re-run `hand-calib`.
- **Fingers stop short of a full close** — read the per-DOF `dex raw vs calib` table printed at
  exit (the camera overlay also shows the live raw angle next to each ratio bar). If the raw angle
  of your fist stays below the calibrated closed angle, the ratio tops out below 1.0 and the hand
  cannot fully close: re-run `hand-calib` squeezing the same fist you use during teleop. WiLoR
  underestimates the curl of a clenched fist, so the closed capture must come from WiLoR's own
  estimate, not from an assumed anatomical angle.
- **Wrong finger moves** — the DOF order is `[little, ring, middle, index, thumb_bend, thumb_rot]`.
  Run `hand-test` to see which physical finger each index drives.
- **Only the thumb rotates the wrong way** — the palm-normal sign comes from the detector's
  handedness. State your hand explicitly with `--primary right` or `--primary left`.
- **Wrist depth looks wrong** — depth fusion only runs with `--source realsense`; on a webcam or
  video the wrist is monocular and up-to-scale, which is why those use `--scale 3`.
- **Slow inference (<20 fps)** — expected on eager PyTorch (~50 ms/frame on a 3090); the control
  loop is decoupled from perception.
