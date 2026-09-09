# xarm-teleop

> 한국어 문서: [README.ko.md](README.ko.md)

Real-time **teleoperation of a UFACTORY xArm7 + Inspire RH56 5-finger hand**. Point an Intel
RealSense **D435** at your hand: the arm's end-effector follows your wrist and each finger of the
RH56 follows the matching finger of your hand.

The camera always supplies *where* the hand is. *How* it is posed comes from one of two sources,
chosen at run time with `--pose-source`:

```
POSITION  D435 color+depth ─► WiLoR wrist pixel + metric depth ─► wrist → TCP ─► safety ─► xArm7

POSE      method 1 (default)  WiLoR 21 keypoints, markerless  ─┐
          method 2            Noitom mocap glove over the LAN ─┴─► fingers → RH56 6 DOF (RS485)
```

**Method 1** needs nothing but the camera. **Method 2** adds a mocap glove whose IMUs measure
finger flexion directly, so the fingers keep working where the camera cannot see them — a fist
closing around an object, fingers behind the palm, motion blur — at the cost of wearing the glove
and running Axis Studio on a Windows laptop cabled to this PC.

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

**The glove (method 2).** Axis Studio on the Windows laptop broadcasts the glove's bone rotations
over the ethernet link; this PC receives them and runs forward kinematics over the hand skeleton
to produce the same 21-keypoint layout WiLoR emits. Everything downstream — curl, thumb rotation,
calibration, the overlay — is then the code above, unchanged. The glove is an IMU device and never
reports a position, which is exactly the half the camera keeps.

The two sensors sit in different frames, so the rotation between them is measured rather than
configured: both see the same physical palm, so a palm frame built the same way from each
keypoint set differs by exactly that rotation. It is averaged over the first `--glove-align-frames`
frames after every re-acquisition, which makes re-indexing the clutch also re-align the glove.
Finger angles do not depend on it and work from the first frame.

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

### 2.1 The glove and the Windows laptop (method 2 only)

You need the Noitom gloves and their hub, the Windows laptop Axis Studio is licensed on, and an
ethernet cable between that laptop and this PC. Skip this whole section for method 1.

**a. Wire the two machines.** A direct cable is enough — no switch, no router, no DHCP — as long
as both ends have a static address on the same subnet. The addresses below are the ones used in
the rest of this section; any private subnet works.

- *Windows*: Settings → Network & Internet → Ethernet → IP assignment → Edit → Manual, IPv4
  on, IP `192.168.2.16`, mask `255.255.255.0`, gateway blank.
- *Here*: `sudo ip link set <iface> up && sudo ip addr add 192.168.2.15/24 dev <iface>`
  (`ip -br link` lists the interfaces), or set the same statically in NetworkManager so it
  survives a reboot.
- Check it both ways: `ping 192.168.2.15` from Windows, `ping 192.168.2.16` from here. Answer any
  Windows firewall prompt with **allow on private networks**.

**b. Axis Studio.** Install it on the Windows laptop with its licence dongle plugged in, power up
the glove hub, pair the gloves, and set **Settings → Working Mode** to the **Hand** mode so the
stream carries finger bones. Then calibrate the glove with Axis Studio's on-screen poses — the
teleop calibration in §3 is a separate, later step and does not replace this one.

**c. BVH Broadcasting.** Open **Settings → BVH Broadcasting**, turn on the toggle at the top
right, and set every field exactly as below. Anything else and the receiver either sees nothing or
sees a skeleton it cannot read:

| Field | Value |
|---|---|
| Frame Format → Type | `Binary`, *Use old header format* unchecked |
| Sync | `GenLock` checked |
| Skeleton | `Axis Studio` |
| BVH Format → Rotation | `XYZ` |
| BVH Format → Displacement | **checked** — this is what carries the bone lengths |
| Coordinate system | `OPT` |
| Protocol | `UDP` |
| Local Address | the Windows LAN address (`192.168.2.16`), port `7001` |
| Destination Address | **this PC's** LAN address (`192.168.2.15`), port `7012` |

Loopback addresses (`127.0.0.1`) never work — the destination must be the address this PC answers
`ping` on. Press OK; broadcasting starts immediately and keeps running while Axis Studio is open.

*Displacement* is the one setting people leave off. Without it the stream carries rotations but no
bone lengths, and the receiver falls back to a nominal hand skeleton (it says so in a warning),
which makes every finger angle approximate.

*TCP instead of UDP*: set Protocol to `TCP` in the same dialog and run with
`--glove-host 192.168.2.16 --glove-port <that port>`; this PC then connects to Axis Studio instead
of listening for its broadcast.

**d. Prove the link** before a camera, an arm or the hand is in the picture:

```bash
python scripts/teleop.py glove-test              # Ctrl+C to stop
```

It refuses to start unless a frame actually arrives, then logs the six raw finger angles (and,
once calibrated, the 0-1 ratios) a few times a second. Curl each finger in turn and watch its own
number move. Add `--glove-hand left` if you glove the other hand, or `--glove-port <n>` if you
changed the destination port.

The receiver uses the Noitom MocapApi wrapper from `mocap_ros_py`, reused in place from
`/home/user/extra_workdir/mocap_ros_py` (see `src/paths.py`); no ROS is involved on either side.

## 3. Calibration

Run once per operator. It records your own open-hand and fist finger angles, which is what turns
raw curl into a usable 0–1 ratio:

```bash
python scripts/teleop.py hand-calib --source realsense                  # method 1
python scripts/teleop.py hand-calib --pose-source glove                 # method 2 (no camera)
```

Each source gets its own file — `data/hand_calib.json` for the camera, `data/glove_hand_calib.json`
for the glove — because they measure the same hand differently; the matching one is picked up
automatically by `sim`/`teleop`, which take `--hand-calib <path>` to point elsewhere and
`--hand-calib none` to force the built-in defaults. The glove capture needs no camera, arm or
hand: only joint angles are recorded, and the glove supplies those on its own.
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

**Method 2** is the same commands with `--pose-source glove` — the camera stays in the loop for
the wrist position, so `--source` is still required:

```bash
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 \
    --pose-source glove --hand-port /dev/ttyUSB0 --display
```

The run refuses to start until the glove sends a frame, the overlay is labelled `pose:glove`, and
the drawn hand skeleton is the glove's own pose reprojected onto your hand — so the video shows
what actually reaches the RH56. If the glove goes quiet for longer than `--glove-timeout`, the
frame counts as untracked: the arm holds and the fingers keep their last pose, exactly as when the
camera loses the hand.

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
| `--pose-source` | `wilor` | `glove` switches the hand pose to the mocap glove (method 2) |
| `--glove-port`, `--glove-host` | 7012, — | UDP listen port; `--glove-host` connects over TCP instead |
| `--glove-hand` | follows `--primary` | which gloved hand to read |
| `--glove-timeout`, `--glove-wait` | 0.3 s, 5 s | staleness cutoff during teleop; wait for the first frame at startup |
| `--glove-align-frames` | 30 | frames averaged for the glove→camera rotation after each re-acquisition |

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
- **`no glove data on udp :7012 after 5s`** — Axis Studio is not reaching this PC. Check BVH
  Broadcasting is enabled with *Destination* = this PC's LAN address and port `7012` (not a
  loopback address), that the cable link is up (`ping` both ways), and that the Windows firewall
  is not blocking the outbound stream. `--glove-port` must match the Destination port.
- **Glove connects but no bones arrive** — the stream carries the other hand, or Axis Studio is
  not in a hand/glove working mode. Pass `--glove-hand left|right`; `glove-test` lists any bones
  missing from the stream at startup.
- **Glove fingers move but the arm rotation is off** — the glove→camera alignment locked on a bad
  frame. Let tracking drop and re-acquire (that re-estimates it), or raise `--glove-align-frames`.
  `--pos-only` ignores hand orientation entirely.
- **Wrong finger moves** — the DOF order is `[little, ring, middle, index, thumb_bend, thumb_rot]`.
  Run `hand-test` to see which physical finger each index drives.
- **Only the thumb rotates the wrong way** — the palm-normal sign comes from the detector's
  handedness. State your hand explicitly with `--primary right` or `--primary left`.
- **Wrist depth looks wrong** — depth fusion only runs with `--source realsense`; on a webcam or
  video the wrist is monocular and up-to-scale, which is why those use `--scale 3`.
- **Slow inference (<20 fps)** — expected on eager PyTorch (~50 ms/frame on a 3090); the control
  loop is decoupled from perception.
