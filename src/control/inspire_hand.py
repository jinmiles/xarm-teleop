"""Inspire Robots RH56 dexterous hand driver (RS485, 6 DOF).

The hand speaks **standard Modbus RTU** over 8N1 at 115200 baud with slave id 1 -- *not* the
``EB 90`` vendor framing printed in the RH56 manual. Framing and the register below were taken
from a bench script verified against an RH56F1:

    write:  <id> 10 <addr_hi> <addr_lo> <n_reg_hi> <n_reg_lo> <n_bytes> <data...> <crc_lo> <crc_hi>
    ack:    <id> 10 <addr_hi> <addr_lo> <n_reg_hi> <n_reg_lo>                      <crc_lo> <crc_hi>
    read:   <id> 03 <addr_hi> <addr_lo> <n_reg_hi> <n_reg_lo>                      <crc_lo> <crc_hi>
    reply:  <id> 03 <n_bytes> <data...>                                            <crc_lo> <crc_hi>

Register values are **big-endian** int16; the CRC-16/MODBUS is appended low byte first.

The 6 DOF are ordered ``[little, ring, middle, index, thumb_bend, thumb_rot]`` and ANGLE_SET
takes raw device units, *not* the 0..1000 scale from the manual. The two poses in ``CMD_OPEN`` /
``CMD_CLOSED`` are the ones verified on hardware, and every command is clamped into the envelope
they span -- direction differs per DOF (the four fingers and the thumb bend close by *decreasing*,
thumb rotation opposes by *increasing*), which the open->closed interpolation in ``apply`` handles
on its own. ``CMD_CLOSED`` is a light grip, i.e. deliberately short of a full fist; widen it only
after checking the real end stops with ``teleop.py hand-test``.

Only ANGLE_SET is a confirmed register address. Speed/force writes and actual-position reads are
therefore opt-in (``speed_reg`` / ``force_reg`` / ``angle_act_reg``) and skipped by default rather
than aimed at guessed addresses -- a blind write into the neighbouring config block can change the
hand's id or baud rate and take the link down.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

FUNC_READ = 0x03           # read holding registers
FUNC_WRITE = 0x10          # write multiple registers
EXCEPTION_FLAG = 0x80      # set in the function code of a Modbus exception response

REG_ANGLE_SET = 1040       # hardware-verified; 6 consecutive int16 registers

N_DOF = 6
DOF_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")

# Hardware-verified poses, in DOF order. Fingers/thumb bend close downward, thumb rotation upward.
CMD_OPEN = np.array([1740, 1740, 1740, 1740, 1350, 1500], dtype=int)
CMD_CLOSED = np.array([1400, 1400, 1400, 1400, 1250, 1650], dtype=int)
CMD_MIN = np.minimum(CMD_OPEN, CMD_CLOSED)
CMD_MAX = np.maximum(CMD_OPEN, CMD_CLOSED)

WRITE_ACK_LEN = 8


def crc16(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _with_crc(body: bytes) -> bytes:
    return body + crc16(body).to_bytes(2, "little")


def pack_int16(values: Sequence[int]) -> bytes:
    return b"".join(int(v).to_bytes(2, "big", signed=True) for v in values)


def unpack_int16(data: bytes) -> list[int]:
    return [int.from_bytes(data[i:i + 2], "big", signed=True) for i in range(0, len(data), 2)]


def build_write(hand_id: int, addr: int, values: Sequence[int]) -> bytes:
    """Modbus function 0x10 -- write multiple holding registers."""
    data = pack_int16(values)
    return _with_crc(bytes([hand_id, FUNC_WRITE]) + addr.to_bytes(2, "big")
                     + len(values).to_bytes(2, "big") + bytes([len(data)]) + data)


def build_read(hand_id: int, addr: int, count: int) -> bytes:
    """Modbus function 0x03 -- read holding registers."""
    return _with_crc(bytes([hand_id, FUNC_READ]) + addr.to_bytes(2, "big")
                     + count.to_bytes(2, "big"))


def _check_frame(frame: bytes, hand_id: int, func: int) -> None:
    if len(frame) < 5:
        raise ValueError(f"short reply: {frame.hex(' ')}")
    if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
        raise ValueError(f"crc mismatch in reply: {frame.hex(' ')}")
    if frame[0] != hand_id:
        raise ValueError(f"reply from id {frame[0]}, expected {hand_id}: {frame.hex(' ')}")
    if frame[1] == func | EXCEPTION_FLAG:
        raise ValueError(f"modbus exception 0x{frame[2]:02x} for function 0x{func:02x}")
    if frame[1] != func:
        raise ValueError(f"unexpected function 0x{frame[1]:02x} in reply: {frame.hex(' ')}")


def parse_write_ack(frame: bytes, hand_id: int, addr: int, count: int) -> None:
    """Validate the 8-byte echo the hand returns for a 0x10 write; raise on anything else."""
    _check_frame(frame, hand_id, FUNC_WRITE)
    if len(frame) != WRITE_ACK_LEN:
        raise ValueError(f"write ack must be {WRITE_ACK_LEN} bytes: {frame.hex(' ')}")
    echo_addr = int.from_bytes(frame[2:4], "big")
    echo_count = int.from_bytes(frame[4:6], "big")
    if (echo_addr, echo_count) != (addr, count):
        raise ValueError(f"ack echoes register {echo_addr}x{echo_count}, sent {addr}x{count}")


def parse_read_reply(frame: bytes, hand_id: int, count: int) -> bytes:
    """Validate a 0x03 reply and return its raw data payload."""
    _check_frame(frame, hand_id, FUNC_READ)
    if len(frame) != 2 * count + 5 or frame[2] != 2 * count:
        raise ValueError(f"expected {count} registers, got: {frame.hex(' ')}")
    return bytes(frame[3:-2])


class InspireHand:
    """Serial driver for an RH56-series hand.

    ``dry_run`` builds and logs frames without opening a port, so the retargeting path can be
    exercised (and the CLI validated) with no hardware attached.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        hand_id: int = 1,
        timeout: float = 0.15,
        min_interval: float = 0.05,
        speed: Optional[int] = None,
        force: Optional[int] = None,
        min_delta: int = 4,
        open_cmd: Optional[Sequence[int]] = None,
        closed_cmd: Optional[Sequence[int]] = None,
        speed_reg: Optional[int] = None,
        force_reg: Optional[int] = None,
        angle_act_reg: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.hand_id = int(hand_id)
        # The ack read returns as soon as its 8 bytes arrive, so a generous timeout costs nothing
        # while the hand is healthy. Bailing out early is what hurts: this is a half-duplex bus, so
        # returning before the hand has finished replying lets the next write collide with the tail
        # of the ack, and the hand drops that command.
        self.timeout = float(timeout)
        self.min_interval = float(min_interval)  # floor on the gap between frames (bus quiet time)
        self.speed = speed
        self.force = force
        self.min_delta = int(min_delta)  # deadband in raw device units (anti-jitter)
        self.open_cmd = np.asarray(CMD_OPEN if open_cmd is None else open_cmd, dtype=int)
        self.closed_cmd = np.asarray(CMD_CLOSED if closed_cmd is None else closed_cmd, dtype=int)
        self.cmd_min = np.minimum(self.open_cmd, self.closed_cmd)
        self.cmd_max = np.maximum(self.open_cmd, self.closed_cmd)
        self.speed_reg = speed_reg
        self.force_reg = force_reg
        self.angle_act_reg = angle_act_reg
        self.dry_run = bool(dry_run)
        self._ser = None
        self._last: Optional[np.ndarray] = None
        self._t_last_write: Optional[float] = None
        self._n_writes = 0
        self._n_write_err = 0

    # --- transport ------------------------------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            logger.warning("inspire hand DRY-RUN: frames are built and logged, not sent")
        else:
            import serial  # pyserial; imported lazily so the repo works without a hand
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout,
                                      write_timeout=max(self.timeout, 0.2))
            logger.info("inspire hand opened: %s @%d (id=%d)", self.port, self.baud, self.hand_id)
        if self.speed is not None and self.speed_reg is not None:
            self.set_speed([self.speed] * N_DOF)
        if self.force is not None and self.force_reg is not None:
            self.set_force([self.force] * N_DOF)
        if (self.speed is not None and self.speed_reg is None) or \
           (self.force is not None and self.force_reg is None):
            logger.info("hand speed/force not written: only ANGLE_SET (register %d) is a verified "
                        "address on this hand; pass the real ones via speed_reg/force_reg",
                        REG_ANGLE_SET)
        setpoint = self.read_angle_set()
        if setpoint is not None:
            logger.info("hand ANGLE_SET at connect: %s",
                        ", ".join(f"{n}={a}" for n, a in zip(DOF_NAMES, setpoint)))

    def _write_registers(self, addr: int, values: Sequence[int]) -> None:
        frame = build_write(self.hand_id, addr, values)
        if self._ser is None:
            logger.debug("dry-run tx %s", frame.hex(" "))
            return
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        self._ser.flush()  # half-duplex RS485: let the frame leave before listening for the ack
        self._n_writes += 1
        reply = self._ser.read(WRITE_ACK_LEN)
        self._t_last_write = time.perf_counter()
        try:
            parse_write_ack(reply, self.hand_id, addr, len(values))
        except ValueError as exc:
            self._n_write_err += 1
            if self._n_write_err in (1, 10) or self._n_write_err % 100 == 0:
                logger.warning("no valid ack from hand for register %d (%d so far): %s",
                               addr, self._n_write_err, exc)
            return
        self._n_write_err = 0

    def _read_registers(self, addr: int, count: int) -> Optional[list[int]]:
        if self._ser is None:
            return None
        self._ser.reset_input_buffer()
        self._ser.write(build_read(self.hand_id, addr, count))
        frame = self._ser.read(2 * count + 5)
        try:
            return unpack_int16(parse_read_reply(frame, self.hand_id, count))
        except ValueError as exc:
            logger.warning("bad reply from hand at register %d: %s", addr, exc)
            return None

    # --- commands -------------------------------------------------------------------------
    def set_angles(self, angles: Sequence[int]) -> None:
        """Write ANGLE_SET for all 6 DOF in raw device units, clamped to the verified envelope."""
        vals = np.clip(np.asarray(angles, dtype=int).reshape(N_DOF), self.cmd_min, self.cmd_max)
        self._write_registers(REG_ANGLE_SET, vals.tolist())
        self._last = vals

    def set_speed(self, speeds: Sequence[int]) -> None:
        if self.speed_reg is None:
            return
        self._write_registers(self.speed_reg, [int(np.clip(s, 0, 1000)) for s in speeds])

    def set_force(self, forces: Sequence[int]) -> None:
        if self.force_reg is None:
            return
        self._write_registers(self.force_reg, [int(np.clip(f, 0, 1000)) for f in forces])

    def read_angle_set(self) -> Optional[list[int]]:
        """Read the commanded setpoints back -- a link check, not the measured finger position."""
        return self._read_registers(REG_ANGLE_SET, N_DOF)

    def read_angles(self) -> Optional[list[int]]:
        """Measured finger angles, if the actual-position register is known for this hand."""
        if self.angle_act_reg is None:
            return None
        return self._read_registers(self.angle_act_reg, N_DOF)

    # --- EndEffector interface ------------------------------------------------------------
    def to_command(self, closed: np.ndarray) -> np.ndarray:
        """Closed-ratios in [0,1] (1 = fully bent) -> raw device units, per DOF."""
        c = np.clip(np.asarray(closed, dtype=float).reshape(N_DOF), 0.0, 1.0)
        return np.rint(self.open_cmd + c * (self.closed_cmd - self.open_cmd)).astype(int)

    def apply(self, closed: np.ndarray) -> None:
        """Command the hand from closed-ratios in [0,1] (1 = fully bent), one per DOF."""
        if (self._t_last_write is not None
                and time.perf_counter() - self._t_last_write < self.min_interval):
            return  # rate limit: leave the bus quiet; the next tick carries a fresher target
        angles = self.to_command(closed)
        if self._last is not None and np.max(np.abs(angles - self._last)) < self.min_delta:
            return  # deadband: skip a redundant frame (less serial traffic, less finger jitter)
        self.set_angles(angles)

    def open_hand(self) -> None:
        self.set_angles(self.open_cmd)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
            # "the fingers never moved" is ambiguous without these: no frames means the retargeting
            # or the deadband held it still, frames with errors means the link is the problem.
            logger.info("inspire hand closed: %s (%d frames written, %d unacked; last command %s)",
                        self.port, self._n_writes, self._n_write_err,
                        self._last.tolist() if self._last is not None else "none")
