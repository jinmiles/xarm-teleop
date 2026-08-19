"""Inspire Robots RH56 dexterous hand driver (RS485, 6 DOF).

Framing follows the vendor manual ("THE DEXTEROUS HAND RH56 SERIES USER MANUAL" V1.0.9,
sec. 2.2) -- a raw register read/write protocol, *not* Modbus RTU, over 8N1 at 115200 baud
with hand id 1 by default:

    write:  EB 90 <id> <len=n+3> 12 <addr_lo> <addr_hi> <data...>      <chk>
    read:   EB 90 <id> 04        11 <addr_lo> <addr_hi> <reg_len>      <chk>
    reply:  90 EB <id> <len>     <11|12> <addr_lo> <addr_hi> <data...> <chk>

<chk> is the low byte of the sum of every byte except the two header bytes; register values are
little-endian int16. The 6 DOF are ordered [little, ring, middle, index, thumb_bend, thumb_rot]
and ANGLE_SET takes 0..1000, where 1000 = fully open and 0 = fully bent (-1 = leave that DOF
untouched). Verify direction and the thumb-rotation sense on hardware with
``teleop.py hand-test`` before running teleop.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

HEADER_TX = b"\xeb\x90"
HEADER_RX = b"\x90\xeb"
CMD_READ = 0x11
CMD_WRITE = 0x12

# register addresses (decimal, per manual sec. 2.4)
REG_CLEAR_ERROR = 1004
REG_SAVE = 1005
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_POS_ACT = 1534
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582
REG_CURRENT = 1594
REG_ERROR = 1606
REG_STATUS = 1612
REG_TEMP = 1618

N_DOF = 6
DOF_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
ANGLE_OPEN = 1000   # fully open / extended
ANGLE_CLOSED = 0    # fully bent
ANGLE_HOLD = -1     # leave this DOF where it is


def _checksum(body: bytes) -> int:
    """Low byte of the sum of everything after the 2-byte header."""
    return sum(body) & 0xFF


def build_write(hand_id: int, addr: int, data: bytes) -> bytes:
    body = bytes([hand_id, len(data) + 3, CMD_WRITE, addr & 0xFF, (addr >> 8) & 0xFF]) + data
    return HEADER_TX + body + bytes([_checksum(body)])


def build_read(hand_id: int, addr: int, length: int) -> bytes:
    body = bytes([hand_id, 0x04, CMD_READ, addr & 0xFF, (addr >> 8) & 0xFF, length])
    return HEADER_TX + body + bytes([_checksum(body)])


def parse_reply(frame: bytes, expect_cmd: int) -> bytes:
    """Validate a reply frame and return its data payload."""
    if len(frame) < 9 or not frame.startswith(HEADER_RX):
        raise ValueError(f"bad reply header: {frame.hex(' ')}")
    body, chk = frame[2:-1], frame[-1]
    if _checksum(body) != chk:
        raise ValueError(f"checksum mismatch in reply: {frame.hex(' ')}")
    if body[2] != expect_cmd:
        raise ValueError(f"unexpected command 0x{body[2]:02x} in reply: {frame.hex(' ')}")
    return bytes(body[5:])


def pack_int16(values: Sequence[int]) -> bytes:
    return b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)


def unpack_int16(data: bytes) -> list[int]:
    return [int.from_bytes(data[i:i + 2], "little", signed=True) for i in range(0, len(data), 2)]


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
        timeout: float = 0.05,
        speed: int = 500,
        force: int = 300,
        min_delta: int = 8,
        dry_run: bool = False,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.hand_id = int(hand_id)
        self.timeout = float(timeout)
        self.speed = int(speed)
        self.force = int(force)
        self.min_delta = int(min_delta)  # deadband on the 0..1000 scale (anti-jitter)
        self.dry_run = bool(dry_run)
        self._ser = None
        self._last: Optional[np.ndarray] = None
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
        self.set_speed([self.speed] * N_DOF)
        self.set_force([self.force] * N_DOF)
        angles = self.read_angles()
        if angles is not None:
            logger.info("hand angles at connect: %s",
                        ", ".join(f"{n}={a}" for n, a in zip(DOF_NAMES, angles)))
        errs = self.read_errors()
        if errs is not None and any(errs):
            logger.warning("hand reports actuator errors %s; clearing", errs)
            self.clear_error()

    def _write_registers(self, addr: int, data: bytes) -> None:
        frame = build_write(self.hand_id, addr, data)
        if self._ser is None:
            logger.debug("dry-run tx %s", frame.hex(" "))
            return
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        reply = self._ser.read(9)  # ack; short timeout keeps the control loop moving
        if len(reply) != 9:
            self._n_write_err += 1
            if self._n_write_err in (1, 10) or self._n_write_err % 100 == 0:
                logger.warning("no ack from hand for register %d (%d so far)",
                               addr, self._n_write_err)
            return
        self._n_write_err = 0

    def _read_registers(self, addr: int, length: int) -> Optional[bytes]:
        if self._ser is None:
            return None
        self._ser.reset_input_buffer()
        self._ser.write(build_read(self.hand_id, addr, length))
        frame = self._ser.read(length + 8)
        if len(frame) != length + 8:
            logger.warning("short read from hand at register %d (%d/%d bytes)",
                           addr, len(frame), length + 8)
            return None
        try:
            return parse_reply(frame, CMD_READ)
        except ValueError as exc:
            logger.warning("bad reply from hand: %s", exc)
            return None

    # --- commands -------------------------------------------------------------------------
    def set_angles(self, angles: Sequence[int]) -> None:
        """Write ANGLE_SET for all 6 DOF (0..1000, or -1 to hold that DOF)."""
        vals = [ANGLE_HOLD if int(a) < 0 else int(np.clip(a, ANGLE_CLOSED, ANGLE_OPEN))
                for a in angles]
        self._write_registers(REG_ANGLE_SET, pack_int16(vals))

    def set_speed(self, speeds: Sequence[int]) -> None:
        self._write_registers(REG_SPEED_SET,
                              pack_int16([int(np.clip(s, 0, 1000)) for s in speeds]))

    def set_force(self, forces: Sequence[int]) -> None:
        self._write_registers(REG_FORCE_SET,
                              pack_int16([int(np.clip(f, 0, 1000)) for f in forces]))

    def clear_error(self) -> None:
        self._write_registers(REG_CLEAR_ERROR, bytes([1]))

    def read_angles(self) -> Optional[list[int]]:
        data = self._read_registers(REG_ANGLE_ACT, 12)
        return unpack_int16(data) if data else None

    def read_forces(self) -> Optional[list[int]]:
        data = self._read_registers(REG_FORCE_ACT, 12)
        return unpack_int16(data) if data else None

    def read_errors(self) -> Optional[list[int]]:
        data = self._read_registers(REG_ERROR, N_DOF)
        return list(data) if data else None

    def read_status(self) -> Optional[list[int]]:
        data = self._read_registers(REG_STATUS, N_DOF)
        return list(data) if data else None

    def read_temperature(self) -> Optional[list[int]]:
        data = self._read_registers(REG_TEMP, N_DOF)
        return list(data) if data else None

    # --- EndEffector interface ------------------------------------------------------------
    def apply(self, closed: np.ndarray) -> None:
        """Command the hand from closed-ratios in [0,1] (1 = fully bent), one per DOF."""
        c = np.clip(np.asarray(closed, dtype=float).reshape(N_DOF), 0.0, 1.0)
        angles = np.rint((1.0 - c) * ANGLE_OPEN).astype(int)
        if self._last is not None and np.max(np.abs(angles - self._last)) < self.min_delta:
            return  # deadband: skip a redundant frame (less serial traffic, less finger jitter)
        self.set_angles(angles.tolist())
        self._last = angles

    def open_hand(self) -> None:
        self.set_angles([ANGLE_OPEN] * N_DOF)
        self._last = np.full(N_DOF, ANGLE_OPEN)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
            logger.info("inspire hand closed: %s", self.port)
