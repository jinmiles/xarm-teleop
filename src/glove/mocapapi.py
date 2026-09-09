"""Project-owned ctypes binding for Noitom's MocapApi (``librobotapi``).

The glove receiver needs four of the SDK's interfaces -- settings, application, avatar, joint --
and nothing else. Binding them here rather than importing Noitom's own python wrapper keeps the
teleop machine free of any external checkout: the shared library ships in ``vendor/noitom/`` and
this file is the only code between it and ``glove/client.py``.

The proc tables below are function-pointer structs read positionally by the library, so every
field of an interface must be declared, in order, even the ones never called -- a missing entry
shifts every later slot onto the wrong function. Handles are opaque uint64 cookies.

Library: MocapApi / librobotapi, redistributed by Noitom at github.com/pnmocap/mocap_ros_py.
"""
from __future__ import annotations

import ctypes
import platform
from ctypes import (CFUNCTYPE, POINTER, Structure, Union, byref, c_bool, c_char_p, c_double,
                    c_float, c_int32, c_uint16, c_uint32, c_uint64, sizeof)
from pathlib import Path
from typing import Optional

from ..log import get_logger

logger = get_logger(__name__)

Handle = c_uint64

ERRORS = (
    "NoError", "MoreEvent", "InsufficientBuffer", "InvalidObject", "InvalidHandle",
    "InvalidParameter", "NotSupported", "IgnoreUDPSettings", "IgnoreTCPSettings",
    "IgnoreBvhSettings", "JointNotFound", "WithoutTransformation", "NoneMessage",
    "NoneParent", "NoneChild", "AddressInUse",
)
NO_ERROR = 0

EVENT_AVATAR_UPDATED = 256
BVH_ROTATION_XYZ = 0

# .so filenames as Noitom ships them, keyed by platform.machine()
_LIB_NAMES = {
    "x86_64": "librobotapi_x86-64.so",
    "aarch64": "librobotapi_arm64.so",
    "arm64": "librobotapi_arm64.so",
    "arm": "librobotapi_arm.so",
}


class MocapApiError(RuntimeError):
    """A MocapApi call returned a non-zero status."""


def _check(err: int, what: str) -> None:
    if err != NO_ERROR:
        name = ERRORS[err] if 0 <= err < len(ERRORS) else str(err)
        raise MocapApiError(f"{what}: {name}")


# --- interface proc tables ------------------------------------------------------------------
class _SettingsApi(Structure):
    _fields_ = [
        ("CreateSettings", CFUNCTYPE(c_int32, POINTER(Handle))),
        ("DestroySettings", CFUNCTYPE(c_int32, Handle)),
        ("SetSettingsUDP", CFUNCTYPE(c_int32, c_uint16, Handle)),
        ("SetSettingsTCP", CFUNCTYPE(c_int32, c_char_p, c_uint16, Handle)),
        ("SetSettingsBvhRotation", CFUNCTYPE(c_int32, c_int32, Handle)),
        ("SetSettingsBvhDisplacement", CFUNCTYPE(c_int32, c_int32, Handle)),
        ("SetSettingsBvhData", CFUNCTYPE(c_int32, c_int32, Handle)),
        ("SetSettingsCalcData", CFUNCTYPE(c_int32, Handle)),
    ]


class _ApplicationApi(Structure):
    _fields_ = [
        ("CreateApplication", CFUNCTYPE(c_int32, POINTER(Handle))),
        ("DestroyApplication", CFUNCTYPE(c_int32, Handle)),
        ("SetApplicationSettings", CFUNCTYPE(c_int32, Handle, Handle)),
        ("SetApplicationRenderSettings", CFUNCTYPE(c_int32, Handle, Handle)),
        ("OpenApplication", CFUNCTYPE(c_int32, Handle)),
        ("EnableApplicationCacheEvents", CFUNCTYPE(c_int32, Handle)),
        ("DisableApplicationCacheEvents", CFUNCTYPE(c_int32, Handle)),
        ("ApplicationCacheEventsIsEnabled", CFUNCTYPE(c_int32, POINTER(c_bool), Handle)),
        ("CloseApplication", CFUNCTYPE(c_int32, Handle)),
        ("GetApplicationRigidBodies", CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), Handle)),
        ("GetApplicationAvatars", CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), Handle)),
        ("PollApplicationNextEvent", CFUNCTYPE(c_int32, POINTER(None), POINTER(c_uint32), Handle)),
    ]


class _AvatarApi(Structure):
    _fields_ = [
        ("GetAvatarIndex", CFUNCTYPE(c_int32, POINTER(c_uint32), Handle)),
        ("GetAvatarRootJoint", CFUNCTYPE(c_int32, POINTER(Handle), Handle)),
        ("GetAvatarJoints", CFUNCTYPE(c_int32, POINTER(Handle), POINTER(c_uint32), Handle)),
        ("GetAvatarJointByName", CFUNCTYPE(c_int32, c_char_p, POINTER(Handle), Handle)),
        ("GetAvatarName", CFUNCTYPE(c_int32, POINTER(c_char_p), Handle)),
        ("GetAvatarRigidBodies", CFUNCTYPE(c_int32, POINTER(Handle), POINTER(c_uint32), Handle)),
        ("GetAvatarJointHierarchy", CFUNCTYPE(c_int32, POINTER(c_char_p))),
        ("GetAvatarPostureIndex", CFUNCTYPE(c_int32, POINTER(c_uint32), POINTER(Handle))),
        ("GetAvatarPostureTimeCode",
         CFUNCTYPE(c_int32, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32),
                   POINTER(c_uint32), POINTER(Handle))),
    ]


class _JointApi(Structure):
    _fields_ = [
        ("GetJointName", CFUNCTYPE(c_int32, POINTER(c_char_p), Handle)),
        ("GetJointLocalRotation",
         CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float),
                   POINTER(c_float), Handle)),
        ("GetJointLocalRotationByEuler",
         CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), Handle)),
        ("GetJointLocalTransformation",
         CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), Handle)),
        ("GetJointDefaultLocalTransformation",
         CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), Handle)),
        ("GetJointChild", CFUNCTYPE(c_int32, POINTER(Handle), POINTER(c_uint32), Handle)),
        ("GetJointBodyPart", CFUNCTYPE(c_int32, POINTER(Handle), Handle)),
        ("GetJointSensorModule", CFUNCTYPE(c_int32, POINTER(Handle), Handle)),
        ("GetJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), Handle)),
        ("GetJointNameByTag", CFUNCTYPE(c_int32, POINTER(c_char_p), c_int32)),
        ("GetJointChildJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), POINTER(c_uint32), c_int32)),
        ("GetJointParentJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), c_int32)),
    ]


class _EventReserved(Structure):
    _fields_ = [(f"reserved{i}", c_uint64) for i in range(6)]


class _EventData(Union):
    _fields_ = [("reserved", _EventReserved), ("avatar_handle", Handle), ("error", c_int32)]


class Event(Structure):
    _fields_ = [("size", c_uint32), ("event_type", c_int32), ("timestamp", c_double),
                ("event_data", _EventData)]


# --- library loading ------------------------------------------------------------------------
_lib = None
_interfaces: dict[bytes, object] = {}


def library_name() -> str:
    """Noitom's .so filename for this CPU architecture."""
    machine = platform.machine()
    name = _LIB_NAMES.get(machine)
    if name is None:
        raise MocapApiError(f"no MocapApi library for this architecture: {machine}")
    return name


def load(lib_path: Optional[Path] = None):
    """Load librobotapi once per process and return the ctypes handle."""
    global _lib
    if _lib is not None:
        return _lib
    if lib_path is None:
        from .. import paths

        lib_path = paths.mocapapi_lib(library_name())
    lib_path = Path(lib_path)
    if not lib_path.exists():
        raise MocapApiError(
            f"MocapApi library not found: {lib_path}. It ships in the repo under "
            f"vendor/noitom/; set XARM_TELEOP_MOCAPAPI to point elsewhere."
        )
    _lib = ctypes.CDLL(str(lib_path))
    _lib.MCPGetGenericInterface.argtypes = [c_char_p, POINTER(ctypes.c_void_p)]
    _lib.MCPGetGenericInterface.restype = c_int32
    logger.info("MocapApi loaded: %s", lib_path)
    return _lib


def _interface(version: bytes, struct_type):
    """Fetch a proc table by interface version string (cached, as the SDK returns a singleton)."""
    if version not in _interfaces:
        lib = load()
        ptr = POINTER(struct_type)()
        _check(lib.MCPGetGenericInterface(c_char_p(version), ctypes.cast(byref(ptr),
                                                                        POINTER(ctypes.c_void_p))),
               f"get interface {version.decode()}")
        _interfaces[version] = ptr
    return _interfaces[version].contents


# --- thin object wrappers -------------------------------------------------------------------
class Settings:
    """Receiver settings: where the BVH stream comes from and how it is encoded."""

    def __init__(self) -> None:
        self._api = _interface(b"PROC_TABLE:IMCPSettings_001", _SettingsApi)
        self.handle = Handle()
        _check(self._api.CreateSettings(byref(self.handle)), "create settings")

    def set_udp(self, port: int) -> None:
        _check(self._api.SetSettingsUDP(c_uint16(int(port)), self.handle), f"set udp port {port}")

    def set_tcp(self, host: str, port: int) -> None:
        _check(self._api.SetSettingsTCP(host.encode("utf8"), c_uint16(int(port)), self.handle),
               f"set tcp {host}:{port}")

    def set_bvh_rotation(self, order: int = BVH_ROTATION_XYZ) -> None:
        _check(self._api.SetSettingsBvhRotation(c_int32(int(order)), self.handle),
               "set bvh rotation")

    def destroy(self) -> None:
        if self.handle:
            self._api.DestroySettings(self.handle)
            self.handle = Handle()


class Joint:
    """One bone of the streamed skeleton."""

    __slots__ = ("_api", "handle")

    def __init__(self, handle: Handle) -> None:
        self._api = _interface(b"PROC_TABLE:IMCPJoint_003", _JointApi)
        self.handle = handle

    def name(self) -> str:
        out = c_char_p()
        _check(self._api.GetJointName(byref(out), self.handle), "get joint name")
        return out.value.decode("utf8") if out.value else ""

    def local_rotation(self) -> tuple[float, float, float, float]:
        """Parent-relative rotation as (w, x, y, z)."""
        x, y, z, w = c_float(), c_float(), c_float(), c_float()
        _check(self._api.GetJointLocalRotation(byref(x), byref(y), byref(z), byref(w), self.handle),
               "get joint rotation")
        return w.value, x.value, y.value, z.value

    def local_position(self) -> Optional[tuple[float, float, float]]:
        """Parent-relative translation, or None when the stream carries no displacement."""
        x, y, z = c_float(), c_float(), c_float()
        if self._api.GetJointLocalTransformation(byref(x), byref(y), byref(z),
                                                 self.handle) != NO_ERROR:
            return None
        return x.value, y.value, z.value

    def children(self) -> list["Joint"]:
        count = c_uint32()
        _check(self._api.GetJointChild(POINTER(Handle)(), byref(count), self.handle),
               "get joint child count")
        if not count.value:
            return []
        buf = (Handle * count.value)()
        _check(self._api.GetJointChild(buf, byref(count), self.handle), "get joint children")
        return [Joint(buf[i]) for i in range(count.value)]


class Avatar:
    """One streamed skeleton; the glove receiver only walks it from the root."""

    __slots__ = ("_api", "handle")

    def __init__(self, handle: Handle) -> None:
        self._api = _interface(b"PROC_TABLE:IMCPAvatar_003", _AvatarApi)
        self.handle = handle

    def root_joint(self) -> Joint:
        out = Handle()
        _check(self._api.GetAvatarRootJoint(byref(out), self.handle), "get avatar root joint")
        return Joint(out)


class Application:
    """The receiver itself: opens the socket and hands out queued events."""

    def __init__(self) -> None:
        self._api = _interface(b"PROC_TABLE:IMCPApplication_002", _ApplicationApi)
        self.handle = Handle()
        _check(self._api.CreateApplication(byref(self.handle)), "create application")
        self._settings: Optional[Settings] = None

    def set_settings(self, settings: Settings) -> None:
        _check(self._api.SetApplicationSettings(settings.handle, self.handle),
               "set application settings")
        self._settings = settings   # keep it alive: the handle is borrowed, not copied

    def open(self) -> None:
        _check(self._api.OpenApplication(self.handle), "open application")

    def close(self) -> None:
        if self.handle:
            self._api.CloseApplication(self.handle)
            self._api.DestroyApplication(self.handle)
            self.handle = Handle()
        if self._settings is not None:
            self._settings.destroy()
            self._settings = None

    def poll_events(self) -> list[Event]:
        """Drain the queue. The first call sizes it, the second fills the caller's buffer."""
        count = c_uint32(0)
        _check(self._api.PollApplicationNextEvent(None, byref(count), self.handle), "poll events")
        if not count.value:
            return []
        buf = (Event * count.value)()
        for evt in buf:
            evt.size = sizeof(Event)   # the SDK rejects events whose size field is unset
        if self._api.PollApplicationNextEvent(ctypes.cast(buf, POINTER(None)), byref(count),
                                              self.handle) != NO_ERROR:
            return []
        return [buf[i] for i in range(count.value)]
