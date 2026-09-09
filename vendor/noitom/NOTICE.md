# Noitom MocapApi (librobotapi)

Third-party binary, vendored so the teleop machine needs no external checkout. It decodes the
BVH stream Axis Studio broadcasts over the LAN; `src/glove/mocapapi.py` is the project-owned
ctypes binding to it, and nothing else in the repo links against it.

| | |
|---|---|
| Origin | <https://github.com/pnmocap/mocap_ros_py> (`lib/`), redistributed by Noitom |
| File | `librobotapi_x86-64.so`, ELF 64-bit x86-64, BuildID `e1f9082f559c9c1bff46614cb2c63ad99ef386b9` |
| Used by | `src/glove/mocapapi.py` → `src/glove/client.py` (method 2 only) |

Other architectures ship under their own names in the upstream repo (`librobotapi_arm64.so`,
`librobotapi_arm.so`); drop one in beside this file and it is picked up automatically. To load a
copy from elsewhere, set `XARM_TELEOP_MOCAPAPI` to the file or to the directory holding it.

Upstream publishes no licence file; the binaries are distributed openly in that public repository.
Check with Noitom before redistributing them beyond this project.
