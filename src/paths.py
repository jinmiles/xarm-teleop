"""Single source of repo paths, external asset locations, and upstream integration shims.

All path handling in the project goes through this module. External assets (MANO, WiLoR
detector weights, sample images) are reused in place from other projects on this machine and
are never copied into or committed to this repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- project paths -----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"      # vendored upstream (submodules), kept clean
XARM7_SCENE = THIRD_PARTY / "mujoco_menagerie" / "ufactory_xarm7" / "scene.xml"
DATA_DIR = REPO_ROOT / "data"                # git-ignored local workspace (inputs)
HAND_CALIB = DATA_DIR / "hand_calib.json"    # per-operator dex-hand calibration (git-ignored)
GLOVE_CALIB = DATA_DIR / "glove_hand_calib.json"  # same, for glove-sourced finger angles
OUTPUT_DIR = REPO_ROOT / "outputs"           # git-ignored local workspace (outputs)
TESTS_DIR = REPO_ROOT / "tests"
DOCS_DIR = REPO_ROOT / "docs"

# --- external assets reused from other projects on this machine (do not copy/commit) -----
EXTERNAL_ROOT = Path("/home/user/extra_workdir")
MANO_RIGHT_PKL = EXTERNAL_ROOT / "EvalAI" / "assets" / "mano" / "MANO_RIGHT.pkl"
WILOR_DETECTOR_PT = EXTERNAL_ROOT / "HaWoR" / "weights" / "external" / "detector.pt"
SAMPLE_IMAGES_DIR = (
    EXTERNAL_ROOT / "MV-SAM3D" / "submodules" / "Dyn-HaMR" / "third-party" / "hamer" / "example_data"
)
SAMPLE_VIDEO = EXTERNAL_ROOT / "HaWoR" / "example" / "video_0.mp4"  # dev-only default source
# Noitom MocapApi python wrapper + librobotapi .so, reused in place (glove pose source).
MOCAP_ROS_PY = EXTERNAL_ROOT / "mocap_ros_py"

# Shared HuggingFace cache so wilor-mini weight downloads are centralized and reusable.
HF_HOME = EXTERNAL_ROOT / "hf_cache"


def configure_hf_cache() -> None:
    """Point the HuggingFace cache at the shared location.

    Must be called before importing wilor-mini / huggingface_hub so downloads land in and
    are served from the shared cache rather than the user home cache.
    """
    if HF_HOME.exists():
        os.environ.setdefault("HF_HOME", str(HF_HOME))


def calib_path(pose_source: str) -> Path:
    """Default finger-calibration file for a pose source.

    The two sources report different raw angles for the same hand, so their open/fist captures
    must not share a file -- loading the wrong one silently saturates every DOF.
    """
    return GLOVE_CALIB if pose_source == "glove" else HAND_CALIB


def import_mocap_api():
    """Import ``mocap_robotapi`` (Noitom MocapApi ctypes wrapper) from the external repo.

    Integration is sys.path injection only; the external repo is used as-is. Upstream's module
    body carries an unused ``from docutils...`` import, so a stub module is registered when
    docutils is absent rather than adding a dependency the wrapper never uses.
    """
    import types

    if not MOCAP_ROS_PY.exists():
        raise FileNotFoundError(
            f"mocap_ros_py not found at {MOCAP_ROS_PY}; it provides the Noitom MocapApi wrapper "
            "and librobotapi .so used by the glove pose source"
        )
    try:
        import docutils.parsers.rst.directives  # noqa: F401
    except ImportError:
        for name in ("docutils", "docutils.parsers", "docutils.parsers.rst",
                     "docutils.parsers.rst.directives"):
            module = sys.modules.setdefault(name, types.ModuleType(name))
            if name.endswith("directives") and not hasattr(module, "encoding"):
                module.encoding = "utf-8"
    if str(MOCAP_ROS_PY) not in sys.path:
        sys.path.insert(0, str(MOCAP_ROS_PY))
    import mocap_robotapi  # noqa: E402  (path injection must happen first)

    return mocap_robotapi


def add_third_party(*names: str) -> None:
    """Inject vendored ``third_party/<name>`` dirs onto sys.path (upstream integration shim).

    Integrate with upstream via sys.path injection only; never edit submodule internals.
    """
    for name in names:
        p = THIRD_PARTY / name
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def ensure_workspace() -> None:
    """Create git-ignored local workspace dirs on demand."""
    for d in (DATA_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
