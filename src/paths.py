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

# Shared HuggingFace cache so wilor-mini weight downloads are centralized and reusable.
HF_HOME = EXTERNAL_ROOT / "hf_cache"


def configure_hf_cache() -> None:
    """Point the HuggingFace cache at the shared location.

    Must be called before importing wilor-mini / huggingface_hub so downloads land in and
    are served from the shared cache rather than the user home cache.
    """
    if HF_HOME.exists():
        os.environ.setdefault("HF_HOME", str(HF_HOME))


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
