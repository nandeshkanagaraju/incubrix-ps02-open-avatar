from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def cfg(tmp_path):
    """A Config rooted at tmp_path so tests never touch the real runs/ or consent/."""
    from openavatar.config import Config

    data = json.loads((REPO / "configs" / "default.json").read_text())
    data["output_root"] = "runs"
    data["consent_registry"] = "consent/consent_registry.json"
    data["reference_dir"] = "consent/reference_images"
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    p = cfg_dir / "test.json"
    p.write_text(json.dumps(data))
    return Config.load(p)


@pytest.fixture
def spec_dir(tmp_path) -> Path:
    """A 2-spec directory copied out of the committed batch A."""
    d = tmp_path / "specs"
    d.mkdir()
    for name in ("a1_studio_portrait.json", "a4_studio_portrait.json"):
        shutil.copy2(REPO / "specs" / "batch_a_fictional" / name, d / name)
    return d
