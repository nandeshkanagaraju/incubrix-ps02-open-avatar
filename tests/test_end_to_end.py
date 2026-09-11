"""End-to-end tests.

``test_full_pipeline_offline`` walks the entire product path - plan, export,
execute, ingest, validate, manifest - using the deterministic stub backend, so
it runs in CI on any machine with no weights and no network.

``test_local_backend_generates_a_real_image`` is the same path against real
diffusers weights; it is marked ``slow`` and skipped unless
OPENAVATAR_RUN_SLOW=1 and the weights are already cached.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from openavatar.backends import StubBackend
from openavatar.jobs import Bundle, plan_batch, zip_bundle
from openavatar.metrics import evaluate_bundle
from openavatar.render import ingest, render_bundle
from openavatar.validate import build_manifest, validate_bundle


def test_full_pipeline_offline(tmp_path, cfg, repo_root):
    """plan -> export -> execute -> ingest -> validate -> manifest, end to end."""
    spec_src = repo_root / "specs" / "batch_a_fictional"
    specs = tmp_path / "specs"
    specs.mkdir()
    for f in sorted(spec_src.glob("*.json"))[:3]:
        shutil.copy2(f, specs / f.name)

    batch = "e2e"
    bundle_dir = tmp_path / "runs" / batch

    # 1. plan
    bundle, safety = plan_batch(specs, bundle_dir, cfg, batch)
    assert len(bundle.jobs) == 3
    assert all(r.decision == "ALLOW" for r in safety)

    # 2. export a portable bundle (the notebook hand-off)
    z = zip_bundle(bundle_dir, tmp_path / f"{batch}.zip")
    assert zipfile.ZipFile(z).read("bundle.json")

    # 3. execute somewhere else, then package results like the notebook does
    remote = tmp_path / "remote"
    remote_bundle = remote / batch
    remote_bundle.mkdir(parents=True)
    (remote_bundle / "bundle.json").write_bytes((bundle_dir / "bundle.json").read_bytes())
    render_bundle(remote_bundle, StubBackend())

    archive_dir = tmp_path / "archive"
    (archive_dir / "images").mkdir(parents=True)
    remote_results = json.loads((remote_bundle / "results.json").read_text())
    remote_results["route"] = "notebook"
    for r in remote_results["results"]:
        src = Path(r["image_path"])
        (archive_dir / "images" / src.name).write_bytes(src.read_bytes())
        r["image_path"] = f"images/{src.name}"
    (archive_dir / "results.json").write_text(json.dumps(remote_results))

    # 4. ingest into the local bundle
    merged = ingest(bundle_dir, archive_dir)
    assert len(merged) == 3
    assert len(list((bundle_dir / "images").glob("*.png"))) == 3

    # 5. validate
    report = validate_bundle(bundle_dir, cfg)
    assert report.failed == 0
    assert report.success_rate == 1.0

    # 6. manifest carries everything an evaluator needs to reproduce a render
    manifest = build_manifest(bundle_dir, cfg)
    assert len(manifest["avatars"]) == 3
    for a in manifest["avatars"]:
        assert a["seed"] and a["model"]["revision"] and a["output"]["sha256"]
        assert a["compute_route"]["backend"] == "stub"
        assert a["synthetic_media"] is True

    # 7. determinism: the same seed through the same backend reproduces bytes
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    (rerun / "bundle.json").write_bytes((bundle_dir / "bundle.json").read_bytes())
    render_bundle(rerun, StubBackend())
    rerun_results = {r["job_id"]: r for r in json.loads((rerun / "results.json").read_text())["results"]}
    for job_id, rec in merged.items():
        assert rerun_results[job_id]["sha256"] == rec["sha256"], f"{job_id} was not reproducible"


def test_cli_end_to_end(tmp_path, repo_root):
    """Drive the real CLI as a subprocess, including the non-zero exit contract."""
    specs = tmp_path / "specs"
    specs.mkdir()
    for f in sorted((repo_root / "specs" / "batch_a_fictional").glob("*.json"))[:2]:
        shutil.copy2(f, specs / f.name)
    runs = tmp_path / "runs"
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "openavatar.cli", *args],
                              capture_output=True, text=True, env=env, cwd=repo_root)

    bundle = runs / "cli_e2e"
    r = cli("plan", str(specs), "--batch-id", "cli_e2e", "--out", str(bundle))
    assert r.returncode == 0, r.stderr
    assert "2 job(s) prepared" in r.stdout

    r = cli("render", str(bundle), "--backend", "stub")
    assert r.returncode == 0, r.stderr

    r = cli("validate", str(bundle))
    assert r.returncode == 0, r.stderr
    assert (bundle / "avatar_manifest.json").exists()

    # corrupting an output must make `validate` exit 2
    b = Bundle.load(bundle)
    p = bundle / "images" / b.jobs[0].output_name
    p.write_bytes(p.read_bytes()[:800])
    r = cli("validate", str(bundle))
    assert r.returncode == 2, "validate must exit 2 when an output fails"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("OPENAVATAR_RUN_SLOW") != "1",
    reason="set OPENAVATAR_RUN_SLOW=1 to run real diffusion (needs cached weights)",
)
def test_local_backend_generates_a_real_image(tmp_path, cfg, repo_root):
    from openavatar.backends import LocalDiffusersBackend

    specs = tmp_path / "specs"
    specs.mkdir()
    src = json.loads((repo_root / "specs" / "batch_a_fictional" / "a1_studio_portrait.json").read_text())
    src["render"].update({"width": 256, "height": 256, "steps": 4})
    (specs / "a1_studio_portrait.json").write_text(json.dumps(src))

    bundle_dir = tmp_path / "runs" / "slow"
    plan_batch(specs, bundle_dir, cfg, "slow")
    render_bundle(bundle_dir, LocalDiffusersBackend(device="auto"))
    report = validate_bundle(bundle_dir, cfg)
    assert report.failed == 0
    metrics = evaluate_bundle(bundle_dir, cfg)
    assert metrics["per_image"], "no image was scored"
    for rec in metrics["per_image"].values():
        assert rec["clip_score"] > 0
