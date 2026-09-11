"""Integration tests: planning, validation and ingestion."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from openavatar.backends import StubBackend
from openavatar.jobs import Bundle, plan_batch, zip_bundle
from openavatar.render import ingest, render_bundle
from openavatar.validate import build_manifest, validate_bundle


def _noisy(w: int, h: int, seed: int):
    """A deterministic non-blank image large enough to clear the byte floor."""
    from PIL import Image

    img = Image.new("RGB", (w, h))
    px = img.load()
    state = seed
    for y in range(h):
        for x in range(w):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            v = state >> 16
            px[x, y] = (v % 256, (v // 256) % 256, (v // 7) % 256)
    return img


def _plan(tmp_path, spec_dir, cfg, batch_id="t", **kw):
    out = tmp_path / "runs" / batch_id
    return plan_batch(spec_dir, out, cfg, batch_id, **kw), out


# --- planning ---------------------------------------------------------------


def test_plan_writes_a_loadable_bundle(tmp_path, spec_dir, cfg):
    (bundle, _), out = _plan(tmp_path, spec_dir, cfg)
    assert len(bundle.jobs) == 2
    reloaded = Bundle.load(out)
    assert {j.job_id for j in reloaded.jobs} == {j.job_id for j in bundle.jobs}
    assert (out / "safety_log.jsonl").exists()
    assert len(list((out / "specs").glob("*.json"))) == 2


def test_plan_pins_model_repo_and_revision(tmp_path, spec_dir, cfg):
    (bundle, _), _ = _plan(tmp_path, spec_dir, cfg)
    for job in bundle.jobs:
        assert job.repo_id
        assert len(job.revision) == 40, "revision must be a full commit sha, not a branch name"


def test_model_key_override_applies_model_defaults(tmp_path, spec_dir, cfg):
    (bundle, _), _ = _plan(tmp_path, spec_dir, cfg, model_key_override="sdturbo")
    for job in bundle.jobs:
        assert job.model_key == "sdturbo"
        assert job.steps == cfg.model("sdturbo")["default_steps"]


def test_held_specs_are_not_turned_into_jobs(tmp_path, cfg, repo_root):
    import shutil

    d = tmp_path / "edge"
    d.mkdir()
    for f in (repo_root / "specs" / "batch_c_edge").glob("*.json"):
        shutil.copy2(f, d / f.name)
    (bundle, results), out = _plan(tmp_path, d, cfg, "edge")
    assert len(bundle.jobs) == 1                      # only c3_valid_control
    assert {h["decision"] for h in bundle.held} == {"REFUSE", "NEEDS_CLARIFICATION"}
    assert (out / "held" / "c2_disallowed.json").exists()
    held = json.loads((out / "held" / "c2_disallowed.json").read_text())
    assert held["safety"]["decision"] == "REFUSE"


def test_unknown_model_key_raises(tmp_path, spec_dir, cfg):
    from openavatar.config import ConfigError

    with pytest.raises(ConfigError, match="unknown model_key"):
        _plan(tmp_path, spec_dir, cfg, model_key_override="does-not-exist")


def test_zip_bundle_contains_inputs_only(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    z = zip_bundle(out, tmp_path / "b.zip")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert "bundle.json" in names
    assert all(not n.startswith("images/") for n in names)


def test_bundle_rejects_wrong_schema_version(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    d = json.loads((out / "bundle.json").read_text())
    d["schema_version"] = 99
    (out / "bundle.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema_version"):
        Bundle.load(out)


# --- validation --------------------------------------------------------------


@pytest.fixture
def rendered(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    render_bundle(out, StubBackend())
    return out


def test_validate_passes_on_clean_outputs(rendered, cfg):
    report = validate_bundle(rendered, cfg)
    assert report.failed == 0
    assert report.success_rate == 1.0


def test_validate_detects_a_corrupted_image(rendered, cfg):
    bundle = Bundle.load(rendered)
    victim = rendered / "images" / bundle.jobs[0].output_name
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])          # truncate mid-stream
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == bundle.jobs[0].job_id)
    assert not item.ok
    assert {"CORRUPT_IMAGE", "TRUNCATED_IMAGE", "HASH_MISMATCH"} & set(item.codes)
    assert report.success_rate < 1.0


def test_validate_detects_missing_file(rendered, cfg):
    bundle = Bundle.load(rendered)
    (rendered / "images" / bundle.jobs[0].output_name).unlink()
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == bundle.jobs[0].job_id)
    assert "MISSING_FILE" in item.codes


def test_validate_detects_wrong_dimensions(rendered, cfg):
    from PIL import Image

    bundle = Bundle.load(rendered)
    p = rendered / "images" / bundle.jobs[0].output_name
    _noisy(64, 64, seed=7).save(p)
    results = json.loads((rendered / "results.json").read_text())
    for r in results["results"]:                      # keep the hash honest
        if r["job_id"] == bundle.jobs[0].job_id:
            from openavatar.backends import sha256_file
            r["sha256"] = sha256_file(p)
    (rendered / "results.json").write_text(json.dumps(results))
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == bundle.jobs[0].job_id)
    assert "WRONG_DIMENSIONS" in item.codes


def test_validate_detects_blank_image(rendered, cfg):
    from PIL import Image

    from openavatar.backends import sha256_file

    bundle = Bundle.load(rendered)
    job = bundle.jobs[0]
    p = rendered / "images" / job.output_name
    Image.new("RGB", (job.width, job.height), (128, 128, 128)).save(p)
    results = json.loads((rendered / "results.json").read_text())
    for r in results["results"]:
        if r["job_id"] == job.job_id:
            r["sha256"] = sha256_file(p)
    (rendered / "results.json").write_text(json.dumps(results))
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == job.job_id)
    assert "BLANK_IMAGE" in item.codes


def test_tiny_blank_image_reports_both_codes(rendered, cfg):
    """Regression: an all-black frame (NaN pipeline output) compresses below the
    byte floor. EMPTY_FILE alone hides the real defect, so both codes must fire."""
    from PIL import Image

    from openavatar.backends import sha256_file

    bundle = Bundle.load(rendered)
    job = bundle.jobs[0]
    p = rendered / "images" / job.output_name
    Image.new("RGB", (job.width, job.height), (0, 0, 0)).save(p)
    assert p.stat().st_size < 1024, "fixture assumption: a black PNG is tiny"
    results = json.loads((rendered / "results.json").read_text())
    for r in results["results"]:
        if r["job_id"] == job.job_id:
            r["sha256"] = sha256_file(p)
    (rendered / "results.json").write_text(json.dumps(results))
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == job.job_id)
    assert "EMPTY_FILE" in item.codes
    assert "BLANK_IMAGE" in item.codes


def test_validate_detects_hash_mismatch(rendered, cfg):
    bundle = Bundle.load(rendered)
    results = json.loads((rendered / "results.json").read_text())
    results["results"][0]["sha256"] = "0" * 64
    (rendered / "results.json").write_text(json.dumps(results))
    report = validate_bundle(rendered, cfg)
    item = next(i for i in report.items if i.job_id == results["results"][0]["job_id"])
    assert "HASH_MISMATCH" in item.codes


def test_validate_without_results_is_a_clear_error(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    with pytest.raises(FileNotFoundError, match="results.json"):
        validate_bundle(out, cfg)


# --- manifest ----------------------------------------------------------------


def test_manifest_carries_full_provenance(rendered, cfg):
    validate_bundle(rendered, cfg)
    manifest = build_manifest(rendered, cfg)
    assert manifest["summary"]["job_success_rate"] == 1.0
    for entry in manifest["avatars"]:
        assert entry["seed"] is not None
        assert entry["prompt"] and entry["negative_prompt"]
        assert len(entry["model"]["revision"]) == 40
        assert entry["output"]["sha256"]
        assert entry["safety"]["decision"] == "ALLOW"
        assert entry["compute_route"]["backend"] == "stub"
        assert entry["synthetic_media"] is True
        assert "synthetic" in entry["content_label"].lower()
        assert entry["spec"] is not None


# --- ingest ------------------------------------------------------------------


def _make_archive(tmp_path, bundle_dir, mutate=None) -> Path:
    """Build a notebook-style results archive from an already-rendered bundle."""
    staging = tmp_path / "nb"
    (staging / "images").mkdir(parents=True, exist_ok=True)
    results = json.loads((bundle_dir / "results.json").read_text())
    results["route"] = "notebook"
    for r in results["results"]:
        if r.get("image_path"):
            src = Path(r["image_path"])
            (staging / "images" / src.name).write_bytes(src.read_bytes())
            r["image_path"] = f"images/{src.name}"
    if mutate:
        mutate(results, staging)
    (staging / "results.json").write_text(json.dumps(results))
    z = tmp_path / "nb.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(staging / "results.json", "results.json")
        for f in (staging / "images").glob("*"):
            zf.write(f, f"images/{f.name}")
    return z


def test_ingest_round_trips_notebook_results(tmp_path, rendered, cfg):
    z = _make_archive(tmp_path, rendered)
    for p in (rendered / "images").glob("*.png"):
        p.unlink()
    (rendered / "results.json").unlink()
    merged = ingest(rendered, z)
    assert len(merged) == 2
    assert all(r["backend"] == "stub" for r in merged.values())
    report = validate_bundle(rendered, cfg)
    assert report.failed == 0


def test_ingest_rejects_a_foreign_batch(tmp_path, rendered):
    z = _make_archive(tmp_path, rendered, mutate=lambda r, s: r.update(batch_id="someone-else"))
    with pytest.raises(ValueError, match="batch_id mismatch"):
        ingest(rendered, z)


def test_ingest_rejects_unknown_job_ids(tmp_path, rendered):
    def mutate(results, staging):
        results["results"][0]["job_id"] = "ghost-job"

    z = _make_archive(tmp_path, rendered, mutate=mutate)
    with pytest.raises(ValueError, match="unknown job_id"):
        ingest(rendered, z)


def test_ingest_rejects_a_claimed_success_with_no_image(tmp_path, rendered):
    def mutate(results, staging):
        for f in (staging / "images").glob("*"):
            f.unlink()

    z = _make_archive(tmp_path, rendered, mutate=mutate)
    with pytest.raises(FileNotFoundError, match="absent"):
        ingest(rendered, z)


def test_ingest_rejects_path_traversal(tmp_path, rendered):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.txt", "nope")
        zf.writestr("results.json", "{}")
    with pytest.raises(ValueError, match="unsafe path"):
        ingest(rendered, z)


# --- resume ------------------------------------------------------------------


def test_render_resumes_and_skips_completed_jobs(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    first = render_bundle(out, StubBackend())
    mtimes = {p.name: p.stat().st_mtime_ns for p in (out / "images").glob("*.png")}
    second = render_bundle(out, StubBackend(), resume=True)
    assert set(first) == set(second)
    assert {p.name: p.stat().st_mtime_ns for p in (out / "images").glob("*.png")} == mtimes


def test_render_reruns_when_a_cached_image_is_tampered_with(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    render_bundle(out, StubBackend())
    victim = next((out / "images").glob("*.png"))
    victim.write_bytes(b"garbage")
    render_bundle(out, StubBackend(), resume=True)
    assert victim.read_bytes() != b"garbage", "tampered cache entry must be re-rendered"


def test_render_only_filters_jobs(tmp_path, spec_dir, cfg):
    (bundle, _), out = _plan(tmp_path, spec_dir, cfg)
    target = bundle.jobs[0].job_id
    results = render_bundle(out, StubBackend(), only=[target])
    assert set(results) == {target}


def test_render_only_with_no_match_raises(tmp_path, spec_dir, cfg):
    (_, _), out = _plan(tmp_path, spec_dir, cfg)
    with pytest.raises(KeyError):
        render_bundle(out, StubBackend(), only=["nope"])
