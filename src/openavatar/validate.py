"""Validation of returned outputs + provenance manifest construction.

The validator is deliberately paranoid because images may arrive from a remote
notebook: it re-opens every file, verifies the decoded dimensions against the
job that requested them, detects truncation/corruption and blank renders, and
checks that the provenance record is complete.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .backends import sha256_file
from .config import Config
from .jobs import Bundle, Job

# Stable failure codes - asserted by tests and referenced in the evidence index.
CODES = {
    "MISSING_FILE": "declared output file does not exist",
    "EMPTY_FILE": "output file is below the minimum byte threshold",
    "CORRUPT_IMAGE": "file could not be decoded as an image",
    "TRUNCATED_IMAGE": "image decoded but pixel data is incomplete",
    "WRONG_DIMENSIONS": "decoded size does not match the requested size",
    "BLANK_IMAGE": "image has near-zero pixel variance",
    "HASH_MISMATCH": "file hash differs from the hash recorded by the executor",
    "PROVENANCE_INCOMPLETE": "execution record is missing required provenance fields",
    "JOB_ERROR": "executor reported a failure for this job",
    "UNKNOWN_JOB": "result references a job that is not in the bundle",
}

REQUIRED_RESULT_FIELDS = ("job_id", "status", "runtime_sec", "device", "backend")


@dataclass
class ItemValidation:
    job_id: str
    ok: bool
    codes: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    batch_id: str
    validated_at: str
    total_jobs: int
    passed: int
    failed: int
    items: list[ItemValidation] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return round(self.passed / self.total_jobs, 4) if self.total_jobs else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        return d


def _pixel_stats(path: Path) -> tuple[int, int, float]:
    """Return (width, height, stddev). Raises on undecodable or truncated data."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        im.verify()  # catches structural corruption
    with Image.open(path) as im:
        im.load()    # catches truncation; PIL raises OSError on short data
        rgb = im.convert("RGB")
        stat = ImageStat.Stat(rgb)
        stddev = sum(stat.stddev) / len(stat.stddev)
        return rgb.width, rgb.height, stddev


def validate_item(job: Job, result: dict, images_dir: Path, rules: dict) -> ItemValidation:
    codes: list[str] = []
    details: dict = {}

    missing = [f for f in REQUIRED_RESULT_FIELDS if f not in result]
    if missing:
        codes.append("PROVENANCE_INCOMPLETE")
        details["missing_fields"] = missing

    if result.get("status") != "ok":
        codes.append("JOB_ERROR")
        details["error"] = result.get("error")
        return ItemValidation(job.job_id, False, codes, details)

    path = Path(result.get("image_path") or (images_dir / job.output_name))
    if not path.is_absolute():
        path = images_dir.parent / path
    if not path.exists():
        path = images_dir / job.output_name
    details["path"] = str(path)

    if not path.exists():
        codes.append("MISSING_FILE")
        return ItemValidation(job.job_id, False, codes, details)

    size = path.stat().st_size
    details["bytes"] = size
    if size < rules.get("min_file_bytes", 1024):
        codes.append("EMPTY_FILE")
        return ItemValidation(job.job_id, False, codes, details)

    try:
        w, h, stddev = _pixel_stats(path)
    except OSError as exc:
        codes.append("TRUNCATED_IMAGE" if "truncated" in str(exc).lower() else "CORRUPT_IMAGE")
        details["decode_error"] = str(exc)
        return ItemValidation(job.job_id, False, codes, details)
    except Exception as exc:  # noqa: BLE001 - PIL raises assorted types on bad data
        codes.append("CORRUPT_IMAGE")
        details["decode_error"] = f"{type(exc).__name__}: {exc}"
        return ItemValidation(job.job_id, False, codes, details)

    details.update({"width": w, "height": h, "pixel_stddev": round(stddev, 3)})

    if rules.get("require_exact_dimensions", True) and (w, h) != (job.width, job.height):
        codes.append("WRONG_DIMENSIONS")
        details["expected"] = [job.width, job.height]

    if stddev < rules.get("min_pixel_stddev", 6.0):
        codes.append("BLANK_IMAGE")

    declared = result.get("sha256")
    actual = sha256_file(path)
    details["sha256"] = actual
    if declared and declared != actual:
        codes.append("HASH_MISMATCH")
        details["declared_sha256"] = declared

    return ItemValidation(job.job_id, not codes, codes, details)


def validate_bundle(bundle_dir: str | Path, cfg: Config) -> ValidationReport:
    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    results_path = bundle_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found - run `openavatar render` or `openavatar ingest` first"
        )
    results = {r["job_id"]: r for r in json.loads(results_path.read_text())["results"]}
    rules = cfg.validation
    images_dir = bundle_dir / "images"

    report = ValidationReport(
        batch_id=bundle.batch_id,
        validated_at=datetime.now(timezone.utc).isoformat(),
        total_jobs=len(bundle.jobs), passed=0, failed=0,
    )

    known = {j.job_id for j in bundle.jobs}
    for stray in sorted(set(results) - known):
        report.items.append(ItemValidation(stray, False, ["UNKNOWN_JOB"], {}))

    for job in bundle.jobs:
        r = results.get(job.job_id)
        if r is None:
            item = ItemValidation(job.job_id, False, ["MISSING_FILE"], {"reason": "no execution record"})
        else:
            item = validate_item(job, r, images_dir, rules)
        report.items.append(item)
        if item.ok:
            report.passed += 1
        else:
            report.failed += 1

    (bundle_dir / "validation.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return report


def build_manifest(bundle_dir: str | Path, cfg: Config) -> dict:
    """Write ``avatar_manifest.json`` - the provenance record required by the brief."""
    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    results = {r["job_id"]: r for r in json.loads((bundle_dir / "results.json").read_text())["results"]}
    validation = json.loads((bundle_dir / "validation.json").read_text())
    val_by_job = {i["job_id"]: i for i in validation["items"]}
    safety = {}
    log_path = bundle_dir / "safety_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                safety[rec["spec_id"]] = rec

    entries = []
    for job in bundle.jobs:
        r = results.get(job.job_id, {})
        v = val_by_job.get(job.job_id, {})
        spec_path = bundle_dir / "specs" / f"{job.spec_id}.json"
        entries.append(
            {
                "job_id": job.job_id,
                "spec_id": job.spec_id,
                "identity_id": job.identity_id,
                "spec": json.loads(spec_path.read_text()) if spec_path.exists() else None,
                "spec_sha256": job.spec_sha256,
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt,
                "seed": job.seed,
                "steps": job.steps,
                "guidance_scale": job.guidance_scale,
                "aspect_ratio": job.aspect_ratio,
                "requested_size": [job.width, job.height],
                "model": {
                    "model_key": job.model_key,
                    "repo_id": job.repo_id,
                    "revision": job.revision,
                    "scheduler": job.scheduler,
                    "lora": job.lora,
                },
                "output": {
                    "file": job.output_name,
                    "sha256": v.get("details", {}).get("sha256"),
                    "bytes": v.get("details", {}).get("bytes"),
                    "decoded_size": [v.get("details", {}).get("width"), v.get("details", {}).get("height")],
                },
                "safety": safety.get(job.spec_id),
                "validation": {"ok": v.get("ok"), "codes": v.get("codes", [])},
                "compute_route": {
                    "backend": r.get("backend"),
                    "device": r.get("device"),
                    "runtime_sec": r.get("runtime_sec"),
                    "peak_rss_mb": r.get("peak_rss_mb"),
                },
                "synthetic_media": True,
                "content_label": "AI-generated synthetic image. Not a photograph of a real person.",
            }
        )

    manifest = {
        "manifest_version": 1,
        "tool": {"name": "openavatar", "version": __version__},
        "batch_id": bundle.batch_id,
        "created_at": bundle.created_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": bundle.host,
        "config_path": str(cfg.path),
        "held_specs": bundle.held,
        "summary": {
            "total_jobs": validation["total_jobs"],
            "passed": validation["passed"],
            "failed": validation["failed"],
            "job_success_rate": validation["success_rate"],
        },
        "avatars": entries,
    }
    (bundle_dir / "avatar_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
