"""Batch execution and result ingestion.

``render_bundle`` drives a local backend with cache/resume semantics; ``ingest``
absorbs results produced elsewhere (the free-accelerator notebook route). Both
converge on the same ``results.json`` shape so downstream stages do not care
which route was used.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .backends import Backend, JobResult, sha256_file
from .jobs import Bundle
from .logging_utils import get_logger

log = get_logger(__name__)


def _results_path(bundle_dir: Path) -> Path:
    return bundle_dir / "results.json"


def load_results(bundle_dir: str | Path) -> dict[str, dict]:
    p = _results_path(Path(bundle_dir))
    if not p.exists():
        return {}
    return {r["job_id"]: r for r in json.loads(p.read_text())["results"]}


def write_results(bundle_dir: str | Path, results: dict[str, dict], route: str) -> Path:
    bundle_dir = Path(bundle_dir)
    payload = {
        "batch_id": Bundle.load(bundle_dir).batch_id,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "results": [results[k] for k in sorted(results)],
    }
    p = _results_path(bundle_dir)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def render_bundle(
    bundle_dir: str | Path,
    backend: Backend,
    resume: bool = True,
    only: list[str] | None = None,
) -> dict[str, dict]:
    """Execute every job in a bundle. A failing job is recorded, not fatal."""
    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    images_dir = bundle_dir / "images"
    existing = load_results(bundle_dir) if resume else {}

    jobs = bundle.jobs if not only else [j for j in bundle.jobs if j.job_id in set(only)]
    if only and not jobs:
        raise KeyError(f"no jobs in bundle matched {only}")

    for i, job in enumerate(jobs, 1):
        prior = existing.get(job.job_id)
        if resume and prior and prior.get("status") == "ok":
            img = images_dir / job.output_name
            if img.exists() and sha256_file(img) == prior.get("sha256"):
                log.info("skip %s (cached)", job.job_id)
                continue
            log.warning("cache miss for %s - re-rendering", job.job_id)

        log.info("render %d/%d %s [%s %sx%s seed=%s steps=%s]",
                 i, len(jobs), job.job_id, job.model_key, job.width, job.height, job.seed, job.steps)
        result: JobResult = backend.run(job, images_dir)
        if result.status == "ok":
            log.info("  ok in %.1fs -> %s", result.runtime_sec, Path(result.image_path).name)
        else:
            log.error("  failed: %s", result.error)
        existing[job.job_id] = result.to_dict()
        write_results(bundle_dir, existing, route=backend.name)

    write_results(bundle_dir, existing, route=backend.name)
    return existing


def ingest(bundle_dir: str | Path, results_archive: str | Path) -> dict[str, dict]:
    """Ingest a notebook result bundle (a .zip or a directory).

    Expects ``results.json`` plus an ``images/`` folder. Every incoming record is
    checked against the local bundle so a mismatched or stray archive is rejected
    before it can pollute the evidence set.
    """
    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    src = Path(results_archive)
    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".zip":
        staging = bundle_dir / "_ingest_staging"
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():
                # Reject absolute paths and traversal before extracting.
                if member.startswith("/") or ".." in Path(member).parts:
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(staging)
        src = staging

    rp = src / "results.json"
    if not rp.exists():
        raise FileNotFoundError(f"{rp} not found in ingest source")
    incoming = json.loads(rp.read_text())

    if incoming.get("batch_id") != bundle.batch_id:
        raise ValueError(
            f"batch_id mismatch: archive={incoming.get('batch_id')!r} bundle={bundle.batch_id!r}"
        )

    known = {j.job_id: j for j in bundle.jobs}
    merged = load_results(bundle_dir)
    imported = 0
    for rec in incoming["results"]:
        jid = rec["job_id"]
        if jid not in known:
            raise ValueError(f"archive contains unknown job_id {jid!r}")
        if rec.get("status") == "ok":
            name = known[jid].output_name
            candidate = src / "images" / name
            if not candidate.exists():
                raise FileNotFoundError(f"archive claims {jid} succeeded but images/{name} is absent")
            (images_dir / name).write_bytes(candidate.read_bytes())
            rec["image_path"] = str(images_dir / name)
        merged[jid] = rec
        imported += 1

    write_results(bundle_dir, merged, route=incoming.get("route", "notebook"))
    log.info("ingested %d result records from %s", imported, results_archive)
    return merged
