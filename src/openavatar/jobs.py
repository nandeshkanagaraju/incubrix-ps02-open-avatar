"""Portable job bundles.

A *bundle* is a self-contained directory that can be zipped and carried to a
free-accelerator notebook. It never contains code or credentials - only the
resolved jobs, so the notebook stays a thin executor and the architecture stays
local.

    runs/<batch_id>/
        bundle.json          resolved jobs + model pin + batch metadata
        specs/*.json         the exact input specs
        safety_log.jsonl     one record per screened spec
        held/               specs that were refused or held for clarification
        images/              populated by `render` or by `ingest`
        results.json         per-job execution records (written by executor)
        validation.json      written by `validate`
        avatar_manifest.json written by `validate`
        metrics.json         written by `evaluate`
"""

from __future__ import annotations

import json
import platform
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Config
from .prompt import build_negative_prompt, build_prompt
from .safety import ConsentRegistry, SafetyLog, SafetyResult, screen
from .spec import AvatarSpec, load_specs

BUNDLE_SCHEMA_VERSION = 2


@dataclass
class Job:
    """One fully-resolved unit of generation work. Nothing is left implicit."""

    job_id: str
    spec_id: str
    identity_id: str | None
    prompt: str
    negative_prompt: str
    width: int
    height: int
    aspect_ratio: str
    seed: int
    steps: int
    guidance_scale: float
    model_key: str
    repo_id: str
    revision: str
    scheduler: str | None
    lora: dict | None
    output_name: str
    region_context: str | None  # bookkeeping only; not in prompt
    spec_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bundle:
    batch_id: str
    created_at: str
    schema_version: int
    tool_version: str
    jobs: list[Job] = field(default_factory=list)
    held: list[dict] = field(default_factory=list)
    host: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "tool_version": self.tool_version,
            "host": self.host,
            "jobs": [j.to_dict() for j in self.jobs],
            "held": self.held,
        }

    @staticmethod
    def load(bundle_dir: str | Path) -> "Bundle":
        p = Path(bundle_dir) / "bundle.json"
        if not p.exists():
            raise FileNotFoundError(f"not a bundle directory (no bundle.json): {bundle_dir}")
        d = json.loads(p.read_text())
        if d.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"bundle schema_version {d.get('schema_version')} != expected {BUNDLE_SCHEMA_VERSION}"
            )
        return Bundle(
            batch_id=d["batch_id"], created_at=d["created_at"],
            schema_version=d["schema_version"], tool_version=d["tool_version"],
            jobs=[Job(**j) for j in d["jobs"]], held=d.get("held", []), host=d.get("host", {}),
        )

    def job(self, job_id: str) -> Job:
        for j in self.jobs:
            if j.job_id == job_id:
                return j
        raise KeyError(job_id)


def _host_info() -> dict:
    import psutil

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_mb": round(psutil.virtual_memory().total / 1e6),
    }


def _spec_sha(spec: AvatarSpec) -> str:
    import hashlib

    payload = json.dumps(spec.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def plan_batch(
    spec_dir: str | Path,
    out_dir: str | Path,
    cfg: Config,
    batch_id: str,
    model_key_override: str | None = None,
) -> tuple[Bundle, list[SafetyResult]]:
    """Screen every spec, resolve allowed ones into jobs, write a portable bundle."""
    out = Path(out_dir)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "held").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    registry_path = cfg.resolve("consent_registry")
    registry = ConsentRegistry(registry_path) if registry_path.exists() else ConsentRegistry(registry_path)
    log = SafetyLog(out / "safety_log.jsonl")

    specs = load_specs(spec_dir)
    bundle = Bundle(
        batch_id=batch_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        schema_version=BUNDLE_SCHEMA_VERSION,
        tool_version=__version__,
        host=_host_info(),
    )
    results: list[SafetyResult] = []

    for spec in specs:
        result = screen(spec, registry)
        log.append(result)
        results.append(result)

        src = Path(spec_dir) / f"{spec.spec_id}.json"
        if not result.allowed:
            target = out / "held" / f"{spec.spec_id}.json"
            payload = {"spec": spec.model_dump(), "safety": result.to_dict()}
            target.write_text(json.dumps(payload, indent=2, sort_keys=True))
            bundle.held.append({"spec_id": spec.spec_id, "decision": result.decision,
                                "codes": [f.code for f in result.findings]})
            continue

        if src.exists():
            shutil.copy2(src, out / "specs" / src.name)
        else:
            (out / "specs" / f"{spec.spec_id}.json").write_text(
                json.dumps(spec.model_dump(), indent=2, sort_keys=True)
            )

        model_key = model_key_override or spec.render.model_key
        m = cfg.model(model_key)
        steps = spec.render.steps if model_key_override is None else m["default_steps"]
        guidance = spec.render.guidance_scale if model_key_override is None else m["default_guidance"]

        bundle.jobs.append(
            Job(
                job_id=f"{spec.spec_id}__{model_key}__s{spec.render.seed}",
                spec_id=spec.spec_id,
                identity_id=spec.identity_id,
                prompt=build_prompt(spec),
                negative_prompt=build_negative_prompt(),
                width=spec.render.width,
                height=spec.render.height,
                aspect_ratio=spec.render.aspect_ratio,
                seed=spec.render.seed,
                steps=steps,
                guidance_scale=guidance,
                model_key=model_key,
                repo_id=m["repo_id"],
                revision=m["revision"],
                scheduler=m.get("scheduler"),
                lora=m.get("lora"),
                output_name=f"{spec.spec_id}__{model_key}__s{spec.render.seed}.png",
                region_context=spec.region_context,
                spec_sha256=_spec_sha(spec),
            )
        )

    (out / "bundle.json").write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))
    return bundle, results


def zip_bundle(bundle_dir: str | Path, zip_path: str | Path) -> Path:
    """Zip the inputs of a bundle (not the images) for transport to a notebook."""
    bundle_dir, zip_path = Path(bundle_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(bundle_dir / "bundle.json", "bundle.json")
        for f in sorted((bundle_dir / "specs").glob("*.json")):
            zf.write(f, f"specs/{f.name}")
    return zip_path
