"""Automated benchmark.

Timing boundary is the ``backend.run`` call only: weight loading is excluded and
measured separately, because a cached pipeline is loaded once per batch while
``time per image`` is the per-job cost a product would pay.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import PIPELINE_WEIGHT_FILES, get_backend
from .config import Config
from .jobs import Bundle
from .logging_utils import get_logger

log = get_logger(__name__)


def _model_size_mb(repo_id: str, revision: str) -> float | None:
    """Size of the weights the pipeline actually loads - not the whole repo.

    Summing every *.safetensors in the SD 1.5 repo reports ~23.6 GB, which is
    wrong by more than 5x: it counts the single-file checkpoints, the non-EMA
    UNet, fp16 duplicates of every shard and the disabled safety checker. Only
    the files in PIPELINE_WEIGHT_FILES are read.

    The local snapshot is measured first, so the reported number is the weight
    set actually resident on the benchmarked machine and the benchmark still
    works offline. The Hub is only consulted as a fallback.
    """
    local = _local_snapshot_size_mb(repo_id, revision)
    if local is not None:
        return local
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        by_name = {s.rfilename: (s.size or 0) for s in info.siblings}
        total = sum(by_name.get(f, 0) for f in PIPELINE_WEIGHT_FILES)
        if not total:
            # LoRA and other single-file repos do not use the subfolder layout.
            total = sum(
                size for name, size in by_name.items()
                if name.endswith(".safetensors") and "/" not in name
            )
        return round(total / 1e6, 1) if total else None
    except Exception as exc:  # noqa: BLE001 - offline benchmarking must still work
        log.warning("could not resolve model size for %s: %s", repo_id, exc)
        return None


def _local_snapshot_size_mb(repo_id: str, revision: str) -> float | None:
    """Measure the cached snapshot on this machine, following symlinks."""
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(repo_id=repo_id, revision=revision,
                                      local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached, or hub unavailable
        return None

    total = 0
    for name in PIPELINE_WEIGHT_FILES:
        f = snap / name
        if f.exists():
            total += f.stat().st_size
    if not total:
        # single-file repo layout (LoRA)
        for f in snap.glob("*.safetensors"):
            total += f.stat().st_size
    return round(total / 1e6, 1) if total else None


def run_benchmark(
    bundle_dir: str | Path,
    cfg: Config,
    backend_name: str = "local",
    repetitions: int = 3,
    warmup: int = 1,
    limit: int = 2,
    device: str = "auto",
) -> dict:
    """Time the first ``limit`` jobs of a bundle ``repetitions`` times each."""
    import psutil

    bundle_dir = Path(bundle_dir)
    bundle = Bundle.load(bundle_dir)
    jobs = bundle.jobs[:limit]
    if not jobs:
        raise ValueError("bundle contains no jobs to benchmark")

    backend = get_backend(backend_name, device=device)
    bench_dir = bundle_dir / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    if hasattr(backend, "pipeline"):
        backend.pipeline(jobs[0])  # force weight load outside the timing boundary
    load_sec = round(time.perf_counter() - t_load0, 3)

    per_job = []
    per_job_rss = []
    for job in jobs:
        for _ in range(warmup):
            backend.run(job, bench_dir)
        samples = []
        rss_samples = []
        for r in range(repetitions):
            res = backend.run(job, bench_dir)
            if res.status != "ok":
                log.error("benchmark job %s failed: %s", job.job_id, res.error)
                continue
            samples.append(res.runtime_sec)
            rss_samples.append(res.peak_rss_mb)
            log.info("bench %s rep %d/%d: %.2fs", job.job_id, r + 1, repetitions, res.runtime_sec)
        if samples:
            per_job_rss.append({"peak_rss_mb": max(rss_samples)})
            per_job.append({
                "job_id": job.job_id,
                "model_key": job.model_key,
                "size": [job.width, job.height],
                "steps": job.steps,
                "repetitions": len(samples),
                "samples_sec": samples,
                "median_sec": round(statistics.median(samples), 3),
                "mean_sec": round(statistics.fmean(samples), 3),
                "stdev_sec": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
                "peak_rss_mb": max(rss_samples),
            })

    medians = [j["median_sec"] for j in per_job]
    proc = psutil.Process()
    peak_rss = max((j["peak_rss_mb"] for j in per_job_rss), default=None)
    models = {}
    for job in jobs:
        if job.model_key not in models:
            models[job.model_key] = {
                "repo_id": job.repo_id,
                "revision": job.revision,
                "size_mb": _model_size_mb(job.repo_id, job.revision),
            }

    report = {
        "benchmarked_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": bundle.batch_id,
        "method": {
            "timing_boundary": "backend.run() only; weight load excluded and reported as pipeline_load_sec",
            "warmup_runs_per_job": warmup,
            "repetitions_per_job": repetitions,
            "memory_measure": "process peak RSS via getrusage + psutil RSS snapshot",
            "seeds": "fixed per job; identical across repetitions",
            "cache": "pipeline cached across jobs; image resume disabled during benchmark",
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_mb": round(psutil.virtual_memory().total / 1e6),
            "device": getattr(backend, "device", "cpu"),
            "backend": backend_name,
            "accelerator_peak_memory_mb": None,
            "accelerator_note": "no CUDA device on this host; peak accelerator memory is not measurable locally",
        },
        "pipeline_load_sec": load_sec,
        "peak_process_rss_mb": peak_rss,
        "current_process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "memory_note": (
            "peak_process_rss_mb is the high-water mark from getrusage and is the "
            "number to quote: it captures the CPU-side peak while weights are loaded "
            "and cast. current_process_rss_mb is much lower on Apple MPS because the "
            "weights are then handed to Metal and live in unified memory that is not "
            "attributed to process RSS. Neither number is a GPU-memory measurement; "
            "peak accelerator memory is only available on the CUDA route and is "
            "recorded per job as peak_accelerator_mb in results.json."
        ),
        "models": models,
        "per_job": per_job,
        "summary": {
            "jobs_timed": len(per_job),
            "median_time_per_image_sec": round(statistics.median(medians), 3) if medians else None,
            "fastest_sec": round(min(medians), 3) if medians else None,
            "slowest_sec": round(max(medians), 3) if medians else None,
        },
    }
    (bundle_dir / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
