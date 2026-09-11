"""Execution backends.

Two routes satisfy the hybrid execution rule:

  ``local``  - diffusers on this laptop (CPU or Apple MPS). Slow but fully
               candidate-executed and needs no accelerator.
  ``notebook`` - the bundle is exported, executed on a disclosed free
               accelerator, and the results are re-ingested by ``ingest``.

Both produce the same :class:`JobResult` record, so validation, manifesting and
evaluation are identical regardless of route.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .jobs import Job


@dataclass
class JobResult:
    job_id: str
    status: str  # "ok" | "error"
    image_path: str | None
    sha256: str | None
    runtime_sec: float
    peak_rss_mb: float
    device: str
    backend: str
    dtype: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: str

    def run(self, job: Job, out_dir: Path) -> JobResult: ...


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _peak_rss_mb() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return usage / 1e6 if usage > 1e7 else usage / 1e3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalDiffusersBackend:
    """CPU/MPS diffusers backend with a pipeline cache so a batch loads weights once."""

    name = "local"

    def __init__(self, device: str = "auto", dtype: str = "auto"):
        self.device = pick_device(device)
        # float32 everywhere by default. float16 is the obvious memory win on
        # MPS (UNet 3.4 GB -> 1.7 GB) but SD 1.5's VAE decoder overflows in fp16
        # and silently emits NaN, which reaches disk as an all-black PNG - see
        # evidence/failures/fp16_mps_nan_black_output.png. Upcasting only the VAE
        # then fails because the UNet hands it fp16 latents, so the choice is
        # all-fp16 (silently wrong) or all-fp32 (correct). Correct wins; the
        # memory is recovered with attention and VAE slicing below.
        # `--dtype float16` remains available for anyone who wants to reproduce
        # the failure. The effective dtype is recorded on every JobResult.
        self.dtype = "float32" if dtype == "auto" else dtype
        self._cache: dict[tuple, object] = {}

    # -- pipeline construction -------------------------------------------
    def _key(self, job: Job) -> tuple:
        lora = (job.lora or {}).get("repo_id")
        return (job.repo_id, job.revision, job.scheduler, lora)

    def _build(self, job: Job):
        import torch
        from diffusers import DiffusionPipeline, LCMScheduler

        dtype = {"float32": torch.float32, "float16": torch.float16}[self.dtype]
        pipe = DiffusionPipeline.from_pretrained(
            job.repo_id,
            revision=job.revision,
            torch_dtype=dtype,
            safety_checker=None,          # we run our own prompt-level safety layer
            requires_safety_checker=False,
            use_safetensors=True,
        )
        if job.scheduler == "LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        if job.lora:
            pipe.load_lora_weights(
                job.lora["repo_id"],
                revision=job.lora.get("revision"),
                weight_name=job.lora.get("weight_name"),
            )
            pipe.fuse_lora()
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        # 8 GB laptop: slice attention and the VAE so peak RSS stays bounded.
        pipe.enable_attention_slicing("max")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        return pipe

    def pipeline(self, job: Job):
        key = self._key(job)
        if key not in self._cache:
            self._cache.clear()  # only one pipeline resident at a time on 8 GB
            self._cache[key] = self._build(job)
        return self._cache[key]

    # -- execution --------------------------------------------------------
    def run(self, job: Job, out_dir: Path) -> JobResult:
        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / job.output_name
        t0 = time.perf_counter()
        try:
            pipe = self.pipeline(job)
            # Seeding on CPU keeps results reproducible across cpu/mps routes.
            generator = torch.Generator(device="cpu").manual_seed(job.seed)
            kwargs = dict(
                prompt=job.prompt,
                width=job.width,
                height=job.height,
                num_inference_steps=job.steps,
                guidance_scale=job.guidance_scale,
                generator=generator,
            )
            # LCM / turbo run without classifier-free guidance, so no negative prompt.
            if job.guidance_scale > 1.0:
                kwargs["negative_prompt"] = job.negative_prompt
            image = pipe(**kwargs).images[0]
            image.save(target, format="PNG")
            dt = time.perf_counter() - t0
            return JobResult(
                job_id=job.job_id, status="ok", image_path=str(target),
                sha256=sha256_file(target), runtime_sec=round(dt, 3),
                peak_rss_mb=round(_peak_rss_mb(), 1), device=self.device, backend=self.name,
                dtype=self.dtype,
            )
        except Exception as exc:  # noqa: BLE001 - a failed job must not kill the batch
            return JobResult(
                job_id=job.job_id, status="error", image_path=None, sha256=None,
                runtime_sec=round(time.perf_counter() - t0, 3),
                peak_rss_mb=round(_peak_rss_mb(), 1), device=self.device,
                backend=self.name, dtype=self.dtype, error=f"{type(exc).__name__}: {exc}",
            )


class StubBackend:
    """Deterministic offline backend used by tests and by `--dry-run`.

    Paints a seed-derived gradient. It produces real, decodable PNGs of the right
    size so the validator, manifest and evaluation paths can be exercised end to
    end without downloading weights.
    """

    name = "stub"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def run(self, job: Job, out_dir: Path) -> JobResult:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / job.output_name
        t0 = time.perf_counter()
        try:
            img = Image.new("RGB", (job.width, job.height))
            px = img.load()
            # Deterministic LCG so the same seed always yields the same bytes,
            # with enough entropy that the PNG is realistically sized - a pure
            # gradient compresses to <2 KB and would not exercise the
            # truncation/decode branches of the validator.
            state = (job.seed * 1103515245 + 12345) & 0x7FFFFFFF
            for y in range(job.height):
                for x in range(job.width):
                    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                    n = (state >> 16) % 48 - 24
                    px[x, y] = (
                        max(0, min(255, (x * 3 + job.seed) % 256 + n)),
                        max(0, min(255, (y * 5 + job.seed // 2) % 256 + n)),
                        max(0, min(255, ((x + y) * 2 + job.seed // 3) % 256 + n)),
                    )
            img.save(target, format="PNG")
            return JobResult(
                job_id=job.job_id, status="ok", image_path=str(target),
                sha256=sha256_file(target), runtime_sec=round(time.perf_counter() - t0, 3),
                peak_rss_mb=round(_peak_rss_mb(), 1), device=self.device, backend=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return JobResult(
                job_id=job.job_id, status="error", image_path=None, sha256=None,
                runtime_sec=round(time.perf_counter() - t0, 3),
                peak_rss_mb=round(_peak_rss_mb(), 1), device=self.device,
                backend=self.name, error=f"{type(exc).__name__}: {exc}",
            )


def get_backend(name: str, device: str = "auto") -> Backend:
    if name == "local":
        return LocalDiffusersBackend(device=device)
    if name == "stub":
        return StubBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'local' or 'stub'")
