#!/usr/bin/env python
"""Build notebooks/kaggle_executor.ipynb from the cell sources below.

The notebook is generated rather than hand-edited so it stays in sync with the
bundle schema and so the repo holds no stray execution_count/output noise.

The notebook is deliberately a THIN EXECUTOR: it reads a bundle produced by
`openavatar plan`, runs each job against the pinned revision recorded *in the
bundle*, and writes a results archive. It contains no prompt construction, no
safety logic and no validation - all of that stays in the local package.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MD_INTRO = """\
# PS02 Track 02 - free-accelerator executor

**Role of this notebook:** execute a job bundle prepared locally by
`openavatar plan`. It is an *execution environment*, not the product
architecture - prompt construction, safety screening, validation, provenance and
evaluation all run on the candidate's laptop.

**Compute disclosure:** Kaggle Notebooks, free GPU tier (T4 x2 or P100).
No payment, no card, no paid API. Kaggle is used because realistic latent
diffusion on an 8 GB CPU laptop is impractical at batch scale.

**Not Google Colab:** Colab's managed-runtime terms prohibit deepfake creation,
and this project includes an individual-avatar capability, so Colab is excluded
on policy grounds even for the fictional batches.

## How to run
1. Create a new Kaggle notebook, set Accelerator = GPU.
2. Upload the bundle zip produced by `openavatar export runs/<batch_id>` as a
   Kaggle Dataset, or drag `bundle.json` into `/kaggle/working`.
3. Set `BUNDLE_JSON` below to its path and Run All.
4. Download `results_<batch_id>.zip` from the output pane.
5. Locally: `openavatar ingest runs/<batch_id> results_<batch_id>.zip`
"""

CELL_ENV = '''\
# Pinned environment. Kaggle images change; pinning keeps runs reproducible.
!pip -q install "diffusers==0.35.1" "transformers==4.56.1" "accelerate==1.10.1" \\
                "peft==0.17.1" "safetensors==0.6.2" "huggingface-hub==0.34.4"
import subprocess, sys, json, platform, torch
print("python  :", platform.python_version())
print("torch   :", torch.__version__)
print("cuda    :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("pip freeze -> requirements_notebook.txt")
open("requirements_notebook.txt","w").write(
    subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True).stdout)
'''

CELL_LOAD = '''\
# ---- load every locally-prepared bundle -----------------------------------
from pathlib import Path
import json, glob

# Upload bundles.zip (produced by scripts/export_all.sh) and unzip it here.
BUNDLE_DIR = "/kaggle/working/bundles"

paths = sorted(glob.glob(f"{BUNDLE_DIR}/*.json"))
assert paths, f"no bundle json found in {BUNDLE_DIR} - did you unzip bundles.zip?"

bundles = []
for p in paths:
    b = json.loads(Path(p).read_text())
    assert b["schema_version"] == 2, f"{p}: unexpected schema_version {b['schema_version']}"
    bundles.append(b)

total = sum(len(b["jobs"]) for b in bundles)
print(f"{len(bundles)} bundle(s), {total} job(s) total\\n")
for b in bundles:
    models = sorted({f"{j['repo_id'].split('/')[-1]}@{j['revision'][:8]}" for j in b["jobs"]})
    print(f"  {b['batch_id']:20s} {len(b['jobs']):2d} jobs  held={len(b.get('held', []))}  {models}")
'''

CELL_PIPE = '''\
# ---- pipeline cache: load each pinned checkpoint at most once -------------
import torch
from diffusers import DiffusionPipeline, LCMScheduler

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}

def get_pipe(job):
    key = (job["repo_id"], job["revision"], job.get("scheduler"),
           (job.get("lora") or {}).get("repo_id"))
    if key in _cache:
        return _cache[key]
    pipe = DiffusionPipeline.from_pretrained(
        job["repo_id"], revision=job["revision"], torch_dtype=DTYPE,
        safety_checker=None, requires_safety_checker=False, use_safetensors=True)
    if job.get("scheduler") == "LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if job.get("lora"):
        pipe.load_lora_weights(job["lora"]["repo_id"],
                               revision=job["lora"].get("revision"),
                               weight_name=job["lora"].get("weight_name"))
        pipe.fuse_lora()
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    _cache.clear()          # one resident pipeline; Kaggle GPUs are shared
    _cache[key] = pipe
    return pipe
'''

CELL_RUN = '''\
# ---- execute every job of every bundle ------------------------------------
# Jobs are grouped by checkpoint so each set of weights is loaded exactly once,
# even when the same model is shared across bundles.
import hashlib, time, json
from pathlib import Path

OUT = Path("/kaggle/working/out")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p,"rb") as fh:
        for c in iter(lambda: fh.read(1<<20), b""): h.update(c)
    return h.hexdigest()

work = [(b["batch_id"], j) for b in bundles for j in b["jobs"]]
work.sort(key=lambda t: (t[1]["repo_id"], t[1]["revision"], str(t[1].get("scheduler")),
                         str((t[1].get("lora") or {}).get("repo_id"))))

results = {b["batch_id"]: [] for b in bundles}
for i, (batch_id, job) in enumerate(work, 1):
    out_dir = OUT/batch_id/"images"; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        pipe = get_pipe(job)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        g = torch.Generator(device="cpu").manual_seed(job["seed"])
        kw = dict(prompt=job["prompt"], width=job["width"], height=job["height"],
                  num_inference_steps=job["steps"],
                  guidance_scale=job["guidance_scale"], generator=g)
        if job["guidance_scale"] > 1.0:
            kw["negative_prompt"] = job["negative_prompt"]
        img = pipe(**kw).images[0]
        target = out_dir/job["output_name"]
        img.save(target, format="PNG")
        peak = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else None
        rec = dict(job_id=job["job_id"], status="ok",
                   image_path=f"images/{job['output_name']}",
                   sha256=sha256_file(target),
                   runtime_sec=round(time.perf_counter()-t0,3),
                   peak_rss_mb=None, peak_accelerator_mb=(round(peak,1) if peak else None),
                   device=DEVICE, backend="notebook", dtype=str(DTYPE).replace("torch.",""))
        print(f"[{i}/{len(work)}] ok   {batch_id}/{job['job_id']}  {rec['runtime_sec']}s  peakGPU={rec['peak_accelerator_mb']}MB")
    except Exception as e:
        rec = dict(job_id=job["job_id"], status="error", image_path=None, sha256=None,
                   runtime_sec=round(time.perf_counter()-t0,3), peak_rss_mb=None,
                   device=DEVICE, backend="notebook", dtype=str(DTYPE).replace("torch.",""),
                   error=f"{type(e).__name__}: {e}")
        print(f"[{i}/{len(work)}] FAIL {batch_id}/{job['job_id']}: {rec['error']}")
    results[batch_id].append(rec)

ok = sum(1 for rs in results.values() for r in rs if r["status"]=="ok")
print(f"\\n{ok}/{len(work)} succeeded")
'''

CELL_PACK = '''\
# ---- package results for local ingestion ----------------------------------
import json, shutil, platform
from pathlib import Path

env = {
    "provider": "kaggle",
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__,
    "python": platform.python_version(),
    "dtype": str(DTYPE).replace("torch.", ""),
}

for b in bundles:
    bid = b["batch_id"]
    (OUT/bid).mkdir(parents=True, exist_ok=True)
    (OUT/bid/"results.json").write_text(json.dumps({
        "batch_id": bid, "route": "notebook",
        "environment": env, "results": results[bid],
    }, indent=2, sort_keys=True))

shutil.copy("requirements_notebook.txt", OUT/"requirements_notebook.txt")
(OUT/"environment.json").write_text(json.dumps(env, indent=2, sort_keys=True))
archive = shutil.make_archive("/kaggle/working/results_all", "zip", OUT)
print("download this file ->", archive)
print()
print("then run locally, from the repo root:")
print("  unzip -o results_all.zip -d results_all")
for b in bundles:
    print(f"  openavatar ingest runs/{b['batch_id']} results_all/{b['batch_id']}")
'''


def cell(source: str, kind: str = "code") -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def main() -> None:
    nb = {
        "cells": [
            cell(MD_INTRO, "markdown"),
            cell("## 1. Pinned environment", "markdown"), cell(CELL_ENV),
            cell("## 2. Load the locally-prepared job bundle", "markdown"), cell(CELL_LOAD),
            cell("## 3. Pipeline cache (pinned revisions from the bundle)", "markdown"), cell(CELL_PIPE),
            cell("## 4. Execute jobs", "markdown"), cell(CELL_RUN),
            cell("## 5. Package results for `openavatar ingest`", "markdown"), cell(CELL_PACK),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "kaggle_executor.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
