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
# ---- load the locally-prepared bundle -------------------------------------
from pathlib import Path
import json

BUNDLE_JSON = "/kaggle/working/bundle.json"   # <-- edit if you uploaded a dataset

bundle = json.loads(Path(BUNDLE_JSON).read_text())
assert bundle["schema_version"] == 2, f"unexpected schema_version {bundle['schema_version']}"
jobs = bundle["jobs"]
print(f"batch_id   : {bundle['batch_id']}")
print(f"jobs       : {len(jobs)}")
print(f"held specs : {len(bundle.get('held', []))}  (refused/ambiguous - not executed here)")
for j in jobs[:3]:
    print(f"  {j['job_id']}  {j['repo_id']}@{j['revision'][:12]}  {j['width']}x{j['height']} seed={j['seed']}")
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
# ---- execute every job ----------------------------------------------------
import hashlib, time, json
from pathlib import Path

OUT = Path("/kaggle/working/out"); (OUT/"images").mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p,"rb") as fh:
        for c in iter(lambda: fh.read(1<<20), b""): h.update(c)
    return h.hexdigest()

results = []
for i, job in enumerate(jobs, 1):
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
        target = OUT/"images"/job["output_name"]
        img.save(target, format="PNG")
        peak = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else None
        rec = dict(job_id=job["job_id"], status="ok",
                   image_path=f"images/{job['output_name']}",
                   sha256=sha256_file(target),
                   runtime_sec=round(time.perf_counter()-t0,3),
                   peak_rss_mb=None, peak_accelerator_mb=(round(peak,1) if peak else None),
                   device=DEVICE, backend="notebook")
        print(f"[{i}/{len(jobs)}] ok   {job['job_id']}  {rec['runtime_sec']}s  peakGPU={rec['peak_accelerator_mb']}MB")
    except Exception as e:
        rec = dict(job_id=job["job_id"], status="error", image_path=None, sha256=None,
                   runtime_sec=round(time.perf_counter()-t0,3), peak_rss_mb=None,
                   device=DEVICE, backend="notebook", error=f"{type(e).__name__}: {e}")
        print(f"[{i}/{len(jobs)}] FAIL {job['job_id']}: {rec['error']}")
    results.append(rec)

print(f"\\n{sum(1 for r in results if r['status']=='ok')}/{len(results)} succeeded")
'''

CELL_PACK = '''\
# ---- package results for local ingestion ----------------------------------
import json, shutil, platform, subprocess, sys
from pathlib import Path

(OUT/"results.json").write_text(json.dumps({
    "batch_id": bundle["batch_id"],
    "route": "notebook",
    "environment": {
        "provider": "kaggle",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    },
    "results": results,
}, indent=2, sort_keys=True))

shutil.copy("requirements_notebook.txt", OUT/"requirements_notebook.txt")
archive = shutil.make_archive(f"/kaggle/working/results_{bundle['batch_id']}", "zip", OUT)
print("download this file ->", archive)
print("then run locally:")
print(f"  openavatar ingest runs/{bundle['batch_id']} results_{bundle['batch_id']}.zip")
print(f"  openavatar validate runs/{bundle['batch_id']}")
print(f"  openavatar evaluate runs/{bundle['batch_id']}")
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
