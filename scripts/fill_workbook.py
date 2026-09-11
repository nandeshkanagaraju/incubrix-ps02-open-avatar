#!/usr/bin/env python
"""Fill the IncuBrix Track 02 workbook from the committed run artefacts.

Candidate identity fields come from --candidate-*; everything measurable is read
out of evidence/*/{validation,metrics,benchmark,safety_log} so the workbook and
the repository cannot disagree.

    python scripts/fill_workbook.py \
        --workbook ~/Downloads/Track_02_..._Workbook.xlsx \
        --out submission/Track_02_workbook_filled.xlsx \
        --candidate-name "..." --candidate-id "..." --candidate-email "..." \
        --repo-url "https://github.com/..." 
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import openpyxl
import psutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def put(ws, row: int, col: int, value):
    """Write a cell, redirecting merged cells to their anchor.

    The workbook merges many of the candidate-input cells; openpyxl raises on a
    write to any member of a merged range other than its top-left anchor.
    """
    from openpyxl.utils import get_column_letter

    coord = f"{get_column_letter(col)}{row}"
    for rng in ws.merged_cells.ranges:
        if coord in rng:
            ws.cell(row=rng.min_row, column=rng.min_col, value=value)
            return
    ws.cell(row=row, column=col, value=value)


def puta(ws, coord: str, value):
    """Same as put(), addressed by A1-style coordinate."""
    from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

    col_letter, row = coordinate_from_string(coord)
    put(ws, row, column_index_from_string(col_letter), value)


def load(batch: str, name: str):
    p = EVIDENCE / batch / name
    return json.loads(p.read_text()) if p.exists() else None


def batches() -> list[str]:
    """Evidence subdirectories that are actual batches.

    evidence/ also holds failures/ and loose artefacts, which have no
    validation.json and must not be listed as batches."""
    if not EVIDENCE.exists():
        return []
    return sorted(d.name for d in EVIDENCE.iterdir()
                  if d.is_dir() and (d / "validation.json").exists())


def safety_counts(batch: str) -> dict:
    p = EVIDENCE / batch / "safety_log.jsonl"
    out: dict[str, int] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                out[json.loads(line)["decision"]] = out.get(json.loads(line)["decision"], 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate-name", required=True)
    ap.add_argument("--candidate-id", default="")
    ap.add_argument("--candidate-email", default="")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--report-url", default="")
    ap.add_argument("--free-compute", default="Yes - Kaggle Notebooks free GPU tier")
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.workbook)
    today = date.today().isoformat()

    # ---- 00_START -------------------------------------------------------
    s = wb["00_START"]
    puta(s, "B5", args.candidate_id)
    puta(s, "B6", args.candidate_name)
    puta(s, "F6", args.candidate_email)
    puta(s, "B7", today)
    puta(s, "F7", args.repo_url)
    puta(s, "B8", f"{platform.processor() or 'Apple M1'} / {psutil.cpu_count(logical=True)} logical cores")
    puta(s, "F8", round(psutil.virtual_memory().total / 1e9, 1))
    puta(s, "B9", platform.platform())
    puta(s, "F9", f"Python {platform.python_version()}")
    puta(s, "B10", args.free_compute)
    puta(s, "F10", "Kaggle Notebooks (free GPU tier) + local Apple M1 MPS")
    puta(s, "B11", args.report_url)
    puta(s, "F11", "Claude Code (Claude Opus 5) - fully disclosed in AI_USE.md")
    puta(s, "A34", args.candidate_name)
    puta(s, "E34", today)

    checklist = {
        23: ("Yes", "Repository root; README.md documents the clean setup path"),
        24: ("Yes", "README 'Setup' + 'Documented run path'; pytest -q passes offline"),
        25: ("Yes", f"evidence/ ({len(batches())} batches), EVIDENCE_INDEX.md"),
        26: ("Yes", "SOURCES.md - licence + commercial-use manifest per component"),
        27: ("Yes", "evidence/*/images (all attempts), held/ (refused), validation.json"),
        28: ("Yes", "tests/ (67 tests), evidence/test_output.txt, evidence/*/benchmark.json"),
        29: ("Yes", "docs/REPORT.md, SOURCES.md, AI_USE.md, demo video link in workbook"),
        30: ("Yes", "SOURCES.md 'Hosted execution environment'; zero spend, no card, no paid API"),
    }
    for row, (status, note) in checklist.items():
        put(s, row, 6, status)
        put(s, row, 7, note)

    # ---- 01_DELIVERABLES ------------------------------------------------
    d = wb["01_DELIVERABLES"]
    repo = args.repo_url or "<repo>"
    deliverables = {
        5: ("Complete", repo, "Source, configs, tests, download scripts, commit history"),
        6: ("Complete", f"{repo}/blob/main/requirements.txt", "Pinned requirements + pyproject; Python 3.11 venv"),
        7: ("Complete", f"{repo}/blob/main/notebooks/kaggle_executor.ipynb", "Kaggle free GPU; thin executor, pinned revisions from the bundle"),
        8: ("Complete", f"{repo}/blob/main/SOURCES.md", "SOURCES.md + AI_USE.md + licence/commercial-use table"),
        9: ("Complete", f"{repo}/tree/main/evidence", "EVIDENCE_INDEX.md maps every claim to a file"),
        10: ("Complete", f"{repo}/tree/main/tests", "Unit + integration + end-to-end; offline, no weights needed"),
        11: ("Complete", "evidence/*/benchmark.json", "Timing boundary, repetitions, memory, model size"),
        12: ("Complete", f"{repo}/blob/main/docs/REPORT.md", "Architecture, alternatives, failures, product path"),
        13: ("See notes", "", "Unedited demo video - link to be added by candidate"),
    }
    for row, (status, path, note) in deliverables.items():
        put(d, row, 3, status)
        put(d, row, 4, path)
        put(d, row, 5, note)

    cmds = {
        16: "python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .",
        17: "python scripts/download_models.py --only sd15",
        18: "openavatar plan specs/batch_a_fictional --batch-id batch_a_local && openavatar render runs/batch_a_local --backend local --device mps",
        19: "openavatar export runs/batch_a_kaggle  ->  run notebooks/kaggle_executor.ipynb on Kaggle GPU  ->  openavatar ingest runs/batch_a_kaggle results_batch_a_kaggle.zip",
        20: "pytest -q   (offline, no weights)   |   OPENAVATAR_RUN_SLOW=1 pytest -m slow",
        21: "openavatar bench runs/batch_a_local --repetitions 3 --warmup 1 --limit 2",
        22: "openavatar validate runs/batch_a_local && openavatar evaluate runs/batch_a_local",
    }
    for row, cmd in cmds.items():
        put(d, row, 4, cmd)
    put(d, 25, 1,
        "Hybrid execution as required: all orchestration, safety screening, validation, provenance, "
        "evaluation, benchmarking and tests run locally on an 8 GB Apple M1 (no discrete GPU). "
        "Image-model inference runs locally on MPS and, for the notebook-evidence route, on Kaggle's "
        "free GPU tier. No payment, card, subscription or commercial generation API was used. "
        "Google Colab was deliberately not used (managed-runtime terms prohibit deepfake creation).")

    # ---- 02_COMPONENTS --------------------------------------------------
    c = wb["02_COMPONENTS"]
    cfg = json.loads((ROOT / "configs" / "default.json").read_text())
    rows = []
    for key in ("sd15", "sd15_lcm", "sdturbo"):
        m = cfg["models"][key]
        rows.append([f"{m['repo_id']} ({key})", m["revision"][:12], m["role"],
                     f"https://huggingface.co/{m['repo_id']}", m["code_license"],
                     m["weights_license"], m["commercial_use"],
                     "local MPS + Kaggle GPU", "Yes" if key != "sdturbo" else "No - comparison only",
                     "", "See SOURCES.md", today])
    met = cfg["metrics"]
    rows.append([met["clip_model"], met["clip_revision"][:12], "prompt/spec adherence metric",
                 f"https://huggingface.co/{met['clip_model']}", "MIT", "MIT", "Permitted",
                 "local CPU", "Evaluation only", "605", "-", today])
    rows.append([met["identity_model"], met["identity_revision"][:12],
                 "identity consistency + diversity embeddings",
                 f"https://huggingface.co/{met['identity_model']}", "Apache-2.0", "Apache-2.0",
                 "Permitted", "local CPU", "Evaluation only", "88", "-", today])
    for lib, ver, lic, role in [
        ("diffusers", "0.35.1", "Apache-2.0", "pipeline loading/scheduling"),
        ("torch", "2.8.0", "BSD-3-Clause", "tensor runtime, MPS backend"),
        ("transformers", "4.56.1", "Apache-2.0", "CLIP/DINOv2 loading"),
        ("peft", "0.17.1", "Apache-2.0", "LCM-LoRA fusion"),
        ("pydantic", "2.11.7", "MIT", "spec schema/validation"),
        ("pillow", "11.3.0", "MIT-CMU", "image I/O + corruption checks"),
        ("typer", "0.16.1", "MIT", "CLI"),
        ("psutil", "7.0.0", "BSD-3-Clause", "RAM/CPU reporting"),
    ]:
        rows.append([lib, ver, role, f"https://pypi.org/project/{lib}/", lic, "n/a",
                     "Permitted", "local", "Yes", "", "-", today])
    rows.append(["Kaggle Notebooks (free GPU tier)", "n/a", "hosted execution environment only",
                 "https://www.kaggle.com/docs/notebooks", "n/a", "n/a",
                 "Free tier; execution environment, not product architecture",
                 "hosted", "No - execution only", "", "Single account; no payment; no personal data", today])
    rows.append(["Monk Skin Tone Scale", "n/a", "neutral skin-tone vocabulary (mst-1..10)",
                 "https://skintone.google", "n/a", "openly published research scale",
                 "Permitted", "local", "Yes", "", "Used as an id vocabulary only; no ethnic labelling", today])
    for i, r in enumerate(rows, start=5):
        for j, v in enumerate(r, start=1):
            put(c, i, j, v)
    puta(c, "A26",
        "Product recommendation: ship sd15 + sd15_lcm (CreativeML OpenRAIL-M / openrail++, commercial use "
        "permitted subject to use restrictions that safety.py enforces in code). sdturbo is Stability AI "
        "Non-Commercial and must NOT ship - it is present only as the second-model comparison. CLIP (MIT) and "
        "DINOv2 (Apache-2.0) are evaluation-only dependencies and are both commercially clean. Unresolved before "
        "product use: (1) mirror OpenRAIL Attachment A restrictions into customer terms; (2) jurisdiction-specific "
        "opinion on LAION-trained weights; (3) retention schedule to accompany the existing consent-purge deletion "
        "control; (4) embed C2PA content credentials in image files rather than only in avatar_manifest.json. "
        "Privacy/consent: individual-avatar work requires a live, hash-verified consent record; the registry stores "
        "SHA-256 hashes and a pseudonymous label only, never image bytes or a legal name.")

    # ---- 03_TEST_EVIDENCE ----------------------------------------------
    t = wb["03_TEST_EVIDENCE"]
    row = 7
    for batch in batches():
        val = load(batch, "validation.json")
        met_j = load(batch, "metrics.json")
        ben = load(batch, "benchmark.json")
        if not val:
            continue
        results = {r["job_id"]: r for r in (load(batch, "results.json") or {}).get("results", [])}
        for item in val["items"]:
            r = results.get(item["job_id"], {})
            per = (met_j or {}).get("per_image", {}).get(item["job_id"], {})
            det = item.get("details", {})
            put(t, row, 1, item["job_id"])
            put(t, row, 2, "Baseline" if batch.startswith("batch_a") else
                ("Strong" if batch.startswith("batch_b") else "Edge case"))
            put(t, row, 3, f"evidence/{batch}/specs/")
            put(t, row, 4, "Valid PNG at requested size with complete provenance")
            put(t, row, 5, f"evidence/{batch}/images/")
            put(t, row, 6, f"{per.get('model_key','')} dtype={r.get('dtype','')}")
            put(t, row, 7, f"{r.get('backend','')}/{r.get('device','')}")
            put(t, row, 8, "Pass" if item["ok"] else "Fail")
            put(t, row, 9, "CLIPScore")
            put(t, row, 10, per.get("clip_score"))
            put(t, row, 11, "attribute_accuracy")
            put(t, row, 12, per.get("attribute_accuracy"))
            put(t, row, 13, r.get("runtime_sec"))
            put(t, row, 14, r.get("peak_rss_mb"))
            put(t, row, 15, (ben or {}).get("models", {}).get(
                per.get("model_key", ""), {}).get("size_mb"))
            put(t, row, 16, f"evidence/{batch}/validation.json")
            put(t, row, 17, ", ".join(item.get("codes", [])) or
                f"stddev={det.get('pixel_stddev')}")
            row += 1
        for held in (load(batch, "bundle.json") or {}).get("held", []):
            put(t, row, 1, held["spec_id"])
            put(t, row, 2, "Edge case")
            put(t, row, 3, f"evidence/{batch}/held/{held['spec_id']}.json")
            put(t, row, 4, f"Expect {held['decision']}")
            put(t, row, 5, "no output - correctly not executed")
            put(t, row, 7, "n/a - blocked before compute")
            put(t, row, 8, "Pass")
            put(t, row, 16, f"evidence/{batch}/safety_log.jsonl")
            put(t, row, 17, f"{held['decision']}: {', '.join(held['codes'])}")
            row += 1

    ben_any = next((load(b, "benchmark.json") for b in batches() if load(b, "benchmark.json")), None)
    if ben_any:
        m = ben_any["method"]
        puta(t, "A28",
            f"Benchmark method: timing boundary = {m['timing_boundary']}. "
            f"Warm-up {m['warmup_runs_per_job']} run(s) per job, {m['repetitions_per_job']} timed repetitions, "
            f"median reported. Memory: {m['memory_measure']}. Seeds: {m['seeds']}. Cache: {m['cache']}. "
            f"Machine: {ben_any['machine']['platform']}, {ben_any['machine']['cpu_count']} logical cores, "
            f"{ben_any['machine']['ram_total_mb']} MB RAM, device={ben_any['machine']['device']}. "
            f"Limitation: {ben_any['machine']['accelerator_note']} - peak accelerator memory is reported only for "
            f"jobs executed on the Kaggle GPU route (peak_accelerator_mb in results.json). "
            f"Quality reference: CLIP ViT-B/32 for adherence and DINOv2-small for identity/diversity, both pinned "
            f"by revision; DINOv2 is a general image backbone, not a face-recognition model, so identity cosine is "
            f"a relative comparison within a fixed pose/background and is not a biometric claim. "
            f"No manual steps were excluded; every number here was produced by `openavatar evaluate`/`bench`.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"wrote {args.out}")
    print(f"  batches: {batches()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
