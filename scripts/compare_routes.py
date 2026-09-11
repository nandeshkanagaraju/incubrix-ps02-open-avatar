#!/usr/bin/env python
"""Quantify reproducibility of the same job bundle across two compute routes.

The Baseline requirement is that batches be reproducible from a clean notebook.
Bit-identical output across different accelerators is not achievable - CUDA and
MPS reduce floating point in different orders - so the honest claim is
*perceptual and metric equivalence*, which this measures:

  * DINOv2 cosine between the two renders of the same job
  * the CLIPScore difference between routes
  * whether the files are byte-identical (expected: no)

    python scripts/compare_routes.py runs/batch_a_local runs/batch_a_sd15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openavatar.config import Config          # noqa: E402
from openavatar.jobs import Bundle            # noqa: E402
from openavatar.metrics import Embedders      # noqa: E402


def route_of(bundle_dir: Path) -> str:
    return json.loads((bundle_dir / "results.json").read_text()).get("route", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = Path(args.a), Path(args.b)
    cfg = Config.load()
    emb = Embedders.load(cfg)

    jobs_a = {j.job_id: j for j in Bundle.load(A).jobs}
    jobs_b = {j.job_id: j for j in Bundle.load(B).jobs}
    shared = sorted(set(jobs_a) & set(jobs_b))
    if not shared:
        print("no job_ids in common", file=sys.stderr)
        return 1

    met_a = json.loads((A / "metrics.json").read_text())["per_image"]
    met_b = json.loads((B / "metrics.json").read_text())["per_image"]

    rows = []
    for jid in shared:
        pa = A / "images" / jobs_a[jid].output_name
        pb = B / "images" / jobs_b[jid].output_name
        if not (pa.exists() and pb.exists()):
            continue
        vecs = emb.identity([pa, pb])
        cos = float(vecs[0] @ vecs[1])
        ca = met_a.get(jid, {}).get("clip_score")
        cb = met_b.get(jid, {}).get("clip_score")
        rows.append({
            "job_id": jid,
            "dinov2_cosine": round(cos, 4),
            "clip_a": ca, "clip_b": cb,
            "clip_delta": round(abs(ca - cb), 3) if (ca and cb) else None,
            "bytes_identical": pa.read_bytes() == pb.read_bytes(),
        })

    cos = [r["dinov2_cosine"] for r in rows]
    deltas = [r["clip_delta"] for r in rows if r["clip_delta"] is not None]
    report = {
        "route_a": {"bundle": str(A), "route": route_of(A)},
        "route_b": {"bundle": str(B), "route": route_of(B)},
        "n_compared": len(rows),
        "mean_dinov2_cosine": round(sum(cos) / len(cos), 4) if cos else None,
        "min_dinov2_cosine": round(min(cos), 4) if cos else None,
        "mean_abs_clip_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "max_abs_clip_delta": round(max(deltas), 3) if deltas else None,
        "any_bytes_identical": any(r["bytes_identical"] for r in rows),
        "note": (
            "Bit-identical output across CUDA and MPS is not achievable - the two "
            "backends reduce floating point in different orders. The reproducibility "
            "claim is therefore metric and perceptual equivalence at a fixed seed, "
            "not byte equality."
        ),
        "per_job": rows,
    }
    out = Path(args.out or ROOT / "evidence" / "route_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"compared {len(rows)} job(s): {report['route_a']['route']} vs {report['route_b']['route']}")
    print(f"  DINOv2 cosine   mean={report['mean_dinov2_cosine']}  min={report['min_dinov2_cosine']}")
    print(f"  |CLIP delta|    mean={report['mean_abs_clip_delta']}  max={report['max_abs_clip_delta']}")
    print(f"  bytes identical: {report['any_bytes_identical']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
