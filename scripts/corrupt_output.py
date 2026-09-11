#!/usr/bin/env python
"""Inject the required 'one corrupted output' test case.

Copies a successful render into a new bundle-local job, truncates it mid-stream,
and leaves the executor's original sha256 in place - exactly what a partial
download from a free accelerator looks like. Run `openavatar validate` after
this to capture the failure evidence.

    python scripts/corrupt_output.py runs/batch_c_edge --fraction 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openavatar.jobs import Bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--job-id", default=None, help="defaults to the first successful job")
    ap.add_argument("--fraction", type=float, default=0.6, help="fraction of bytes to keep")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    bundle = Bundle.load(bundle_dir)
    results = json.loads((bundle_dir / "results.json").read_text())
    ok = [r for r in results["results"] if r["status"] == "ok"]
    if not ok:
        print("no successful jobs to corrupt", file=sys.stderr)
        return 1

    target = next((r for r in ok if r["job_id"] == args.job_id), ok[0])
    job = bundle.job(target["job_id"])
    path = bundle_dir / "images" / job.output_name
    data = path.read_bytes()
    keep = int(len(data) * args.fraction)
    path.write_bytes(data[:keep])

    print(f"corrupted {path.name}: {len(data)} -> {keep} bytes "
          f"(executor sha256 {target['sha256'][:12]}... left intact)")
    print(f"now run: openavatar validate {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
