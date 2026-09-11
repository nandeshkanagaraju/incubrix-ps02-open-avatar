#!/usr/bin/env bash
# One documented run path: prepares, executes, validates, evaluates and
# benchmarks every batch, then collects evidence and regenerates the index.
#
#   ./scripts/run_all.sh [DEVICE]     DEVICE defaults to auto (mps on Apple Silicon)
#
# Safe to re-run: `render` resumes from results.json and skips jobs whose image
# is already present and hash-matching.
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${1:-auto}"
OA=".venv/bin/openavatar"
PY=".venv/bin/python"

echo "=== 1/6  Baseline: six fictional avatars (sd15) ==================="
$OA plan specs/batch_a_fictional --batch-id batch_a_sd15 -m sd15
$OA render runs/batch_a_sd15 --backend local --device "$DEVICE"

echo "=== 2/6  Strong: controlled single-attribute changes, two approaches ==="
$OA plan specs/batch_b_attrctl --batch-id batch_b_sd15 -m sd15
$OA render runs/batch_b_sd15 --backend local --device "$DEVICE"
$OA plan specs/batch_b_attrctl --batch-id batch_b_lcm -m sd15_lcm
$OA render runs/batch_b_lcm --backend local --device "$DEVICE"

echo "=== 3/6  Edge cases: ambiguous / disallowed / corrupted output ====="
$OA plan specs/batch_c_edge --batch-id batch_c_edge
$OA render runs/batch_c_edge --backend local --device "$DEVICE"
$PY scripts/corrupt_output.py runs/batch_c_edge

echo "=== 4/6  Validate + evaluate ======================================"
for b in batch_a_sd15 batch_b_sd15 batch_b_lcm batch_c_edge; do
  # validate exits 2 when an output fails; batch_c_edge is expected to fail.
  $OA validate "runs/$b" || echo "   (validate reported failures for $b - expected for batch_c_edge)"
  $OA evaluate "runs/$b"
done

echo "=== 5/6  Benchmark ================================================"
$OA bench runs/batch_a_sd15 --device "$DEVICE" --repetitions 3 --warmup 1 --limit 2

echo "=== 6/6  Collect evidence ========================================="
$PY -m pytest -q > evidence/test_output.txt 2>&1 || true
tail -1 evidence/test_output.txt
$PY scripts/build_evidence.py runs/batch_a_sd15 runs/batch_b_sd15 runs/batch_b_lcm runs/batch_c_edge

echo
echo "Done. See EVIDENCE_INDEX.md"
