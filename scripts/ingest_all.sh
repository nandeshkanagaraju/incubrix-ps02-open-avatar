#!/usr/bin/env bash
# Ingest a results_all.zip produced by the Kaggle executor, then validate,
# evaluate and collect evidence for every batch.
#
#   ./scripts/ingest_all.sh ~/Downloads/results_all.zip
#
# batch_c_edge is expected to fail validation: scripts/corrupt_output.py
# deliberately truncates one of its outputs to evidence the corrupted-output case.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${1:?usage: ingest_all.sh <results_all.zip>}"
OA=".venv/bin/openavatar"
PY=".venv/bin/python"
STAGE="submission/results_all"

rm -rf "$STAGE"
mkdir -p "$STAGE"
unzip -qo "$ARCHIVE" -d "$STAGE"
echo "staged: $(ls "$STAGE")"

BATCHES=$(find "$STAGE" -name results.json -maxdepth 2 -exec dirname {} \; | xargs -n1 basename | sort)
echo "batches found: $BATCHES"

for b in $BATCHES; do
  echo "=== ingest $b ==="
  $OA ingest "runs/$b" "$STAGE/$b"
done

echo "=== inject the corrupted-output test case ==="
$PY scripts/corrupt_output.py runs/batch_c_edge

for b in $BATCHES; do
  echo "=== validate + evaluate $b ==="
  $OA validate "runs/$b" || echo "   (validation failures recorded for $b)"
  $OA evaluate "runs/$b"
done

echo "=== tests ==="
mkdir -p evidence
$PY -m pytest -q > evidence/test_output.txt 2>&1 || true
tail -1 evidence/test_output.txt

echo "=== collect evidence ==="
$PY scripts/build_evidence.py $(for b in $BATCHES; do echo -n "runs/$b "; done)

echo
echo "Done. Review EVIDENCE_INDEX.md"
