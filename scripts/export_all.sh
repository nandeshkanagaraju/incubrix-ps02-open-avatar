#!/usr/bin/env bash
# Prepare every batch locally and package the job bundles for the free-accelerator run.
# Produces submission/bundles.zip -> upload to Kaggle, unzip to /kaggle/working/bundles
set -euo pipefail
cd "$(dirname "$0")/.."
OA=".venv/bin/openavatar"

rm -rf runs/batch_a_sd15 runs/batch_a_sdturbo runs/batch_b_sd15 runs/batch_b_lcm runs/batch_c_edge

# Baseline: six fictional avatars, primary model
$OA plan specs/batch_a_fictional --batch-id batch_a_sd15    -m sd15
# Strong: same six specs on a second open model (licence-restricted comparison)
$OA plan specs/batch_a_fictional --batch-id batch_a_sdturbo -m sdturbo
# Strong: controlled single-attribute changes under two conditioning approaches
$OA plan specs/batch_b_attrctl   --batch-id batch_b_sd15    -m sd15
$OA plan specs/batch_b_attrctl   --batch-id batch_b_lcm     -m sd15_lcm
# Edge cases: ambiguous + disallowed are held locally; only the control executes
$OA plan specs/batch_c_edge      --batch-id batch_c_edge    -m sd15

mkdir -p submission/bundles
for b in batch_a_sd15 batch_a_sdturbo batch_b_sd15 batch_b_lcm batch_c_edge; do
  cp "runs/$b/bundle.json" "submission/bundles/$b.json"
done

rm -f submission/bundles.zip
(cd submission && zip -qr bundles.zip bundles)
echo
echo "Job count per bundle:"
for f in submission/bundles/*.json; do
  python3 -c "import json,sys;d=json.load(open('$f'));print(f\"  {d['batch_id']:18s} {len(d['jobs']):2d} jobs, {len(d.get('held',[]))} held\")"
done
echo
echo "Upload submission/bundles.zip to Kaggle."
