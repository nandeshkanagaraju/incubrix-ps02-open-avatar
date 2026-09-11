#!/usr/bin/env bash
# Demo driver. Run it, read the SAY lines aloud, press Enter between steps.
#   ./scripts/demo.sh
# Total command runtime ~90s; the rest is you talking. Target 6 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
OA=".venv/bin/openavatar"
PY=".venv/bin/python"

step() { printf '\n\033[1;36m════ %s ════\033[0m\n' "$1"; }
say()  { printf '\033[1;37m  • %s\033[0m\n' "$1"; }
pause() { printf '\n\033[0;90m   ── press Enter ──\033[0m\n'; read -r _ || true; }

# Demo specs: the committed ones at 384px/20 steps take ~50s each, which is dead
# air on camera. These are the same specs at 256px/8 steps - same code path,
# same pinning, just faster. Generated here so nothing extra is committed.
DEMO=$(mktemp -d)/specs; mkdir -p "$DEMO" "$DEMO"_edge
$PY - "$DEMO" <<'PYEOF'
import json, sys, pathlib
dst = pathlib.Path(sys.argv[1])
fast = {"width": 256, "height": 256, "steps": 8}
src = pathlib.Path("specs/batch_a_fictional/a1_studio_portrait.json")
d = json.loads(src.read_text()); d["render"].update(fast)
(dst / src.name).write_text(json.dumps(d, indent=2, sort_keys=True))
edge = pathlib.Path(str(dst) + "_edge")
for f in pathlib.Path("specs/batch_c_edge").glob("*.json"):
    e = json.loads(f.read_text()); e["render"].update(fast)
    (edge / f.name).write_text(json.dumps(e, indent=2, sort_keys=True))
PYEOF

clear
step "1. What this is"
say "PS02 Track 02. Local-first orchestrator for open-weight avatar generation."
say "Everything except the diffusion call runs on this 8GB M1 with no GPU."
say "Inference runs locally on MPS or on Kaggle free tier. No payment, no commercial API."
pause

step "2. Structured input, not a prompt string"
cat specs/batch_a_fictional/a1_studio_portrait.json
say "The input is a structured spec with closed vocabularies, not free text."
pause

step "3. Plan: spec -> fully pinned job"
$OA plan "$DEMO" --batch-id demo_a -m sd15 2>&1 | tail -8
$PY -c "import json;j=json.load(open('runs/demo_a/bundle.json'))['jobs'][0];print(json.dumps({k:j[k] for k in ['job_id','seed','steps','repo_id','revision','prompt']},indent=1))"
say "Planning resolves the spec into a job pinning prompt, seed and a 40-char commit SHA."
say "Nothing is looked up at execution time. That is what makes it reproducible on Kaggle."
pause

step "4. Safety: refusal before any compute is spent"
$OA plan "$DEMO"_edge --batch-id demo_c 2>&1 | tail -10
echo; cat runs/demo_c/safety_log.jsonl | $PY -c "import sys,json;[print(' ',d['spec_id'],'->',d['decision'],[f['code'] for f in d['findings']]) for d in map(json.loads,sys.stdin)]"
say "Three specs in, one job out. Disallowed is refused with stable codes."
say "Ambiguous is held for clarification, not silently guessed. Both kept as evidence."
pause

step "5. THE CODE DECISION: nationality labels can never reach a prompt"
sed -n '/^def build_prompt/,/^    return/p' src/openavatar/prompt.py | head -32
$PY -m pytest tests/test_prompt.py::test_region_context_never_reaches_the_prompt -q 2>&1 | tail -2
say "region_context records coverage for reporting. build_prompt only reads appearance and scene."
say "There is no access path from a region label to a prompt."
say "The test asserts the prompt is BYTE-IDENTICAL with and without the label."
say "That turns 'we do not stereotype' into something you can check."
pause

step "6. Live generation on this laptop"
$OA render runs/demo_a --backend local --device mps 2>&1 | grep -E "ok in|failed"
say "Real generation on an 8GB M1 with no GPU. 256px / 8 steps to keep this short;"
say "the committed evidence batches are 384px / 20 steps through the same code path."
NEW="runs/demo_a/images/a1_studio_portrait__sd15__s1001.png"
REF="evidence/batch_a_sd15/images/a1_studio_portrait__sd15__s1001.png"
$PY scripts/compare_image.py "$NEW" "$REF" /tmp/demo_compare.png \
  --left-label "JUST GENERATED - this laptop, Apple M1 MPS, 256px/8 steps" \
  --right-label "COMMITTED EVIDENCE - Kaggle Tesla T4, 384px/20 steps" \
  --note "Same spec, same seed 1001, same pinned revision 451f4fe1. Cross-route DINOv2 cosine 0.9995." >/dev/null
open /tmp/demo_compare.png
say "Opening it next to the committed Kaggle render of the same spec and seed ->"
pause

step "7. Output validation and provenance"
$OA validate runs/demo_a 2>&1 | tail -6
$PY -c "
import json;a=json.load(open('runs/demo_a/avatar_manifest.json'))['avatars'][0]
print(json.dumps({'seed':a['seed'],'model':a['model'],'output':a['output'],'compute_route':a['compute_route'],'synthetic_media':a['synthetic_media']},indent=1))"
say "Validation re-opens every file, re-hashes it, checks decoded size against the job."
say "It does not trust the executor's own report, because these images cross a network."
pause

step "8. FAILURE AND RECOVERY (the real one)"
open evidence/failures/fp16_mps_nan_black_output.png
$PY -c "
from PIL import Image,ImageStat;im=Image.open('evidence/failures/fp16_mps_nan_black_output.png').convert('RGB')
print('  stddev:',[round(x,1) for x in ImageStat.Stat(im).stddev],'<- pure black')"
say "First local run 'succeeded' and wrote a pure black PNG. No exception raised."
say "SD 1.5 fp16 on MPS emits NaN in the UNET latents - not the VAE, I checked."
say "The cause was attention slicing, which the brief itself recommends."
echo
sed -n '/Attention slicing is the standard/,/tests\/test_jobs/p' src/openavatar/backends.py
say "fp16 with slicing: 8.1s and NaN. Without: 5.9s and correct. Slower AND wrong."
pause

step "9. ...and the bug it exposed in my own validator"
$PY -m pytest tests/test_jobs_and_validate.py::test_tiny_blank_image_reports_both_codes -q 2>&1 | tail -2
say "The validator caught the black frames but reported only EMPTY_FILE."
say "A black PNG compresses under 1KB, tripping the size floor, and I returned early -"
say "so BLANK_IMAGE, the code that names the real defect, never ran."
say "A validator that rejects the right file for the wrong reason looks handled. That is worse."
pause

step "10. Corrupted output fails the batch"
$OA render runs/demo_c --backend local --device mps 2>&1 | grep -E "ok in|rendered"
$PY scripts/corrupt_output.py runs/demo_c
$OA validate runs/demo_c > /dev/null 2>&1; echo "  validate exit code = $?"
say "Exit code 2. A corrupted output fails the batch instead of passing quietly."
pause

step "11. Tests and benchmark"
$PY -m pytest -q 2>&1 | tail -2
say "67 tests, six seconds, no model weights and no network."
say "The backend is a Protocol with one method, so a stub replaces diffusers in CI."
echo
$PY -c "
import json;b=json.load(open('evidence/batch_a_local/benchmark.json'))
print('  median s/image :',b['summary']['median_time_per_image_sec'])
print('  peak RSS MB    :',b['peak_process_rss_mb'],' (current',b['current_process_rss_mb'],'- MPS weights sit in Metal unified memory)')
print('  model size MB  :',b['models']['sd15']['size_mb'])"
say "I had two misleading numbers here. Model size said 23.6 GB - it was summing"
say "checkpoints the pipeline never loads. Real figure 4265 MB. RSS said 36 MB while"
say "holding a 4 GB model. Both fixed; a misleading benchmark is an eligibility gate."
pause

step "12. The six baseline avatars"
$PY scripts/contact_sheet.py evidence/batch_a_sd15 /tmp/demo_sheet.png \
  --title "Baseline: 6 fictional avatars - neutral attributes, 6 coverage contexts, no nationality in any prompt" >/dev/null
open /tmp/demo_sheet.png
say "Six avatars spanning 6 skin tones, 4 age bands, 3 presentations, varied hair and attire."
say "Each caption is the SPEC that produced it. No prompt contained a nationality."
say "Visible misses too: a1 asked for feminine presentation and produced masculine."
say "That is the miss my attribute probe was failing to measure - see the report."
pause

step "13. Results and the product call"
$PY -c "
import json
rows=[('batch_a_local','M1 MPS'),('batch_a_sd15','Kaggle T4'),('batch_a_sdturbo','Kaggle T4'),('batch_b_sd15','Kaggle T4'),('batch_b_lcm','Kaggle T4')]
print(f\"  {'batch':18s}{'route':11s}{'CLIP':>7s}{'attr':>8s}\")
for b,r in rows:
    m=json.load(open(f'evidence/{b}/metrics.json'));k=list(m['by_model'])[0];s=m['by_model'][k]
    print(f'  {b:18s}{r:11s}{s[\"mean_clip_score\"]:7.2f}{s[\"mean_attribute_accuracy\"]:8.3f}')
rc=json.load(open('evidence/route_comparison.json'))
print('\n  cross-route DINOv2 cosine:',rc['mean_dinov2_cosine'],'(min',str(rc['min_dinov2_cosine'])+')')"
say "Same bundle, same seeds, M1 versus Tesla T4: DINOv2 cosine 0.9995."
say "Bit-identical is impossible across CUDA and MPS, so I measured equivalence instead."
say "sd-turbo scores best and CANNOT ship - Stability Non-Commercial. sd15 is the"
say "commercial path and it costs quality. That trade-off is the product decision."
say "pose scores 0 of 5 on both models - that is my metric failing, not the model,"
say "and I report it as such rather than hiding it in an average."
pause

step "Done"
say "AI_USE.md discloses I used Claude Code as a primary implementation author,"
say "and records what it got wrong along the way."
rm -rf runs/demo_a runs/demo_c
