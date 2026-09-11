# openavatar — PS02 / Track 02: Open-Source AI Human-Avatar Generation

A local-first orchestrator for generating human avatars with open-weight
diffusion models. Everything except image-model inference runs on a non-GPU
laptop: spec validation, safety and consent screening, job preparation, output
validation, provenance, evaluation, benchmarking and tests.

Inference runs either **locally** (CPU or Apple MPS, slow but fully
self-contained) or on a **disclosed free accelerator** (Kaggle GPU) via a thin
executor notebook. Both routes produce identical records, so validation and
scoring do not care which was used.

---

## Architecture

```
 specs/*.json ──► plan ──► runs/<batch>/bundle.json ──┬─► render  (local CPU/MPS)
   AvatarSpec      │                                  │
   (pydantic,      │  safety.screen()                 └─► export ─► Kaggle notebook
    closed vocab)  │  ├─ REFUSE   → held/                             │
                   │  ├─ NEEDS_CLARIFICATION → held/                  ▼
                   │  └─ ALLOW    → Job (prompt, seed, pinned      results_<batch>.zip
                   │                 repo_id@revision)                │
                   ▼                                                  ▼
            safety_log.jsonl                                       ingest
                                                                      │
                                          results.json ◄──────────────┘
                                                │
                                                ▼
                                     validate ──► validation.json
                                                  avatar_manifest.json
                                                │
                                                ▼
                                     evaluate ──► metrics.json
                                     bench    ──► benchmark.json
```

| Module | Responsibility |
|---|---|
| `spec.py` | Input contract. Closed vocabularies, `extra="forbid"`, cross-field coherence rules. |
| `attributes.py` | Neutral appearance vocabulary + prompt phrase tables. |
| `prompt.py` | Deterministic prompt/negative-prompt construction; attribute probes for scoring. |
| `safety.py` | Refusal rules, ambiguity detection, hash-backed consent registry, JSONL safety log. |
| `jobs.py` | Portable job bundles; resolves each spec to a fully-pinned `Job`. |
| `backends.py` | `local` (diffusers CPU/MPS) and `stub` (deterministic, offline) executors. |
| `render.py` | Batch driver with cache/resume; `ingest` for notebook results. |
| `validate.py` | Output validation (corruption, dimensions, blank, hash) + `avatar_manifest.json`. |
| `metrics.py` | CLIP adherence, DINOv2 identity consistency, diversity/coverage. |
| `bench.py` | Repeated-run timing, memory, model size. |
| `cli.py` | The documented CLI contract. |

### Two design decisions worth reading the code for

**1. Nationality labels can never reach a prompt.**
`AvatarSpec.region_context` records which geographic/cultural coverage slot a
spec fills. `prompt.build_prompt()` reads only `appearance` and `scene` — it has
no access path to `region_context` or `notes`. `tests/test_prompt.py::
test_region_context_never_reaches_the_prompt` asserts that adding a region label
produces a byte-identical prompt. Appearance is described with neutral traits
(Monk Skin Tone ids, hair length/texture, attire) chosen per-spec by a human, so
the system never encodes "nationality ⇒ appearance".

**2. Provenance is reconstructed from the artefacts, not trusted.**
`validate` re-opens every PNG, re-hashes it, and compares the decoded size
against the job that requested it, because images may arrive from a remote
notebook. `ingest` rejects foreign `batch_id`s, unknown `job_id`s, zip path
traversal, and any record that claims success without a matching image.

---

## Setup (CPU laptop, no GPU required)

```bash
git clone <repo-url> && cd ps02-open-avatar
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_models.py            # ~4.3 GB for sd15; add --only sd15
```

Verify with no weights and no network:

```bash
pytest -q                                    # 67 tests, stub backend only
openavatar plan specs/batch_c_edge --batch-id demo
openavatar render runs/demo --backend stub
openavatar validate runs/demo
```

## CLI contract

| Command | Input | Output | Exit |
|---|---|---|---|
| `plan <spec-dir> --batch-id ID` | directory of `*.json` specs | `runs/ID/{bundle.json,safety_log.jsonl,specs/,held/}` | 0, 1 on bad input |
| `export <bundle>` | bundle dir | `<bundle>.zip` (inputs only) | 0 |
| `render <bundle> --backend local\|stub --device auto\|cpu\|mps` | bundle dir | `images/*.png`, `results.json` | 0 |
| `ingest <bundle> <archive>` | bundle dir + notebook `.zip`/dir | `images/*.png`, `results.json` | 0, 1 on mismatch |
| `validate <bundle>` | bundle dir | `validation.json`, `avatar_manifest.json` | 0, **2 if any output failed** |
| `evaluate <bundle>` | validated bundle | `metrics.json` | 0 |
| `bench <bundle> -r N --limit K` | bundle dir | `benchmark.json` | 0 |
| `consent add\|revoke\|purge\|list` | — | consent registry | 0, 1 on unknown id |
| `report <bundle>` | bundle dir | stdout summary | 0 |

`--config` overrides the config file on every command; `OPENAVATAR_CONFIG` does
the same via the environment. No path or model id is hardcoded in the pipeline.

## Documented run path

```bash
# Baseline — six fictional avatars, local execution
openavatar plan specs/batch_a_fictional --batch-id batch_a_local
openavatar render  runs/batch_a_local --backend local --device mps
openavatar validate runs/batch_a_local
openavatar evaluate runs/batch_a_local

# Same batch on the free accelerator (reproducible second route)
openavatar export runs/batch_a_kaggle
#   -> upload bundle.json to notebooks/kaggle_executor.ipynb, Run All, download zip
openavatar ingest  runs/batch_a_kaggle results_batch_a_kaggle.zip
openavatar validate runs/batch_a_kaggle

# Strong — controlled single-attribute changes + second model comparison
openavatar plan specs/batch_b_attrctl --batch-id batch_b_sd15   -m sd15
openavatar plan specs/batch_b_attrctl --batch-id batch_b_lcm    -m sd15_lcm

# Edge cases: ambiguous / disallowed / corrupted output
openavatar plan specs/batch_c_edge --batch-id batch_c_edge
openavatar render runs/batch_c_edge --backend local --device mps
python scripts/corrupt_output.py runs/batch_c_edge
openavatar validate runs/batch_c_edge          # exits 2, logs CORRUPT_IMAGE

# Benchmark
openavatar bench runs/batch_a_local --repetitions 3 --warmup 1 --limit 2
```

## Consent and privacy controls

Individual-avatar work (`intent: "individual"`) is **refused** unless a live
consent record covers every reference image:

```bash
openavatar consent add subj-01 --subject-ref "subject-A" \
  --scope "PS02 assessment demo only" --image consent/reference_images/r1.jpg
openavatar consent list
openavatar consent revoke subj-01     # subsequent jobs are refused
openavatar consent purge  subj-01     # deletes the images and the record
```

The registry stores **SHA-256 hashes only** — never image bytes, never a legal
name. Swapping a reference file after consent was granted invalidates the record
(`CONSENT_INVALID`). Reference images are git-ignored and never committed.

All generated images are labelled as synthetic in `avatar_manifest.json`
(`synthetic_media: true` plus a human-readable `content_label`).

## Safety decisions

| Decision | Meaning | Effect |
|---|---|---|
| `ALLOW` | passes every rule | job prepared |
| `NEEDS_CLARIFICATION` | ambiguous/subjective spec, or references without `intent="individual"` | held in `held/`, not executed |
| `REFUSE` | minors, sexual content, gore, impersonation/deepfake, real-person targeting, ethnic essentialism, biometric inference, missing/revoked consent | never prepared |

Every decision is appended to `safety_log.jsonl` with stable reason codes.

## Testing

```bash
pytest -q                                   # 67 offline tests
OPENAVATAR_RUN_SLOW=1 pytest -m slow        # real diffusion, needs cached weights
```

Unit (spec/prompt/safety) · integration (plan/validate/ingest/resume) ·
end-to-end (full pipeline offline + CLI subprocess incl. the exit-2 contract).

## Licensing summary

See `SOURCES.md` for the full manifest. Short version: the **commercially
reusable path is `sd15` / `sd15_lcm`** (CreativeML OpenRAIL-M + openrail++,
commercial use permitted subject to use restrictions). **`sdturbo` is
research-only** (Stability AI Non-Commercial Research Community License) and is
included solely as a second-model comparison — it must not ship in a product.

## Disclosure

- Free compute: Kaggle Notebooks free GPU tier. No payment, no card, no paid API.
- Google Colab is deliberately **not** used (managed-runtime terms prohibit
  deepfake creation and this project has an individual-avatar capability).
- AI coding assistance: see `AI_USE.md`.
