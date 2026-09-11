# Technical report — PS02 / Track 02: Open-Source AI Human-Avatar Generation

> Numbers in this report are filled from the committed run artefacts. Every
> figure is traceable through `EVIDENCE_INDEX.md` to the JSON file it came from.

## 1. Problem and approach

Generate human avatars from a structured specification using open-weight models,
support inclusive appearance controls without encoding nationality stereotypes,
and keep everything except image inference runnable on a non-GPU laptop.

The shape of the solution follows from one observation: **the interesting
engineering is not the diffusion call.** Calling `pipe(prompt=...)` is four
lines. The work is everything around it — turning a structured spec into a
defensible prompt, deciding what to refuse, keeping provenance that survives a
round trip through someone else's GPU, and proving the output is what was asked
for. So the architecture puts the model at the edge (a swappable backend) and
everything else in the middle, locally.

## 2. Architecture

See README for the pipeline diagram. Four boundaries carry the design:

**Spec → Prompt.** `AvatarSpec` is a closed schema (`extra="forbid"`, `Literal`
vocabularies, cross-field coherence rules). Prompt construction is a pure
function of `appearance` + `scene`. This is what makes the anti-stereotyping
property *testable* rather than aspirational — see §4.

**Screen → Job.** `safety.screen()` returns one of three decisions, and only
`ALLOW` produces a `Job`. Refused and ambiguous specs are written to `held/`
with the verdict attached, so the evidence of a refusal is as durable as the
evidence of a success.

**Job → Bundle.** A `Job` is fully resolved: prompt, negative prompt, seed,
steps, guidance, and `repo_id` + a 40-character revision SHA. Nothing is looked
up at execution time. This is what lets the same bundle run locally and on a
Kaggle GPU and be compared meaningfully.

**Result → Verdict.** `validate` does not trust `results.json`. It re-opens
every PNG, re-hashes it, and compares decoded dimensions against the requesting
job, because images may have crossed a network.

### Backend abstraction

`Backend` is a `Protocol` with one method. Three consequences:
- `local` (diffusers, CPU/MPS) and `stub` (deterministic, offline) are
  interchangeable, so the **entire pipeline is testable without weights** — the
  64-test suite runs in ~6 s with no network.
- The notebook route is not a special case; it produces the same `JobResult`
  records and enters through `ingest`.
- Adding a third backend is one class.

## 3. Execution routes and the hybrid constraint

| | Local (M1, MPS) | Kaggle free GPU |
|---|---|---|
| Role | primary, quota-free | scale + required notebook evidence |
| dtype | float16 on MPS, float32 on CPU | float16 |
| Product logic | all of it | none — thin executor |

The dtype split is a real engineering decision: SD 1.5's UNet is ~3.4 GB in
float32, which does not leave room for the text encoder and activations on an
8 GB machine. float16 on MPS halves residency; CPU stays float32 because
float16 matmuls are not accelerated there. The effective dtype is recorded on
every `JobResult` and surfaces in `avatar_manifest.json`, so a benchmark number
can never be quoted without the precision it was measured at.

Attention slicing and VAE slicing are enabled unconditionally on the local
backend, per the brief's guidance to prefer memory-efficient attention and scale
only after the pipeline is stable.

**Colab is excluded on policy grounds**, not convenience: its managed-runtime
terms prohibit deepfake creation and this project ships an individual-avatar
capability.

## 4. Inclusive appearance controls without stereotyping

The requirement is to "separate appearance attributes from nationality labels".
The implementation makes that separation structural rather than a convention:

- Appearance is described with neutral traits: Monk Skin Tone ids (`mst-1..10`),
  hair length/texture/colour, attire, age band, presentation.
- `region_context` exists, but only as **coverage bookkeeping**. It records
  which geographic/cultural slot a spec fills so `metrics.json` can report
  coverage.
- `prompt.build_prompt()` reads `appearance` and `scene`. It has no access path
  to `region_context` or `notes`.
- `tests/test_prompt.py::test_region_context_never_reaches_the_prompt` asserts
  that adding a region label yields a **byte-identical** prompt.

The attribute combinations for the six coverage contexts were chosen by hand, so
the system never learns or asserts "nationality ⇒ appearance". It cannot: the
label is not in the generative path.

The residual risk is the base model's own priors, which this codebase cannot fix
and does not claim to. That is precisely what dimension 3 of
`docs/HUMAN_REVIEW_PROTOCOL.md` exists to surface.

## 5. Safety and consent

| Decision | Trigger | Effect |
|---|---|---|
| `REFUSE` | minors, sexual content, gore, deepfake/impersonation, real-person targeting, ethnic essentialism ("a typical X face"), biometric inference, missing/revoked/mismatched consent | no job created |
| `NEEDS_CLARIFICATION` | subjective terms ("professional-looking", "attractive"), references without `intent="individual"` | held, not executed |
| `ALLOW` | everything else | job created |

Refusal precedence is deliberate: a spec that is both unsafe and ambiguous is
refused, never merely queried.

**Consent.** `intent="individual"` requires a consent record that exists, is not
revoked, and whose SHA-256 hashes still match the files on disk. Swapping a
reference image after consent invalidates it. The registry holds hashes and a
pseudonymous label only — no image bytes, no legal name — so the registry itself
is not a biometric store. `consent purge` deletes the reference files and the
record (right-to-erasure). `consent/reference_images/` is git-ignored.

**Synthetic-media labelling.** Every manifest entry carries
`synthetic_media: true` and a human-readable `content_label`.

## 6. Evaluation

| Metric | Model | Why this model |
|---|---|---|
| Prompt/spec adherence (CLIPScore) | CLIP ViT-B/32 | standard, cheap on CPU |
| Per-attribute adherence | CLIP ViT-B/32 | discriminative probe, see below |
| Identity consistency | **DINOv2-small** | independent of CLIP *and* of the generator |
| Diversity | DINOv2-small | mean pairwise embedding distance |
| Coverage | — | literal counts over the requested matrix |

**Why a discriminative probe.** Raw CLIPScore against a 40-word prompt is a
blunt instrument: it moves with photographic style as much as with the requested
attributes. So for each controlled attribute the evaluator builds a *pair* —
the phrase that was requested and a contrasting phrase from the same vocabulary
— and asks CLIP which it prefers. `attribute_accuracy` is the hit rate. This
turns "is the score high?" into "did the model actually produce the requested
attire rather than some other attire?", which is the question the spec asks.

**Why DINOv2 rather than CLIP for identity.** CLIP already grades adherence.
Using it for identity too would let one model both describe and score the
output. DINOv2 is self-supervised with a different objective and is not in the
generation stack, so it is an genuinely independent embedding view.

**Limitations, stated plainly.** DINOv2 is a general image backbone, not a face
recognition model: its cosine similarity is influenced by pose, crop and
background as well as by facial identity. It is adequate for *relative*
comparison within a fixed pose/background (which is how batch B uses it) and is
not a biometric identity claim. A dedicated face-embedding model would be the
right choice for the Exceptional-level individual-avatar work.

## 7. Results

All figures below were read out of `evidence/*/metrics.json`,
`evidence/*/benchmark.json` and `evidence/route_comparison.json`.
`EVIDENCE_INDEX.md` maps each one to its file.

### Compute routes

| | Local | Free accelerator |
|---|---|---|
| Hardware | Apple M1, 8 GB, 8 logical cores | Kaggle Tesla T4 (free tier) |
| Runtime | torch 2.8.0, MPS, float16, Python 3.11 | torch 2.10.0+cu128, CUDA, float16, Python 3.12 |
| Median time/image (384², 20 steps) | **18.53 s** | **2.21 s** |
| Pipeline load | 15.25 s | once per checkpoint |
| Peak process RSS | **1662.8 MB** | n/a |
| Peak accelerator memory | not measurable on MPS | **2520 MB** (sd15) |
| Loaded weights | 4265.1 MB | same pinned revision |

### Batch results

| Batch | Route | Model | Jobs | Pass | CLIPScore | Attribute accuracy | Diversity |
|---|---|---|---|---|---|---|---|
| `batch_a_local` | M1 MPS | sd15 | 6 | 6/6 | 28.83 | 0.580 | 0.565 |
| `batch_a_sd15` | T4 | sd15 | 6 | 6/6 | 28.84 | 0.580 | 0.565 |
| `batch_a_sdturbo` | T4 | sd-turbo | 6 | 6/6 | **31.15** | **0.673** | 0.661 |
| `batch_b_sd15` | T4 | sd15 | 5 | 5/5 | 27.44 | 0.639 | 0.155 |
| `batch_b_lcm` | T4 | sd15 + LCM-LoRA | 5 | 5/5 | 27.39 | 0.586 | 0.171 |
| `batch_c_edge` | T4 | sd15 | 1 | 0/1 | — | — | — |

Job success rate is 23/24 across all batches. The single failure is deliberate:
`scripts/corrupt_output.py` truncates one output so the corrupted-output case has
real evidence, and `validate` exits 2 on it.

### Cross-route reproducibility

The same bundle, same seeds, executed on M1 MPS and on a Tesla T4:

| Measure | Value |
|---|---|
| Mean DINOv2 cosine between routes | **0.9995** |
| Minimum DINOv2 cosine | 0.9978 |
| Mean absolute CLIPScore difference | 0.083 |
| Byte-identical files | **No** |

Bit-identical output across CUDA and MPS is not achievable — the two backends
reduce floating point in a different order. Rather than claim reproducibility we
cannot demonstrate, the claim made here is *metric and perceptual equivalence at
a fixed seed*, and `scripts/compare_routes.py` measures it. A cosine of 0.9995 on
an independent embedding model means the two routes produce the same picture.

### Coverage (Strong requirement)

`batch_a_sd15` spans six coverage contexts, each with a distinct neutral
attribute combination — and, per §4, the context label itself never enters a
prompt:

- Contexts: west-africa, south-asia, east-asia, northern-europe, latin-america, MENA — 1 each
- Skin tone: mst-2, mst-4, mst-5, mst-6, mst-7, mst-9 — 6 distinct points on a 10-point scale
- Age band: young-adult, adult ×3, middle-aged, senior
- Also varied: presentation, hair length/texture, attire, background, pose

### Controlled single-attribute changes

Five renders share `identity_id` and seed; each variant changes exactly one
attribute. Per-attribute probe hit rates:

| Attribute | sd15 | sd15 + LCM |
|---|---|---|
| age_band | 5/5 | 5/5 |
| skin_tone | 5/5 | 5/5 |
| hair_length | 5/5 | 5/5 |
| attire | 5/5 | 4/5 |
| eyewear | 1/1 | 0/1 |
| background | 2/5 | 2/5 |
| pose | **0/5** | **0/5** |
| expression | **0/5** | **0/5** |

Identity consistency within the group: **0.845** mean pairwise DINOv2 cosine for
sd15 (min 0.699), **0.829** for LCM (min 0.733) — the renders hold the same
person while one attribute moves.

**The `pose` and `expression` rows are the honest finding here.** Zero hits out of
five, on both models, is not a model failure — it is a *metric* failure. CLIP
ViT-B/32 at 384² cannot discriminate "front-facing headshot" from "three-quarter
headshot", or "neutral expression" from "a slight closed-mouth smile"; the
contrast phrases are too close in its text embedding space. The right reading is
that the probe is valid for coarse attributes (age, skin tone, hair, attire) and
uninformative for fine ones. Reporting a 0.580 mean adherence without saying that
two of the eight attributes are being measured by a broken ruler would be
misleading. Fixing it needs either a higher-resolution CLIP variant or a
pose/landmark estimator, which is listed in §12.

### Model comparison

sd-turbo scores highest on both adherence (31.15 vs 28.84) and diversity (0.661
vs 0.565), and is ~5× faster (0.45 s vs 2.21 s per image). **It is also the one
model here that cannot ship** — Stability AI Non-Commercial Research Community
License. The commercially usable path is sd15, which costs quality and speed.
That trade-off is the product decision, and it is why the licence status is
carried in `configs/default.json` next to the weights rather than only in prose.

LCM-LoRA cuts sd15 to 6 steps and is ~3× faster per image on the T4 (0.76 s vs
2.21 s) at roughly equal CLIPScore (27.39 vs 27.44), but attribute accuracy drops
(0.586 vs 0.639), concentrated in `attire` and `eyewear`. Few-step distillation
is worth it for volume, not for fine attribute control.

## 8. Failures and what they taught

Every failure below is preserved in the repository with the artefact that proves
it. None was discovered by reading code — each came from a run.

### 1. Silent all-black output on Apple MPS (the expensive one)

The first real local render "succeeded" in 63.6 s and wrote a PNG that was pure
black. The pipeline raised nothing; the only hint was a numpy
`invalid value encountered in cast` warning.

Diagnosis: SD 1.5 in float16 on MPS emits **NaN in the UNet latents**. The usual
suspect is the VAE decoder, which has a well-known fp16 overflow and an easy fix
(upcast the VAE). That fix was tried and failed — the UNet hands the VAE fp16
latents, so an fp32 VAE just raises a dtype error. Probing the latents directly
settled it: `torch.isnan(latents).any()` was already `True` before the VAE ran.

The actual cause was narrower and more surprising — **attention slicing**:

| Configuration | Time | Latent NaN | Result |
|---|---|---|---|
| fp16 + attention slicing | 8.1 s | **True** | black image |
| fp16, no attention slicing | 5.9 s | False | valid image |

Attention slicing is the standard memory-efficiency step, and **the assessment
brief explicitly recommends it** ("use low resolution, few steps and
memory-efficient attention first"). On this hardware it is strictly worse on
every axis: slower *and* silently wrong. So the backend disables attention
slicing specifically on MPS+fp16 and keeps it everywhere else, with the
measurement recorded in a comment next to the branch. `attention_slicing` is now
written into every `JobResult` and into the manifest, because a benchmark number
is meaningless without it.

Evidence: `evidence/failures/fp16_mps_nan_black_output.png`,
`evidence/failures/fp16_mps_nan_validation.json`.

### 2. The validator caught it, then hid what it caught

The black frames *were* rejected — but the report said only `EMPTY_FILE`, never
`BLANK_IMAGE`. An all-black PNG compresses to under 1 KB, tripping the byte-size
floor, and `validate_item` returned early on that check. The code that names the
actual defect never ran.

The check was right and the control flow was wrong. `EMPTY_FILE` is now recorded
without short-circuiting, so a tiny blank frame reports **both** codes and the
diagnosis survives. Regression test:
`tests/test_jobs_and_validate.py::test_tiny_blank_image_reports_both_codes`.

This is the failure worth the most: a validator that rejects the right files for
the wrong stated reason is more dangerous than one that misses them, because the
failure looks handled.

### 3. Two misleading benchmark numbers, self-inflicted

Both found by reading output that looked implausible rather than by a test.

- **Model size reported 23.6 GB.** `_model_size_mb` summed every `*.safetensors`
  in the repo. The SD 1.5 repo ships single-file checkpoints
  (`v1-5-pruned.safetensors`, 7.7 GB), a non-EMA UNet, fp16 duplicates of every
  shard and a safety checker this pipeline disables — roughly 19 GB that is never
  read. The real figure is **4265.1 MB**. Fixed by measuring exactly the three
  files the pipeline loads, from a single canonical list
  (`backends.PIPELINE_WEIGHT_FILES`) shared with the download script so the two
  cannot drift.
- **Process RSS reported 36 MB** while holding a 4 GB model. Not a bug in psutil:
  on MPS the weights are handed to Metal and live in unified memory that is not
  attributed to process RSS. Resident memory genuinely drops after load. The
  benchmark now reports **peak** RSS from `getrusage` (**1662.8 MB**) as the
  headline, keeps current RSS beside it, and carries a `memory_note` explaining
  why they differ and that neither is a GPU-memory measurement.

A misleading benchmark is an explicit eligibility gate in this assessment. Both
numbers would have passed review unchallenged, which is precisely the problem.

### 4. Disk and download waste

The first `download_models.py` allow-list pulled ~12 GB of single-file
checkpoints `from_pretrained` never reads. An attempted cleanup then deleted blob
files that were symlink targets, corrupting the cache and forcing a full
re-download. Fixed with exact per-file allow-listing; a clean setup now fetches
~4.3 GB.

### 5. Honest limitations that are not bugs

- `pose` and `expression` adherence is unmeasurable with CLIP ViT-B/32 at this
  resolution (0/5 on both models) — see §7.
- DINOv2 is a general image backbone, not a face recogniser. Its cosine is
  influenced by pose, crop and background as well as identity. Valid for relative
  comparison at fixed pose; **not** a biometric claim.
- Human review was performed by a single non-independent reviewer (the
  candidate). The protocol in `docs/HUMAN_REVIEW_PROTOCOL.md` states the required
  fix: two reviewers from different backgrounds plus an inter-rater agreement
  check.
- The safety layer screens *intent* at the prompt level, before compute is spent.
  It does not inspect pixels. A production system needs both.

## 9. Alternatives considered

| Decision | Alternative | Why rejected |
|---|---|---|
| SD 1.5 as primary | SDXL / SDXL-Turbo | 6.9 GB UNet does not fit the 8 GB laptop budget; SDXL-Turbo is non-commercial |
| SD-Turbo as comparison | a second commercially-licensed model | few-step distillation is the more informative comparison axis, and flagging a research-only model *as* research-only is itself part of the licensing deliverable |
| DINOv2 for identity | ArcFace / InsightFace | genuinely better for faces, but adds a heavy non-pip-clean dependency; documented as the right upgrade for Exceptional-level work |
| Pydantic closed schema | free-text prompts | free text makes the anti-stereotyping property untestable |
| JSON config | YAML | one fewer dependency; the config is machine-written as often as hand-written |
| `region_context` retained for bookkeeping | drop nationality entirely | coverage reporting is a stated requirement; the fix is to keep the label out of the *prompt*, not out of the *records* |

## 10. Security and privacy

- `ingest` rejects zip path traversal, foreign `batch_id`s, unknown `job_id`s,
  and any record claiming success without a matching image.
- No secrets, tokens or credentials in the repository; no provider account
  identifiers in committed logs.
- The diffusers built-in safety checker is disabled and replaced with a
  prompt-level safety layer that runs *before* compute is spent, and whose
  decisions are logged with stable codes. Trade-off: this screens intent, not
  pixels. A production system would want both.
- Consent data is hash-only. Reference images are never committed.

## 11. Product recommendation

Ship the `sd15` + `sd15_lcm` path (OpenRAIL-M / openrail++, commercial use
permitted subject to restrictions the safety layer already enforces). Keep
`sdturbo` out of any product build — it is non-commercial and is present only as
a comparison. Before launch: mirror the OpenRAIL use restrictions into customer
terms, obtain a jurisdiction-specific opinion on LAION-trained weights, set a
retention schedule to go with the existing deletion control, and embed C2PA
content credentials in the image files rather than only in the manifest.

## 12. Next steps

1. Swap DINOv2 for a face-embedding model for individual-avatar identity scoring.
2. IP-Adapter / reference-conditioned generation to complete the Exceptional level.
3. C2PA content credentials embedded at save time.
4. Two independent human reviewers with an inter-rater agreement check.
5. A held-out prompt-injection suite for the free-text `extra_prompt` field.
