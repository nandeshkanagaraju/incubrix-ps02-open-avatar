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

<!-- RESULTS -->

## 8. Failures and what they taught

<!-- FAILURES -->

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
