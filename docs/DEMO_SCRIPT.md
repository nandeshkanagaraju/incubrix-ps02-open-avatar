# Demo video script (unedited, ≤ 8 minutes)

The brief requires: a clean run, output, a failure and recovery, tests, and one
code-level decision. Record in one take with no cuts. Below is a timed running
order that fits in eight minutes.

Before recording: `git status` should be clean, the terminal font large enough to
read, and `runs/` already populated (the demo re-runs a *small* slice, not the
whole batch — say so out loud).

---

## 0:00 – 0:30 — What this is

> "PS02 Track 02. This is `openavatar`, a local-first orchestrator for
> open-weight avatar generation. Everything except the diffusion call runs on
> this 8 GB M1 with no discrete GPU: spec validation, safety screening, job
> preparation, output validation, provenance, evaluation and tests. Image
> inference runs either locally on MPS or on Kaggle's free GPU tier, both
> disclosed. No payment, no card, no commercial API."

## 0:30 – 1:30 — Clean run: plan

```bash
cat specs/batch_a_fictional/a1_studio_portrait.json
.venv/bin/openavatar plan specs/batch_a_fictional --batch-id demo_a -m sd15
```

Point at the decision table, then:

```bash
python -c "import json;b=json.load(open('runs/demo_a/bundle.json'));print(json.dumps(b['jobs'][0],indent=2))"
```

> "The spec is structured, not a prompt string. Planning resolves it into a job
> that pins the prompt, the seed, and the model repo plus a 40-character commit
> SHA. Nothing is looked up at execution time — that's what makes the same
> bundle reproducible here and on Kaggle."

## 1:30 – 2:30 — Safety: the refusal path

```bash
.venv/bin/openavatar plan specs/batch_c_edge --batch-id demo_c
cat runs/demo_c/held/c2_disallowed.json
cat runs/demo_c/safety_log.jsonl
```

> "Three specs in, one job out. The disallowed one is refused with stable reason
> codes — `IMPERSONATION`, `REAL_PERSON_TARGETING` — before any compute is spent.
> The ambiguous one is held for clarification, not silently guessed at. Both are
> written to `held/` with the verdict attached, so a refusal is as durable a
> piece of evidence as a success."

## 2:30 – 3:30 — The code-level decision *(this is the required one)*

Open `src/openavatar/prompt.py` and `src/openavatar/spec.py` side by side.

> "The Strong requirement is to separate appearance attributes from nationality
> labels. I made that structural rather than a convention. `region_context` on
> the spec records which coverage slot a spec fills, so I can report geographic
> coverage. But `build_prompt` only reads `appearance` and `scene` — there is no
> access path from a region label to a prompt."

```bash
.venv/bin/python -m pytest tests/test_prompt.py::test_region_context_never_reaches_the_prompt -v
```

> "The test asserts the prompt is *byte-identical* with and without the region
> label. That's what turns 'we don't stereotype' from a claim into something an
> evaluator can check. What it does not fix is the base model's own priors —
> that's what dimension 3 of the human review protocol is for."

## 3:30 – 4:30 — Output and provenance

```bash
.venv/bin/openavatar validate runs/batch_a_sd15
python -c "import json;m=json.load(open('runs/batch_a_sd15/avatar_manifest.json'));print(json.dumps(m['avatars'][0],indent=2)[:1500])"
```

Open two or three PNGs.

> "Validation re-opens every file, re-hashes it and checks the decoded size
> against the job that asked for it — it does not trust the executor's own
> report, because these images crossed a network. The manifest carries the spec,
> seed, model revision, safety verdict, compute route and a synthetic-media
> label for every image."

## 4:30 – 5:45 — Failure and recovery *(required)*

Use the real one:

```bash
open evidence/failures/fp16_mps_nan_black_output.png
```

> "First real local run gave me this — pure black. SD 1.5 in fp16 on Apple MPS
> emits NaN in the UNet latents. I checked whether it was the VAE, which is the
> usual culprit and has an easy fix —"

```bash
.venv/bin/python /tmp/mps_probe.py   # or show the recorded probe output
```

> "— it isn't. The latents are already NaN before the VAE sees them, so upcasting
> the VAE can't help. The fix was the dtype default. But the more interesting
> part is what it exposed in my own code:"

```bash
git log --oneline -- src/openavatar/validate.py
git show <commit> -- src/openavatar/validate.py | head -40
```

> "The validator *caught* the black frames, but reported only `EMPTY_FILE` —
> because an all-black PNG compresses below the 1 KB floor, and I was returning
> early on that check. It was hiding `BLANK_IMAGE`, which is the code that
> actually names the defect. I stopped short-circuiting, so both fire, and added
> a regression test."

```bash
.venv/bin/python -m pytest tests/test_jobs_and_validate.py::test_tiny_blank_image_reports_both_codes -v
```

Then the injected corruption:

```bash
python scripts/corrupt_output.py runs/batch_c_edge
.venv/bin/openavatar validate runs/batch_c_edge; echo "exit=$?"
```

> "Exit code 2. A corrupted output fails the batch rather than passing quietly."

## 5:45 – 6:45 — Tests and benchmark

```bash
.venv/bin/python -m pytest -q
```

> "65 tests, about six seconds, and — this is the part I care about — no model
> weights and no network. The backend is a Protocol with one method, so a
> deterministic stub stands in for diffusers and the entire pipeline is testable
> in CI."

```bash
cat runs/batch_a_sd15/benchmark.json | python -m json.tool | head -30
```

> "The timing boundary is the generate call only; weight loading is measured
> separately, because a batch pays it once. Peak accelerator memory is null
> locally and populated on the Kaggle route — I'd rather report null than a
> number I can't measure."

## 6:45 – 7:30 — Evaluation

```bash
.venv/bin/openavatar evaluate runs/batch_b_sd15
```

> "Adherence uses CLIP, but not as a raw score — for each controlled attribute it
> builds a pair, the requested phrase against a contrasting one from the same
> vocabulary, and asks which CLIP prefers. Identity uses DINOv2, deliberately a
> different family, so the model grading identity isn't the one describing the
> image. DINOv2 is a general backbone, not a face model, so this is a relative
> comparison within a fixed pose — not a biometric claim."

## 7:30 – 8:00 — Close

> "Two compute routes, both disclosed, no spend. The commercially reusable path
> is SD 1.5 plus LCM-LoRA; sd-turbo is in there as a second-model comparison and
> is non-commercial, which is flagged in the config so the restriction travels
> with it. AI_USE.md discloses that I used Claude Code as a primary implementation
> author, and what it got wrong along the way."

---

## Recording checklist

- [ ] One take, no cuts, no edits
- [ ] Terminal readable at the recorded resolution
- [ ] No tokens, credentials or account identifiers on screen
- [ ] Under 8:00
- [ ] Say out loud when something was pre-computed rather than run live
