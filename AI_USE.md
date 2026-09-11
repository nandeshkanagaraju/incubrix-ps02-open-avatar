# AI_USE.md — disclosure of AI assistance

Required by the assessment: *"AI coding assistants may be used only if fully
disclosed in AI_USE.md. The candidate remains responsible for correctness and
must be able to modify the implementation live."*

## Tool used

| Tool | Model | Where |
|---|---|---|
| Claude Code (CLI) | Claude Opus 5 | Interactive terminal session on the candidate's laptop, 11 September 2026 |

No other AI tool, autocomplete assistant or code-generation service was used.

## Scope — what the assistant produced

This is a full and deliberately unflattering disclosure. The assistant was used
as a **primary implementation author**, not merely as autocomplete. It drafted
initial versions of:

- `src/openavatar/` — all ten modules (`spec`, `attributes`, `prompt`, `safety`,
  `config`, `jobs`, `backends`, `render`, `validate`, `metrics`, `bench`, `cli`)
- `tests/` — all 64 tests
- `scripts/` — `download_models.py`, `make_specs.py`, `build_notebook.py`,
  `corrupt_output.py`
- `notebooks/kaggle_executor.ipynb` (via the generator script)
- `README.md`, `SOURCES.md`, this file, and `docs/REPORT.md`

## Direction given by the candidate

The candidate set the scope and the design constraints that shaped the result:

- Target Baseline + Strong; build the consent machinery regardless so the
  refusal-without-consent path is real and tested.
- Hybrid execution: local Apple-MPS route **and** a Kaggle notebook route, so
  the submission does not depend on accelerator quota being available.
- Keep the notebook a thin executor; the architecture stays local.
- Keep the implementation lean — meet the stated requirements, do not gold-plate.

## Material corrections made during the session

Recording these because they are the honest record of what was wrong and how it
was caught:

1. **Fabricated model revisions.** The assistant's first `configs/default.json`
   contained invented commit SHAs. They were replaced by querying
   `HfApi().model_info()` for the real revision of each repo before any run.
   Nothing in the final config is guessed.
2. **Broken model download.** The first `download_models.py` allow-list pulled
   the ~12 GB single-file checkpoints that `from_pretrained` never reads, then an
   attempted cleanup deleted blobs that were symlink targets and corrupted the
   cache. The cache was wiped and re-fetched with an exact per-subfolder
   allow-list.
3. **Two failing tests on the first run of the validator suite.** The stub
   backend emitted a pure gradient that compressed below the 1 KB `EMPTY_FILE`
   floor, so truncating it tripped the byte check instead of the decode check the
   tests were targeting. Fixed at the source — the stub now emits deterministic
   LCG noise so its PNGs are realistically sized — rather than by loosening the
   assertion.

## Verification performed

- `pytest -q` — 64 tests pass offline with no weights and no network.
- Every generated image in `evidence/` came from an actual local execution of
  this code; none is illustrative or hand-made.
- All metrics in `docs/REPORT.md` and the workbook were produced by
  `openavatar evaluate` / `openavatar bench` and are reproducible from the
  committed specs and seeds.
- Safety refusal behaviour was verified against the committed
  `specs/batch_c_edge` batch, not only in unit tests.

## Candidate responsibility

The candidate is responsible for the correctness of everything submitted and for
being able to explain, trace and modify any part of it during live validation.
AI assistance does not transfer that responsibility.
