"""openavatar command line interface.

Documented contract (also in README.md):

    plan      spec-dir            -> runs/<batch>/bundle.json, safety_log.jsonl, held/
    export    bundle              -> <bundle>.zip  (portable job bundle for a notebook)
    render    bundle              -> images/*.png, results.json
    ingest    bundle + archive    -> images/*.png, results.json   (notebook route)
    validate  bundle              -> validation.json, avatar_manifest.json
    evaluate  bundle              -> metrics.json
    bench     bundle              -> benchmark.json
    consent   add|revoke|purge|list
    report    bundle              -> human-readable summary on stdout

Exit codes: 0 success, 1 usage/config error, 2 validation failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, ConfigError
from .logging_utils import configure, get_logger

app = typer.Typer(add_completion=False, help="Open-source human-avatar generation orchestrator.")
consent_app = typer.Typer(help="Manage consent records for individual-avatar work.")
app.add_typer(consent_app, name="consent")

console = Console()
log = get_logger("cli")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to a config JSON file.")
BundleArg = typer.Argument(..., help="Path to a bundle directory (runs/<batch_id>).")


def _cfg(path: str | None) -> Config:
    try:
        return Config.load(path)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    configure(verbose=verbose)


@app.command()
def version():
    """Print the tool version."""
    console.print(f"openavatar {__version__}")


@app.command()
def plan(
    spec_dir: Path = typer.Argument(..., help="Directory of *.json avatar specs."),
    batch_id: str = typer.Option(..., "--batch-id", "-b", help="Identifier for this batch."),
    out: Path = typer.Option(None, "--out", "-o", help="Bundle directory (default runs/<batch_id>)."),
    model_key: str = typer.Option(None, "--model-key", "-m", help="Override every spec's model."),
    config: str = ConfigOpt,
):
    """Screen specs for safety/consent and build a portable job bundle."""
    from .jobs import plan_batch

    cfg = _cfg(config)
    out_dir = Path(out) if out else cfg.resolve("output_root") / batch_id
    configure(log_file=out_dir / "run.log")
    if model_key and model_key not in cfg.model_keys:
        console.print(f"[red]unknown --model-key {model_key!r}[/red]; configured: {cfg.model_keys}")
        raise typer.Exit(1)
    try:
        bundle, results = plan_batch(spec_dir, out_dir, cfg, batch_id, model_key)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        console.print(f"[red]plan failed:[/red] {exc}")
        raise typer.Exit(1)

    t = Table(title=f"plan: {batch_id}")
    t.add_column("spec_id"); t.add_column("decision"); t.add_column("codes")
    for r in results:
        colour = {"ALLOW": "green", "NEEDS_CLARIFICATION": "yellow", "REFUSE": "red"}[r.decision]
        t.add_row(r.spec_id, f"[{colour}]{r.decision}[/{colour}]",
                  ", ".join(f.code for f in r.findings) or "-")
    console.print(t)
    console.print(f"{len(bundle.jobs)} job(s) prepared, {len(bundle.held)} held -> {out_dir}")


@app.command()
def export(bundle: Path = BundleArg, out: Path = typer.Option(None, "--out", "-o")):
    """Zip a bundle's inputs for transport to a free-accelerator notebook."""
    from .jobs import zip_bundle

    target = Path(out) if out else bundle.with_suffix(".zip")
    p = zip_bundle(bundle, target)
    console.print(f"exported -> {p}")


@app.command()
def render(
    bundle: Path = BundleArg,
    backend: str = typer.Option("local", "--backend", help="'local' (diffusers) or 'stub' (offline)."),
    device: str = typer.Option("auto", "--device", help="auto|cpu|mps"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Re-render jobs that already succeeded."),
    only: list[str] = typer.Option(None, "--only", help="Render only these job_ids."),
    config: str = ConfigOpt,
):
    """Execute a bundle locally (CPU or Apple MPS)."""
    from .backends import get_backend
    from .render import render_bundle

    _cfg(config)
    configure(log_file=bundle / "run.log")
    try:
        be = get_backend(backend, device=device)
        results = render_bundle(bundle, be, resume=not no_resume, only=list(only) if only else None)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"[red]render failed:[/red] {exc}")
        raise typer.Exit(1)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    console.print(f"rendered {ok}/{len(results)} job(s) ok -> {bundle/'results.json'}")


@app.command()
def ingest(
    bundle: Path = BundleArg,
    archive: Path = typer.Argument(..., help="results .zip or directory from the notebook run."),
):
    """Ingest results produced on a free accelerator."""
    from .render import ingest as do_ingest

    configure(log_file=bundle / "run.log")
    try:
        results = do_ingest(bundle, archive)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]ingest failed:[/red] {exc}")
        raise typer.Exit(1)
    console.print(f"ingested {len(results)} record(s)")


@app.command()
def validate(bundle: Path = BundleArg, config: str = ConfigOpt):
    """Validate returned outputs and write the provenance manifest."""
    from .validate import build_manifest, validate_bundle

    cfg = _cfg(config)
    configure(log_file=bundle / "run.log")
    try:
        report = validate_bundle(bundle, cfg)
        build_manifest(bundle, cfg)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]validate failed:[/red] {exc}")
        raise typer.Exit(1)

    t = Table(title=f"validation: {report.batch_id}")
    t.add_column("job_id"); t.add_column("ok"); t.add_column("codes"); t.add_column("size")
    for i in report.items:
        d = i.details
        size = f"{d.get('width','?')}x{d.get('height','?')}"
        t.add_row(i.job_id, "[green]yes[/green]" if i.ok else "[red]no[/red]",
                  ", ".join(i.codes) or "-", size)
    console.print(t)
    console.print(f"passed {report.passed}/{report.total_jobs} (success rate {report.success_rate})")
    console.print(f"manifest -> {bundle/'avatar_manifest.json'}")
    if report.failed:
        raise typer.Exit(2)


@app.command()
def evaluate(bundle: Path = BundleArg, config: str = ConfigOpt):
    """Score adherence, identity consistency and diversity/coverage."""
    from .metrics import evaluate_bundle

    cfg = _cfg(config)
    configure(log_file=bundle / "run.log")
    try:
        out = evaluate_bundle(bundle, cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]evaluate failed:[/red] {exc} (run `validate` first)")
        raise typer.Exit(1)

    t = Table(title=f"metrics: {out['batch_id']}")
    t.add_column("model"); t.add_column("n"); t.add_column("mean CLIPScore"); t.add_column("attr accuracy")
    for k, v in sorted(out["by_model"].items()):
        t.add_row(k, str(v["n"]), str(v["mean_clip_score"]), str(v["mean_attribute_accuracy"]))
    console.print(t)
    console.print(f"diversity (mean pairwise distance): {out['diversity'].get('mean_pairwise_distance')}")
    for ident, v in out["identity_consistency"].items():
        console.print(f"identity {ident}: n={v['n']} mean_cosine={v.get('mean_cosine')}")
    console.print(f"metrics -> {bundle/'metrics.json'}")


@app.command()
def bench(
    bundle: Path = BundleArg,
    backend: str = typer.Option("local", "--backend"),
    device: str = typer.Option("auto", "--device"),
    repetitions: int = typer.Option(3, "--repetitions", "-r"),
    warmup: int = typer.Option(1, "--warmup"),
    limit: int = typer.Option(2, "--limit", help="How many jobs to time."),
    config: str = ConfigOpt,
):
    """Run the automated benchmark."""
    from .bench import run_benchmark

    cfg = _cfg(config)
    configure(log_file=bundle / "run.log")
    try:
        rep = run_benchmark(bundle, cfg, backend, repetitions, warmup, limit, device)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]bench failed:[/red] {exc}")
        raise typer.Exit(1)
    s = rep["summary"]
    console.print(f"median time per image: {s['median_time_per_image_sec']}s "
                  f"(range {s['fastest_sec']}-{s['slowest_sec']}s over {s['jobs_timed']} job(s))")
    console.print(f"pipeline load: {rep['pipeline_load_sec']}s | "
                  f"peak process RSS {rep['peak_process_rss_mb']} MB "
                  f"(current {rep['current_process_rss_mb']} MB)")
    for key, m in sorted(rep["models"].items()):
        console.print(f"model {key}: {m['size_mb']} MB of loaded weights @ {m['revision'][:12]}")
    console.print(f"benchmark -> {bundle/'benchmark.json'}")


@app.command()
def report(bundle: Path = BundleArg):
    """Print a one-screen summary of a completed batch."""
    b = Path(bundle)
    for name in ("validation.json", "metrics.json", "benchmark.json"):
        p = b / name
        console.print(f"[bold]{name}[/bold]: {'present' if p.exists() else '[yellow]missing[/yellow]'}")
    if (b / "validation.json").exists():
        v = json.loads((b / "validation.json").read_text())
        console.print(f"  job success rate: {v['success_rate']} ({v['passed']}/{v['total_jobs']})")
    if (b / "metrics.json").exists():
        m = json.loads((b / "metrics.json").read_text())
        console.print(f"  models compared: {sorted(m['by_model'])}")
        console.print(f"  coverage/region_context: {m['coverage']['region_context']}")


# --- consent -----------------------------------------------------------------


@consent_app.command("add")
def consent_add(
    consent_id: str = typer.Argument(...),
    subject_ref: str = typer.Option(..., "--subject-ref", help="Pseudonymous label, not a legal name."),
    scope: str = typer.Option(..., "--scope", help="What the subject consented to."),
    images: list[Path] = typer.Option(..., "--image", help="Reference image path (repeatable)."),
    config: str = ConfigOpt,
):
    """Record consent for an identifiable person's reference images (hashes only)."""
    from .safety import ConsentError, ConsentRegistry

    cfg = _cfg(config)
    reg = ConsentRegistry(cfg.resolve("consent_registry"))
    try:
        rec = reg.add(consent_id, subject_ref, [str(p) for p in images], scope)
    except ConsentError as exc:
        console.print(f"[red]consent add failed:[/red] {exc}")
        raise typer.Exit(1)
    console.print(json.dumps(rec, indent=2))


@consent_app.command("revoke")
def consent_revoke(consent_id: str = typer.Argument(...), config: str = ConfigOpt):
    """Revoke a consent record. Subsequent jobs using it are refused."""
    from .safety import ConsentError, ConsentRegistry

    cfg = _cfg(config)
    try:
        rec = ConsentRegistry(cfg.resolve("consent_registry")).revoke(consent_id)
    except ConsentError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"revoked {consent_id} at {rec['revoked_at']}")


@consent_app.command("purge")
def consent_purge(consent_id: str = typer.Argument(...), config: str = ConfigOpt):
    """Delete the reference images and the record (right-to-erasure control)."""
    from .safety import ConsentError, ConsentRegistry

    cfg = _cfg(config)
    try:
        deleted = ConsentRegistry(cfg.resolve("consent_registry")).purge(
            consent_id, cfg.resolve("reference_dir")
        )
    except ConsentError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"purged {consent_id}; deleted {len(deleted)} file(s)")


@consent_app.command("list")
def consent_list(config: str = ConfigOpt):
    """List consent records (hashes only; no images, no names)."""
    from .safety import ConsentRegistry

    cfg = _cfg(config)
    reg = ConsentRegistry(cfg.resolve("consent_registry"))
    records = reg._data["records"]
    if not records:
        console.print("no consent records")
        return
    t = Table(title="consent records")
    t.add_column("consent_id"); t.add_column("subject_ref"); t.add_column("images"); t.add_column("revoked")
    for cid, r in sorted(records.items()):
        t.add_row(cid, r["subject_ref"], str(len(r["image_sha256"])), str(r["revoked"]))
    console.print(t)


def run() -> None:
    sys.exit(app())


if __name__ == "__main__":
    run()
