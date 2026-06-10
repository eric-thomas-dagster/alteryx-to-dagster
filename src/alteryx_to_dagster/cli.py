"""Standalone CLI: `alteryx-to-dagster import workflow.yxmd ...`.

Intentionally separate from the dagster-community-components CLI so this
migration tool can evolve at its own pace and ship to users who don't
need the broader registry.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import click

from . import __version__, import_workflow


@click.group(
    help="Convert Alteryx workflows (.yxmd / .yxmz / .yxzp) into Dagster projects.",
    invoke_without_command=False,
)
@click.version_option(__version__, "-V", "--version", prog_name="alteryx-to-dagster")
def main() -> None:
    pass


@main.command(
    "import",
    help=(
        "Convert one Alteryx workflow OR a folder of workflows into a Dagster project. "
        "If --out-dir doesn't exist it's scaffolded as a fresh Dagster project; if it "
        "doesn't have pandas/dagster-webserver/etc. they're added automatically."
    ),
)
@click.argument(
    "source",
    type=click.Path(exists=True),  # file OR directory
    metavar="WORKFLOW_OR_FOLDER",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=(
        "Where to write the Dagster project. Defaults to a directory next to the "
        "source — `./<workflow_stem>/` for a single file, `./<folder_basename>/` for "
        "a folder of workflows. Scaffolded as a fresh Dagster project if it doesn't "
        "already exist."
    ),
)
@click.option(
    "--pkg",
    default=None,
    help=(
        "Python package name under src/. Defaults to a sanitized form of --out-dir's "
        "basename (e.g. `--out-dir my-proj` → `--pkg my_proj`)."
    ),
)
@click.option(
    "--install/--no-install",
    "install",
    default=True,
    show_default=True,
    help=(
        "Run `dagster-component add <id> --auto-install` for each registry component "
        "used, and ensure pandas / numpy / dagster-webserver / dagster-dg-cli are in "
        "pyproject.toml. Use --no-install to skip (useful for read-only previews)."
    ),
)
@click.option(
    "--llm-translate",
    default=None,
    metavar="MODEL",
    help=(
        "Enable v1.5 LLM-assisted translation of Alteryx-only formula expressions "
        "(IIF / Contains / DateTimeAdd / etc.). Pass a LiteLLM model id "
        "(e.g. gpt-4o-mini, claude-haiku-4-5-20251001, gemini/gemini-2.5-flash). "
        "LLM is used at import time ONLY — the resulting Dagster project carries "
        "no LLM dependency at materialization time."
    ),
)
@click.option(
    "--llm-api-key-env",
    default=None,
    metavar="ENV_VAR",
    help="Env var holding the LLM API key (e.g. OPENAI_API_KEY).",
)
@click.option(
    "--llm-score-threshold",
    default=0.8,
    type=float,
    show_default=True,
    help="Translations with combined_score below this stay flagged in MIGRATION.md (not emitted).",
)
def import_cmd(
    source: str,
    out_dir: str | None,
    pkg: str | None,
    install: bool,
    llm_translate: str | None,
    llm_api_key_env: str | None,
    llm_score_threshold: float,
) -> None:
    source_path = Path(source)

    # File OR folder? Discover the .yxmd / .yxzp / .yxmz workflows to process.
    if source_path.is_dir():
        workflows = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in (".yxmd", ".yxmz", ".yxzp") and p.is_file()
        )
        if not workflows:
            click.echo(f"No .yxmd / .yxzp / .yxmz workflows found in {source_path}.", err=True)
            sys.exit(2)
        default_outdir_basename = source_path.name
    else:
        workflows = [source_path]
        default_outdir_basename = source_path.stem

    # Auto-derive --out-dir / --pkg if user didn't pass them.
    out_dir_path = Path(out_dir) if out_dir else Path.cwd() / _sanitize_dir(default_outdir_basename)
    if pkg is None:
        pkg = _sanitize_pkg(out_dir_path.name)

    # Scaffold a Dagster project if --out-dir doesn't exist (or isn't one).
    if install:
        _ensure_dagster_project(out_dir_path, pkg)

    if len(workflows) > 1:
        click.echo(f"Importing {len(workflows)} workflow(s) from {source_path} → {out_dir_path} (pkg={pkg})")
    else:
        click.echo(f"Importing {workflows[0]} → {out_dir_path} (pkg={pkg})")
    if llm_translate:
        click.echo(f"  LLM-assisted translation: model={llm_translate}, threshold={llm_score_threshold}")

    all_component_ids: list[str] = []
    for wf in workflows:
        if len(workflows) > 1:
            click.echo(f"\n── {wf.name} ──")
        result = import_workflow(
            yxmd_path=str(wf),
            out_dir=str(out_dir_path),
            pkg=pkg,
            llm_translate=llm_translate,
            llm_api_key_env=llm_api_key_env,
            llm_score_threshold=llm_score_threshold,
        )
        click.echo(f"  ✓ mapped tools:    {result['mapped_count']}")
        click.echo(f"  ✓ unmapped tools:  {result['unmapped_count']}")
        click.echo(f"  ✓ components used: {', '.join(result['component_ids']) or '(none)'}")
        click.echo(f"  ✓ migration report: {result['migration_report']}")
        for cid in result["component_ids"]:
            if cid not in all_component_ids:
                all_component_ids.append(cid)

    if install and all_component_ids:
        click.echo("\nInstalling registry components into the project...")
        _install_components(out_dir_path, pkg, all_component_ids)
        _ensure_runtime_deps(out_dir_path)
        click.echo(
            f"\nNext: `cd {out_dir_path} && uv run dg dev` to open the asset graph at "
            "http://localhost:3000."
        )
    elif all_component_ids:
        click.echo("\nNext (with --install we'd run these for you):")
        for cid in all_component_ids:
            click.echo(f"  dagster-component add {cid} --auto-install")


_PKG_INVALID = __import__("re").compile(r"[^A-Za-z0-9_]+")


def _sanitize_dir(name: str) -> str:
    """Lower-case, replace hostile chars with `-` (matches create-dagster's
    naming convention)."""
    s = __import__("re").sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return s.lower() or "dagster_project"


def _sanitize_pkg(name: str) -> str:
    """Python package name: lowercase + alphanum + underscore only."""
    s = _PKG_INVALID.sub("_", name).strip("_").lower()
    if not s or s[0].isdigit():
        s = f"pkg_{s}" if s else "alteryx_project"
    return s


def _ensure_dagster_project(out_dir: Path, pkg: str) -> None:
    """Scaffold a fresh Dagster project if `out_dir` isn't already one.

    Detection: `out_dir/pyproject.toml` contains `[tool.dg.project]`.
    Bootstrap: `uvx create-dagster project <name> --uv-sync`.
    """
    pyproj = out_dir / "pyproject.toml"
    if pyproj.exists() and "[tool.dg.project]" in pyproj.read_text():
        return  # already a Dagster project
    if out_dir.exists() and any(out_dir.iterdir()):
        # Non-empty, non-Dagster directory — bail rather than scaffold over
        # the user's files.
        raise click.ClickException(
            f"{out_dir} exists and isn't a Dagster project. Either rm -rf it, "
            "point --out-dir somewhere fresh, or pre-scaffold with "
            "`uvx create-dagster project <name>` and re-run with --no-install "
            "to skip the auto-bootstrap."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Scaffolding fresh Dagster project at {out_dir} ...")
    uvx = shutil.which("uvx")
    if not uvx:
        raise click.ClickException(
            "`uvx` not on PATH — install uv (https://docs.astral.sh/uv/) or "
            "pre-scaffold the Dagster project yourself and re-run with --no-install."
        )
    # create-dagster project takes the dir name as the project name. We
    # let create-dagster create the directory itself — pre-creating it
    # makes create-dagster bail with "already exists".
    proc = subprocess.run(
        [uvx, "create-dagster@latest", "project", out_dir.name, "--uv-sync"],
        cwd=out_dir.parent, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise click.ClickException(
            f"create-dagster failed:\n{proc.stderr[-500:]}"
        )


def _ensure_runtime_deps(out_dir: Path) -> None:
    """Add the runtime Python deps every imported workflow needs.

    `dagster-component add --auto-install` copies the component file but
    does NOT propagate the component's `requirements.txt` into the user's
    pyproject.toml. So pandas, numpy, and the dg CLI all need to be
    added explicitly here.
    """
    uv = shutil.which("uv")
    if not uv:
        click.echo(
            "  ⚠ `uv` not on PATH — skipping `uv add pandas numpy dagster-webserver`. "
            "Run those manually before `uv run dg dev`.",
            err=True,
        )
        return
    click.echo("  Adding pandas, numpy, openpyxl, dagster-webserver, dagster-dg-cli ...")
    # openpyxl is a transitive runtime dep of any workflow that reads/writes
    # .xlsx (Alteryx Excel Input → dataframe_from_excel, etc.). pandas itself
    # doesn't bundle it, and `dagster-component add` doesn't propagate
    # component requirements.txt, so we add it here unconditionally.
    subprocess.run(
        [uv, "add", "pandas", "numpy", "openpyxl"],
        cwd=out_dir, capture_output=True, text=True,
    )
    subprocess.run(
        [uv, "add", "--dev", "dagster-dg-cli", "dagster-webserver"],
        cwd=out_dir, capture_output=True, text=True,
    )


def _install_components(out_dir: Path, pkg: str, component_ids: List[str]) -> None:
    """Shell out to `dagster-component add` for each id.

    We deliberately do NOT depend on dagster-community-components-cli as a
    library import — that'd defeat the standalone-tool point. If the user
    doesn't have it on PATH we tell them how to get it (`uvx --from
    dagster-community-components-cli dagster-component …`).
    """
    cli = shutil.which("dagster-component")
    base_cmd: List[str]
    if cli:
        base_cmd = [cli]
    else:
        # Fallback to uvx — most users will have uv installed.
        uvx = shutil.which("uvx")
        if not uvx:
            click.echo(
                "  ✗ Neither `dagster-component` nor `uvx` is on PATH. "
                "Install one and run `dagster-component add <id> --auto-install` "
                "manually for each component listed above.",
                err=True,
            )
            sys.exit(2)
        base_cmd = [uvx, "--from", "dagster-community-components-cli", "dagster-component"]

    for cid in component_ids:
        click.echo(f"  → add {cid}")
        cmd = base_cmd + ["add", cid, "--auto-install"]
        rc = subprocess.run(cmd, cwd=str(out_dir)).returncode
        if rc != 0:
            click.echo(f"  ✗ add {cid} exited {rc}", err=True)
            sys.exit(rc)

        # `add` drops a starter defs.yaml at src/<pkg>/defs/<cid>/defs.yaml
        # that references the registry component's example asset — that'll
        # collide with the importer's emitted defs.yaml. Clean up.
        starter_dir = out_dir / "src" / pkg / "defs" / cid
        starter_yaml = starter_dir / "defs.yaml"
        if starter_yaml.exists():
            starter_yaml.unlink()
            try:
                starter_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
