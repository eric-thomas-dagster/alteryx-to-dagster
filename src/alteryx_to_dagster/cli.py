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


@main.command("import", help="Parse an Alteryx workflow and emit a Dagster project.")
@click.argument("yxmd_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory — typically the root of a freshly-scaffolded create-dagster project.",
)
@click.option(
    "--pkg",
    required=True,
    help="Python package name under src/ (matches your create-dagster project's pkg name).",
)
@click.option(
    "--install",
    is_flag=True,
    help="After emitting defs.yamls, run `dagster-component add <id> --auto-install` for each component used.",
)
def import_cmd(yxmd_path: str, out_dir: str, pkg: str, install: bool) -> None:
    click.echo(f"Importing {yxmd_path} → {out_dir} (pkg={pkg})")
    result = import_workflow(yxmd_path=yxmd_path, out_dir=out_dir, pkg=pkg)
    click.echo(f"  ✓ mapped tools:    {result['mapped_count']}")
    click.echo(f"  ✓ unmapped tools:  {result['unmapped_count']}")
    click.echo(f"  ✓ components used: {', '.join(result['component_ids']) or '(none)'}")
    click.echo(f"  ✓ migration report: {result['migration_report']}")

    if install and result["component_ids"]:
        click.echo("")
        click.echo("Installing components into the project...")
        _install_components(Path(out_dir), pkg, result["component_ids"])
    elif result["component_ids"]:
        click.echo("")
        click.echo("Next:")
        for cid in result["component_ids"]:
            click.echo(f"  dagster-component add {cid} --auto-install")


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
