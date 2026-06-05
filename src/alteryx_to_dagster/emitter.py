"""Emit per-tool defs.yaml (or inline .py) into a scaffolded Dagster project.

Layout produced under `<out_dir>/src/<pkg>/defs/`:
    <asset_name>/defs.yaml      ← one folder per Alteryx tool that mapped
    <asset_name>.py             ← inline Python for unmapped-but-emittable tools

`type:` lines use `<pkg>.components.<component_id>.component.<ClassName>`
so they work after `dagster-component add <component_id>` has been run
in the project. The runner takes care of running those adds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .mapper import MappedTool


# Map registry id → ComponentClassName for the `type:` line.
#
# ⚠️ Case matters AND the `*Component` suffix is NOT universal —
# `append_fields`, `dataframe_join`, `dataframe_union` drop it. When adding
# new entries, grep the installed source:
#   grep '^class .*Component\|^class [A-Za-z]\+(Component' .../component.py
_COMPONENT_CLASS_NAMES: Dict[str, str] = {
    "filter": "FilterComponent",
    "formula": "FormulaComponent",
    "multi_field_formula": "MultiFieldFormulaComponent",
    "summarize": "SummarizeComponent",
    "sort": "SortComponent",
    "unique_dedup": "UniqueDedupComponent",
    "select_columns": "SelectColumnsComponent",
    "sample": "SampleComponent",
    "record_id": "RecordIdComponent",
    "running_total": "RunningTotalComponent",
    "pivot": "PivotComponent",
    "unpivot": "UnpivotComponent",
    "append_fields": "AppendFields",       # no -Component suffix
    "dataframe_join": "DataframeJoin",     # no -Component suffix
    "dataframe_union": "DataframeUnion",   # no -Component suffix
    "dataframe_from_csv": "DataframeFromCsvComponent",
    "dataframe_to_csv": "DataframeToCsvComponent",
    "dataframe_to_excel": "DataframeToExcelComponent",
    "dataframe_to_parquet": "DataframeToParquetComponent",
}


def emit_yaml(
    out_root: Path,
    pkg: str,
    component_id: str,
    asset_name: str,
    attributes: Dict[str, object],
    schema_url: str | None = None,
) -> Path:
    """Write src/<pkg>/defs/<asset_name>/defs.yaml. Returns the file path."""
    class_name = _COMPONENT_CLASS_NAMES.get(component_id)
    if class_name is None:
        raise ValueError(
            f"emitter has no class name for component_id={component_id!r}. "
            "Add an entry to _COMPONENT_CLASS_NAMES."
        )

    defs_dir = out_root / "src" / pkg / "defs" / asset_name
    defs_dir.mkdir(parents=True, exist_ok=True)
    defs_path = defs_dir / "defs.yaml"

    type_line = f"{pkg}.components.{component_id}.component.{class_name}"
    # asset_name lives on MappedTool (and is what we used to pick the folder
    # name above) but most components also require it as an attribute. Merge
    # it in unless the caller already supplied one in `attributes`.
    merged_attrs = dict(attributes)
    merged_attrs.setdefault("asset_name", asset_name)
    body = {
        "type": type_line,
        # Drop None / "" values so YAML stays clean.
        "attributes": {k: v for k, v in merged_attrs.items() if v is not None and v != ""},
    }
    header = ""
    if schema_url:
        header = f"# yaml-language-server: $schema={schema_url}\n"
    defs_path.write_text(header + yaml.safe_dump(body, sort_keys=False))
    return defs_path


def emit_inline_python(out_root: Path, pkg: str, asset_name: str, py_source: str) -> Path:
    """Write src/<pkg>/defs/<asset_name>.py for tools we can't express as YAML."""
    defs_dir = out_root / "src" / pkg / "defs"
    defs_dir.mkdir(parents=True, exist_ok=True)
    py_path = defs_dir / f"{asset_name}.py"
    py_path.write_text(py_source)
    return py_path


def emit_migration_report(
    out_root: Path,
    *,
    yxmd_source: str,
    mapped: List[tuple],   # list of (tool_id, plugin_short, component_id, asset_name, notes)
    unmapped: List[tuple], # list of (tool_id, plugin, reason, suggestion)
) -> Path:
    """Write MIGRATION.md summarizing what was converted and what wasn't."""
    out_root.mkdir(parents=True, exist_ok=True)
    md = out_root / "MIGRATION.md"
    lines = [
        f"# Alteryx → Dagster migration report",
        "",
        f"Source workflow: `{yxmd_source}`",
        "",
        f"- Tools mapped: **{len(mapped)}**",
        f"- Tools unmapped: **{len(unmapped)}**",
        "",
        "## Mapped tools",
        "",
        "| Tool ID | Alteryx plugin | Component id | Asset name |",
        "|---|---|---|---|",
    ]
    for tool_id, plugin, comp_id, asset_name, _notes in mapped:
        lines.append(f"| {tool_id} | `{plugin}` | `{comp_id}` | `{asset_name}` |")

    note_rows = [(tid, plg, an, n) for (tid, plg, _ci, an, ns) in mapped for n in ns]
    if note_rows:
        lines += [
            "",
            "## Notes / TODOs from translation",
            "",
        ]
        for tool_id, plugin, asset_name, note in note_rows:
            lines.append(f"- **tool {tool_id}** (`{plugin}`, asset `{asset_name}`): {note}")

    if unmapped:
        lines += [
            "",
            "## Unmapped tools — manual conversion required",
            "",
            "| Tool ID | Alteryx plugin | Why | Suggestion |",
            "|---|---|---|---|",
        ]
        for tool_id, plugin, reason, suggestion in unmapped:
            lines.append(f"| {tool_id} | `{plugin}` | {reason} | {suggestion} |")

    lines += [
        "",
        "## Next steps",
        "",
        "1. Inspect each generated `defs.yaml` — bracket-stripped expressions",
        "   may need small tweaks for `pandas.eval` syntax.",
        "2. For unmapped tools, run `dagster-component search <keyword>` to find",
        "   the closest existing component, then write a defs.yaml by hand.",
        "3. Run `dg check defs` to validate every YAML loads.",
        "4. Run `dg dev` and visualize the imported asset graph at http://localhost:3000.",
        "5. v1.5 (LLM-assisted): re-run with `--llm-assist openai` to translate the",
        "   flagged Alteryx-only expressions automatically.",
        "",
    ]
    md.write_text("\n".join(lines) + "\n")
    return md
