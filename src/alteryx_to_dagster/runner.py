"""End-to-end driver: parse → map → emit.

Top-level entrypoint:

    import_workflow(
        yxmd_path="workflow.yxmd",
        out_dir="my-project/",
        pkg="my_project",                # python package name under src/
    )

Caller is responsible for scaffolding the `create-dagster` project + running
`dagster-component add <id>` for each registry id this importer emits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .emitter import emit_inline_python, emit_migration_report, emit_yaml
from .mapper import MappedTool, UnmappedTool, map_tool
from .parser import AlteryxNode, AlteryxWorkflow, parse_workflow


SCHEMA_URL_BASE = (
    "https://raw.githubusercontent.com/eric-thomas-dagster/"
    "dagster-component-templates/main/manifest.json"
)


def _topo_sort(wf: AlteryxWorkflow) -> List[AlteryxNode]:
    """Kahn's algorithm — Alteryx workflows are DAGs by construction."""
    incoming: Dict[str, int] = {n.tool_id: 0 for n in wf.nodes}
    for e in wf.edges:
        if e.dest_tool in incoming:
            incoming[e.dest_tool] += 1
    queue = [n for n in wf.nodes if incoming[n.tool_id] == 0]
    order: List[AlteryxNode] = []
    by_id = wf.by_id()
    while queue:
        # Stable order: lowest tool_id first.
        queue.sort(key=lambda n: int(n.tool_id) if n.tool_id.isdigit() else n.tool_id)
        node = queue.pop(0)
        order.append(node)
        for e in wf.downstreams_of(node.tool_id):
            if e.dest_tool in incoming:
                incoming[e.dest_tool] -= 1
                if incoming[e.dest_tool] == 0:
                    queue.append(by_id[e.dest_tool])
    if len(order) != len(wf.nodes):
        # Cycle (shouldn't happen with a valid Alteryx workflow), fall back to source order.
        return wf.nodes
    return order


def import_workflow(
    yxmd_path: str | Path,
    out_dir: str | Path,
    pkg: str,
) -> Dict[str, object]:
    """Parse the .yxmd / .yxmz and emit defs.yaml + .py files under out_dir.

    Returns a summary dict: {
        "mapped_count": int,
        "unmapped_count": int,
        "component_ids": [...],
        "migration_report": Path,
        "files_written": [Path, ...],
    }
    """
    yxmd_path = Path(yxmd_path)
    out_dir = Path(out_dir)

    wf = parse_workflow(yxmd_path)
    ordered = _topo_sort(wf)

    # Asset names get assigned during mapping; we need them to resolve
    # downstream `upstream_asset_key` references.
    tool_to_asset: Dict[str, str] = {}

    mapped_results: List[Tuple[str, str, str, str, List[str]]] = []   # for the report
    unmapped_results: List[Tuple[str, str, str, str]] = []
    component_ids_used: List[str] = []
    files_written: List[Path] = []

    for node in ordered:
        # Resolve upstreams in connection-anchor order so e.g. Join's Left/Right
        # arrive deterministically.
        incoming_edges = sorted(
            wf.upstreams_of(node.tool_id),
            key=lambda e: (e.dest_anchor, e.origin_tool),
        )
        upstreams = [tool_to_asset.get(e.origin_tool, "") for e in incoming_edges]

        result = map_tool(node, upstreams)
        if isinstance(result, UnmappedTool):
            unmapped_results.append((node.tool_id, node.plugin, result.reason, result.suggestion))
            # We can't link downstreams to a missing asset — leave tool_to_asset empty
            # for this tool. Downstream tools that consume it will get "" as their
            # upstream and the user will see the gap when they try to materialize.
            continue

        assert isinstance(result, MappedTool)
        tool_to_asset[node.tool_id] = result.asset_name

        if result.inline_python:
            path = emit_inline_python(out_dir, pkg, result.asset_name, result.inline_python)
        else:
            schema_url = (
                f"https://raw.githubusercontent.com/eric-thomas-dagster/"
                f"dagster-component-templates/main/assets/"
                f"_/{result.component_id}/schema.json"  # placeholder — schema lives under category/, see CLI
            )
            path = emit_yaml(
                out_dir, pkg,
                component_id=result.component_id,
                asset_name=result.asset_name,
                attributes=result.attributes,
                schema_url=None,    # `dagster-component add` rewrites this anyway
            )
            if result.component_id not in component_ids_used and result.component_id != "(inline_python)":
                component_ids_used.append(result.component_id)
        files_written.append(path)

        mapped_results.append((
            node.tool_id,
            node.plugin_short,
            result.component_id,
            result.asset_name,
            result.notes,
        ))

    # Surface bundled files (only present when the source was .yxzp / .yxmz) as
    # "couldn't convert" rows — `.yxdb` is a proprietary binary format and
    # macros need separate parsing.
    for yxdb in wf.bundled_data_files:
        unmapped_results.append((
            "(bundled)",
            "yxdb data file",
            f"Bundled .yxdb data file in the .yxzp/.yxmz package: `{yxdb}`",
            "Convert the .yxdb to CSV / Parquet manually (open in Alteryx, "
            "Output Data tool → CSV) then reference it from a `dataframe_from_csv` asset.",
        ))
    for yxmc in wf.bundled_macros:
        unmapped_results.append((
            "(bundled)",
            "yxmc macro",
            f"Bundled custom macro in the .yxzp/.yxmz package: `{yxmc}`",
            "Macros are nested workflows. Either inline the macro's logic as additional "
            "assets here, or re-import the .yxmc separately with `alteryx-import`.",
        ))

    report = emit_migration_report(
        out_dir,
        yxmd_source=str(yxmd_path),
        mapped=mapped_results,
        unmapped=unmapped_results,
    )
    files_written.append(report)

    return {
        "mapped_count": len(mapped_results),
        "unmapped_count": len(unmapped_results),
        "component_ids": component_ids_used,
        "migration_report": report,
        "files_written": files_written,
    }
