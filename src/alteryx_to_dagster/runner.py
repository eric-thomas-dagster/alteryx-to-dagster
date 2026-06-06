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
from .macro_splicer import splice_macros
from .mapper import MappedTool, UnmappedTool, _stock_macro_basenames, map_tool
from .parser import AlteryxNode, AlteryxWorkflow, parse_workflow


SCHEMA_URL_BASE = (
    "https://raw.githubusercontent.com/eric-thomas-dagster/"
    "dagster-component-templates/main/manifest.json"
)


_BRACKETED_FIELD_RE = __import__("re").compile(r"\[([^\[\]]+)\]")
_FIELD_ATTR_RE = __import__("re").compile(r"\bfield=\"([^\"]+)\"")
_FIELD2_ATTR_RE = __import__("re").compile(r"\bfield2=\"([^\"]+)\"")
_RENAME_ATTR_RE = __import__("re").compile(r"\brename=\"([^\"]+)\"")
_EXPRESSION_ATTR_RE = __import__("re").compile(r"\bexpression=\"([^\"]+)\"")
_NAME_ATTR_RE = __import__("re").compile(r"\bname=\"([^\"]+)\"")


def _columns_needed_downstream(wf: AlteryxWorkflow, origin_tool_id: str) -> list:
    """Scan every downstream tool of `origin_tool_id` and return the
    deduplicated list of column names they reference. Used by the
    placeholder source emitter to size the stub schema."""
    import xml.etree.ElementTree as _ET
    by_id = wf.by_id()
    visited: set = set()
    stack = [e.dest_tool for e in wf.downstreams_of(origin_tool_id)]
    cols: set = set()
    while stack:
        tid = stack.pop()
        if tid in visited or tid not in by_id:
            continue
        visited.add(tid)
        node = by_id[tid]
        # Stringify the node's configuration so the regex can scrape it.
        try:
            cfg_text = _ET.tostring(node.config, encoding="unicode")
        except Exception:
            cfg_text = ""
        cols.update(_BRACKETED_FIELD_RE.findall(cfg_text))
        cols.update(_FIELD_ATTR_RE.findall(cfg_text))
        cols.update(_FIELD2_ATTR_RE.findall(cfg_text))
        cols.update(_EXPRESSION_ATTR_RE.findall(cfg_text))
        cols.update(_RENAME_ATTR_RE.findall(cfg_text))
        # Don't walk Field name="..." for tools that DEFINE columns (TextInput).
        # Recurse into THIS node's own downstreams (transitively).
        stack.extend(e.dest_tool for e in wf.downstreams_of(tid))
    # Filter junk
    out: list = []
    for c in cols:
        if not c or "*" in c or "{" in c or "(" in c or c.startswith("\\") or len(c) > 80:
            continue
        # Reject regex / expression noise
        if any(s in c for s in [">", "<", "&", "|", "=", "+", "-", "/", "*", '"', "'", "\n"]):
            continue
        if c not in out:
            out.append(c)
    return out


def _stub_value_literal_for(col: str) -> str:
    """Pick a Python literal value for a stub-row column based on the
    column name. Returns a string that can be embedded directly in `repr`."""
    n = col.lower()
    if "date" in n or "time" in n or "_at" in n:
        return '"2020-01-01"'
    if any(t in n for t in ("count", "qty", "quantity", "amount", "price", "id", "num", "score", "rate", "total", "sum", "rank")):
        return "0"
    if "lat" in n:
        return "0.0"
    if "lon" in n or "lng" in n:
        return "0.0"
    return '""'


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
    *,
    llm_translate: str | None = None,
    llm_api_key_env: str | None = None,
    llm_score_threshold: float = 0.8,
) -> Dict[str, object]:
    """Parse the .yxmd / .yxmz / .yxzp and emit defs.yaml + .py files under out_dir.

    `llm_translate`: when set to a LiteLLM model id (e.g. "gpt-4o-mini",
    "claude-haiku-4-5-20251001"), the importer makes two LLM calls per
    flagged Alteryx-only formula expression (translate + independent
    score). Translations meeting `llm_score_threshold` get baked into
    the emitted YAML / .py. **No LLM dependency at materialization time.**

    Returns a summary dict: {
        "mapped_count": int,
        "unmapped_count": int,
        "component_ids": [...],
        "migration_report": Path,
        "files_written": [Path, ...],
        "llm_calls_made": int,
    }
    """
    yxmd_path = Path(yxmd_path)
    out_dir = Path(out_dir)

    translator = None
    if llm_translate:
        from .llm_translator import LLMTranslator
        translator = LLMTranslator(
            model=llm_translate,
            api_key_env_var=llm_api_key_env,
            score_threshold=llm_score_threshold,
        )

    wf = parse_workflow(yxmd_path)

    # Inline any nested macros (recursive — child macros expand too). The
    # splicer skips macros routed to stock registry components (cleanse,
    # etc.); those land in the parent's node list and the mapper handles
    # the routing.
    source_dir = wf.source_dir or yxmd_path.parent
    wf = splice_macros(
        wf,
        source_dir=source_dir,
        stock_macro_basenames=_stock_macro_basenames(),
    )

    ordered = _topo_sort(wf)

    tool_to_asset: Dict[str, str] = {}
    mapped_results: List[Tuple[str, str, str, str, List[str]]] = []   # for the report
    unmapped_results: List[Tuple[str, str, str, str]] = []
    component_ids_used: List[str] = []
    files_written: List[Path] = []

    placeholder_assets_emitted: set = set()  # tool_ids we've stubbed a source for

    def _emit_placeholder_source(origin_tool_id: str) -> str:
        """When the upstream node was unmapped, emit a placeholder source
        asset that returns a one-row DataFrame whose columns match what
        downstream consumers reference. Lets downstream assets actually
        execute on the stub rather than KeyError on the first column lookup.

        Column extraction: walks the parser's edge list to find every
        downstream node consuming this tool, then scrapes column references
        from each downstream node's <Configuration> XML — bracketed `[Field]`,
        Sort/Summarize/Select `field=` attrs, etc.
        """
        ph_name = f"unmapped_upstream_for_tool_{origin_tool_id}"
        if origin_tool_id in placeholder_assets_emitted:
            return ph_name
        placeholder_assets_emitted.add(origin_tool_id)

        cols = _columns_needed_downstream(wf, origin_tool_id)
        if not cols:
            cols = ["col1", "col2", "col3"]  # safe non-empty default

        # Synthesize a sensible single value per column (date-ish names get
        # an ISO timestamp; id/count/qty get 0; lat/lon get 0.0; else "").
        row_literal = "[" + ", ".join(_stub_value_literal_for(c) for c in cols) + "]"

        py = f'''"""Placeholder for Alteryx tool {origin_tool_id} (was unmapped).

The original Alteryx tool's mapping wasn't found in alteryx_to_dagster.mapper.
This stub returns a 1-row DataFrame whose schema matches what downstream
tools reference, so the graph at least loads + executes. Replace this with
the real tool's logic (see MIGRATION.md).
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={ph_name!r},
    group_name="alteryx_unmapped",
    description="Placeholder for unmapped Alteryx tool {origin_tool_id}. Replace.",
)
def {ph_name}() -> pd.DataFrame:
    return pd.DataFrame([{row_literal}], columns={cols!r})
'''
        emit_inline_python(out_dir, pkg, ph_name, py)
        return ph_name

    for node in ordered:
        # Resolve upstreams in connection-anchor order so e.g. Join's Left/Right
        # arrive deterministically.
        incoming_edges = sorted(
            wf.upstreams_of(node.tool_id),
            key=lambda e: (e.dest_anchor, e.origin_tool),
        )
        upstreams: List[str] = []
        for e in incoming_edges:
            up = tool_to_asset.get(e.origin_tool, "")
            if not up:
                # Upstream node wasn't mapped — emit a placeholder so the
                # downstream asset still resolves at build time.
                up = _emit_placeholder_source(e.origin_tool)
                tool_to_asset[e.origin_tool] = up  # cache so siblings share
            upstreams.append(up)

        result = map_tool(node, upstreams, translator=translator)
        if isinstance(result, UnmappedTool):
            unmapped_results.append((node.tool_id, node.plugin, result.reason, result.suggestion))
            # Don't record into tool_to_asset — downstream tools that
            # consume this will trigger _emit_placeholder_source on their
            # own incoming-edge walk.
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

    # Surface bundled files (only present when the source was .yxzp / .yxmz).
    # We can now NATIVELY read .yxdb — point at `dataframe_from_yxdb` instead
    # of telling people to manually convert.
    for yxdb in wf.bundled_data_files:
        unmapped_results.append((
            "(bundled)",
            "yxdb data file",
            f"Bundled .yxdb data file in the .yxzp/.yxmz package: `{yxdb}`",
            "Read natively via the `dataframe_from_yxdb` registry component, "
            "or convert to Parquet for portability: "
            "`python -c \"import import_yxdb; import_yxdb.to_dataframe('%s').to_parquet('%s.parquet')\"`. "
            "Cloud-deployable: copy the file (or its parquet equivalent) into "
            "S3 / GCS / ADLS and update `file_path` in the emitted defs.yaml." % (yxdb, yxdb),
        ))
    for yxmc in wf.bundled_macros:
        unmapped_results.append((
            "(bundled)",
            "yxmc macro",
            f"Bundled custom macro in the .yxzp/.yxmz package: `{yxmc}`",
            "Macros are nested workflows. Either inline the macro's logic as additional "
            "assets here, or re-import the .yxmc separately with `alteryx-import`.",
        ))

    # Cloud-portability warning: scan emitted defs for absolute local paths
    # so customers don't deploy a project that breaks the moment it leaves
    # the developer's laptop.
    _emit_local_path_warning(mapped_results, files_written, out_dir)

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


# --------------------------------------------------------------- helpers

_LOCAL_PATH_RE = __import__("re").compile(
    r"""(?:^|['"= ])([A-Z]:[\\/][^'"\n]+|/[A-Za-z0-9_./-]+|~/[^'"\n]+)""",
    __import__("re").MULTILINE,
)


def _emit_local_path_warning(mapped_results, files_written, out_dir: Path) -> None:
    """Scan emitted defs.yaml + .py files for absolute local paths. If any,
    inject a single CLOUD_PORTABILITY.md alongside MIGRATION.md explaining
    that local paths won't survive a Dagster Cloud / k8s deployment and
    recommending S3 / GCS / ADLS / Snowflake-stage equivalents.

    Doesn't modify the emitted YAML — that's a human judgment call (which
    cloud, which bucket layout, which credentials). Just surfaces the gap.
    """
    found_paths: List[str] = []
    for p in files_written:
        if not p.exists() or p.name.endswith((".md",)):
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        for m in _LOCAL_PATH_RE.finditer(text):
            path = m.group(1)
            # Filter out things that are obviously not data-file paths
            # (e.g. `/tmp` chunks of regex, dotted Python imports).
            if path.startswith(("/usr", "/etc", "/var/log")):
                continue
            if "." not in path.rsplit("/", 1)[-1]:  # no extension → unlikely a file
                continue
            found_paths.append(f"  {p.relative_to(out_dir)}: {path}")

    if not found_paths:
        return

    md = out_dir / "CLOUD_PORTABILITY.md"
    body = (
        "# Cloud-portability notice\n\n"
        "The imported project references **local filesystem paths** in one or "
        "more emitted assets. Local paths work for `dg dev` on your laptop, "
        "but break the moment the project deploys anywhere else — Dagster+ "
        "Hybrid / Serverless, Kubernetes, a CI runner — because none of those "
        "environments share your laptop's filesystem.\n\n"
        "**Recommended fix**: copy the referenced files (or their converted "
        "equivalents — Parquet is usually the right swap for .yxdb / .csv) into "
        "cloud object storage and update `file_path` in each defs.yaml:\n\n"
        "| Cloud | URL shape |\n"
        "|---|---|\n"
        "| AWS S3 | `s3://my-bucket/alteryx-exports/customers.parquet` |\n"
        "| Google Cloud Storage | `gs://my-bucket/alteryx-exports/customers.parquet` |\n"
        "| Azure Blob | `abfs://container@account.dfs.core.windows.net/path/file.parquet` |\n"
        "| Snowflake stage | `@MY_STAGE/path/customers.parquet` (via `sql_transform`) |\n\n"
        "Most `dataframe_from_*` components accept the same URL forms pandas "
        "/ pyarrow do, so the swap is usually one line per asset.\n\n"
        "## Local paths detected in this import\n\n"
    )
    body += "\n".join(sorted(set(found_paths))) + "\n"
    md.write_text(body)
    files_written.append(md)
