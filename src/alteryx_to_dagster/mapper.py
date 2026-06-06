"""Alteryx tool → Dagster community component mapping.

A `ToolMapping` callable takes the parser's AlteryxNode + the inferred
upstream asset names (in connection-anchor order) and returns either:

  - a MappedTool (component_id + asset_name + attributes dict + notes)
  - None, signalling "no mapping for this tool" → flagged in MIGRATION.md
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .expr_translator import ExprTranslation, translate as _det_translate
from .macro_splicer import MACRO_PLUGIN_PREFIX
from .parser import AlteryxNode


@dataclass
class MappedTool:
    component_id: str                       # e.g. "filter" — registry id
    asset_name: str                         # e.g. "high_volume_orders"
    attributes: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)   # caveats surfaced in MIGRATION.md
    inline_python: Optional[str] = None     # if non-None, emit a .py file instead of defs.yaml


@dataclass
class UnmappedTool:
    reason: str                             # why we couldn't map it
    suggestion: str = ""                    # how the user might fix it manually


# ---------------------------------------------------------------- helpers

_BRACKETED_FIELD = re.compile(r"\[([^\[\]]+)\]")


def _strip_field_brackets(expr: str) -> str:
    """Alteryx wraps field refs in [Brackets]; pandas eval just uses the bare name."""
    return _BRACKETED_FIELD.sub(r"\1", expr)


def _translate_expr(expr: str) -> ExprTranslation:
    """Run the deterministic translator. Returns an ExprTranslation; callers
    decide what to do with `is_python` / `fully` (route through eval, PYTHON
    path, or fall back to LLM).

    Thin wrapper kept here so the mapper has one canonical entry point.
    """
    return _det_translate(expr)


def _ascii_safe(s: str) -> str:
    """Asset-name-safe identifier (lower_snake_case, ASCII)."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "asset"


def _asset_name_for(node: AlteryxNode) -> str:
    """Prefer the annotation; fall back to plugin_short + tool_id."""
    if node.annotation:
        return _ascii_safe(node.annotation)
    return _ascii_safe(f"{node.plugin_short}_{node.tool_id}")


def _single_upstream(upstreams: List[str]) -> str:
    if not upstreams:
        return ""
    return upstreams[0]


# ---------------------------------------------------------------- mappers

# Alteryx field types → pandas dtypes for inline_dataframe's dtypes dict.
_ALTERYX_INT_TYPES = {"byte", "int16", "int32", "int64"}
_ALTERYX_FLOAT_TYPES = {"float", "double", "decimal", "fixeddecimal"}
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")


def _infer_dtype_from_values(values: List[str]) -> Optional[str]:
    """If every non-empty value in `values` looks like an int / float /
    ISO datetime, return the corresponding pandas dtype. Else None.

    Alteryx's runtime sometimes promotes V_String columns to numeric / date
    at execution time based on the actual data. The Alteryx XML almost
    always declares them V_String. Without this we end up comparing strings
    to ints downstream and crash.
    """
    non_empty = [v for v in values if v != "" and v is not None]
    if not non_empty:
        return None
    # ISO datetime?
    if all(_ISO_DATETIME_RE.match(v) for v in non_empty):
        return "datetime64[ns]"
    # All ints?
    try:
        [int(v) for v in non_empty]
        return "Int64"
    except (ValueError, TypeError):
        pass
    # All floats?
    try:
        [float(v) for v in non_empty]
        return "float64"
    except (ValueError, TypeError):
        pass
    return None


def _map_text_input(node: AlteryxNode, _upstreams: List[str]) -> MappedTool:
    """Alteryx Text Input → `inline_dataframe` registry component.

    Field types come from the Alteryx Field type attribute first
    (Int32 / Double / etc.). For V_String fields, we ALSO scan the actual
    row values — Alteryx's runtime auto-types V_String at execution time,
    so a column declared V_String whose values are all ISO datetimes /
    ints / floats gets upgraded. Without this, downstream filters and
    formulas crash on string-vs-int comparisons.
    """
    cfg = node.config
    name = _asset_name_for(node)

    field_specs: List[tuple[str, str]] = []
    fields_el = cfg.find("Fields")
    if fields_el is not None:
        for f in fields_el.findall("Field"):
            fn = f.attrib.get("name")
            if fn:
                field_specs.append((fn, f.attrib.get("type", "V_String")))

    rows: List[List[str]] = []
    data_el = cfg.find("Data")
    if data_el is not None:
        for r in data_el.findall("r"):
            rows.append([(c.text or "") for c in r.findall("c")])

    column_names = [fn for fn, _t in field_specs]

    # Pass 1 — declared Alteryx type → pandas dtype.
    dtypes: Dict[str, str] = {}
    override_count = 0
    for col_idx, (fn, atype) in enumerate(field_specs):
        lower = atype.lower()
        if lower in _ALTERYX_INT_TYPES:
            dtypes[fn] = "Int64"
            override_count += 1
        elif lower in _ALTERYX_FLOAT_TYPES:
            dtypes[fn] = "float64"
            override_count += 1
        elif lower.startswith("date") or lower == "datetime":
            dtypes[fn] = "datetime64[ns]"
            override_count += 1
        else:
            # V_String — scan the values to see if Alteryx would have
            # runtime-promoted this column.
            col_values = [r[col_idx] if col_idx < len(r) else "" for r in rows]
            inferred = _infer_dtype_from_values(col_values)
            if inferred is not None:
                dtypes[fn] = inferred
                override_count += 1
            # else: leave dtype unset; inline_dataframe defaults to string.

    return MappedTool(
        component_id="inline_dataframe",
        asset_name=name,
        attributes={
            # Order matters for diff-friendly YAML output — keep it stable
            # with what consumers have been seeing: asset_name first,
            # rows/cols in the middle, dtypes at the end.
            "asset_name": name,
            "columns": column_names,
            "rows": rows,
            "group_name": "alteryx_imported",
            "description": f"Alteryx Text Input (tool {node.tool_id})",
            "dtypes": dtypes or None,
        },
        notes=[
            f"Tool {node.tool_id} (Text Input) mapped to `inline_dataframe`. "
            f"{len(rows)} row(s) × {len(column_names)} column(s) preserved; "
            f"{override_count} numeric dtype override(s) emitted."
        ],
    )


def _map_filter(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Filter → `filter` registry component.

    The registry's `filter` component handles BOTH pandas-eval strings AND
    PYTHON-path Series expressions (via an eval() fallback with df/pd/np in
    scope). So we always emit a `filter` defs.yaml regardless of which path
    the translator produces — runtime stays deterministic, no inline .py
    needed for Contains / DateTimeAdd / IIF cases.
    """
    expr_el = node.config.find("Expression")
    raw_expr = (expr_el.text or "").strip() if expr_el is not None else ""
    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)

    # Empty Alteryx filter — emit `True` so the filter passes through.
    if not raw_expr:
        return MappedTool(
            component_id="filter",
            asset_name=asset_name,
            attributes={
                "upstream_asset_key": upstream,
                "condition": "True",
                "group_name": "alteryx_imported",
            },
            notes=[],
        )

    tr = _translate_expr(raw_expr)
    notes = list(tr.notes)

    if not tr.fully:
        notes.append(
            f"Filter expression on tool {node.tool_id}: at least one Alteryx-only "
            f"function couldn't be deterministically translated. Original: `{raw_expr}`. "
            f"Best-effort emitted: `{tr.pandas_expr}`. Review by hand or re-run with "
            f"`--llm-translate <model>`."
        )

    return MappedTool(
        component_id="filter",
        asset_name=asset_name,
        attributes={
            "upstream_asset_key": upstream,
            "condition": tr.pandas_expr,
            "group_name": "alteryx_imported",
        },
        notes=notes,
    )


def _map_formula(node: AlteryxNode, upstreams: List[str], translator=None) -> MappedTool:
    """Alteryx Formula → `formula` component (pandas eval), with optional
    v1.5 LLM-assisted translation for the expressions v1 had to drop.

    Three buckets per FormulaField:

      1. **Fully deterministic** (math + comparison + bracket-strip only) →
         emitted into the `formula` component's `expressions` dict.
      2. **Needs LLM, translator returned a pandas-eval expression with
         combined_score ≥ threshold** → also into `expressions`. Both
         original Alteryx + translated pandas surface in MIGRATION.md.
      3. **Needs LLM, translator returned PYTHON-path (Series-based)** →
         collected in `python_exprs`. If ANY column needs PYTHON path,
         the whole formula tool emits as an inline @dg.asset .py file
         (mixing eval-friendly cols + Series-style cols there).
      4. **Needs LLM, no translator OR score below threshold** → flagged
         in MIGRATION.md, NOT emitted. Run won't crash; user re-runs
         with `--llm-translate <model>` or fixes by hand.
    """
    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)

    # The registry's `formula` component handles BOTH pandas-eval strings AND
    # PYTHON-path expressions (df["…"].dt.year, np.where(...), etc.) via an
    # eval() fallback. So we keep PYTHON-path translations inside the same
    # `expressions` dict instead of escaping to an inline @dg.asset .py.
    expressions: Dict[str, str] = {}
    notes: List[str] = []

    ff_el = node.config.find("FormulaFields")
    if ff_el is not None:
        for f in ff_el.findall("FormulaField"):
            out_field = f.attrib.get("field", "?")
            expr = f.attrib.get("expression", "")
            tr = _translate_expr(expr)
            notes.extend(tr.notes)

            if tr.fully:
                expressions[out_field] = tr.pandas_expr
                continue

            if translator is None:
                # Emit the best-effort translation so the user sees what we
                # came up with; the formula component's fallback eval may
                # still execute it at runtime.
                expressions[out_field] = tr.pandas_expr
                notes.append(
                    f"Formula on tool {node.tool_id} → column {out_field!r}: "
                    f"Alteryx expression `{expr}` uses functions our "
                    f"deterministic translator doesn't fully cover. Emitted "
                    f"best-effort `{tr.pandas_expr}`. Re-run with "
                    f"`--llm-translate <model>` to refine."
                )
                continue

            try:
                r = translator.translate_and_score(expr)
            except Exception as e:  # noqa: BLE001
                expressions[out_field] = tr.pandas_expr
                notes.append(
                    f"Formula on tool {node.tool_id} → column {out_field!r}: "
                    f"LLM translation FAILED ({e!s}). Original Alteryx: `{expr}`. "
                    "Emitted best-effort deterministic translation; review."
                )
                continue

            if r.combined_score < translator.score_threshold:
                expressions[out_field] = tr.pandas_expr
                notes.append(
                    f"Formula on tool {node.tool_id} → column {out_field!r}: "
                    f"LLM translated `{expr}` → `{r.pandas_expr}` "
                    f"(combined_score={r.combined_score:.2f} < threshold "
                    f"{translator.score_threshold:.2f}). Fell back to "
                    f"deterministic translation; review. Scorer: {r.score_reason}"
                )
                continue

            expressions[out_field] = r.pandas_expr
            notes.append(
                f"Formula on tool {node.tool_id} → column {out_field!r}: "
                f"LLM-translated `{expr}` → `{r.pandas_expr}` "
                f"({'PYTHON path' if r.is_python else 'pandas eval'}, "
                f"score={r.combined_score:.2f}). {r.reasoning}"
            )

    return MappedTool(
        component_id="formula",
        asset_name=asset_name,
        attributes={
            "upstream_asset_key": upstream,
            "expressions": expressions,
            "group_name": "alteryx_imported",
        },
        notes=notes,
    )


def _emit_formula_as_python(
    *,
    node: AlteryxNode,
    upstream: str,
    asset_name: str,
    eval_exprs: Dict[str, str],
    python_exprs: Dict[str, str],
    notes: List[str],
) -> MappedTool:
    """Emit a formula tool as a small inline @dg.asset .py when at least
    one output column needs the PYTHON path. Result is fully deterministic
    at runtime — no LLM calls during materialization."""
    lines = []
    for col, expr in eval_exprs.items():
        # Keep these consistent with the `formula` component's semantics: df.eval(expr).
        lines.append(f'    df[{col!r}] = df.eval({expr!r})')
    for col, py_expr in python_exprs.items():
        lines.append(f'    df[{col!r}] = {py_expr}')
    body = "\n".join(lines) if lines else "    pass"

    py = f'''"""Alteryx Formula (tool {node.tool_id}) — emitted as inline Python.

At least one output column required the PYTHON path (pandas Series ops
like .str / .dt that pandas eval can't compile), so the whole formula
landed here as a single @dg.asset. The result is fully deterministic at
runtime — no LLM calls happen during materialization.

LLM was used at import time only to translate Alteryx-only functions to
their pandas equivalents. See MIGRATION.md for the score per expression.
"""
import dagster as dg
import numpy as np  # noqa: F401  (available for PYTHON-path expressions)
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Formula (tool {node.tool_id}) — LLM-assisted inline Python.",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
{body}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        notes=notes,
        inline_python=py,
    )


def _map_summarize(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Summarize → our `summarize` component.

    Shape: `aggregations` is a dict, not a list. Two forms:
      - Simple: `{revenue: sum}` — aggregate that column with that func, output column
        keeps the source name.
      - Named:  `{total_revenue: {col: revenue, agg: sum}}` — output a named column
        from a chosen source. Use the named form whenever Alteryx supplies a rename.
    """
    group_by: List[str] = []
    aggs: Dict[str, object] = {}
    sf_el = node.config.find("SummarizeFields")
    if sf_el is not None:
        for f in sf_el.findall("SummarizeField"):
            field_name = f.attrib.get("field", "")
            action = f.attrib.get("action", "").lower()
            rename = f.attrib.get("rename") or None
            if action == "groupby":
                group_by.append(field_name)
            else:
                if rename and rename != field_name:
                    aggs[rename] = {"col": field_name, "agg": action}
                else:
                    aggs[field_name] = action
    return MappedTool(
        component_id="summarize",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "group_by": group_by,
            "aggregations": aggs,
            "group_name": "alteryx_imported",
        },
    )


def _map_join(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    join_fields_l: List[str] = []
    join_fields_r: List[str] = []
    jf_el = node.config.find("JoinInfo")
    if jf_el is not None:
        for f in jf_el.findall("Field"):
            l = f.attrib.get("field")
            r = f.attrib.get("field2") or l
            if l:
                join_fields_l.append(l)
            if r:
                join_fields_r.append(r)
    return MappedTool(
        component_id="dataframe_join",
        asset_name=_asset_name_for(node),
        attributes={
            "left_asset_key": upstreams[0] if upstreams else "",
            "right_asset_key": upstreams[1] if len(upstreams) > 1 else "",
            "left_on": join_fields_l,
            "right_on": join_fields_r,
            "how": "inner",
            "group_name": "alteryx_imported",
        },
        notes=[
            f"Join on tool {node.tool_id}: Alteryx's Join also emits Left-Unjoined "
            "and Right-Unjoined anchors. The default mapping is inner-join only. "
            "If downstream tools consume the L/R anchors, you'll need additional "
            "filter assets to recreate the antijoin behaviour."
        ] if jf_el is not None else [],
    )


def _map_union(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    return MappedTool(
        component_id="dataframe_union",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_keys": upstreams,
            "group_name": "alteryx_imported",
        },
    )


def _map_sort(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    by: List[str] = []
    ascending: List[bool] = []
    sf_el = node.config.find("SortInfo")
    if sf_el is not None:
        for f in sf_el.findall("Field"):
            fn = f.attrib.get("field")
            order = f.attrib.get("order", "Ascending")
            if fn:
                by.append(fn)
                ascending.append(order.lower().startswith("asc"))
    # The `sort` component takes EITHER `ascending: bool` (single direction
    # applied to all `by` columns) OR `ascending_per_column: List[bool]`.
    # Match Alteryx's per-column control: scalar for one column, list for
    # multiple — keeps the emitted YAML compact for the common single-key sort.
    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "by": by,
        "group_name": "alteryx_imported",
    }
    if len(by) <= 1:
        attrs["ascending"] = ascending[0] if ascending else True
    else:
        attrs["ascending_per_column"] = ascending
    return MappedTool(
        component_id="sort",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_unique(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    fields: List[str] = []
    uf_el = node.config.find("UniqueFields")
    if uf_el is not None:
        for f in uf_el.findall("Field"):
            fn = f.attrib.get("field")
            if fn:
                fields.append(fn)
    return MappedTool(
        component_id="unique_dedup",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "subset": fields,
            "group_name": "alteryx_imported",
        },
    )


def _map_input_csv(node: AlteryxNode, _upstreams: List[str]) -> MappedTool:
    """Alteryx Input Data tool. Routes by file extension:

      .csv / .tsv / .txt         → dataframe_from_csv
      .parquet                   → dataframe_from_parquet
      .xlsx / .xls               → dataframe_from_excel
      .yxdb                      → dataframe_from_yxdb (native Alteryx binary)
      anything else              → dataframe_from_csv (with a note)

    The `<File>` element only carries text — `find("File") or find("Connection")`
    silently drops File on text-only elements because ElementTree's
    Element.__bool__ is False when there are no children. Use explicit
    `is not None` instead.
    """
    file_el = node.config.find("File")
    if file_el is None:
        file_el = node.config.find("Connection")
    file_path = (file_el.text or "").strip() if file_el is not None else ""

    # Alteryx Excel paths include a `|||Sheet1$` (or `|||<NamedRange>`)
    # suffix to specify the worksheet. The DataframeFromExcelComponent
    # accepts a sheet_name field separately, so strip the suffix off the
    # file path and forward the sheet via attributes below.
    sheet_name: Optional[str] = None
    if "|||" in file_path:
        file_path, raw_sheet = file_path.split("|||", 1)
        # Strip the trailing `$` Alteryx adds to sheet references.
        sheet_name = raw_sheet.rstrip("$") or None

    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    notes = []
    file_attr = "file_path"
    extra_attrs: Dict[str, object] = {}

    if ext == "parquet":
        component_id = "dataframe_from_parquet"
    elif ext in ("xlsx", "xls"):
        # Use the registry's polymorphic file_ingestion source component —
        # it handles Excel natively (and CSV / Parquet / JSON / etc. via
        # format=auto on the extension).
        component_id = "file_ingestion"
        extra_attrs["format"] = "excel"
        # file_ingestion.read_excel reads the first sheet by default; it
        # doesn't accept sheet_name. If Alteryx pointed at a specific sheet
        # via `|||Sheet$` suffix, surface it as a note for the user.
        if sheet_name:
            notes.append(
                f"Input Data on tool {node.tool_id}: Alteryx targeted Excel sheet "
                f"{sheet_name!r}. `file_ingestion` reads the first sheet — to read "
                "the original sheet, save it as the first sheet OR replace this "
                "component with an inline @dg.asset using "
                "`pd.read_excel(path, sheet_name=...)`."
            )
    elif ext == "yxdb":
        component_id = "dataframe_from_yxdb"
    else:
        component_id = "dataframe_from_csv"

    if ext not in ("csv", "tsv", "txt", "parquet", "xlsx", "xls", "yxdb"):
        notes.append(
            f"Input Data on tool {node.tool_id}: unknown extension {ext!r}; "
            f"defaulted to dataframe_from_csv. Adjust if your file is a different format."
        )
    attrs: Dict[str, object] = {
        file_attr: file_path,
        "group_name": "alteryx_imported",
        **extra_attrs,
    }
    return MappedTool(
        component_id=component_id,
        asset_name=_asset_name_for(node),
        attributes=attrs,
        notes=notes,
    )


def _map_output_csv(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Output Data tool writing a delimited file."""
    file_el = node.config.find("File")
    if file_el is None:
        file_el = node.config.find("Connection")
    file_path = (file_el.text or "").strip() if file_el is not None else ""
    # Sniff format by extension; default to CSV.
    fmt = "csv"
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext in ("xlsx", "xls"):
        fmt = "excel"
    elif ext == "parquet":
        fmt = "parquet"
    component_id = {
        "csv": "dataframe_to_csv",
        "excel": "dataframe_to_excel",
        "parquet": "dataframe_to_parquet",
    }[fmt]
    return MappedTool(
        component_id=component_id,
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            # `dataframe_to_csv` / `_excel` / `_parquet` all use `file_path`, not `path`.
            "file_path": file_path,
            "group_name": "alteryx_imported",
        },
    )


# ================================================================ In-DB tools
#
# Alteryx In-DB tools push compute into the warehouse — the chain of
# tools assembles a single SQL query that executes on the source DB
# (Snowflake / Postgres / BigQuery / etc.). The "ideal" mapping is a
# single Dagster `sql_transform` asset per In-DB subgraph that emits the
# assembled query as a single push-down statement (preserves Alteryx's
# performance story).
#
# v1 of this importer does the simpler thing: tool-by-tool, each In-DB
# tool becomes its own `sql_transform` asset that creates an intermediate
# table that the next tool selects from. Loses single-query pushdown but
# is unambiguous and easy to read. Future v2 can detect connected In-DB
# subgraphs and collapse them into a single sql_transform with CTEs.


def _indb_destination_table(node: AlteryxNode) -> str:
    """Stable intermediate-table name per Alteryx tool."""
    return f"alteryx_indb_tool_{node.tool_id}_{_asset_name_for(node)}"


def _indb_connection_env_var(node: AlteryxNode) -> str:
    """Pick the connection-URL env var. Alteryx config carries a connection
    name in <Connection> or <ConnectionString>; we slugify it into an env
    var so customers can wire one connection per Alteryx In-DB workflow.
    """
    cfg = node.config
    for tag in ("Connection", "ConnectionString", "ConnectionName"):
        el = cfg.find(tag)
        if el is not None:
            txt = (el.text or el.attrib.get("value") or "").strip()
            if txt:
                slug = re.sub(r"[^A-Za-z0-9]+", "_", txt).strip("_").upper()
                if slug:
                    return f"{slug}_URL"
    return "INDB_CONNECTION_URL"


def _indb_upstream_table(upstream_asset_name: str) -> str:
    """If an upstream is also an In-DB asset, its destination_table follows
    the same naming convention as us. We don't have direct access to the
    upstream's node here, so we just use the asset-name suffix and hope
    it matches. (For tool-by-tool emission this works because the runner
    builds the asset name and we wrap it.)"""
    # By convention upstream In-DB destination tables are named
    # `alteryx_indb_tool_<id>_<asset_name>` — but here we only have the
    # asset_name. The downstream sql_transform's `upstream_asset_keys`
    # carries the Dagster lineage; for the SQL FROM we reference the
    # destination_table value. Caller passes that in via `_make_indb_sql`
    # below using upstream_dest_tables tracked at the runner level.
    return upstream_asset_name


def _make_indb_mapped(
    node: AlteryxNode,
    sql: str,
    upstreams: List[str],
    *,
    return_dataframe: bool = False,
    extra_notes: Optional[List[str]] = None,
) -> MappedTool:
    return MappedTool(
        component_id="sql_transform",
        asset_name=_asset_name_for(node),
        attributes={
            "connection_url_env_var": _indb_connection_env_var(node),
            "destination_table": _indb_destination_table(node),
            "sql": sql,
            "return_dataframe": return_dataframe,
            "if_exists": "replace",
            "upstream_asset_keys": list(upstreams) if upstreams else None,
            "group_name": "alteryx_imported_indb",
        },
        notes=[
            f"In-DB tool {node.tool_id}: each Alteryx In-DB tool materializes "
            "its own intermediate table. To preserve Alteryx's single-query "
            "pushdown, future versions will collapse chains into one sql_transform "
            "with CTEs — for now you can hand-merge sibling assets if perf matters."
        ] + (extra_notes or []),
    )


def _map_indb_input(node: AlteryxNode, _upstreams: List[str]) -> MappedTool:
    cfg = node.config
    query_el = cfg.find("Query") or cfg.find("TableSelect") or cfg.find("Table")
    raw_sql = (query_el.text or "").strip() if query_el is not None and query_el.text else ""
    if not raw_sql:
        # Some Alteryx In-DB Input tools store the query as a TableName attribute.
        table_el = cfg.find("TableName") or cfg.find("Source")
        table_name = (table_el.text or "").strip() if table_el is not None and table_el.text else "TODO_source_table"
        raw_sql = f"SELECT * FROM {table_name}"
    return _make_indb_mapped(node, raw_sql, [])


def _map_indb_filter(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    expr_el = cfg.find("Expression")
    raw = (expr_el.text or "").strip() if expr_el is not None and expr_el.text else "TRUE"
    # In-DB expressions are already SQL — strip Alteryx [brackets] to bare names.
    sql_expr = _strip_field_brackets(raw)
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    sql = f"SELECT * FROM {upstream_table} WHERE {sql_expr}"
    return _make_indb_mapped(node, sql, upstreams)


def _map_indb_formula(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    ff_el = cfg.find("FormulaFields")
    select_extra = []
    if ff_el is not None:
        for f in ff_el.findall("FormulaField"):
            col = f.attrib.get("field", "?")
            expr = _strip_field_brackets(f.attrib.get("expression", ""))
            select_extra.append(f"({expr}) AS {col}")
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    select_clause = "*"
    if select_extra:
        select_clause = "*, " + ", ".join(select_extra)
    sql = f"SELECT {select_clause} FROM {upstream_table}"
    return _make_indb_mapped(node, sql, upstreams)


def _map_indb_summarize(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    group_by = []
    aggs = []
    sf_el = cfg.find("SummarizeFields")
    if sf_el is not None:
        for f in sf_el.findall("SummarizeField"):
            field_name = f.attrib.get("field", "")
            action = f.attrib.get("action", "").lower()
            rename = f.attrib.get("rename") or None
            if action == "groupby":
                group_by.append(field_name)
            else:
                out_name = rename or f"{action}_{field_name}"
                sql_fn = action.upper()
                # Alteryx → SQL aggregate name mappings
                sql_fn = {"AVG": "AVG", "SUM": "SUM", "COUNT": "COUNT",
                          "MIN": "MIN", "MAX": "MAX", "MEDIAN": "MEDIAN"}.get(sql_fn, sql_fn)
                aggs.append(f"{sql_fn}({field_name}) AS {out_name}")
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    select_clause = ", ".join(group_by + aggs) or "*"
    group_clause = f"GROUP BY {', '.join(group_by)}" if group_by else ""
    sql = f"SELECT {select_clause} FROM {upstream_table} {group_clause}".strip()
    return _make_indb_mapped(node, sql, upstreams)


def _map_indb_join(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    join_left = []
    join_right = []
    jf_el = cfg.find("JoinInfo")
    if jf_el is not None:
        for f in jf_el.findall("Field"):
            l = f.attrib.get("field")
            r = f.attrib.get("field2") or l
            if l:
                join_left.append(l)
                join_right.append(r)
    left = upstreams[0] if upstreams else "LEFT_TABLE"
    right = upstreams[1] if len(upstreams) > 1 else "RIGHT_TABLE"
    on_clause = " AND ".join(f"l.{l} = r.{r}" for l, r in zip(join_left, join_right)) or "1=1"
    sql = f"SELECT * FROM {left} l JOIN {right} r ON {on_clause}"
    return _make_indb_mapped(node, sql, upstreams,
        extra_notes=["In-DB Join: emitted as INNER JOIN — adjust to LEFT/RIGHT/FULL by hand if needed."])


def _map_indb_union(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    if not upstreams:
        sql = "-- TODO: no upstreams wired"
    else:
        sql = "\nUNION ALL\n".join(f"SELECT * FROM {u}" for u in upstreams)
    return _make_indb_mapped(node, sql, upstreams)


def _map_indb_sample(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    n_el = cfg.find("N") or cfg.find("Records")
    n = int(n_el.text) if n_el is not None and n_el.text and n_el.text.isdigit() else 100
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    sql = f"SELECT * FROM {upstream_table} LIMIT {n}"
    return _make_indb_mapped(node, sql, upstreams,
        extra_notes=["In-DB Sample: emitted as LIMIT — random / weighted sampling needs ORDER BY RANDOM() (Postgres) or SAMPLE (Snowflake)."])


def _map_indb_select(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    cols = []
    sf_el = cfg.find("SelectFields") or cfg.find("Fields")
    if sf_el is not None:
        for f in sf_el.findall("SelectField") + sf_el.findall("Field"):
            fn = f.attrib.get("field")
            sel = f.attrib.get("selected", "True").lower() != "false"
            rename = f.attrib.get("rename")
            if fn and sel and fn != "*Unknown":
                cols.append(f"{fn} AS {rename}" if rename and rename != fn else fn)
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    select_clause = ", ".join(cols) or "*"
    sql = f"SELECT {select_clause} FROM {upstream_table}"
    return _make_indb_mapped(node, sql, upstreams)


def _map_indb_streamout(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Stream Out of In-DB → execute the upstream query, pull result into pandas."""
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    sql = f"SELECT * FROM {upstream_table}"
    return _make_indb_mapped(node, sql, upstreams, return_dataframe=True,
        extra_notes=["StreamOut: this is the boundary where data leaves the warehouse and lands in a pandas DataFrame. Downstream non-In-DB tools consume that DataFrame."])


def _map_indb_writedata(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """In-DB Write Data → CTAS into a final user-specified table."""
    cfg = node.config
    dest_el = cfg.find("TableName") or cfg.find("Destination") or cfg.find("OutputTable")
    dest_table = (dest_el.text or "TODO_dest_table").strip() if dest_el is not None and dest_el.text else "TODO_dest_table"
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    sql = f"SELECT * FROM {upstream_table}"
    return MappedTool(
        component_id="sql_transform",
        asset_name=_asset_name_for(node),
        attributes={
            "connection_url_env_var": _indb_connection_env_var(node),
            "destination_table": dest_table,
            "sql": sql,
            "return_dataframe": False,
            "if_exists": "replace",
            "upstream_asset_keys": list(upstreams) if upstreams else None,
            "group_name": "alteryx_imported_indb",
        },
        notes=[f"In-DB WriteData → CTAS into final table `{dest_table}`."],
    )


def _map_indb_connect(node: AlteryxNode, _upstreams: List[str]):
    """In-DB Connection Manager — sets up a named DB connection. No standalone
    Dagster asset; surfaced in MIGRATION.md so the user can wire the matching
    env var."""
    env_var = _indb_connection_env_var(node)
    return UnmappedTool(
        reason=(
            f"In-DB connection manager tool {node.tool_id}: doesn't materialize "
            "anything — it just declares a named connection for downstream In-DB "
            "tools. Set the corresponding env var on your Dagster deployment."
        ),
        suggestion=(
            f"Export `{env_var}` to a SQLAlchemy connection URL "
            "(e.g. `snowflake://user:pwd@account/db/schema?warehouse=…`). "
            "Every downstream In-DB tool references this env var via its "
            "`connection_url_env_var` field."
        ),
    )


# Alteryx format-spec characters → Python strftime codes. Order matters
# (longer keys first) — multi-char codes like `yyyy` must be tried before `yy`.
_ALTERYX_TO_PY_FORMAT: List[tuple[str, str]] = [
    ("yyyy", "%Y"),
    ("yy", "%y"),
    ("MMMM", "%B"),
    ("MMM", "%b"),
    ("MM", "%m"),
    ("dddd", "%A"),
    ("ddd", "%a"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("hh", "%I"),
    ("mm", "%M"),  # only after MM/MMM/MMMM matched (minutes vs months)
    ("ss", "%S"),
    ("AM/PM", "%p"),
    ("AMPM", "%p"),
    ("tt", "%p"),
]


def _translate_alteryx_format(fmt: str) -> str:
    """Translate an Alteryx datetime format spec (`yyyy-MM-dd HH:mm:ss`)
    to Python strftime codes (`%Y-%m-%d %H:%M:%S`). If the spec already
    contains `%` codes, leave it alone (caller passed through Python codes).
    """
    if "%" in fmt:
        return fmt
    out = fmt
    for alt, py in _ALTERYX_TO_PY_FORMAT:
        out = out.replace(alt, py)
    return out


def _map_datetime_tool(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx DateTime tool → `datetime_parser` registry component.

    Handles both directions (parse string → datetime, format datetime → string)
    by setting input_format or output_format as appropriate.
    """
    cfg = node.config
    field_el = cfg.find("InputFieldName") or cfg.find("Field")
    new_el = cfg.find("OutputFieldName") or cfg.find("NewField")
    fmt_el = cfg.find("Format")
    direction_el = cfg.find("IsFrom")
    field_name = (field_el.text or "Date").strip() if field_el is not None and field_el.text else "Date"
    new_field = (new_el.text or f"{field_name}_Out").strip() if new_el is not None and new_el.text else f"{field_name}_Out"
    raw_fmt = (fmt_el.text or "%Y-%m-%d").strip() if fmt_el is not None and fmt_el.text else "%Y-%m-%d"
    fmt = _translate_alteryx_format(raw_fmt)
    direction = (direction_el.text or "DateTime").strip() if direction_el is not None and direction_el.text else "DateTime"

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "date_column": field_name,
        "output_column": new_field,
        "group_name": "alteryx_imported",
    }
    if direction.lower().startswith("string"):
        # datetime → formatted string
        attrs["output_format"] = fmt
    else:
        # string → datetime
        attrs["input_format"] = fmt

    return MappedTool(
        component_id="datetime_parser",
        asset_name=_asset_name_for(node),
        attributes=attrs,
        notes=[
            f"DateTime on tool {node.tool_id}: Alteryx format codes are "
            "translated to Python strftime (yyyy→%Y, MM→%m, etc.). Spot-check "
            "format string if the column doesn't parse as expected."
        ],
    )


def _map_datetime_tool_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline-python implementation; kept for reference, NOT registered."""
    cfg = node.config
    field_el = cfg.find("InputFieldName") or cfg.find("Field")
    new_el = cfg.find("OutputFieldName") or cfg.find("NewField")
    fmt_el = cfg.find("Format")
    direction_el = cfg.find("IsFrom")
    field_name = (field_el.text or "Date").strip() if field_el is not None and field_el.text else "Date"
    new_field = (new_el.text or f"{field_name}_Out").strip() if new_el is not None and new_el.text else f"{field_name}_Out"
    raw_fmt = (fmt_el.text or "%Y-%m-%d").strip() if fmt_el is not None and fmt_el.text else "%Y-%m-%d"
    fmt = _translate_alteryx_format(raw_fmt)
    direction = (direction_el.text or "DateTime").strip() if direction_el is not None and direction_el.text else "DateTime"
    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)
    if direction.lower().startswith("string"):
        body = f'    df[{new_field!r}] = df[{field_name!r}].dt.strftime({fmt!r})'
        descr = f"DateTime format ({field_name} -> {new_field}, format={fmt})"
    else:
        body = f'    df[{new_field!r}] = pd.to_datetime(df[{field_name!r}], format={fmt!r}, errors="coerce")'
        descr = f"DateTime parse ({field_name} -> {new_field}, format={fmt})"
    py = f'''"""Alteryx DateTime tool (tool {node.tool_id}) — inline pandas.

{descr}. The Alteryx format-spec characters mostly match Python's strftime
codes, but a few differ (Alteryx `yyyy` ≈ Python `%Y`). Spot-check.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx DateTime (tool {node.tool_id}): {descr}",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
{body}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"DateTime on tool {node.tool_id}: emitted as inline pandas "
            f"({'strftime' if direction.lower().startswith('string') else 'to_datetime'}). "
            "Alteryx format codes ≈ Python strftime, but spot-check `yyyy` vs `%Y` etc."
        ],
    )


def _map_regex_tool(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Regex tool → `regex_parser` registry component.

    Mode map: Parse → extract, Replace → replace, Match → match,
    Tokenize → split (component's `mode=split` handles split-into-rows).
    """
    cfg = node.config
    field_el = cfg.find("Field")
    # Alteryx stores the regex in <RegExExpression value="..."/> as an
    # ATTRIBUTE (not element text), with <Pattern>x</Pattern> as a fallback.
    regex_attr_el = cfg.find("RegExExpression")
    pattern_el = cfg.find("Pattern")
    method_el = cfg.find("Method")
    new_el = cfg.find("NewField")
    # <Replace expression="..."/> — replacement is on the attribute.
    repl_attr_el = cfg.find("Replace")
    field_name = (field_el.text or "Field").strip() if field_el is not None and field_el.text else "Field"
    if regex_attr_el is not None and "value" in regex_attr_el.attrib:
        pattern = regex_attr_el.attrib["value"]
    elif pattern_el is not None and pattern_el.text:
        pattern = pattern_el.text.strip()
    else:
        pattern = ".*"  # safe build-time default; tool was misconfigured
    method = (method_el.text or "Replace").strip() if method_el is not None and method_el.text else "Replace"
    new_field = (new_el.text or f"{field_name}_Out").strip() if new_el is not None and new_el.text else f"{field_name}_Out"
    replacement = ""
    if repl_attr_el is not None:
        replacement = repl_attr_el.attrib.get("expression", "")
        if not replacement and repl_attr_el.text:
            replacement = repl_attr_el.text.strip()

    # ParseComplex == multi-group capture (extract); ParseSimple == split.
    # Map Alteryx's method names onto regex_parser's mode field.
    mode_map = {
        "parse": "extract",
        "parsecomplex": "extract",
        "parsesimple": "split",
        "replace": "replace",
        "match": "match",
        "tokenize": "split",
    }
    mode = mode_map.get(method.lower(), "extract")

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "column": field_name,
        "pattern": pattern,
        "mode": mode,
        "group_name": "alteryx_imported",
    }
    if mode == "replace":
        attrs["replacement"] = replacement
        attrs["output_column"] = new_field
    elif mode in ("match", "extract"):
        attrs["output_column"] = new_field
    return MappedTool(
        component_id="regex_parser",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_regex_tool_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    cfg = node.config
    field_el = cfg.find("Field")
    pattern_el = cfg.find("Pattern")
    method_el = cfg.find("Method")
    new_el = cfg.find("NewField")
    repl_el = cfg.find("Replace")
    field_name = (field_el.text or "Field").strip() if field_el is not None and field_el.text else "Field"
    pattern = (pattern_el.text or "").strip() if pattern_el is not None and pattern_el.text else ""
    method = (method_el.text or "Replace").strip() if method_el is not None and method_el.text else "Replace"
    new_field = (new_el.text or f"{field_name}_Out").strip() if new_el is not None and new_el.text else f"{field_name}_Out"
    replacement = (repl_el.text or "").strip() if repl_el is not None and repl_el.text else ""

    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)
    m = method.lower()

    if m == "replace":
        body = f'    df[{new_field!r}] = df[{field_name!r}].str.replace({pattern!r}, {replacement!r}, regex=True)'
        descr = f"Regex replace ({field_name} → {new_field}, /{pattern}/ → {replacement!r})"
    elif m == "match":
        body = f'    df[{new_field!r}] = df[{field_name!r}].str.match({pattern!r})'
        descr = f"Regex match ({field_name} → {new_field}, /{pattern}/)"
    elif m == "tokenize":
        body = (
            f'    parts = df[{field_name!r}].str.split({pattern!r}, regex=True, expand=True)\n'
            f'    parts.columns = [f"{{{field_name!r}}}_{{i+1}}" for i in range(parts.shape[1])]\n'
            f'    df = pd.concat([df, parts], axis=1)'
        )
        descr = f"Regex tokenize ({field_name} split on /{pattern}/)"
    elif m == "parse":
        # Parse = extract groups from regex into separate columns
        body = (
            f'    parts = df[{field_name!r}].str.extract({pattern!r})\n'
            f'    parts.columns = [f"{{{field_name!r}}}_g{{i+1}}" for i in range(parts.shape[1])]\n'
            f'    df = pd.concat([df, parts], axis=1)'
        )
        descr = f"Regex parse-groups ({field_name} via /{pattern}/)"
    else:
        body = f'    raise NotImplementedError("Unknown Alteryx Regex method: {method!r}")'
        descr = f"Regex (unknown method {method!r})"

    py = f'''"""Alteryx Regex (tool {node.tool_id}) — inline pandas.

{descr}.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Regex (tool {node.tool_id}): {descr}",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
{body}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
    )


def _map_json_parse(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx JSON Parse → `json_flatten` registry component."""
    return MappedTool(
        component_id="json_flatten",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "group_name": "alteryx_imported",
        },
    )


def _map_json_parse_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    cfg = node.config
    field_el = cfg.find("Field")
    field_name = (field_el.text or "JSON_Name").strip() if field_el is not None and field_el.text else "JSON_Name"

    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)

    py = f'''"""Alteryx JSON Parse (tool {node.tool_id}) — pd.json_normalize.

Flattens the {field_name!r} JSON column into new columns. Behavior matches
Alteryx's JSON Parse for most JSON shapes; deeply-nested arrays may need
explicit `max_level` or a follow-up `explode`.
"""
import dagster as dg
import json
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx JSON Parse (tool {node.tool_id}) on column {field_name!r}",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
    # Tolerant: already-parsed dicts pass through; strings get json.loads'd.
    payloads = df[{field_name!r}].apply(
        lambda v: json.loads(v) if isinstance(v, str) else v
    )
    parsed = pd.json_normalize(payloads.tolist())
    parsed.columns = [f"{field_name}_{{c}}" for c in parsed.columns]
    return pd.concat([df.drop(columns=[{field_name!r}]), parsed], axis=1)
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
    )


def _map_xml_parse(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx XML Parse → `xml_parser` registry component.

    Alteryx config doesn't include xpath_expressions (it auto-extracts
    direct children); emit an empty dict — the user fills in per-column
    xpaths in the defs.yaml.
    """
    cfg = node.config
    field_el = cfg.find("Field")
    field_name = (field_el.text or "XML").strip() if field_el is not None and field_el.text else "XML"
    return MappedTool(
        component_id="xml_parser",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "xml_column": field_name,
            "xpath_expressions": {},
            "group_name": "alteryx_imported",
        },
        notes=[
            f"XMLParse on tool {node.tool_id}: emitted with empty "
            "`xpath_expressions`. The Alteryx tool auto-extracts direct child "
            "elements; for the registry's xml_parser you need to specify "
            "explicit xpath_expressions like `{out_col: '//Tag/text()'}` in "
            "the emitted defs.yaml."
        ],
    )


def _map_xml_parse_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    cfg = node.config
    field_el = cfg.find("Field")
    field_name = (field_el.text or "XML").strip() if field_el is not None and field_el.text else "XML"

    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)

    py = f'''"""Alteryx XML Parse (tool {node.tool_id}) — inline pandas + xml.etree.

Extracts every direct child element's text from the {field_name!r} column
into its own column. For attribute-aware or namespace-heavy XML, tweak
the parser callable below.
"""
import dagster as dg
import pandas as pd
import xml.etree.ElementTree as ET


def _xml_to_dict(xml_str):
    if not isinstance(xml_str, str) or not xml_str.strip():
        return {{}}
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return {{}}
    out = {{}}
    for child in root:
        out[child.tag] = (child.text or "").strip()
    return out


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx XML Parse (tool {node.tool_id}) on column {field_name!r}",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
    parsed = pd.json_normalize(df[{field_name!r}].apply(_xml_to_dict).tolist())
    parsed.columns = [f"{field_name}_{{c}}" for c in parsed.columns]
    return pd.concat([df.drop(columns=[{field_name!r}]), parsed], axis=1)
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"XMLParse on tool {node.tool_id}: stripped-down parser — extracts "
            "direct-child-element text only. For attribute / namespace handling, "
            "edit the `_xml_to_dict` function in the emitted .py."
        ],
    )


def _map_text_to_columns(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Text To Columns → `text_to_columns` registry component."""
    cfg = node.config
    field_el = cfg.find("Field")
    delim_el = cfg.find("Delimeters") or cfg.find("Delimiters")
    cols_el = cfg.find("NumFields") or cfg.find("NumColumns")
    field_name = (field_el.text or "Field1").strip() if field_el is not None and field_el.text else "Field1"
    delim = (delim_el.text or ",").strip() if delim_el is not None and delim_el.text else ","
    n_cols = int(cols_el.text) if cols_el is not None and cols_el.text and cols_el.text.isdigit() else None
    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "column": field_name,
        "separator": delim,
        "group_name": "alteryx_imported",
    }
    if n_cols:
        attrs["max_splits"] = n_cols - 1
    return MappedTool(
        component_id="text_to_columns",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_text_to_columns_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    field_el = node.config.find("Field")
    delim_el = node.config.find("Delimeters") or node.config.find("Delimiters")
    cols_el = node.config.find("NumFields") or node.config.find("NumColumns")
    asset_name = _asset_name_for(node)
    upstream = _single_upstream(upstreams)
    field_name = (field_el.text or "Field1").strip() if field_el is not None else "Field1"
    delim = (delim_el.text or ",").strip() if delim_el is not None else ","
    n_cols = int(cols_el.text) if cols_el is not None and cols_el.text and cols_el.text.isdigit() else None

    n_cols_arg = f"n={n_cols - 1}" if n_cols else ""
    py = f'''"""Alteryx Text To Columns (tool {node.tool_id}) — inline pandas split.

Splits the {field_name!r} column on delimiter {delim!r} into N columns.
No 1:1 registry component for this shape; emitted as inline Python so
runtime stays deterministic.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Text To Columns (tool {node.tool_id})",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
    parts = df[{field_name!r}].str.split({delim!r}, expand=True{', ' + n_cols_arg if n_cols_arg else ''})
    parts.columns = [f"{{ {field_name!r} }}{{i + 1}}" for i in range(parts.shape[1])]
    return pd.concat([df, parts], axis=1)
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"TextToColumns on tool {node.tool_id}: column-naming follows "
            f"Alteryx's {field_name!r}1 / {field_name!r}2 / … convention. "
            "Adjust if your downstream expects different names."
        ],
    )


def _map_data_cleansing(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Data Cleansing → `data_cleansing` registry component.

    Reads the Alteryx flag XML (TrimWhitespace, ReplaceNullsString, ChangeCase)
    and maps to the component's pydantic fields.
    """
    cfg = node.config

    def _flag(name: str) -> bool:
        el = cfg.find(name)
        if el is None:
            return False
        v = (el.text or "").strip().lower() if el.text else (el.attrib.get("value", "") or "").lower()
        return v in ("true", "1", "yes")

    trim_ws = _flag("TrimWhitespace") or _flag("RemoveTabs") or _flag("RemoveDuplicateWhitespace")
    fill_str = _flag("ReplaceNullsString")
    fill_num = _flag("ReplaceNullsNumeric")
    case_el = cfg.find("ChangeCase")
    case_op = (case_el.text or "").strip().lower() if case_el is not None and case_el.text else ""

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "group_name": "alteryx_imported",
    }
    if trim_ws:
        attrs["trim_whitespace"] = True
    if fill_str or fill_num:
        attrs["null_handling"] = "fill"
        if fill_str:
            attrs["null_fill_value"] = ""
        # Numeric fill (0) is handled automatically by the registry component
        # when null_fill_value is None — but we set "" above for the string side.
        # That means numeric NaN stays as NaN in this branch (an Alteryx quirk
        # where you can independently toggle string vs numeric).
    if case_op == "upper":
        attrs["normalize_case"] = "upper"
    elif case_op == "lower":
        attrs["normalize_case"] = "lower"
    elif case_op == "title":
        attrs["normalize_case"] = "title"
    return MappedTool(
        component_id="data_cleansing",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_data_cleansing_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    cfg = node.config
    asset_name = _asset_name_for(node)
    upstream = _single_upstream(upstreams)

    def _flag(name: str) -> bool:
        el = cfg.find(name)
        if el is None:
            return False
        v = (el.text or "").strip().lower() if el.text else (el.attrib.get("value", "") or "").lower()
        return v in ("true", "1", "yes")

    trim_ws = _flag("TrimWhitespace")
    remove_tabs = _flag("RemoveTabs")
    remove_dupws = _flag("RemoveDuplicateWhitespace")
    fill_str = _flag("ReplaceNullsString")
    fill_num = _flag("ReplaceNullsNumeric")
    case_el = cfg.find("ChangeCase")
    case_op = (case_el.text or "").strip().lower() if case_el is not None and case_el.text else ""

    steps = []
    if trim_ws:
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.strip()')
    if remove_tabs:
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.replace("\\t", "", regex=False)')
    if remove_dupws:
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.replace(r"\\s+", " ", regex=True)')
    if fill_str:
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].fillna("")')
    if fill_num:
        steps.append('    for c in df.select_dtypes(include="number").columns:\n        df[c] = df[c].fillna(0)')
    if case_op == "upper":
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.upper()')
    elif case_op == "lower":
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.lower()')
    elif case_op == "title":
        steps.append('    for c in df.select_dtypes(include="object").columns:\n        df[c] = df[c].str.title()')

    body = "\n".join(steps) if steps else "    pass   # no cleansing flags set on the Alteryx tool"

    py = f'''"""Alteryx Data Cleansing (tool {node.tool_id}) — inline pandas.

Implements the cleansing flags that were enabled on the Alteryx tool —
trim whitespace, remove tabs/dup whitespace, fill nulls, change case.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Data Cleansing (tool {node.tool_id})",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
{body}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
    )


def _map_date_filter(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Date Filter → `filter` with a between-style pandas-eval predicate."""
    field_el = node.config.find("Field")
    from_el = node.config.find("FromDate")
    to_el = node.config.find("ToDate")
    field_name = (field_el.text or "Date").strip() if field_el is not None else "Date"
    from_d = (from_el.text or "").strip() if from_el is not None else ""
    to_d = (to_el.text or "").strip() if to_el is not None else ""

    parts = []
    if from_d:
        parts.append(f'{field_name} >= "{from_d}"')
    if to_d:
        parts.append(f'{field_name} <= "{to_d}"')
    condition = " & ".join(f"({p})" for p in parts) or "True"
    return MappedTool(
        component_id="filter",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "condition": condition,
            "group_name": "alteryx_imported",
        },
        notes=[
            f"DateFilter on tool {node.tool_id}: emitted as a `filter` with a "
            f"between predicate ({condition}). Make sure {field_name!r} is "
            "datetime-typed upstream — pandas string-vs-date comparison can be a footgun."
        ],
    )


def _map_join_multiple(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Join Multiple → inline pandas chained .merge() across all upstreams.

    JoinMultiple takes ≥ 2 inputs + a configured join field. We don't have an
    N-way `dataframe_join`, so emit inline Python that merges them all in order.
    """
    cfg = node.config
    asset_name = _asset_name_for(node)
    join_field_el = cfg.find("JoinByRecordPosition") or cfg.find("JoinField")
    join_field = None
    if join_field_el is not None and join_field_el.text:
        join_field = join_field_el.text.strip()

    if not upstreams:
        return MappedTool(
            component_id="(inline_python)",
            asset_name=asset_name,
            inline_python="# no upstreams — this Alteryx JoinMultiple tool wasn't wired up.\n",
        )

    ins_block = ", ".join(
        f'"u{i}": dg.AssetIn(key=dg.AssetKey({u!r}))' for i, u in enumerate(upstreams)
    )
    args_decl = ", ".join(f"u{i}: pd.DataFrame" for i in range(len(upstreams)))
    if join_field:
        merge_chain = f"    df = u0\n    for next_df in [{', '.join(f'u{i}' for i in range(1, len(upstreams)))}]:\n        df = df.merge(next_df, on={join_field!r}, how='inner')"
    else:
        merge_chain = (
            "    # Alteryx default is positional join when no key is set. Recreate via index alignment:\n"
            "    df = u0.reset_index(drop=True)\n"
            f"    for next_df in [{', '.join(f'u{i}' for i in range(1, len(upstreams)))}]:\n"
            "        df = pd.concat([df, next_df.reset_index(drop=True)], axis=1)"
        )

    py = f'''"""Alteryx Join Multiple (tool {node.tool_id}) — inline pandas chained merge.

No N-way `dataframe_join` in the registry; this asset chains pandas
.merge() across all upstreams in connection order. Output is deterministic.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{ {ins_block} }},
    description="Alteryx Join Multiple (tool {node.tool_id}) — {len(upstreams)} inputs",
)
def {asset_name}({args_decl}) -> pd.DataFrame:
{merge_chain}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"JoinMultiple on tool {node.tool_id}: chained {len(upstreams)} inputs via pandas .merge(). "
            "Alteryx supports more advanced match logic (output-only-from-one-side, etc.) — "
            "this is the default inner-join chain."
        ],
    )


def _map_tile(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Tile → `tile_binning` registry component.

    Alteryx supports several tile methods; we map EqualRecords →
    `equal_records` (quantile-based) and everything else → `equal_width`.
    Component's `method` field also accepts `manual_cutoffs` and
    `smart_quantile` for advanced cases — surfaced in MIGRATION.md.
    """
    cfg = node.config
    method_el = cfg.find("Method")
    field_el = cfg.find("Field")
    num_el = cfg.find("NumTiles")
    out_field = cfg.find("OutputField")
    method_a = (method_el.text or "EqualRecords").strip() if method_el is not None and method_el.text else "EqualRecords"
    field_name = (field_el.text or "value").strip() if field_el is not None and field_el.text else "value"
    n_tiles = int(num_el.text) if num_el is not None and num_el.text and num_el.text.isdigit() else 4
    out_col = (out_field.text or "Tile_Num").strip() if out_field is not None and out_field.text else "Tile_Num"
    method_b = "equal_records" if "Records" in method_a else "equal_width"
    return MappedTool(
        component_id="tile_binning",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "column": field_name,
            "n_bins": n_tiles,
            "method": method_b,
            "output_column": out_col,
            "group_name": "alteryx_imported",
        },
        notes=[
            f"Tile on tool {node.tool_id}: Alteryx method={method_a!r} → "
            f"`tile_binning` method={method_b!r}. EqualSums / SmartTile / "
            "ManualCutoffs need the component's manual_cutoffs or "
            "smart_quantile mode — edit the emitted defs.yaml if needed."
        ],
    )


def _map_tile_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    cfg = node.config
    asset_name = _asset_name_for(node)
    upstream = _single_upstream(upstreams)
    method_el = cfg.find("Method")
    field_el = cfg.find("Field")
    num_el = cfg.find("NumTiles")
    out_field = cfg.find("OutputField")

    method = (method_el.text or "EqualRecords").strip() if method_el is not None and method_el.text else "EqualRecords"
    field_name = (field_el.text or "value").strip() if field_el is not None and field_el.text else "value"
    n_tiles = int(num_el.text) if num_el is not None and num_el.text and num_el.text.isdigit() else 4
    out_col = (out_field.text or "Tile_Num").strip() if out_field is not None and out_field.text else "Tile_Num"

    if "Equal" in method and "Records" in method:
        # Equal records → percentile-based → qcut
        body = f'    df[{out_col!r}] = pd.qcut(df[{field_name!r}], q={n_tiles}, labels=False, duplicates="drop") + 1'
    else:
        # EqualSums / SmartTile / etc. — default to equal-width bins
        body = f'    df[{out_col!r}] = pd.cut(df[{field_name!r}], bins={n_tiles}, labels=False) + 1'

    py = f'''"""Alteryx Tile (tool {node.tool_id}) — inline pandas qcut / cut.

Assigns a {n_tiles}-tile bucket label to each row based on {field_name!r}.
Method: {method}.
"""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Tile (tool {node.tool_id}, {n_tiles}-tile {method})",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
{body}
    return df
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"Tile on tool {node.tool_id}: Alteryx supports several tile "
            "methods (EqualRecords / EqualSums / SmartTile / ManualCutoffs). "
            "We mapped 'EqualRecords' → pd.qcut and everything else → pd.cut. "
            "Review if you need SmartTile or ManualCutoffs."
        ],
    )


def _map_sample(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Sample → `sample` registry component.

    The component's `method` field handles random / head / tail / every_nth /
    skip_head. Alteryx's mode names map cleanly: First→head, Last→tail,
    EveryNth→every_nth, Random→random, 1-in-N→random+frac, Skip→skip_head.
    """
    mode_el = node.config.find("Mode")
    n_el = node.config.find("N")
    mode = (mode_el.text or "First").strip() if mode_el is not None else "First"
    n = int(n_el.text) if n_el is not None and n_el.text and n_el.text.isdigit() else 1
    mode_lower = mode.lower().replace("n", "")  # "firstn" -> "first"

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "group_name": "alteryx_imported",
    }
    if mode_lower in ("first", ""):
        attrs["method"] = "head"
        attrs["sample_size"] = n
    elif mode_lower == "last":
        attrs["method"] = "tail"
        attrs["sample_size"] = n
    elif mode_lower in ("everynth", "every"):
        attrs["method"] = "every_nth"
        attrs["sample_size"] = n
    elif mode_lower in ("skip", "skipfirst"):
        attrs["method"] = "skip_head"
        attrs["sample_size"] = n
    elif mode_lower in ("random", "randomn"):
        attrs["method"] = "random"
        attrs["sample_size"] = n
    elif mode_lower in ("1in", "1innn"):
        attrs["method"] = "random"
        attrs["frac"] = 1.0 / max(n, 1)
    else:
        # Unknown — default to head; note in MIGRATION.md.
        attrs["method"] = "head"
        attrs["sample_size"] = n
    return MappedTool(
        component_id="sample",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_record_id(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Record ID → `record_id` (adds a sequence column)."""
    start_el = node.config.find("StartValue")
    name_el = node.config.find("FieldName")
    start = int(start_el.text) if start_el is not None and start_el.text and start_el.text.lstrip("-").isdigit() else 1
    col_name = (name_el.text or "RecordID").strip() if name_el is not None else "RecordID"
    return MappedTool(
        component_id="record_id",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "output_column": col_name,            # component calls it `output_column`, not `column_name`
            "start": start,
            "group_name": "alteryx_imported",
        },
    )


def _map_running_total(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Running Total → `running_total` (cumulative sum by group)."""
    group_by: List[str] = []
    fields: List[str] = []
    rt_el = node.config.find("RunningTotalFields")
    if rt_el is not None:
        for f in rt_el.findall("Field"):
            fn = f.attrib.get("field")
            if fn:
                fields.append(fn)
    gb_el = node.config.find("GroupBy")
    if gb_el is not None:
        for f in gb_el.findall("Field"):
            fn = f.attrib.get("field")
            if fn:
                group_by.append(fn)
    # RunningTotalComponent takes ONE value_column. Alteryx allows multiple
    # accumulator columns per tool — if we got multiple, emit the first and
    # surface the rest in MIGRATION.md (user should split into N RunningTotal
    # tools, one per column).
    primary = fields[0] if fields else ""
    notes = []
    if len(fields) > 1:
        notes.append(
            f"RunningTotal on tool {node.tool_id} accumulated multiple "
            f"columns ({fields}); registry's running_total takes ONE "
            f"value_column. Emitted for {primary!r}; add separate tools "
            f"for {fields[1:]}."
        )
    return MappedTool(
        component_id="running_total",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "value_column": primary,
            "group_by": group_by or None,
            "group_name": "alteryx_imported",
        },
        notes=notes,
    )


def _map_append_fields(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Append Fields → cartesian product. The component's actual
    field names are `upstream_asset_key` (Target = big input) and
    `source_asset_key` (Source = small DataFrame whose columns broadcast)."""
    return MappedTool(
        component_id="append_fields",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": upstreams[0] if upstreams else "",
            "source_asset_key": upstreams[1] if len(upstreams) > 1 else "",
            "group_name": "alteryx_imported",
        },
        notes=[
            f"AppendFields on tool {node.tool_id}: produces a cartesian product. "
            "Alteryx warns past ~16 source rows; same caveat applies here."
        ],
    )


def _map_cross_tab(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx CrossTab → `pivot` (or our `cross_tab` if you prefer that name)."""
    group_by: List[str] = []
    header_field = None
    value_field = None
    method = "sum"
    cfg = node.config
    for f in cfg.findall("GroupFields/Field"):
        fn = f.attrib.get("field")
        if fn:
            group_by.append(fn)
    hf = cfg.find("HeaderField")
    if hf is not None:
        header_field = hf.attrib.get("field") or (hf.text or "").strip() or None
    vf = cfg.find("DataField")
    if vf is not None:
        value_field = vf.attrib.get("field") or (vf.text or "").strip() or None
    methods = cfg.find("Methods")
    if methods is not None:
        m = methods.find("Method")
        if m is not None and m.text:
            method = m.text.strip().lower()
    return MappedTool(
        component_id="pivot",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            # Registry's PivotComponent field names — these don't match pandas
            # (or what _map_cross_tab emitted in v0.4). Spec:
            #   index_columns: List[str] → group keys that stay as rows
            #   pivot_column: str        → column whose values become new headers
            #   value_column: str        → column whose values fill the pivoted cells
            #   agg_func: str            → 'sum' / 'mean' / 'count' / 'min' / 'max' / 'first' / 'last'
            "index_columns": group_by,
            "pivot_column": header_field,
            "value_column": value_field,
            "agg_func": method,
            "group_name": "alteryx_imported",
        },
    )


def _map_transpose(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Transpose → `unpivot` (wide → long).

    Filters Alteryx's `*Unknown` placeholder out of both key + data field
    lists — `*Unknown` is Alteryx's wildcard for "all other columns" and
    isn't a real column name. If `*Unknown` appears in the data fields,
    drop the whole list (signals "melt all non-id columns", which unpivot
    handles natively when value_columns is None).
    """
    key_fields: List[str] = []
    data_fields: List[str] = []
    data_has_wildcard = False
    cfg = node.config
    for f in cfg.findall("KeyFields/Field"):
        fn = f.attrib.get("field")
        if fn and fn != "*Unknown":
            key_fields.append(fn)
    for f in cfg.findall("DataFields/Field"):
        fn = f.attrib.get("field")
        if not fn or f.attrib.get("selected", "True").lower() == "false":
            continue
        if fn == "*Unknown":
            data_has_wildcard = True
            continue
        data_fields.append(fn)
    if data_has_wildcard:
        # Signal to unpivot: "use all non-id columns" by passing None.
        data_fields = []
    # pd.melt rejects var_name / value_name that match an existing column.
    # Pick defaults unlikely to collide — "Name" and "Value" are super-common
    # column names; "Variable" + "MeltedValue" are safer.
    return MappedTool(
        component_id="unpivot",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "id_columns": key_fields,
            "value_columns": data_fields or None,
            "var_name": "Variable",
            "value_name": "MeltedValue",
            "group_name": "alteryx_imported",
        },
        notes=[
            f"Transpose on tool {node.tool_id}: emitted with var_name='Variable' "
            "and value_name='MeltedValue' to avoid collisions with common "
            "column names. Alteryx's defaults are 'Name' / 'Value' — rename "
            "if downstream tools expect those exact column names."
        ],
    )


def _map_count_records(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Count Records → `summarize` with a trivial Count aggregation."""
    out_name_el = node.config.find("FieldName")
    out_name = (out_name_el.text or "Count").strip() if out_name_el is not None else "Count"
    return MappedTool(
        component_id="summarize",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "group_by": [],
            "aggregations": {out_name: "size"},
            "group_name": "alteryx_imported",
        },
    )


def _map_multi_field_formula(node: AlteryxNode, upstreams: List[str], translator=None) -> MappedTool:
    """Alteryx Multi-Field Formula → our `multi_field_formula` component.
    Applies the same expression to a list of columns. Same translator
    flow as `_map_formula`."""
    expr_el = node.config.find("Expression")
    raw_expr = (expr_el.text or "").strip() if expr_el is not None else ""
    fields: List[str] = []
    fl_el = node.config.find("Fields")
    if fl_el is not None:
        for f in fl_el.findall("Field"):
            fn = f.attrib.get("field")
            if fn and f.attrib.get("selected", "True").lower() != "false":
                fields.append(fn)
    tr = _translate_expr(raw_expr)
    translated = tr.pandas_expr
    notes: List[str] = list(tr.notes)

    if not tr.fully and translator is not None:
        try:
            r = translator.translate_and_score(raw_expr)
            if r.combined_score >= translator.score_threshold and not r.is_python:
                translated = r.pandas_expr
                notes.append(
                    f"MultiFieldFormula on tool {node.tool_id}: LLM-translated "
                    f"`{raw_expr}` → `{translated}` (score={r.combined_score:.2f})."
                )
            else:
                notes.append(
                    f"MultiFieldFormula on tool {node.tool_id}: LLM rejected "
                    f"(score={r.combined_score:.2f} or PYTHON path; "
                    f"`multi_field_formula` only takes pandas-eval). Dropped. "
                    f"Original: `{raw_expr}`."
                )
        except Exception as e:  # noqa: BLE001
            notes.append(f"MultiFieldFormula LLM call failed: {e!s}")
    elif not tr.fully:
        notes.append(
            f"MultiFieldFormula on tool {node.tool_id}: Alteryx-only function in "
            f"`{raw_expr}` not deterministically translatable. Re-run with "
            f"`--llm-translate <model>` or edit by hand."
        )
    if tr.is_python:
        # multi_field_formula component is pandas-eval-only — can't emit
        # PYTHON-path. Flag and don't emit a broken expression.
        notes.append(
            f"MultiFieldFormula on tool {node.tool_id}: deterministic translation "
            "needs PYTHON path (Series ops), which `multi_field_formula` doesn't "
            "support. Consider rewriting as N single-column Formula tools."
        )
        translated = ""
    return MappedTool(
        component_id="multi_field_formula",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "fields": fields,
            "expression": translated,
            "group_name": "alteryx_imported",
        },
        notes=notes,
    )


def _map_generate_rows(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Generate Rows → `generate_rows` registry component.

    Common case: append a counter per input row. The component's
    `mode=from_start` + `n=1` matches that. For Alteryx setups with a
    non-default UpdateExpression / Condition, the emitted defs.yaml needs
    a tweak — surface in MIGRATION.md notes.
    """
    cfg = node.config
    field_el = cfg.find("CreateField_Name") or cfg.find("Field") or cfg.find("FieldName")
    field_name = (field_el.text or "rownum").strip() if field_el is not None and field_el.text else "rownum"
    return MappedTool(
        component_id="generate_rows",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "mode": "from_start",
            "n": 1,
            "group_name": "alteryx_imported",
        },
        notes=[
            f"GenerateRows on tool {node.tool_id}: emitted with mode='from_start' (one new row per input row). "
            f"Original Alteryx field was {field_name!r}; adjust the emitted defs.yaml if your "
            "Alteryx workflow used a custom condition or update expression."
        ],
    )


def _map_find_replace(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Find Replace → `find_replace` registry component.

    Two inputs: main (the rows being edited) + lookup (find/replace pairs).
    FieldFind / FieldReplace / FieldSearch elements name the columns.
    """
    cfg = node.config
    find_field_el = cfg.find("FieldFind")
    replace_field_el = cfg.find("FieldReplace")
    search_field_el = cfg.find("FieldSearch")
    find_field = (find_field_el.text or "find").strip() if find_field_el is not None and find_field_el.text else "find"
    replace_field = (replace_field_el.text or "replace").strip() if replace_field_el is not None and replace_field_el.text else "replace"
    search_field = (search_field_el.text or "value").strip() if search_field_el is not None and search_field_el.text else "value"
    return MappedTool(
        component_id="find_replace",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": upstreams[0] if upstreams else "",
            "lookup_asset_key": upstreams[1] if len(upstreams) > 1 else "",
            "lookup_key_column": find_field,
            "lookup_value_column": replace_field,
            "target_column": search_field,
            "group_name": "alteryx_imported",
        },
    )


def _map_create_points(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Create Points → `points_from_latlon` registry component.

    Alteryx config: `<Fields fieldX="lon_col" fieldY="lat_col"/>`. We map
    to the registry's PointsFromLatLonComponent.
    """
    cfg = node.config
    fields_el = cfg.find("Fields")
    if fields_el is not None:
        lon_col = fields_el.attrib.get("fieldX") or "Longitude"
        lat_col = fields_el.attrib.get("fieldY") or "Latitude"
    else:
        lon_col = "Longitude"
        lat_col = "Latitude"
    return MappedTool(
        component_id="points_from_latlon",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "longitude_column": lon_col,
            "latitude_column": lat_col,
            "geometry_column": "geometry",
            "crs": "EPSG:4326",
            "group_name": "alteryx_imported",
        },
    )


def _map_poly_split(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Poly-Split → inline pandas exploder.

    Splits a polyline / polygon into N rows (one per vertex / part).
    Without a proper geometry stack we emit a placeholder that respects
    a `geometry_column` config and uses shapely to access coordinates.
    """
    cfg = node.config
    asset_name = _asset_name_for(node)
    upstream = _single_upstream(upstreams)
    geom_el = cfg.find("GeometryField") or cfg.find("Field")
    geom_col = (geom_el.text or "geometry").strip() if geom_el is not None and geom_el.text else "geometry"

    py = f'''"""Alteryx Poly-Split (tool {node.tool_id}) — explode geometry to per-vertex rows."""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{"upstream": dg.AssetIn(key=dg.AssetKey({upstream!r}))}},
    description="Alteryx Poly-Split (tool {node.tool_id}) on column {geom_col!r}",
)
def {asset_name}(upstream: pd.DataFrame) -> pd.DataFrame:
    df = upstream.copy()
    def _explode(geom):
        if geom is None or getattr(geom, "is_empty", False):
            return []
        try:
            coords = list(geom.coords)
        except (NotImplementedError, AttributeError):
            # Polygon: walk the exterior ring.
            coords = list(getattr(geom.exterior, "coords", []))
        return coords
    df["_coords"] = df[{geom_col!r}].apply(_explode)
    df = df.explode("_coords").reset_index(drop=True)
    df["x"] = df["_coords"].apply(lambda c: c[0] if c is not None else None)
    df["y"] = df["_coords"].apply(lambda c: c[1] if c is not None else None)
    return df.drop(columns=["_coords"])
'''
    return MappedTool(
        component_id="(inline_python)",
        asset_name=asset_name,
        inline_python=py,
        notes=[
            f"PolySplit on tool {node.tool_id}: emits one row per geometry "
            "vertex with `x` / `y` columns. Alteryx also supports a "
            "per-segment split — extend the `_explode` callable if needed."
        ],
    )


# ----------------------------- macros (stock + custom)
#
# Stock macros are .yxmc files Alteryx ships that wrap a fixed tool chain.
# We route them to dedicated registry components instead of inlining the
# macro's XML (cleaner output + matches user intent better — `cleanse.yxmc`
# IS the data_cleansing component).

_STOCK_MACRO_COMPONENTS: Dict[str, Any] = {
    # Map lowercase basename → component-config dict. Field names match the
    # registry data_cleansing component's pydantic Field declarations
    # (`null_handling` / `null_fill_value` / `trim_whitespace` /
    # `normalize_case` / `remove_punctuation`), not Alteryx's per-checkbox
    # flag names. Defaults model what Alteryx Cleanse does most of the
    # time: trim whitespace + fill nulls with empty string (downstream
    # numeric coerce produces NaN where needed).
    "cleanse.yxmc": {
        "component_id": "data_cleansing",
        "attributes": {
            "trim_whitespace": True,
            "null_handling": "fill",
            "null_fill_value": "",
        },
    },
}


def _stock_macro_basenames() -> set:
    """Lowercase basenames of macros routed to stock components — used by
    macro_splicer to skip inlining them."""
    return set(_STOCK_MACRO_COMPONENTS.keys())


def _map_alteryx_macro(node: AlteryxNode, upstreams: List[str]):
    """Handle nodes whose plugin is the synthetic `AlteryxMacro::<basename>`.

    Two paths:
      - Stock macro in `_STOCK_MACRO_COMPONENTS` → emit the dedicated
        registry component with sensible defaults.
      - Anything else (custom user macro) → return UnmappedTool so the
        user knows it didn't inline.

    macro_splicer will have already inlined any non-stock custom macros it
    could resolve on disk — by the time map_tool sees a Macro:: node, it's
    EITHER stock OR a custom macro that couldn't be resolved.
    """
    macro_basename = node.plugin[len("AlteryxMacro::"):]
    stock = _STOCK_MACRO_COMPONENTS.get(macro_basename.lower())
    if stock:
        component_id = stock["component_id"]
        defaults = dict(stock.get("attributes") or {})  # type: ignore[arg-type]
        defaults.setdefault("upstream_asset_key", _single_upstream(upstreams))
        defaults.setdefault("group_name", "alteryx_imported")
        return MappedTool(
            component_id=str(component_id),
            asset_name=_asset_name_for(node),
            attributes=defaults,
            notes=[
                f"Macro {macro_basename!r} routed to stock registry "
                f"component `{component_id}` with sensible defaults. "
                "Tweak the emitted defs.yaml if your usage differs."
            ],
        )
    return UnmappedTool(
        reason=(
            f"Custom macro {macro_basename!r} couldn't be inlined "
            "(file not found alongside the workflow). Either copy the "
            ".yxmc next to the .yxmd and re-import, or rebuild the macro's "
            "logic by hand using `dagster-component search`."
        ),
        suggestion=(
            f"Place `{macro_basename}` in the same directory as the source "
            "workflow (or in a `macros/` subdirectory) and re-run the importer."
        ),
    )


def _map_select(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Select tool — keep/rename/reorder columns."""
    keep: List[str] = []
    rename_map: Dict[str, str] = {}
    sf_el = node.config.find("SelectFields") or node.config.find("Fields")
    if sf_el is not None:
        for f in sf_el.findall("SelectField") + sf_el.findall("Field"):
            fn = f.attrib.get("field")
            selected = f.attrib.get("selected", "True").lower() != "false"
            rename = f.attrib.get("rename")
            if fn and selected and fn != "*Unknown":
                keep.append(rename or fn)
                if rename and rename != fn:
                    rename_map[fn] = rename
    return MappedTool(
        component_id="select_columns",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "columns": keep,
            "rename": rename_map or None,
            "group_name": "alteryx_imported",
        },
    )


# ---------------------------------------------------------------- registry

ToolMapping = Callable[[AlteryxNode, List[str]], MappedTool]

PLUGIN_REGISTRY: Dict[str, ToolMapping] = {
    # Inputs / outputs
    "AlteryxBasePluginsGui.TextInput.TextInput": _map_text_input,
    "AlteryxBasePluginsGui.DbFileInput.DbFileInput": _map_input_csv,
    "AlteryxBasePluginsGui.DbFileOutput.DbFileOutput": _map_output_csv,
    # Row / column selection
    "AlteryxBasePluginsGui.Filter.Filter": _map_filter,
    "AlteryxBasePluginsGui.Select.Select": _map_select,
    "AlteryxBasePluginsGui.AlteryxSelect.AlteryxSelect": _map_select,
    "AlteryxBasePluginsGui.Sort.Sort": _map_sort,
    "AlteryxBasePluginsGui.Unique.Unique": _map_unique,
    "AlteryxBasePluginsGui.Sample.Sample": _map_sample,
    "AlteryxBasePluginsGui.DateFilter.DateFilter": _map_date_filter,
    "AlteryxBasePluginsGui.Tile.Tile": _map_tile,
    # Transforms
    "AlteryxBasePluginsGui.Formula.Formula": _map_formula,
    "AlteryxBasePluginsGui.MultiFieldFormula.MultiFieldFormula": _map_multi_field_formula,
    "AlteryxBasePluginsGui.RecordID.RecordID": _map_record_id,
    "AlteryxBasePluginsGui.RecordId.RecordId": _map_record_id,
    "AlteryxBasePluginsGui.RunningTotal.RunningTotal": _map_running_total,
    "AlteryxBasePluginsGui.DataCleansing.DataCleansing": _map_data_cleansing,
    "AlteryxBasePluginsGui.GenerateRows.GenerateRows": _map_generate_rows,
    "AlteryxBasePluginsGui.FindReplace.FindReplace": _map_find_replace,
    # Aggregates / reshape
    "AlteryxBasePluginsGui.Summarize.Summarize": _map_summarize,
    # The spatial plugins ship their own Summarize engine that's wire-compatible
    # with the base-plugins one (same SummarizeFields XML shape).
    "AlteryxSpatialPluginsGui.Summarize.Summarize": _map_summarize,
    "AlteryxBasePluginsGui.CountRecords.CountRecords": _map_count_records,
    "AlteryxBasePluginsGui.CrossTab.CrossTab": _map_cross_tab,
    "AlteryxBasePluginsGui.Transpose.Transpose": _map_transpose,
    # Parse
    "AlteryxBasePluginsGui.TextToColumns.TextToColumns": _map_text_to_columns,
    "AlteryxBasePluginsGui.DateTime.DateTime": _map_datetime_tool,
    "AlteryxBasePluginsGui.RegEx.RegEx": _map_regex_tool,
    "AlteryxBasePluginsGui.RegExSpawned.RegExSpawned": _map_regex_tool,
    "AlteryxBasePluginsGui.JSONParse.JSONParse": _map_json_parse,
    "AlteryxBasePluginsGui.XMLParse.XMLParse": _map_xml_parse,
    # Multi-input
    "AlteryxBasePluginsGui.Join.Join": _map_join,
    "AlteryxBasePluginsGui.JoinMultiple.JoinMultiple": _map_join_multiple,
    "AlteryxBasePluginsGui.Union.Union": _map_union,
    "AlteryxBasePluginsGui.Append.Append": _map_union,    # alias of Union
    "AlteryxBasePluginsGui.AppendFields.AppendFields": _map_append_fields,
    # Spatial — drop-ins for Alteryx spatial tools
    "AlteryxSpatialPluginsGui.CreatePoints.CreatePoints": _map_create_points,
    "AlteryxSpatialPluginsGui.PolySplit.PolySplit": _map_poly_split,
    # In-DB tools (multiple plugin namespaces in the wild — match all)
    "AlteryxConnectorsGui.InDbConnectionManager.InDbConnectionManager": _map_indb_connect,
    "AlteryxConnectorsGui.InDbInput.InDbInput": _map_indb_input,
    "AlteryxConnectorsGui.InDbFilter.InDbFilter": _map_indb_filter,
    "AlteryxConnectorsGui.InDbFormula.InDbFormula": _map_indb_formula,
    "AlteryxConnectorsGui.InDbSelect.InDbSelect": _map_indb_select,
    "AlteryxConnectorsGui.InDbSummarize.InDbSummarize": _map_indb_summarize,
    "AlteryxConnectorsGui.InDbJoin.InDbJoin": _map_indb_join,
    "AlteryxConnectorsGui.InDbUnion.InDbUnion": _map_indb_union,
    "AlteryxConnectorsGui.InDbSample.InDbSample": _map_indb_sample,
    "AlteryxConnectorsGui.InDbStreamOut.InDbStreamOut": _map_indb_streamout,
    "AlteryxConnectorsGui.InDbWriteData.InDbWriteData": _map_indb_writedata,
    # Some versions use a different namespace prefix.
    "AlteryxConnectorsGui.InDb.InDbInput": _map_indb_input,
    "AlteryxConnectorsGui.InDb.InDbFilter": _map_indb_filter,
}


# Fuzzier match for In-DB tools whose plugin id varies across Alteryx versions
# (the canonical match goes through PLUGIN_REGISTRY first; this is a fallback).
def _fuzzy_indb_match(plugin: str) -> Optional[ToolMapping]:
    if "indb" not in plugin.lower():
        return None
    name = plugin.lower()
    if "connectionmanager" in name or "indbconnect" in name:
        return _map_indb_connect
    if "input" in name:
        return _map_indb_input
    if "filter" in name:
        return _map_indb_filter
    if "formula" in name:
        return _map_indb_formula
    if "select" in name:
        return _map_indb_select
    if "summarize" in name:
        return _map_indb_summarize
    if "join" in name:
        return _map_indb_join
    if "union" in name:
        return _map_indb_union
    if "sample" in name:
        return _map_indb_sample
    if "streamout" in name:
        return _map_indb_streamout
    if "writedata" in name or "output" in name:
        return _map_indb_writedata
    return None


# A subset of mappers accepts a `translator` kwarg (the LLM-assisted ones).
# We dispatch by introspecting the function — keeps the public ToolMapping
# type clean and means mappers that don't need translation don't have to
# care about it.
_TRANSLATOR_AWARE_MAPPERS = frozenset({"_map_formula", "_map_multi_field_formula"})


_CONTROL_FLOW_PLUGINS = {
    # Each of these has a natural Dagster-native equivalent that's *implicit* in
    # the DAG — we emit no Dagster asset for them, just a MIGRATION.md note.
    "AlteryxBasePluginsGui.BlockUntilDone.BlockUntilDone": (
        "Block Until Done — Dagster's DAG already waits for upstream assets "
        "before scheduling downstream. No mapping needed: the DAG topology "
        "we emit preserves the ordering this tool was enforcing."
    ),
    "AlteryxBasePluginsGui.CacheDataset.CacheDataset": (
        "Cache Dataset — Dagster's IO managers persist asset values between "
        "runs by default (LocalFileSystem / S3 / GCS / etc., depending on "
        "config). No mapping needed."
    ),
    "AlteryxBasePluginsGui.MessageBus.MessageBus": (
        "Message tool — used for in-flow logging. Drop or replace with "
        "context.log.info() calls in adjacent Python assets if needed."
    ),
    "AlteryxBasePluginsGui.Browse.Browse": (
        "Browse tool — Alteryx-specific data preview. Dagster's UI shows "
        "asset previews in materialization metadata automatically; "
        "delete this tool from the migrated project."
    ),
    "AlteryxBasePluginsGui.BrowseV2.BrowseV2": (
        "Browse V2 — same as Browse (data preview). Dagster's UI shows "
        "asset previews in materialization metadata automatically; "
        "delete this tool from the migrated project."
    ),
    "AlteryxGuiToolkit.TextBox.TextBox": (
        "Text Box — annotation only (no data flow). Drop; if you want "
        "the comment preserved, add it to the asset's `description` field "
        "in defs.yaml or as a docstring on the inline-py asset."
    ),
    "AlteryxGuiToolkit.Detour.Detour": (
        "Detour — conditional bypass. Dagster equivalent: AutomationCondition "
        "on the downstream asset, or a `dagster.skip_if` pattern in a sensor. "
        "Not auto-mapped because the condition expression usually needs "
        "human judgment for the Dagster-native shape."
    ),
    "AlteryxGuiToolkit.Detour.DetourEnd": (
        "Detour End — companion to the Detour tool. Drop after wiring up "
        "the Detour's downstream AutomationCondition."
    ),
}


def map_tool(node: AlteryxNode, upstreams: List[str], translator=None):
    """Returns either a MappedTool or an UnmappedTool.

    `translator`: optional LLMTranslator. When supplied, the formula /
    multi-field-formula mappers send Alteryx-only expressions
    (IIF / Contains / DateTimeAdd / …) through the LLM at import time.
    Other mappers ignore it.
    """
    # Control-flow tools first — these are intentionally "unmapped" with
    # a clear explanation in MIGRATION.md (Dagster handles the equivalent
    # implicitly).
    cf_note = _CONTROL_FLOW_PLUGINS.get(node.plugin)
    if cf_note:
        return UnmappedTool(
            reason=cf_note,
            suggestion=(
                "Safe to drop — the Alteryx tool's behavior is implicit in "
                "the Dagster DAG / IO manager / automation_condition layer."
            ),
        )

    # Macro references — synthetic plugin string set by the parser. The
    # splicer has already inlined custom macros where possible; what's
    # left here is either stock-macro routing (cleanse, etc.) or a custom
    # macro we couldn't resolve.
    if node.plugin.startswith(MACRO_PLUGIN_PREFIX):
        return _map_alteryx_macro(node, upstreams)

    fn = PLUGIN_REGISTRY.get(node.plugin)
    if fn is None:
        # Fuzzy match for In-DB plugins whose namespace varies across versions.
        fuzzy = _fuzzy_indb_match(node.plugin)
        if fuzzy is not None:
            fn = fuzzy
        else:
            return UnmappedTool(
                reason=f"No mapping for plugin {node.plugin!r}",
                suggestion=(
                    "Register a mapper in alteryx_to_dagster.mapper.PLUGIN_REGISTRY, "
                    "or rebuild this tool's logic manually using "
                    "`dagster-component search <keyword>` to find an equivalent."
                ),
            )
    if translator is not None and fn.__name__ in _TRANSLATOR_AWARE_MAPPERS:
        return fn(node, upstreams, translator=translator)
    return fn(node, upstreams)
