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


def _find_first(cfg, *names):
    """Return the first non-None element among the given tag names.

    Python's `_find_first(cfg, "A", "B")` is BROKEN for ElementTree —
    Element instances are falsy when they have no children, so a leaf
    element like `<A>text</A>` evaluates falsy and the `or` falls through
    to find("B"). This helper uses explicit `is not None` checks so leaf
    elements with only text content are kept.
    """
    for name in names:
        el = cfg.find(name)
        if el is not None:
            return el
    return None


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
    # All ints? Leading-zero values (e.g. '01234') are typed as strings —
    # they're almost always ZIP codes / SSN / IDs where preserving the
    # leading zero matters, and downstream .str ops on them break under Int64.
    try:
        [int(v) for v in non_empty]
        if any(v.startswith("0") and v != "0" and not v.startswith("0.") for v in non_empty):
            return None
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

    Two Alteryx filter modes:
      - Custom: <Expression>...</Expression> (translated below).
      - Simple: <Mode>Simple</Mode><Simple><Operator>=</Operator><Field>X</Field>
                <Operands><Operand>val</Operand></Operands></Simple>
        Reconstruct the pandas expression from the structured form.
    """
    expr_el = node.config.find("Expression")
    raw_expr = (expr_el.text or "").strip() if expr_el is not None else ""
    upstream = _single_upstream(upstreams)
    asset_name = _asset_name_for(node)

    # Simple-mode filter: rebuild a pandas expression from the <Simple>
    # block when no <Expression> is present. Already valid pandas; skip
    # the Alteryx-expr translator entirely so it isn't mistranslated
    # (`df["X"]` is not Alteryx syntax — the translator would double-wrap it).
    simple_pandas: Optional[str] = None
    if not raw_expr:
        simple = node.config.find("Simple")
        if simple is not None:
            op_el = simple.find("Operator")
            field_el = simple.find("Field")
            operands_el = simple.find("Operands")
            op = (op_el.text or "=").strip() if op_el is not None and op_el.text else "="
            field = (field_el.text or "").strip() if field_el is not None and field_el.text else ""
            operand_el = operands_el.find("Operand") if operands_el is not None else None
            operand = (operand_el.text or "").strip() if operand_el is not None and operand_el.text else ""
            if field:
                # Map Alteryx operators → pandas comparison operators.
                _OP = {
                    "=": "==", "==": "==", "!=": "!=", "<>": "!=",
                    ">": ">", ">=": ">=", "<": "<", "<=": "<=",
                }
                py_op = _OP.get(op, "==")
                # Quote string operands; numeric stays bare.
                try:
                    float(operand)
                    operand_lit = operand
                except (ValueError, TypeError):
                    operand_lit = repr(operand)
                simple_pandas = f'df[{field!r}] {py_op} {operand_lit}'

    if simple_pandas is not None:
        return MappedTool(
            component_id="filter",
            asset_name=asset_name,
            attributes={
                "upstream_asset_key": upstream,
                "condition": simple_pandas,
                "group_name": "alteryx_imported",
            },
            notes=[],
        )

    # Empty Alteryx filter (no Expression AND no Simple) — emit `True`.
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
    group_by_rename: Dict[str, str] = {}
    aggs: Dict[str, object] = {}
    sf_el = node.config.find("SummarizeFields")
    if sf_el is not None:
        for f in sf_el.findall("SummarizeField"):
            field_name = f.attrib.get("field", "")
            action = f.attrib.get("action", "").lower()
            rename = f.attrib.get("rename") or None
            # `*Unknown` is Alteryx's wildcard — never a real column name.
            if field_name == "*Unknown":
                continue
            if action == "groupby":
                group_by.append(field_name)
                # Alteryx allows renaming the group-by output column inline.
                # Forward via summarize's group_by_rename field.
                if rename and rename != field_name:
                    group_by_rename[field_name] = rename
            else:
                # Alteryx's "Count" action counts ROWS per group, not non-null
                # values of a specific column. pandas's `groupby.agg("count")`
                # counts non-nulls — so when Alteryx's field name conflicts
                # with the desired output name (the common "Count": Count"
                # pattern), the agg blows up looking for a missing source col.
                # Map Alteryx Count → "size" (pandas row-count semantic) so
                # the result is correct AND doesn't depend on the source col.
                _agg = "size" if action == "count" else action
                _out = rename or field_name
                if _agg == "size":
                    # Single-arg form: summarize component treats `{out: size}`
                    # as row-count per group regardless of source col.
                    aggs[_out] = "size"
                elif rename and rename != field_name:
                    aggs[rename] = {"col": field_name, "agg": action}
                else:
                    aggs[field_name] = action
    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "group_by": group_by,
        "aggregations": aggs,
        "group_name": "alteryx_imported",
    }
    if group_by_rename:
        attrs["group_by_rename"] = group_by_rename
    return MappedTool(
        component_id="summarize",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_join(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    join_fields_l: List[str] = []
    join_fields_r: List[str] = []
    # Positional join (joinByRecordPos="True"): Alteryx stitches rows by
    # position with no key. Emit an inline @dg.asset that does
    # pd.concat([left, right], axis=1) — the closest pandas equivalent.
    cfg_attrib = node.config.attrib.get("joinByRecordPos", "").lower()
    if cfg_attrib == "true" and len(upstreams) >= 2:
        asset_name = _asset_name_for(node)
        left_key, right_key = upstreams[0], upstreams[1]
        py = f'''"""Alteryx positional Join (tool {node.tool_id}) — pd.concat by row position."""
import dagster as dg
import pandas as pd


@dg.asset(
    name={asset_name!r},
    group_name="alteryx_imported",
    ins={{
        "left": dg.AssetIn(key=dg.AssetKey({left_key!r})),
        "right": dg.AssetIn(key=dg.AssetKey({right_key!r})),
    }},
    description="Alteryx positional Join (tool {node.tool_id}) — joined by record position.",
)
def {asset_name}(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True).add_prefix("Right_")
    return pd.concat([left, right], axis=1)
'''
        return MappedTool(
            component_id="(inline_python)",
            asset_name=asset_name,
            inline_python=py,
            notes=[
                f"Join on tool {node.tool_id}: joinByRecordPos='True' — positional join. "
                "Emitted as inline pandas pd.concat(..., axis=1) with right cols Right_-prefixed."
            ],
        )

    # Alteryx joins encode keys as TWO sibling <JoinInfo> blocks —
    # connection="Left" and connection="Right" — each with one or more
    # <Field field="X"/> children. Read both side-by-side so cross-name
    # joins (left "Tm" → right "Teams") get the correct right key.
    left_info = None
    right_info = None
    for ji in node.config.findall("JoinInfo"):
        conn = (ji.attrib.get("connection") or "").lower()
        if conn == "left":
            left_info = ji
        elif conn == "right":
            right_info = ji
    if left_info is not None and right_info is not None:
        left_fields = [f.attrib.get("field") for f in left_info.findall("Field") if f.attrib.get("field")]
        right_fields = [f.attrib.get("field") for f in right_info.findall("Field") if f.attrib.get("field")]
        for l, r in zip(left_fields, right_fields):
            if l == "*Unknown" or r == "*Unknown":
                continue
            join_fields_l.append(l)
            join_fields_r.append(r)
    else:
        # Fallback: older/simpler JoinInfo blocks list both via field/field2.
        jf_el = node.config.find("JoinInfo")
        if jf_el is not None:
            for f in jf_el.findall("Field"):
                l = f.attrib.get("field")
                r = f.attrib.get("field2") or l
                if l == "*Unknown" or r == "*Unknown":
                    continue
                if l:
                    join_fields_l.append(l)
                if r:
                    join_fields_r.append(r)
    left_key = upstreams[0] if upstreams else ""
    right_key = upstreams[1] if len(upstreams) > 1 else ""

    # Alteryx Joins embed a <SelectConfiguration><SelectFields/> block that
    # renames/drops cols inline. Right-side cols arrive with a Right_ prefix
    # in Alteryx's output; SelectField rename="X" then gives them final names.
    # Forward this to dataframe_join via right_prefix + rename + drop_columns.
    # Parsed BEFORE the self-join short-circuit so self-joins inherit the same
    # rename/drop wiring.
    embedded_select = node.config.find("SelectConfiguration")
    if embedded_select is None:
        embedded_select = node.config.find(".//SelectFields")
    post_rename: Dict[str, str] = {}
    post_drop: List[str] = []
    has_explicit_select_rules = False
    if embedded_select is not None:
        sf_el = embedded_select if embedded_select.tag == "SelectFields" else embedded_select.find(".//SelectFields")
        if sf_el is not None:
            for sf in sf_el.findall("SelectField"):
                fname = sf.attrib.get("field", "")
                if not fname or fname == "*Unknown":
                    continue
                selected = sf.attrib.get("selected", "True").lower() == "true"
                rename = sf.attrib.get("rename", "")
                has_explicit_select_rules = True
                if not selected:
                    post_drop.append(fname)
                elif rename and rename != fname:
                    post_rename[fname] = rename

    # Self-join: when both sides reference the same upstream asset, the
    # dataframe_join component supports it natively — right_asset_key gets
    # omitted (or set equal to left_asset_key), the component clones `left`
    # in-process and prefixes its non-key columns with `right_prefix`.
    # Downstream Alteryx tools that reference `Right_<col>` resolve correctly.
    if left_key and right_key and left_key == right_key:
        self_attrs: Dict[str, Any] = {
            "left_asset_key": left_key,
            # right_asset_key intentionally omitted → self-join mode.
            "left_on": join_fields_l,
            "right_on": join_fields_r,
            "how": "inner",
            "right_prefix": "Right_",
            "group_name": "alteryx_imported",
        }
        if post_rename:
            self_attrs["rename"] = post_rename
        if post_drop:
            self_attrs["drop_columns"] = post_drop
        return MappedTool(
            component_id="dataframe_join",
            asset_name=_asset_name_for(node),
            attributes=self_attrs,
            notes=[
                f"Join on tool {node.tool_id}: self-join detected (both inputs "
                f"are {left_key!r}). Mapped to dataframe_join's self-join mode "
                "with right_prefix='Right_'."
            ],
        )

    attrs: Dict[str, Any] = {
        "left_asset_key": left_key,
        "right_asset_key": right_key,
        "left_on": join_fields_l,
        "right_on": join_fields_r,
        "how": "inner",
        "group_name": "alteryx_imported",
    }
    if has_explicit_select_rules:
        attrs["right_prefix"] = "Right_"
        if post_rename:
            attrs["rename"] = post_rename
        if post_drop:
            attrs["drop_columns"] = post_drop

    return MappedTool(
        component_id="dataframe_join",
        asset_name=_asset_name_for(node),
        attributes=attrs,
        notes=[
            f"Join on tool {node.tool_id}: Alteryx's Join also emits Left-Unjoined "
            "and Right-Unjoined anchors. The default mapping is inner-join only. "
            "If downstream tools consume the L/R anchors, you'll need additional "
            "filter assets to recreate the antijoin behaviour."
        ] if (left_info is not None or node.config.find("JoinInfo") is not None) else [],
    )


def _map_union(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    # Alteryx Union has two modes: ByName (default) and ByPos (Join By
    # Record Position). ByPos takes column names from the first input
    # and aligns subsequent inputs positionally — common when the user
    # unions sources with different first-col names (drivers.csv has
    # "Driver" as col1, tires.csv has "Tire", etc.).
    _mode_el = node.config.find("Mode")
    _mode = (_mode_el.text or "").strip() if _mode_el is not None and _mode_el.text else "ByName"
    attrs: Dict[str, object] = {
        "upstream_asset_keys": upstreams,
        "group_name": "alteryx_imported",
    }
    if _mode == "ByPos":
        attrs["mode"] = "by_position"
    return MappedTool(
        component_id="dataframe_union",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_sort(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    by: List[str] = []
    ascending: List[bool] = []
    sf_el = node.config.find("SortInfo")
    if sf_el is not None:
        for f in sf_el.findall("Field"):
            fn = f.attrib.get("field")
            order = f.attrib.get("order", "Ascending")
            if fn and fn != "*Unknown":
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
            if fn and fn != "*Unknown":
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

    # Database connection-string forms used by Input Data:
    #   odbc:DSN=mydsn;UID=u;PWD=p|||SELECT * FROM tbl
    #   oledb:Provider=SQLOLEDB.1;Data Source=host;Initial Catalog=db|||SELECT ...
    #   aka:My_Alias\nSourceData=...\nDSN=mydsn|||SELECT ...
    #   Provider=SQLOLEDB.1;...|||SELECT ...     (raw OLE-DB; no prefix)
    # Detect these BEFORE the per-extension routing — the path has no file
    # extension but does have `|||<query>` separating connection from SQL.
    _looks_like_dsn = (
        file_path.lower().startswith(("odbc:", "oledb:", "aka:"))
        or "Provider=" in file_path
        or "DSN=" in file_path
        or file_path.lower().startswith(("mssql+", "postgresql+", "mysql+",
                                         "oracle+", "snowflake://", "redshift+"))
    )
    if "|||" in file_path and _looks_like_dsn:
        _conn_str, _query = file_path.split("|||", 1)
        _query = _query.strip()
        # Synthesize a stable env-var name from the connection string so
        # users can keep secrets out of source. Hash to keep the var name
        # short / consistent across re-imports of the same connection.
        import hashlib as _hl
        _conn_id = _hl.sha1(_conn_str.encode("utf-8")).hexdigest()[:8].upper()
        _env_var = f"ALTERYX_DB_{_conn_id}_URL"
        return MappedTool(
            component_id="dataframe_from_sql",
            asset_name=_asset_name_for(node),
            attributes={
                "query": _query,
                "database_url_env_var": _env_var,
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Input Data on tool {node.tool_id}: Alteryx connection string "
                f"detected ({_conn_str[:80]!r}). Mapped to dataframe_from_sql. "
                f"Set ${_env_var} to a SQLAlchemy URL — for ODBC sources use "
                f"`mssql+pyodbc:///?odbc_connect=<urlencoded connection-string>` "
                f"(other drivers: `postgresql+psycopg2://...`, "
                f"`oracle+cx_oracle://...`, `snowflake://...`). "
                f"Original query: {_query[:120]}"
            ],
        )

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
    # Alteryx Input Data emits a FileName metadata column with the source
    # basename — required by chains that group_by FileName (e.g. Mario Kart
    # unions multiple CSVs and groups by source). The XML may name this
    # column via `<File OutputFileName="...">`; default to "FileName".
    if component_id == "dataframe_from_csv":
        attrs["add_filename_column"] = True
        if file_el is not None:
            _fn_col = file_el.attrib.get("OutputFileName") or "FileName"
            attrs["filename_column_name"] = _fn_col
        # Parse <FormatSpecificOptions> for delimiter / encoding / header_row.
        # Alteryx Mario Kart sample uses `;` delimiter + cp1252 (CodePage=28591).
        _fopts = node.config.find("FormatSpecificOptions")
        if _fopts is not None:
            _delim_el = _fopts.find("Delimeter") or _fopts.find("Delimiter")
            if _delim_el is not None and _delim_el.text:
                _delim = _delim_el.text
                # Alteryx encodes tab as `\t`; pandas accepts literal tab.
                if _delim == "\\t":
                    _delim = "\t"
                attrs["delimiter"] = _delim
            _cp_el = _fopts.find("CodePage")
            if _cp_el is not None and _cp_el.text:
                # Common code pages → Python encoding names
                _CP_MAP = {
                    "65001": "utf-8",
                    "28591": "latin-1",   # ISO-8859-1
                    "1252": "cp1252",
                    "1250": "cp1250",
                    "20127": "ascii",
                }
                _enc = _CP_MAP.get(_cp_el.text.strip())
                if _enc:
                    attrs["encoding"] = _enc
            _hdr_el = _fopts.find("HeaderRow")
            if _hdr_el is not None and (_hdr_el.text or "").strip().lower() == "false":
                attrs["skiprows"] = 0  # not really right but preserves intent
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
    query_el = _find_first(cfg, "Query", "TableSelect", "Table")
    raw_sql = (query_el.text or "").strip() if query_el is not None and query_el.text else ""
    if not raw_sql:
        # Some Alteryx In-DB Input tools store the query as a TableName attribute.
        table_el = _find_first(cfg, "TableName", "Source")
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
    n_el = _find_first(cfg, "N", "Records")
    n = int(n_el.text) if n_el is not None and n_el.text and n_el.text.isdigit() else 100
    upstream_table = upstreams[0] if upstreams else "UPSTREAM_TABLE"
    sql = f"SELECT * FROM {upstream_table} LIMIT {n}"
    return _make_indb_mapped(node, sql, upstreams,
        extra_notes=["In-DB Sample: emitted as LIMIT — random / weighted sampling needs ORDER BY RANDOM() (Postgres) or SAMPLE (Snowflake)."])


def _map_indb_select(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    cfg = node.config
    cols = []
    sf_el = _find_first(cfg, "SelectFields", "Fields")
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


def _map_data_stream_in(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Data Stream In → `dataframe_to_table` sink.

    Inverse of StreamOut: takes an upstream pandas DataFrame (from a
    non-InDB transform chain) and uploads it to a warehouse table so
    downstream InDB tools can join / filter / aggregate against it.

    Alteryx writes to a temp / user-named table; we emit
    `dataframe_to_table` writing to `<asset_name>_stage` by default —
    downstream InDB tools then `SELECT * FROM <asset_name>_stage` in
    their CTE chain (indb_collapse recognizes the staging asset as a
    warehouse source).

    The connection is resolved the same way as InDB tools — slugified
    from <Connection> → env var (e.g. `SNOWFLAKE_PROD_URL`).
    """
    cfg = node.config
    # Alteryx stores the destination table under TableName / Destination /
    # OutputTable depending on tool version. Fallback to a derived name.
    table_el = _find_first(cfg, "TableName", "Destination", "OutputTable", "Table")
    if table_el is not None and table_el.text:
        dest_table = table_el.text.strip()
    else:
        dest_table = f"{_asset_name_for(node)}_stage"

    return MappedTool(
        component_id="dataframe_to_table",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "table": dest_table,
            "database_url_env_var": _indb_connection_env_var(node),
            "if_exists": "replace",
            "group_name": "alteryx_imported_indb",
        },
        notes=[
            f"Data Stream In on tool {node.tool_id}: upstream pandas "
            f"DataFrame is uploaded to warehouse table `{dest_table}` via "
            "`dataframe_to_table`. Downstream In-DB tools reference this "
            "table as a warehouse source — indb_collapse handles the CTE "
            "wiring automatically."
        ],
    )


def _map_indb_writedata(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """In-DB Write Data → CTAS into a final user-specified table."""
    cfg = node.config
    dest_el = _find_first(cfg, "TableName", "Destination", "OutputTable")
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
    # Alteryx also uses `Mon` / `MON` / `Month` for month names — same as MMM/MMMM
    # in some non-canonical workflows. Order matters: must come before MM.
    ("Month", "%B"),
    ("MONTH", "%B"),
    ("Mon", "%b"),
    ("MON", "%b"),
    ("MM", "%m"),
    ("dddd", "%A"),
    ("ddd", "%a"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("hh", "%I"),
    ("mm", "%M"),  # only after MM/MMM/MMMM/Mon matched (minutes vs months)
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
    field_el = _find_first(cfg, "InputFieldName", "Field")
    new_el = _find_first(cfg, "OutputFieldName", "NewField")
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
    # Alteryx DateTime tool is forgiving — values that don't match the format
    # land in a separate "B" anchor (error rows) instead of crashing. Mirror
    # that by defaulting to errors='coerce' (bad rows → NaT) so a single
    # malformed value doesn't fail the whole asset.
    attrs["on_parse_error"] = "coerce"

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
    field_el = _find_first(cfg, "InputFieldName", "Field")
    new_el = _find_first(cfg, "OutputFieldName", "NewField")
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

    # Both ParseSimple and ParseComplex are EXTRACT operations in Alteryx —
    # they apply the regex to the column and emit one output column per
    # match group, named via <RootName>N or via explicit <ParseComplex>/<Field>
    # entries. Tokenize is the only true split (one row per token).
    mode_map = {
        "parse": "extract",
        "parsecomplex": "extract",
        "parsesimple": "extract",
        "replace": "replace",
        "match": "match",
        "tokenize": "split",
    }
    mode = mode_map.get(method.lower(), "extract")

    # ParseSimple's output columns are <RootName>1, <RootName>2, ...
    # ParseComplex lists each output col explicitly via <Field field=X type=...>.
    output_columns: List[str] = []
    if method.lower() == "parsesimple":
        ps_el = cfg.find("ParseSimple")
        if ps_el is not None:
            root_el = ps_el.find("RootName")
            num_el = ps_el.find("NumFields")
            root_name = root_el.text.strip() if root_el is not None and root_el.text else field_name
            num_fields = int(num_el.attrib.get("value", "1")) if num_el is not None else 1
            output_columns = [f"{root_name}{i + 1}" for i in range(num_fields)]
    elif method.lower() == "parsecomplex":
        pc_el = cfg.find("ParseComplex")
        if pc_el is not None:
            for fld in pc_el.findall("Field"):
                fname = fld.attrib.get("field") or ""
                if fname and fname != "No Marked Groups Found":
                    output_columns.append(fname)

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
    elif mode in ("match",):
        attrs["output_column"] = new_field
    elif mode == "extract":
        if output_columns:
            attrs["output_columns"] = output_columns
        else:
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
    """Alteryx JSON Parse → `json_flatten` registry component.

    Alteryx's JSONParse emits LONG-format output: one row per (key-path,
    value) pair with `JSON_Name` + `JSON_ValueString` columns. The
    json_flatten component's `output_format=long` matches that shape so
    downstream tools that filter on `JSON_Name` resolve.
    """
    cfg = node.config
    # Alteryx XML uses <JSON_Field> (Designer 2018+) — fall back to <Field>
    # for older / hand-written variants. Empty-element trick: ElementTree's
    # Element.__bool__ is False when there are no children, so `or` would
    # silently fall through on text-only elements. Use `is not None`.
    field_el = cfg.find("JSON_Field")
    if field_el is None:
        field_el = cfg.find("Field")
    src_col = (field_el.text or "").strip() if field_el is not None and field_el.text else None
    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "output_format": "long",
        "group_name": "alteryx_imported",
    }
    if src_col:
        attrs["column"] = src_col
    return MappedTool(
        component_id="json_flatten",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_jupyter_code(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx JupyterCode tool → `jupyter_notebook` (backend='exec').

    The notebook source is stored as a JSON `.ipynb` payload inside the
    `<Notebook>` CDATA of the tool's Configuration. We concatenate every
    `code` cell into a single inline Python block and rewrite Alteryx-only
    calls so the code runs against the upstream DataFrame:

      - `Alteryx.read("#1")` / `Alteryx.read(1)` → `df` (the upstream)
      - `Alteryx.read("#N")` (N > 1) → kept as-is (multi-input not modeled
        in a single-asset Dagster output; user wires manually)
      - `Alteryx.write(<expr>, N)` → `out_df = <expr>` for the LAST write
        call (which one matters depends on the downstream chain; we surface
        all writes in MIGRATION.md so the user can split into multi_asset
        if needed)

    Multi-output JupyterCode tools (with multiple `Alteryx.write(_, N)` calls
    going to separate downstream chains) collapse to a single asset that
    emits the LAST write target — surface remaining outputs in MIGRATION.md.
    """
    nb_el = node.config.find("Notebook")
    code: str = ""
    output_count = 0
    extracted = False
    if nb_el is not None and nb_el.text:
        import json as _json
        import re as _re
        try:
            nb = _json.loads(nb_el.text)
            cells = nb.get("cells", [])
            code_lines: List[str] = []
            last_write_expr: Optional[str] = None
            seen_writes: List[tuple] = []  # (expr, anchor)
            for cell in cells:
                if cell.get("cell_type") != "code":
                    continue
                src = cell.get("source", [])
                if isinstance(src, list):
                    src = "".join(src)
                # Rewrite Alteryx.read("#N") / Alteryx.read(N) → df (input 1)
                # OR keep N>1 as a placeholder.
                def _read_repl(m):
                    n = m.group(1).strip().strip('"').strip("'").lstrip("#")
                    return "df" if n in ("", "1") else f'df  # Alteryx.read("#{n}") — multi-input, wire manually'
                src = _re.sub(r"Alteryx\.read\(\s*([^)]+?)\s*\)", _read_repl, src)
                # Capture Alteryx.write(expr, N) calls; keep them as comments
                # and assign the LAST one to out_df.
                def _write_repl(m):
                    nonlocal last_write_expr, output_count
                    expr = m.group(1).strip()
                    anchor = m.group(2).strip()
                    seen_writes.append((expr, anchor))
                    last_write_expr = expr
                    output_count += 1
                    return f"# Alteryx.write({expr}, {anchor})  — output anchor #{anchor}"
                src = _re.sub(
                    r"Alteryx\.write\(\s*([^,]+?)\s*,\s*(\d+)\s*\)", _write_repl, src
                )
                # Drop pip-install boilerplate the Alteryx Jupyter wrapper adds.
                src = _re.sub(r"from ayx import Package\s*\n?", "", src)
                src = _re.sub(r"from ayx import Alteryx\s*\n?", "", src)
                src = _re.sub(r"Package\.installPackages\([^)]*\)\s*\n?", "", src)
                if src.strip():
                    code_lines.append(src.rstrip())
            if last_write_expr:
                code_lines.append(f"out_df = {last_write_expr}")
                extracted = True
            code = "\n".join(code_lines).strip()
        except Exception:
            code = ""

    if not extracted:
        # Older variants: <Code> or <Script> contained inline Python directly.
        for tag in ("Code", "Script", "NotebookSource"):
            _el = node.config.find(tag)
            if _el is not None and _el.text:
                code = _el.text.strip()
                extracted = True
                break
    if not code:
        code = "out_df = df  # original JupyterCode body not captured"

    _notes = []
    if output_count > 1:
        _notes.append(
            f"JupyterCode on tool {node.tool_id}: notebook has {output_count} "
            f"separate `Alteryx.write(_, N)` outputs. This asset emits the LAST "
            f"one ({last_write_expr}); the others are commented in the code. "
            f"To restore Alteryx's multi-output topology, split this asset into "
            f"a Dagster @multi_asset (one output per `Alteryx.write(_, N)`)."
        )
    elif output_count == 1:
        _notes.append(
            f"JupyterCode on tool {node.tool_id}: notebook body extracted "
            f"from `<Notebook>` JSON ({output_count} output)."
        )
    else:
        _notes.append(
            f"JupyterCode on tool {node.tool_id}: couldn't extract notebook "
            f"body — placeholder `out_df = df` in place. Fill in manually."
        )
    return MappedTool(
        component_id="jupyter_notebook",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "backend": "exec",
            "code": code,
            "group_name": "alteryx_imported",
        },
        notes=_notes,
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
    """Alteryx Text To Columns → `text_to_columns` registry component.

    Alteryx names output columns `<RootName>1`, `<RootName>2`, etc., with
    `<RootName>` defaulting to the source field name. The registry component
    auto-generates `<column>_0`, `<column>_1` which won't match downstream
    Alteryx-style references. Emit explicit `output_columns` to preserve.

    Alteryx delimiter is on the `value=` attribute of `<Delimeters>` /
    `<Delimiters>` (not element text). Same for `<NumFields value=N/>`.
    """
    cfg = node.config
    field_el = cfg.find("Field")
    delim_el = _find_first(cfg, "Delimeters", "Delimiters")
    cols_el = _find_first(cfg, "NumFields", "NumColumns")
    root_el = cfg.find("RootName")
    field_name = (field_el.text or "Field1").strip() if field_el is not None and field_el.text else "Field1"
    delim = ","
    if delim_el is not None:
        delim = delim_el.attrib.get("value") or (delim_el.text or "").strip() or ","
    n_cols = None
    if cols_el is not None:
        n_raw = cols_el.attrib.get("value") or (cols_el.text or "").strip()
        if n_raw and n_raw.isdigit():
            n_cols = int(n_raw)
    root_name = (root_el.text or field_name).strip() if root_el is not None and root_el.text else field_name

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "column": field_name,
        "separator": delim,
        "group_name": "alteryx_imported",
    }
    if n_cols:
        attrs["max_splits"] = n_cols - 1
        # Alteryx 1-indexed convention: <root>1, <root>2, ..., <root>N.
        attrs["output_columns"] = [f"{root_name}{i}" for i in range(1, n_cols + 1)]
    return MappedTool(
        component_id="text_to_columns",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


def _map_text_to_columns_DEPRECATED(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Old inline impl — NOT registered."""
    field_el = node.config.find("Field")
    delim_el = _find_first(node.config, "Delimeters", "Delimiters")
    cols_el = _find_first(node.config, "NumFields", "NumColumns")
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
    join_field_el = _find_first(cfg, "JoinByRecordPosition", "JoinField")
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
    # Alteryx nests the target field inside the method-specific sub-element
    # (e.g. <SmartTile><Field>X</Field></SmartTile>) but the simpler
    # EqualRecords form puts it at the top of the config. Try both.
    field_el = cfg.find("Field")
    if field_el is None or not (field_el.text and field_el.text.strip()):
        field_el = cfg.find(".//Field")
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
            # Alteryx names the running-total output column `RunTot_<field>`;
            # downstream tools reference that exact name. Override the
            # component's default (`running_<field>`) to match.
            "output_column": f"RunTot_{primary}" if primary else None,
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
        if fn and fn != "*Unknown":
            group_by.append(fn)
    hf = cfg.find("HeaderField")
    if hf is not None:
        header_field = hf.attrib.get("field") or (hf.text or "").strip() or None
    vf = cfg.find("DataField")
    if vf is not None:
        value_field = vf.attrib.get("field") or (vf.text or "").strip() or None
    methods = cfg.find("Methods")
    method_display = ""  # capitalized form for column-prefix (e.g. "Avg")
    if methods is not None:
        m = methods.find("Method")
        if m is not None:
            # Alteryx stores the value on the attribute (method="First")
            # OR as the element text — handle both.
            method_raw = m.attrib.get("method") or (m.text or "")
            if method_raw:
                method_display = method_raw.strip()
                method = method_display.lower()
    # Normalize Alteryx aggregate names to pandas pivot_table conventions.
    method = {"countdistinct": "nunique", "concat": "first"}.get(method, method)
    # Alteryx CrossTab column naming: prefer Alteryx's own saved <MetaInfo>
    # field names as the source of truth. The engine emits column names like
    # `Avg_Speed` for some methods (Avg / Median / StdDev) and bare header
    # values like `Burger` / `2014_` for others (Sum / Concat / First / etc.).
    # If MetaInfo contains a non-group field that starts with `<Method>_`,
    # Alteryx is using the prefix and we should match. Otherwise emit bare.
    _attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "index_columns": group_by,
        "pivot_column": header_field,
        "value_column": value_field,
        "agg_func": method,
        "group_name": "alteryx_imported",
    }
    if method_display:
        _emitted_names = node.output_field_names()
        _group_set = set(group_by)
        _non_group = [n for n in _emitted_names if n not in _group_set]
        _alteryx_uses_prefix = any(
            n.startswith(f"{method_display}_") for n in _non_group
        )
        if _alteryx_uses_prefix:
            _attrs["column_prefix"] = f"{method_display}_"
    return MappedTool(
        component_id="pivot",
        asset_name=_asset_name_for(node),
        attributes=_attrs,
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
    # Match Alteryx's default output column names — "Name" + "Value" —
    # because downstream tools in the source workflow reference those
    # exact names. If the upstream happens to also have a "Value" or
    # "Name" column, pd.melt raises and the user manually renames in
    # the emitted defs.yaml (or upstream Select drops the conflict).
    return MappedTool(
        component_id="unpivot",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "id_columns": key_fields,
            "value_columns": data_fields or None,
            "var_name": "Name",
            "value_name": "Value",
            "group_name": "alteryx_imported",
        },
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
            # Component's field is `columns`, not `fields` (was a naming
            # mismatch). Provide a no-op default expression when the
            # translation failed so the asset still validates.
            "columns": fields,
            "expression": translated or "{col}",
            "group_name": "alteryx_imported",
        },
        notes=notes,
    )


def _map_generate_rows(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Generate Rows → `generate_rows` registry component OR an
    `inline_dataframe` when there's no upstream (Alteryx GenerateRows can
    run rootless — generate N rows from nothing).

    Common cases:
    - Upstream present: append a counter per input row (`mode=from_start`).
    - No upstream: emit an inline_dataframe with N rows seeded from the
      configured field name (one column, sequential integers).
    """
    cfg = node.config
    field_el = _find_first(cfg, "CreateField_Name", "Field", "FieldName")
    field_name = (field_el.text or "rownum").strip() if field_el is not None and field_el.text else "rownum"

    upstream = _single_upstream(upstreams)
    if not upstream:
        # Rootless GenerateRows — emit inline_dataframe seeded with 10
        # sequential rows. User adjusts row count + start value in defs.yaml.
        return MappedTool(
            component_id="inline_dataframe",
            asset_name=_asset_name_for(node),
            attributes={
                "asset_name": _asset_name_for(node),
                "columns": [field_name],
                "rows": [[i] for i in range(10)],
                "group_name": "alteryx_imported",
                "description": f"Alteryx GenerateRows (tool {node.tool_id}) — rootless; 10-row seed",
            },
            notes=[
                f"GenerateRows on tool {node.tool_id}: NO upstream input in the "
                "Alteryx workflow — emitted as inline_dataframe with 10 sequential "
                f"integer rows in column {field_name!r}. Edit `rows:` in defs.yaml "
                "to match your Alteryx workflow's intended range."
            ],
        )

    # Expression-driven loop expansion: Alteryx GenerateRows with Init/Cond/Loop
    # expressions emits one new row per loop iteration per upstream row. Map
    # to generate_rows component's loop_expression mode after translating each
    # Alteryx expr to Python (bracketed [Col] → row['Col'], create field → value).
    init_el = cfg.find("Expression_Init")
    cond_el = cfg.find("Expression_Cond")
    loop_el = cfg.find("Expression_Loop")
    has_loop_form = (
        init_el is not None and init_el.text and
        cond_el is not None and cond_el.text and
        loop_el is not None and loop_el.text
    )
    if has_loop_form:
        from .expr_translator import translate as _expr_translate
        def _xlate(raw: str) -> str:
            # Bracket → row['X'] for upstream cols; bare field_name (the
            # CreateField) → value (the loop variable).
            t = _expr_translate(raw).pandas_expr
            # Convert df["Col"] → row['Col'] (eval scope uses row dict).
            t = re.sub(r"""df\[(['"])([^'"]+)\1\]""", r"row['\2']", t)
            # Bare field_name not in brackets → value
            t = re.sub(rf"\b{re.escape(field_name)}\b", "value", t)
            return t

        init_py = _xlate(init_el.text)
        cond_py = _xlate(cond_el.text)
        loop_py = _xlate(loop_el.text)
        return MappedTool(
            component_id="generate_rows",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": upstream,
                "mode": "loop_expression",
                "create_column": field_name,
                "init_expression": init_py,
                "condition_expression": cond_py,
                "loop_expression": loop_py,
                "group_name": "alteryx_imported",
            },
            notes=[
                f"GenerateRows on tool {node.tool_id}: loop expansion. "
                f"create_column={field_name!r}, init/cond/loop translated from Alteryx exprs."
            ],
        )

    return MappedTool(
        component_id="generate_rows",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": upstream,
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

    Two inputs: Source (the find/replace lookup pairs) + Targets (the main
    records being edited). Alteryx anchors are literally `Source` and `Targets`.

    Alteryx XML field naming is COUNTERINTUITIVE — verified against a real
    workflow where Source had ['Stop_Words', 'Blank'] and Targets had ['lyric'],
    with FieldFind=lyric and FieldSearch=Stop_Words:
      <FieldFind>X</FieldFind>            — "Find Within Field": column in
                                            TARGETS (main records) to scan.
      <FieldSearch>Y</FieldSearch>        — "Find Value": column in SOURCE
                                            (lookup) providing values to find.
      <ReplaceFoundField>Z</ReplaceFoundField> — column in SOURCE (lookup) with
                                            replacement values.
    The element name `ReplaceFoundField` is canonical; older/custom configs
    occasionally use `FieldReplace` as an alias.
    """
    cfg = node.config
    find_field_el = cfg.find("FieldFind")
    # `ReplaceFoundField` is the canonical Alteryx element; FieldReplace is a
    # secondary name some older or custom configs use. Try both via _find_first.
    replace_field_el = _find_first(cfg, "ReplaceFoundField", "FieldReplace")
    search_field_el = cfg.find("FieldSearch")
    # ReplaceMode controls whether the lookup value overwrites the target
    # column ("Replace") or appends as a new column named for the lookup's
    # value field ("Append"). The XML may also list explicit appended fields
    # under `<ReplaceAppendFields>`; we honor either signal.
    mode_el = cfg.find("ReplaceMode")
    append_fields_el = cfg.find("ReplaceAppendFields")
    _has_append_field = append_fields_el is not None and any(
        _f.attrib.get("field") for _f in append_fields_el.findall("Field")
    )
    is_append_mode = (
        (mode_el is not None and (mode_el.text or "").strip().lower() == "append")
        or _has_append_field
    )
    find_field = (find_field_el.text or "find").strip() if find_field_el is not None and find_field_el.text else "find"
    replace_field = (replace_field_el.text or "replace").strip() if replace_field_el is not None and replace_field_el.text else "replace"
    search_field = (search_field_el.text or "value").strip() if search_field_el is not None and search_field_el.text else "value"
    # Anchor ordering: runner sorts upstreams by dest_anchor alphabetically.
    # Alteryx FindReplace has anchors `F` (Find — the lookup list) and `R`
    # (Records — the main data flow). Sorted alphabetically: F first, R second.
    # So upstreams[0] is the lookup table and upstreams[1] is the records.
    if len(upstreams) >= 2:
        _lookup_key, _main_key = upstreams[0], upstreams[1]
    elif len(upstreams) == 1:
        # Single-input variant — Alteryx allows a self-replace mode. Use the
        # one stream for both slots (component will resolve at runtime).
        _lookup_key = _main_key = upstreams[0]
    else:
        _lookup_key = _main_key = ""
    _attrs: Dict[str, object] = {
        "upstream_asset_key": _main_key,
        "lookup_asset_key": _lookup_key,
        # FieldSearch (Source's value-column) → key column in lookup table
        # FieldFind (Targets' search-within column) → target column in main
        "lookup_key_column": search_field,
        "lookup_value_column": replace_field,
        "target_column": find_field,
        "group_name": "alteryx_imported",
    }
    # In Append mode, the lookup value lands in a NEW column named for the
    # lookup's value field instead of overwriting the target. Set
    # `output_column` to surface it; downstream tools then reference the
    # value-field name directly.
    if is_append_mode:
        _attrs["output_column"] = replace_field
    return MappedTool(
        component_id="find_replace",
        asset_name=_asset_name_for(node),
        attributes=_attrs,
    )


def _map_create_points(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Create Points → `points_from_latlon` registry component.

    Alteryx config: `<Fields fieldX="lon_col" fieldY="lat_col"/>`. Output
    column defaults to `Centroid` (Alteryx's hardcoded convention — not
    configurable in the tool UI). Downstream Alteryx tools reference
    "Centroid", so we emit `geometry_column: "Centroid"` to keep the chain
    intact.
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
            "geometry_column": "Centroid",  # Alteryx CreatePoints default
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
    geom_el = _find_first(cfg, "GeometryField", "Field")
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
    def _to_geom(v):
        if v is None:
            return None
        # Already a shapely geometry?
        if hasattr(v, "geom_type"):
            return v
        # WKT / GeoJSON string fall-through
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return None
        try:
            if s.startswith("{{"):
                import json
                from shapely.geometry import shape
                return shape(json.loads(s))
            from shapely import wkt as _wkt
            return _wkt.loads(s)
        except Exception:
            return None
    def _explode(geom):
        geom = _to_geom(geom)
        if geom is None or getattr(geom, "is_empty", False):
            return []
        try:
            coords = list(geom.coords)
        except (NotImplementedError, AttributeError):
            # Polygon: walk the exterior ring.
            coords = list(getattr(geom.exterior, "coords", []))
        return coords
    # Fall back to any geometry-typed col if the configured one is missing
    # (upstream may emit Centroid / SpatialObj / etc. instead of {geom_col!r}).
    _geom_col = {geom_col!r}
    if _geom_col not in df.columns:
        for _alt in ("geometry", "Centroid", "SpatialObj", "geom", "shape"):
            if _alt in df.columns:
                _geom_col = _alt
                break
    if _geom_col not in df.columns:
        return df  # nothing to explode; downstream stays consistent
    df["_coords"] = df[_geom_col].apply(_explode)
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
    # Alteryx Predictive Tools macros — route to dedicated registry components.
    # date_column / value_column defaults are placeholders; the user needs
    # to edit defs.yaml to match the upstream column names.
    "arima.yxmc": {
        "component_id": "arima_forecast",
        "attributes": {"forecast_periods": 12, "date_column": "date", "value_column": "value"},
    },
    "ets.yxmc": {
        "component_id": "ets_forecast",
        "attributes": {"forecast_periods": 12, "date_column": "date", "value_column": "value"},
    },
    "ts_forecast.yxmc": {
        "component_id": "arima_forecast",
        "attributes": {"forecast_periods": 12, "date_column": "date", "value_column": "value"},
    },
    "timeseriesfiller.yxmc": {
        "component_id": "select_columns",  # passthrough (fills gaps inline)
        "attributes": {},
    },
    "imputation_v2.yxmc": {
        "component_id": "data_cleansing",
        "attributes": {"null_handling": "fill", "null_fill_value": "", "trim_whitespace": False},
    },
    "field_summary_report.yxmc": {
        "component_id": "select_columns",  # passthrough; profiling is metadata
        "attributes": {},
    },
    "oversample_field.yxmc": {
        "component_id": "select_columns",  # passthrough; oversample is statsmodels-shaped
        "attributes": {},
    },
    "histogram.yxmc": {
        "component_id": "select_columns",  # passthrough; histogram is a visual
        "attributes": {},
    },
    # CReW community macros — common utilities; map best-effort to passthrough
    # so the chain doesn't break.
    "crew_expectequal.yxmc": {
        "component_id": "select_columns",
        "attributes": {},
    },
    "crew_ensurefields.yxmc": {
        "component_id": "select_columns",
        "attributes": {},
    },
    # Generic helper macros.
    "selectrecords.yxmc": {
        "component_id": "sample",  # SelectRecords picks first N — sample fits
        "attributes": {"method": "head", "sample_size": 10},
    },
    "countrecords.yxmc": {
        "component_id": "summarize",
        "attributes": {"group_by": [], "aggregations": {}},  # whole-frame size
    },
    "weightedavg.yxmc": {
        "component_id": "weighted_average",
        # Required attributes pulled from the parent <Configuration> at map
        # time — see _map_alteryx_macro special-case below.
        "attributes": {},
    },
    "randomrecords.yxmc": {
        "component_id": "sample",
        # Number/NNumber/Percent/NPercent/Seed pulled from <Configuration>
        # at map time — see _map_alteryx_macro.
        "attributes": {"method": "random"},
    },
    "frequency.yxmc": {
        "component_id": "summarize",
        # Whole-frame freq table; user typically wants group_by=<col>, agg=count
        # — left as a placeholder so dg check doesn't error and user can fill in.
        "attributes": {"group_by": [], "aggregations": {}},
    },
    "ifs.yxmc": {
        # Alteryx IFS = cascading if/elif. Closest registry equivalent is the
        # `formula` component (which already accepts np.select-style
        # expressions). Emit a stub the user fills in.
        "component_id": "formula",
        "attributes": {"expressions": {}},
    },
    # Alteryx Predictive Tools — stock macros that wrap proprietary algos.
    # Each maps to a dedicated registry component we already ship. Most
    # require feature_columns / target_column which the user must set in
    # the emitted defs.yaml (we can't reliably auto-derive from the
    # macro call config without parsing field-selector controls).
    "find_nearest_neighbors.yxmc": {
        "component_id": "nearest_neighbors",
        "attributes": {"feature_columns": ["TODO_set_feature_columns"]},
    },
    "forest_model.yxmc": {
        "component_id": "random_forest_model",
        "attributes": {
            "target_column": "TODO_set_target_column",
            "feature_columns": ["TODO_set_feature_columns"],
        },
    },
    "decision_tree.yxmc": {
        "component_id": "decision_tree_model",
        "attributes": {
            "target_column": "TODO_set_target_column",
            "feature_columns": ["TODO_set_feature_columns"],
        },
    },
    "linear_regression.yxmc": {
        "component_id": "linear_regression_model",
        "attributes": {
            "target_column": "TODO_set_target_column",
            "feature_columns": ["TODO_set_feature_columns"],
        },
    },
    "logistic_regression.yxmc": {
        "component_id": "logistic_regression_model",
        "attributes": {
            "target_column": "TODO_set_target_column",
            "feature_columns": ["TODO_set_feature_columns"],
        },
    },
    "k-centroids_cluster_analysis.yxmc": {
        "component_id": "k_means_clustering",
        "attributes": {"feature_columns": ["TODO_set_feature_columns"]},
    },
    "k-centroids_diagnostics.yxmc": {
        "component_id": "k_means_clustering",
        "attributes": {"feature_columns": ["TODO_set_feature_columns"]},
    },
    "append_cluster.yxmc": {
        "component_id": "append_cluster",
        "attributes": {"feature_columns": ["TODO_set_feature_columns"]},
    },
    "mb_rules.yxmc": {
        "component_id": "market_basket_rules",
        "attributes": {
            "basket_column": "TODO_set_basket_column",
            "item_column": "TODO_set_item_column",
        },
    },
    "mb_inspect.yxmc": {
        "component_id": "market_basket_rules",
        "attributes": {
            "basket_column": "TODO_set_basket_column",
            "item_column": "TODO_set_item_column",
        },
    },
    "model comparison.yxmc": {
        "component_id": "model_comparison",
        "attributes": {
            "target_column": "TODO_set_target_column",
            "feature_columns": ["TODO_set_feature_columns"],
        },
    },
    "create_samples.yxmc": {
        "component_id": "create_samples",
        # Default Alteryx Create_Samples is 60/20/20 train/valid/test;
        # registry default is 80/20 train/test — user adjusts.
        "attributes": {"test_size": 0.2},
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
    # Strip any path prefix (e.g. `Predictive Tools\ARIMA.yxmc` → `ARIMA.yxmc`)
    # so the lookup matches by bare filename only. Windows backslash + POSIX
    # forward slash both normalized.
    _bare = macro_basename.replace("\\", "/").split("/")[-1]
    stock = (
        _STOCK_MACRO_COMPONENTS.get(macro_basename.lower())
        or _STOCK_MACRO_COMPONENTS.get(_bare.lower())
    )
    if stock:
        component_id = stock["component_id"]
        defaults = dict(stock.get("attributes") or {})  # type: ignore[arg-type]
        defaults.setdefault("upstream_asset_key", _single_upstream(upstreams))
        defaults.setdefault("group_name", "alteryx_imported")
        # Stock macros that need parent-Configuration-driven attributes:
        # SelectRecords.yxmc reads `<Value name="Ranges">3-21</Value>` for the
        # 1-indexed row range. Map to sample(method='range', range_start, range_end).
        # Default to head if Ranges isn't a simple `<start>-<end>` (e.g. `1+`
        # which means "from row 1 to end" — keep all rows).
        if component_id == "sample" and _bare.lower() == "selectrecords.yxmc":
            cfg = node.config
            _ranges = None
            for _v in cfg.findall("Value"):
                if _v.attrib.get("name") == "Ranges" and _v.text:
                    _ranges = _v.text.strip()
                    break
            if _ranges:
                import re as _re
                _m = _re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", _ranges)
                if _m:
                    defaults.pop("sample_size", None)
                    defaults["method"] = "range"
                    defaults["range_start"] = int(_m.group(1))
                    defaults["range_end"] = int(_m.group(2))
        # --- Helpers for Predictive Tools macro config extraction ---
        def _cfg_value(name: str) -> str | None:
            """Read `<Value name="X">text</Value>` from the macro CALL config."""
            for _v in node.config.findall("Value"):
                if _v.attrib.get("name") == name and _v.text is not None:
                    return _v.text.strip()
            return None

        def _parse_plus_list(s: str | None) -> list[str]:
            """Split `'A + B + C'` (or `'A+B+C'`) → ['A', 'B', 'C']."""
            if not s:
                return []
            return [t.strip() for t in s.split("+") if t.strip()]

        def _parse_select_fields(s: str | None) -> list[str]:
            """Parse `'col1=True,col2=False,col3=True'` → ['col1', 'col3']."""
            if not s:
                return []
            result = []
            for token in s.split(","):
                if "=" in token:
                    k, v = token.split("=", 1)
                    if v.strip().lower() == "true":
                        result.append(k.strip())
            return result

        # --- Find_Nearest_Neighbors: select.fields / the_k / standardize / algo.*
        if component_id == "nearest_neighbors" and _bare.lower() == "find_nearest_neighbors.yxmc":
            feats = _parse_select_fields(_cfg_value("select.fields"))
            if feats:
                defaults["feature_columns"] = feats
            k_val = _cfg_value("the_k")
            if k_val and k_val.isdigit():
                defaults["n_neighbors"] = int(k_val)
            if (_cfg_value("standardize") or "").lower() == "true":
                defaults["normalize"] = True
            elif (_cfg_value("standardize") or "").lower() == "false":
                defaults["normalize"] = False
            # algorithm: kd_tree | cover_tree | VR | CR | brute. Map to
            # nearest_neighbors component's enum: auto|ball_tree|kd_tree|brute.
            for _algo, _mapped in (
                ("algo.kd_tree", "kd_tree"),
                ("algo.cover_tree", "ball_tree"),  # closest sklearn analogue
                ("algo.brute", "brute"),
            ):
                if (_cfg_value(_algo) or "").lower() == "true":
                    defaults["algorithm"] = _mapped
                    break

        # --- Forest_Model / Decision_Tree / Linear/Logistic_Regression:
        #     Y Var (target) + X Vars (`+ `-separated features)
        if component_id in (
            "random_forest_model", "decision_tree_model",
            "linear_regression_model", "logistic_regression_model",
        ):
            y_var = _cfg_value("Y Var")
            x_vars = _parse_plus_list(_cfg_value("X Vars"))
            if y_var:
                defaults["target_column"] = y_var
            if x_vars:
                defaults["feature_columns"] = x_vars
            if component_id == "random_forest_model":
                n_trees = _cfg_value("Num Trees")
                if n_trees and n_trees.isdigit():
                    defaults["n_estimators"] = int(n_trees)
                max_d = _cfg_value("max.depth")
                if max_d and max_d.isdigit():
                    defaults["max_depth"] = int(max_d)

        # --- Model Comparison: same Y Var / X Vars convention
        if component_id == "model_comparison":
            y_var = _cfg_value("Y Var")
            x_vars = _parse_plus_list(_cfg_value("X Vars"))
            if y_var:
                defaults["target_column"] = y_var
            if x_vars:
                defaults["feature_columns"] = x_vars

        # --- K-Centroids_Cluster_Analysis / K-Centroids_Diagnostics:
        #     `the fields` (`+ ` features) + `num clus` + normalization
        if component_id == "k_means_clustering":
            feats = _parse_plus_list(_cfg_value("the fields"))
            if feats:
                defaults["feature_columns"] = feats
            n_clus = _cfg_value("num clus")
            if n_clus and n_clus.isdigit():
                defaults["n_clusters"] = int(n_clus)
            if (_cfg_value("standardize") or "").lower() == "true":
                defaults["normalize"] = True
            elif (_cfg_value("standardize") or "").lower() == "false":
                defaults["normalize"] = False

        # --- MB_Rules / MB_Inspect:
        #     basket-id field + item field. Alteryx names vary; cover the
        #     two common patterns (`Items` / `IDs` and `item` / `basket`).
        if component_id == "market_basket_rules":
            for _basket_key in ("IDs", "basket", "ID"):
                v = _cfg_value(_basket_key)
                if v:
                    defaults["basket_column"] = v
                    break
            for _item_key in ("Items", "item", "Item"):
                v = _cfg_value(_item_key)
                if v:
                    defaults["item_column"] = v
                    break

        # --- Create_Samples: estimation% / validation% / holdout%
        if component_id == "create_samples":
            est = _cfg_value("Estimation Percent")
            val = _cfg_value("Validation Percent")
            hold = _cfg_value("Holdout Percent")
            # Alteryx default is 50/30/20 train/valid/test. Convert to
            # registry's test_size + validation_size.
            try:
                est_f = float(est) if est else None
                val_f = float(val) if val else None
                hold_f = float(hold) if hold else None
            except (ValueError, TypeError):
                est_f = val_f = hold_f = None
            if hold_f is not None:
                defaults["test_size"] = hold_f / 100.0
            if val_f is not None:
                defaults["validation_size"] = val_f / 100.0

        if component_id == "sample" and _bare.lower() == "randomrecords.yxmc":
            # RandomRecords.yxmc config:
            #   <Value name="Number">True</Value>      ← absolute-count mode
            #   <Value name="NNumber">5000</Value>     ← record count
            #   <Value name="Percent">False</Value>    ← percent mode
            #   <Value name="NPercent">99</Value>      ← percentage value
            #   <Value name="Deterministic">False</Value>
            #   <Value name="Seed">458676342</Value>
            cfg = node.config
            _mode_num = _mode_pct = False
            _n = _pct = _seed = None
            _deterministic = False
            for _v in cfg.findall("Value"):
                _name = _v.attrib.get("name", "")
                _text = (_v.text or "").strip()
                if _name == "Number" and _text.lower() == "true":
                    _mode_num = True
                elif _name == "Percent" and _text.lower() == "true":
                    _mode_pct = True
                elif _name == "NNumber":
                    try:
                        _n = int(_text)
                    except (ValueError, TypeError):
                        pass
                elif _name == "NPercent":
                    try:
                        _pct = float(_text)
                    except (ValueError, TypeError):
                        pass
                elif _name == "Deterministic" and _text.lower() == "true":
                    _deterministic = True
                elif _name == "Seed":
                    try:
                        _seed = int(_text)
                    except (ValueError, TypeError):
                        pass
            if _mode_num and _n is not None:
                defaults["sample_size"] = _n
            elif _mode_pct and _pct is not None:
                defaults["frac"] = _pct / 100.0
            if _deterministic and _seed is not None:
                defaults["random_state"] = _seed
        if component_id == "weighted_average":
            cfg = node.config
            _val = _weight = _group = _out = None
            for _v in cfg.findall("Value"):
                _name = _v.attrib.get("name", "")
                _text = (_v.text or "").strip()
                if _name == "Value":
                    _val = _text
                elif _name == "Weight":
                    _weight = _text
                elif _name == "GroupFields":
                    _group = _text
                elif _name == "OutputFieldName":
                    _out = _text
            if _val:
                defaults["value_column"] = _val
            if _weight:
                defaults["weight_column"] = _weight
            if _group:
                # GroupFields is comma-separated in Alteryx XML.
                defaults["group_by"] = [g.strip() for g in _group.split(",") if g.strip()]
            if _out:
                defaults["output_column"] = _out
        # Flag fields the user must set before materialization.
        _todo_fields = [
            k for k, v in defaults.items()
            if (isinstance(v, str) and v.startswith("TODO_set_"))
            or (isinstance(v, list) and any(isinstance(x, str) and x.startswith("TODO_set_") for x in v))
        ]
        _note = (
            f"Macro {macro_basename!r} routed to stock registry "
            f"component `{component_id}` with sensible defaults. "
            "Tweak the emitted defs.yaml if your usage differs."
        )
        if _todo_fields:
            _note += (
                f" REQUIRED before materialization: replace `TODO_set_*` "
                f"placeholders in field(s) {_todo_fields} with the actual "
                "column names from your upstream DataFrame. (Schema check "
                "passes either way — the failure surfaces at runtime when "
                "the component looks up the placeholder column name.)"
            )
        return MappedTool(
            component_id=str(component_id),
            asset_name=_asset_name_for(node),
            attributes=defaults,
            notes=[_note],
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


def _predictive_y_x_vars(node: AlteryxNode) -> tuple:
    """Extract Y Var (target) + X Vars (features, comma-separated) from
    Alteryx Predictive XML config. Returns (target_col, feature_cols)."""
    cfg = node.config
    target = ""
    features: List[str] = []
    for v in cfg.findall("Value"):
        name = v.attrib.get("name", "")
        text = (v.text or "").strip() if v.text else ""
        if name in ("Y Var", "Target", "Target_Variable", "target"):
            target = text
        elif name in ("X Vars", "Predictors", "X_Variables", "features"):
            features = [c.strip() for c in text.split(",") if c.strip()]
    return target, features


def _make_predictive_mapper(component_id: str, task_type: Optional[str] = None):
    """Build a mapper function for an Alteryx predictive plugin that targets
    a sklearn-backed registry component with the standard predictive shape
    (target_column + feature_columns + optional task_type)."""
    def _mapper(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
        target, features = _predictive_y_x_vars(node)
        attrs: Dict[str, object] = {
            "upstream_asset_key": _single_upstream(upstreams),
            "target_column": target,
            "feature_columns": features,
            "group_name": "alteryx_imported",
        }
        if task_type is not None:
            attrs["task_type"] = task_type
        return MappedTool(
            component_id=component_id,
            asset_name=_asset_name_for(node),
            attributes=attrs,
            notes=[
                f"Predictive {node.plugin}: Alteryx-side hyperparameters "
                "(regularization, CV folds, etc.) NOT translated — registry "
                "component defaults apply. Tune in defs.yaml if needed."
            ],
        )
    _mapper.__name__ = f"_map_predictive_{component_id}"
    return _mapper


def _map_score(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Score → `model_score` registry component.

    Score has TWO upstreams: a fitted model + a DataFrame. The DataFrame
    upstream is the one we wire as `upstream_asset_key`. The model is
    expected at `model_path` (a local pickle/joblib file) — Alteryx
    bundles it as a .yxdb model artifact, which doesn't translate; the
    user needs to point at a serialized sklearn / statsmodels model file.
    """
    cfg = node.config
    out_col = "predicted"
    for v in cfg.findall("Value"):
        if v.attrib.get("name") in ("Output Field", "Score Field"):
            out_col = (v.text or "predicted").strip() if v.text else "predicted"
    # Find features the upstream tools predicted on. Best effort.
    return MappedTool(
        component_id="model_score",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": upstreams[1] if len(upstreams) > 1 else _single_upstream(upstreams),
            "model_path": "TODO_set_path_to_serialized_sklearn_or_statsmodels_model",
            "feature_columns": [],
            "output_column": out_col,
            "group_name": "alteryx_imported",
        },
        notes=[
            f"Score on tool {node.tool_id}: set `model_path` to a serialized "
            "sklearn (joblib) or statsmodels (.save()) model file. Alteryx's "
            "in-flow model passing doesn't translate — the upstream predictive "
            "tool's `model_path` (if set) is the source path."
        ],
    )


def _map_select(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx Select tool — keep/rename/reorder columns.

    Honors Alteryx's `*Unknown` semantic: when present + selected, it means
    "ALSO keep all other (unlisted) upstream columns". In that case we emit
    `columns: None` (the select_columns component interprets None as
    keep-all-upstream-columns), apply renames, and surface explicit drops
    separately if the user deselected specific fields.
    """
    keep: List[str] = []
    rename_map: Dict[str, str] = {}
    drop: List[str] = []
    unknown_selected = False  # True means "keep all OTHER upstream cols too"

    sf_el = _find_first(node.config, "SelectFields", "Fields")
    if sf_el is not None:
        for f in sf_el.findall("SelectField") + sf_el.findall("Field"):
            fn = f.attrib.get("field")
            selected = f.attrib.get("selected", "True").lower() != "false"
            rename = f.attrib.get("rename")
            if not fn:
                continue
            if fn == "*Unknown":
                if selected:
                    unknown_selected = True
                continue
            if selected:
                # Always append the ORIGINAL field name to `columns:` — the
                # registry's select_columns applies `columns:` (keep filter)
                # BEFORE `rename:`, so the original name must be in the
                # keep-list for the rename to actually fire on it.
                keep.append(fn)
                if rename and rename != fn:
                    rename_map[fn] = rename
            else:
                drop.append(fn)

    attrs: Dict[str, object] = {
        "upstream_asset_key": _single_upstream(upstreams),
        "group_name": "alteryx_imported",
    }
    if unknown_selected:
        # `*Unknown` checked: keep ALL upstream columns + apply renames + drops.
        # `columns: None` tells select_columns to keep everything; `drop_columns:`
        # is the registry component's field name for explicit drops.
        attrs["rename"] = rename_map or None
        if drop:
            attrs["drop_columns"] = drop
        # Don't set `columns:` — None default keeps all.
    else:
        # No wildcard: only keep the explicitly-selected fields.
        attrs["columns"] = keep
        attrs["rename"] = rename_map or None
    return MappedTool(
        component_id="select_columns",
        asset_name=_asset_name_for(node),
        attributes=attrs,
    )


# Translation table for Alteryx MultiRowFormula's [Row±N:Col] references
# → pandas equivalents. Each side of the table is the "Col" name.
_ROW_REF_RE = re.compile(r"\[Row([+\-])(\d+):([^\]]+)\]")
# Bare [Col] (no Row prefix) refers to this row's value of Col, including
# the CreateField target itself (current row's value of the new column).
_BARE_FIELD_RE = re.compile(r"\[([^\[\]]+)\]")


def _translate_mrf_expression(expr: str, group_cols: List[str]) -> str:
    """Translate an Alteryx MultiRowFormula expression to a pandas one.

    [Row-1:X] / [Row+1:X] → df['X'].shift(1) / shift(-1)
    [Row-N:X] / [Row+N:X] → shift(N) / shift(-N)
    [X]                   → df['X']

    When group_cols is non-empty, every shift becomes
    df.groupby(group_cols)['X'].shift(...) so windows respect the partition.
    """
    grp = f"df.groupby({group_cols!r}, dropna=False)" if group_cols else None

    # Single regex pass: match either [Row±N:Col] (with capture groups
    # 1,2,3) or bare [Col] (group 4). Operating on the raw string in one
    # sweep avoids re-matching the brackets we just emitted.
    _MRF_TOKEN_RE = re.compile(
        r"\[Row([+\-])(\d+):([^\]]+)\]"   # rowref groups 1, 2, 3
        r"|\[([^\[\]]+)\]"                # bare col group 4
    )

    def _sub(m):
        if m.group(1) is not None:
            sign, n, col = m.group(1), int(m.group(2)), m.group(3)
            shift_n = -n if sign == "+" else n
            if grp:
                return f"{grp}[{col!r}].shift({shift_n})"
            return f"df[{col!r}].shift({shift_n})"
        col = m.group(4)
        return f"df[{col!r}]"

    return _MRF_TOKEN_RE.sub(_sub, expr)


_PURE_ROW_REF_RE = re.compile(r"^\s*\[Row([+\-])(\d+):([^\]]+)\]\s*$")


def _detect_forward_fill(raw_expr: str, out_col: str) -> Optional[str]:
    r"""Detect the classic Alteryx forward-fill pattern and return the
    target column to apply pandas `.ffill()` to.

    Both Alteryx idioms below mean "if X is null, use the previous row's
    X, else keep X" — which in pandas is just `df['X'].ffill()`. The
    naive `np.where(isna, shift(1), X)` translation is silently WRONG
    for consecutive nulls (shift returns the original null, not the
    forward-filled value), so detecting + rewriting matters.

      IF IsNull([X]) THEN [Row-1:X] ELSE [X] ENDIF
      IIF(IsNull([X]), [Row-1:X], [X])

    Returns the column name `X` if matched, else None. The caller emits
    `formula` with `expressions: {out_col: df['X'].ffill()}`.
    """
    # Strip whitespace + collapse internal whitespace for simpler matching.
    s = re.sub(r"\s+", " ", raw_expr.strip())
    patterns = [
        # IF IsNull([X]) THEN [Row-1:X] ELSE [X] ENDIF — case-insensitive
        # since real workflows mix "IsNull" / "isnull" / "ISNULL".
        re.compile(
            r"^IF\s+(?:IsNull|isnull)\s*\(\s*\[([^\]]+)\]\s*\)\s+"
            r"THEN\s+\[Row-1:\1\]\s+"
            r"ELSE\s+\[\1\]\s+ENDIF$",
            flags=re.IGNORECASE,
        ),
        # IIF(IsNull([X]), [Row-1:X], [X])
        re.compile(
            r"^IIF\s*\(\s*(?:IsNull|isnull)\s*\(\s*\[([^\]]+)\]\s*\)\s*,\s*"
            r"\[Row-1:\1\]\s*,\s*"
            r"\[\1\]\s*\)$",
            flags=re.IGNORECASE,
        ),
    ]
    for pat in patterns:
        m = pat.match(s)
        if m:
            return m.group(1)
    return None


def _has_self_reference(raw_expr: str, out_col: str) -> bool:
    """True if the expression references `[Row±N:out_col]` — i.e. depends
    on the previously-COMPUTED value of the output column.

    Pandas `.shift(N)` returns the ORIGINAL previous value, not the
    computed one from the prior row. Alteryx evaluates row-by-row so
    its semantics are iterative — when the user writes
    `IIF(IsNull([Year]), [Row-1:Year] + 1, [Year])` and Year is null
    for 3 consecutive rows, Alteryx fills with prev+1, prev+2, prev+3.
    Pandas only fills with the original null.

    Detection is conservative — we only flag when the output column is
    actually referenced in a row-shift; the caller emits a translation
    AND attaches a warning note so the user knows to validate / replace
    with a Python loop if needed.
    """
    for m in _ROW_REF_RE.finditer(raw_expr):
        ref_col = m.group(3).strip()
        if ref_col == out_col:
            return True
    return False


def _map_multi_row_formula(node: AlteryxNode, upstreams: List[str]) -> MappedTool:
    """Alteryx MultiRowFormula → window_calculation when the expression is a
    pure [Row±N:Col] reference, else formula component with a translated
    pandas .shift() expression.

    Reads:
      <UpdateField value="True|False"/>
      <UpdateField_Name>Col</UpdateField_Name>      (when updating)
      <CreateField_Name>NewCol</CreateField_Name>   (when creating)
      <Expression>[Row-1:Col]</Expression>
      <GroupByFields><Field field="X"/></GroupByFields>
    """
    cfg = node.config
    update_el = cfg.find("UpdateField")
    is_update = update_el is not None and update_el.attrib.get("value", "False").lower() == "true"
    if is_update:
        upd_name_el = cfg.find("UpdateField_Name")
        out_col = upd_name_el.text if upd_name_el is not None and upd_name_el.text else None
    else:
        crt_name_el = cfg.find("CreateField_Name")
        out_col = crt_name_el.text if crt_name_el is not None and crt_name_el.text else None
    expr_el = cfg.find("Expression")
    raw_expr = (expr_el.text if expr_el is not None and expr_el.text else "").strip()

    group_cols: List[str] = []
    gbf = cfg.find("GroupByFields")
    if gbf is not None:
        for f in gbf.findall("Field"):
            fn = f.attrib.get("field")
            if fn:
                group_cols.append(fn)

    if not out_col or not raw_expr:
        return MappedTool(
            component_id="formula",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "expressions": {},
                "group_name": "alteryx_imported",
            },
            notes=[f"MultiRowFormula on tool {node.tool_id}: no output column or expression — emitted empty."],
        )

    # `[_CurrentField_]` is an Alteryx pseudo-token meaning "the column
    # being updated" (only present when UpdateField=True). Translating
    # it literally yields `df['_CurrentField_']` which fails at runtime.
    # Substitute with the actual output column before the rest of the
    # pipeline sees it.
    if "[_CurrentField_]" in raw_expr and is_update and out_col:
        raw_expr = raw_expr.replace("[_CurrentField_]", f"[{out_col}]")

    # Forward-fill shortcut: detect the canonical
    # `IF IsNull([X]) THEN [Row-1:X] ELSE [X] ENDIF` / IIF form and emit
    # `df['X'].ffill()` — semantically correct for consecutive nulls,
    # which the naive np.where+shift translation gets wrong.
    ff_col = _detect_forward_fill(raw_expr, out_col)
    if ff_col:
        ff_expr = (
            f"df.groupby({group_cols!r}, dropna=False)[{ff_col!r}].ffill()"
            if group_cols
            else f"df[{ff_col!r}].ffill()"
        )
        return MappedTool(
            component_id="formula",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "expressions": {out_col: ff_expr},
                "group_name": "alteryx_imported",
            },
            notes=[
                f"MultiRowFormula on tool {node.tool_id}: detected forward-fill "
                f"pattern (IF IsNull THEN [Row-1:X] ELSE [X]) — emitted "
                f"`{ff_col}.ffill()` which correctly handles consecutive "
                "nulls (unlike the naive shift(1)+np.where translation)."
            ],
        )

    # Pure single-token [Row±N:Col]? Map to window_calculation.lag/lead.
    pure = _PURE_ROW_REF_RE.match(raw_expr)
    if pure:
        sign, n, col = pure.group(1), int(pure.group(2)), pure.group(3)
        func = "lag" if sign == "-" else "lead"
        return MappedTool(
            component_id="window_calculation",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "partition_by": group_cols or None,
                "operations": [{
                    "output": out_col,
                    "func": func,
                    "column": col,
                    "periods": n,
                }],
                "group_name": "alteryx_imported",
            },
            notes=[
                f"MultiRowFormula on tool {node.tool_id}: pure [Row{sign}{n}:{col}] mapped "
                f"to window_calculation.{func}(periods={n})"
                + (f" partitioned by {group_cols}" if group_cols else "")
                + "."
            ],
        )

    # Compound expression — run through expr_translator first (which handles
    # IF/THEN/ELSE → np.where, operators, etc.), then post-process the
    # df["Row±N:Col"] patterns it emits into proper .shift() calls.
    from .expr_translator import translate as _expr_translate
    pre = _expr_translate(raw_expr)
    translated = pre.pandas_expr
    grp = f"df.groupby({group_cols!r}, dropna=False)" if group_cols else None

    def _shift_sub(m):
        sign, n, col = m.group(1), int(m.group(2)), m.group(3)
        shift_n = -n if sign == "+" else n
        if grp:
            return f"{grp}[{col!r}].shift({shift_n})"
        return f"df[{col!r}].shift({shift_n})"

    _DF_ROW_RE = re.compile(r"""df\[["']Row([+\-])(\d+):([^"']+)["']\]""")
    translated = _DF_ROW_RE.sub(_shift_sub, translated)

    # If anything remains as bare [Row±N:Col] (translator left it through),
    # convert directly too.
    translated = _ROW_REF_RE.sub(
        lambda m: (
            (f"{grp}[" if grp else "df[")
            + repr(m.group(3))
            + f"].shift({-int(m.group(2)) if m.group(1) == '+' else int(m.group(2))})"
        ),
        translated,
    )

    _notes = [
        f"MultiRowFormula on tool {node.tool_id}: compound expression — translated "
        f"IF/THEN/ELSE + [Row±N:Col] to np.where + df['Col'].shift(N)"
        + (f" grouped by {group_cols}" if group_cols else "")
        + "."
    ]
    if _has_self_reference(raw_expr, out_col):
        _notes.append(
            f"WARNING: expression references [Row±N:{out_col}] — the OUTPUT "
            "column itself. Alteryx evaluates row-by-row so the reference "
            "sees the COMPUTED value from the prior row; pandas `.shift(N)` "
            "returns the ORIGINAL value. For consecutive computations the "
            "result will diverge. If you need true iterative semantics (e.g. "
            "cumulative state, gap-fill with increment), replace this "
            "emitted formula with an inline `@dg.asset` that loops row-by-row "
            "or uses `.cumsum()` / `.cummax()` / a custom accumulator."
        )
    return MappedTool(
        component_id="formula",
        asset_name=_asset_name_for(node),
        attributes={
            "upstream_asset_key": _single_upstream(upstreams),
            "expressions": {out_col: translated},
            "group_name": "alteryx_imported",
        },
        notes=_notes,
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
    "AlteryxBasePluginsGui.BlobConvert.BlobConvert": (
        # BlobConvert → blob_convert (new registry component).
        # Alteryx config: <Field>X</Field>, <ConversionMode>StringToBlob/...</ConversionMode>.
        lambda node, upstreams: (lambda field_el, mode_el: MappedTool(
            component_id="blob_convert",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "input_column": (field_el.text or "blob").strip() if field_el is not None and field_el.text else "blob",
                # Alteryx mode names → blob_convert ops
                "operation": {
                    "stringtoblob": "to_bytes",
                    "blobtostring": "to_text",
                    "base64encode": "to_base64",
                    "base64decode": "from_base64",
                    "blobtohex": "to_hex",
                    "hextoblob": "from_hex",
                }.get(
                    (mode_el.text or "").strip().lower() if mode_el is not None and mode_el.text else "",
                    "to_base64",  # safe default
                ),
                "group_name": "alteryx_imported",
            },
        ))(node.config.find("Field"), node.config.find("ConversionMode"))
    ),
    # Portfolio Composer (Image/Render) → pdf_report.
    "PortfolioPluginsGui.ComposerRender.PortfolioComposerRender": (
        lambda node, upstreams: MappedTool(
            component_id="pdf_report",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "file_path": f"./reports/{_asset_name_for(node)}.pdf",
                "title": (node.annotation or _asset_name_for(node)).strip(),
                "template": "table",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Portfolio Composer Render on tool {node.tool_id}: emitted as "
                "pdf_report (tabular DataFrame → PDF). For richer Alteryx-style "
                "layouts (sections / charts / branding), switch template to "
                "'template_html' and supply an html_template in defs.yaml."
            ],
        )
    ),
    "PortfolioPluginsGui.ComposerImage.PortfolioComposerImage": (
        # Composer Image renders an image section that the downstream Composer
        # Render bundles into a PDF. Two shapes:
        #   - WITH upstream: passthrough (so downstream Union/Join still sees a
        #     DataFrame, not a PDF path).
        #   - WITHOUT upstream: standalone image source. Emit a tiny 1-row
        #     placeholder DataFrame so downstream Union has something to concat.
        lambda node, upstreams: (
            MappedTool(
                component_id="select_columns",
                asset_name=_asset_name_for(node),
                attributes={
                    "upstream_asset_key": _single_upstream(upstreams),
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Image on tool {node.tool_id}: emitted as "
                    "passthrough — image-section rendering has no Dagster-native "
                    "equivalent mid-chain. Embed via the terminal pdf_report's "
                    "template='template_html' with <img> tags."
                ],
            ) if upstreams else MappedTool(
                component_id="inline_dataframe",
                asset_name=_asset_name_for(node),
                attributes={
                    "asset_name": _asset_name_for(node),
                    "columns": ["Image"],
                    "rows": [[""]],
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Image on tool {node.tool_id}: no upstream — "
                    "emitted as a 1-row placeholder DataFrame so downstream report "
                    "tools (Render / Union) still have a DataFrame to consume."
                ],
            )
        )
    ),
    "PortfolioPluginsGui.ComposerTable.PortfolioComposerTable": (
        # Portfolio Composer Table is typically a mid-chain styling step in a
        # multi-section report — the data still flows downstream into a final
        # Composer Render. Emit as a passthrough so downstream Joins/Unions
        # consume the upstream DataFrame; the table-styling is captured as
        # a MIGRATION.md note for the user to re-apply at the final render.
        # If the source workflow wired the Composer Table as a leaf
        # source-style step with no inbound anchor, fall back to a 1-row
        # inline_dataframe so the asset still materializes.
        lambda node, upstreams: (
            MappedTool(
                component_id="inline_dataframe",
                asset_name=_asset_name_for(node),
                attributes={
                    "columns": ["placeholder"],
                    "rows": [[""]],
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Table on tool {node.tool_id}: no upstream — "
                    "emitted as a 1-row inline placeholder. Wire the table render into "
                    "the terminal pdf_report with template='template_html'."
                ],
            )
            if not upstreams else
            MappedTool(
                component_id="select_columns",
                asset_name=_asset_name_for(node),
                attributes={
                    "upstream_asset_key": _single_upstream(upstreams),
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Table on tool {node.tool_id}: emitted as "
                    "passthrough — table-styling has no Dagster-native equivalent "
                    "mid-chain. Wire the table render into the terminal pdf_report "
                    "(set template='template_html' with a custom HTML template for "
                    "styled tables)."
                ],
            )
        )
    ),
    "AlteryxBasePluginsGui.DynamicSelect.DynamicSelect": (
        # DynamicSelect picks columns by type or by a formula on the column
        # NAME. We can't deterministically pre-compute which cols match
        # (depends on runtime dtypes), so emit a passthrough select_columns
        # with the original Mode/Expression in a note. User refines in YAML.
        lambda node, upstreams: (lambda mode_el, expr_el, types_el: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"DynamicSelect on tool {node.tool_id}: Alteryx mode="
                f"{((mode_el.text or '').strip() if mode_el is not None and mode_el.text else 'unknown')!r}, "
                f"expression={((expr_el.text or '').strip() if expr_el is not None and expr_el.text else '')!r}, "
                f"field_types={((types_el.text or '').strip() if types_el is not None and types_el.text else '')!r}. "
                "Emitted as passthrough select_columns — set `columns:` / "
                "`drop_columns:` in defs.yaml once you know the upstream's "
                "runtime dtype / column-name set."
            ],
        ))(node.config.find("Mode"), node.config.find("Expression"), node.config.find("FieldTypes"))
    ),
    "PlotlyCharting": (
        # Plotly Charting tool — emits a chart spec as metadata, passes
        # the upstream DataFrame through. Downstream tools still get the data.
        lambda node, upstreams: MappedTool(
            component_id="select_columns",  # passthrough
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
                "description": (
                    f"Alteryx Plotly Chart (tool {node.tool_id}): "
                    "passthrough today; chart spec recorded in MIGRATION.md note. "
                    "For interactive charting in Dagster, build a Plotly figure in "
                    "a separate asset that consumes this one + emit it as "
                    "MetadataValue.json(fig.to_json())."
                ),
            },
            notes=[
                f"PlotlyCharting on tool {node.tool_id}: passthrough emitted "
                "(downstream tools see the upstream DataFrame unchanged). "
                "Move the Plotly trace config into a Dagster asset that emits "
                "MetadataValue.json / MetadataValue.md for inline rendering."
            ],
        )
    ),
    "AlteryxBasePluginsGui.RTool.RTool": (
        # R Tool → r_script component. Runs the original R script natively
        # via Rscript subprocess (or rpy2 if installed). User picks backend
        # in the emitted defs.yaml.
        lambda node, upstreams: (lambda code_el: MappedTool(
            component_id="r_script",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "script": (
                    (code_el.text or "out_df <- df  # original R script not captured").strip()
                    if code_el is not None and code_el.text else "out_df <- df"
                ),
                "backend": "rscript",
                "intermediate_format": "parquet",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"R Tool on tool {node.tool_id}: original R script ported to "
                "r_script component (backend='rscript'). Confirm Rscript is on PATH "
                "(or set backend='rpy2' if you have rpy2 installed). The script "
                "must assign its result to `out_df`."
            ],
        ))(node.config.find("RCode") or node.config.find("Code") or node.config.find("Script"))
    ),
    "AlteryxRPluginsGui.RTool.RTool": (
        # Alternate plugin path (newer Alteryx releases route through RPluginsGui).
        lambda node, upstreams: (lambda code_el: MappedTool(
            component_id="r_script",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "script": (
                    (code_el.text or "out_df <- df").strip()
                    if code_el is not None and code_el.text else "out_df <- df"
                ),
                "backend": "rscript",
                "intermediate_format": "parquet",
                "group_name": "alteryx_imported",
            },
            notes=[f"R Tool on tool {node.tool_id}: ported to r_script (backend='rscript')."],
        ))(node.config.find("RCode") or node.config.find("Code") or node.config.find("Script"))
    ),
    "JupyterCode": _map_jupyter_code,
    "AlteryxSpatialPluginsGui.SpatialProcess.SpatialProcess": (
        # SpatialProcess covers single-geom transforms. Alteryx <Method> values:
        #   CreateCentroid → centroid
        #   CreateBoundary / ConvertPolygonsToPolylines → boundary
        #   ConvertPolygonsToPoints → polygon_to_points
        #   ConvertPolylinesToPolygons → line_to_polygon
        #   CreateConvexHull → convex_hull
        #   CreateBoundingRectangle → envelope
        #   Simplify → simplify
        #   Buffer → buffer
        #   SetCompression → set_precision
        lambda node, upstreams: (lambda method_el, spatial_el: (lambda _m: MappedTool(
            component_id="spatial_process",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "method": {
                    "createcentroid": "centroid",
                    "createboundary": "boundary",
                    "convertpolygonstopolylines": "polygon_to_lines",
                    "convertpolygonstopoints": "polygon_to_points",
                    "convertpolylinestopolygons": "line_to_polygon",
                    "createconvexhull": "convex_hull",
                    "createboundingrectangle": "envelope",
                    "simplify": "simplify",
                    "buffer": "buffer",
                    "setcompression": "set_precision",
                }.get(_m.lower() if _m else "centroid", "centroid"),
                "geometry_column": (spatial_el.attrib.get("field") if spatial_el is not None else "geometry") or "geometry",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"SpatialProcess on tool {node.tool_id}: Alteryx method={_m!r} → "
                f"spatial_process method. If `buffer` / `simplify` / `set_precision`, "
                "set the corresponding tolerance / buffer_distance / precision_decimals "
                "field in defs.yaml."
            ],
        ))((method_el.text or "").strip() if method_el is not None and method_el.text else ""))(
            node.config.find("Method"), node.config.find("SpatialObj") or node.config.find("Field")
        )
    ),
    "AlteryxSpatialPluginsGui.Optimization.Optimization": (
        lambda node, upstreams: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Optimization on tool {node.tool_id}: linear/integer program "
                "not auto-translated. Replace with a pulp / scipy.optimize.linprog "
                "asset that consumes the same upstream."
            ],
        )
    ),
    "Optimization": (
        lambda node, upstreams: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Optimization on tool {node.tool_id}: linear/integer program "
                "not auto-translated. Replace with pulp / scipy.optimize.linprog."
            ],
        )
    ),
    "AlteryxBasePluginsGui.BasicDataProfile.BasicDataProfile": (
        # Field Summary report — pandas equivalent is df.describe() in
        # metadata. Passthrough so downstream tools still get the data.
        lambda node, upstreams: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"BasicDataProfile on tool {node.tool_id}: passthrough. "
                "For statistical summary, materialize the asset and consume "
                "df.describe() / df.info() output in Dagster metadata."
            ],
        )
    ),
    "AlteryxBasePluginsGui.JupyterCode.JupyterCode": (
        # Fully-qualified alias — same target as the bare 'JupyterCode' key above.
        lambda node, upstreams: (lambda code_el: MappedTool(
            component_id="jupyter_notebook",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "backend": "exec",
                "code": (
                    (code_el.text or "out_df = df").strip()
                    if code_el is not None and code_el.text
                    else "out_df = df  # original JupyterCode body not captured"
                ),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"JupyterCode on tool {node.tool_id}: ported to jupyter_notebook "
                "(backend='exec'). Must assign `out_df`. For real notebook "
                "execution, switch to backend='papermill' + notebook_path."
            ],
        ))(node.config.find("Code") or node.config.find("Script") or node.config.find("NotebookSource"))
    ),
    "PortfolioPluginsGui.ComposerText.PortfolioComposerText": (
        # Same reasoning as ComposerTable — text sections are mid-chain
        # styling, not terminal output. Emit as passthrough so downstream
        # Join/Union still receives a DataFrame. When the text tool has
        # NO upstream (source-style text block), emit a 1-row inline_dataframe
        # so the asset is still materializable as an empty placeholder.
        lambda node, upstreams: (
            MappedTool(
                component_id="select_columns",
                asset_name=_asset_name_for(node),
                attributes={
                    "upstream_asset_key": _single_upstream(upstreams),
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Text on tool {node.tool_id}: emitted as "
                    "passthrough — headline-text report sections have no "
                    "Dagster-native equivalent. Re-attach text content "
                    "to the terminal pdf_report's html_template."
                ],
            )
            if upstreams else
            MappedTool(
                component_id="inline_dataframe",
                asset_name=_asset_name_for(node),
                attributes={
                    "columns": ["text"],
                    "rows": [["placeholder"]],
                    "group_name": "alteryx_imported",
                },
                notes=[
                    f"Portfolio Composer Text on tool {node.tool_id}: no upstream — "
                    "emitted as a 1-row inline_dataframe placeholder so the "
                    "graph still materializes. Re-attach text content downstream."
                ],
            )
        )
    ),
    "AlteryxBasePluginsGui.DynamicRename.DynamicRename": (
        # DynamicRename → dynamic_rename (NEW registry component).
        # Alteryx <RenameMode> values map to dynamic_rename modes:
        #   FirstRow → first_row
        #   Add Prefix → add_prefix (with <Prefix>...</Prefix>)
        #   Add Suffix → add_suffix (with <Suffix>...</Suffix>)
        #   Remove Prefix → replace (pattern = "^<prefix>")
        #   Take Field Names from Right Input Rows → mapping_from_column
        #   Take Field Names from First Row of Data → first_row
        lambda node, upstreams: (lambda rm_el, pfx_el, sfx_el: MappedTool(
            component_id="dynamic_rename",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "mode": (
                    {
                        "firstrow": "first_row",
                        "addprefix": "add_prefix",
                        "addsuffix": "add_suffix",
                        "removeprefix": "replace",
                        "removesuffix": "replace",
                        "takefieldnamesfromrightinputrows": "mapping_from_column",
                        "takefromrow": "first_row",
                    }.get(
                        ((rm_el.text or "").strip().replace(" ", "").lower())
                        if rm_el is not None and rm_el.text else "",
                        "first_row",  # safe default — most common Alteryx use
                    )
                ),
                **(
                    {"prefix": pfx_el.text.strip()}
                    if pfx_el is not None and pfx_el.text else {}
                ),
                **(
                    {"suffix": sfx_el.text.strip()}
                    if sfx_el is not None and sfx_el.text else {}
                ),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"DynamicRename on tool {node.tool_id}: Alteryx RenameMode "
                f"{(rm_el.text if rm_el is not None and rm_el.text else 'FirstRow')!r} → "
                f"dynamic_rename mode. For 'replace' modes, also set `pattern:` "
                "in defs.yaml. For 'mapping_from_column', set "
                "mapping_asset_key + mapping_key_column + mapping_value_column."
            ],
        ))(node.config.find("RenameMode"), node.config.find("Prefix"), node.config.find("Suffix"))
    ),
    "AlteryxBasePluginsGui.MultiRowFormula.MultiRowFormula": _map_multi_row_formula,
    "AlteryxConnectorGui.Download.Download": (
        # Alteryx Download → per_row_http_fetcher (per-row HTTP GET).
        # Reads <URLField>X</URLField> for the URL column; falls back to "URL".
        lambda node, upstreams: (lambda url_el: MappedTool(
            component_id="per_row_http_fetcher",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "url_column": (url_el.text or "URL").strip() if url_el is not None and url_el.text else "URL",
                "method": "GET",
                "timeout_seconds": 30,
                "output_prefix": "DownloadData",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Download on tool {node.tool_id}: url_column from <URLField/> or "
                "defaults to 'URL'. output_prefix='DownloadData' (Alteryx convention). "
                "Confirm headers / auth / body in defs.yaml."
            ],
        ))(node.config.find("URLField") or node.config.find("UrlField") or node.config.find("URL"))
    ),
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
    "AlteryxSpatialPluginsGui.Distance.Distance": (
        # Distance: distance between two geometry columns (Origin / Destination).
        # Alteryx config has <SpatialObjSource>Origin_Geo</SpatialObjSource>
        # and <SpatialObjDest>Destination_Geo</SpatialObjDest>. The registry's
        # distance_calculator wants lat/lng pairs, not geometry columns — so
        # we synthesize geometry.x / geometry.y assumption (Alteryx geometries
        # commonly come from Create Points where x=lng, y=lat).
        lambda node, upstreams: (lambda src_el, dst_el, unit_el, metric_el: MappedTool(
            component_id="distance_calculator",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                # Use the geometry column name + _y / _x suffixes; user adds
                # an upstream Formula that does `df["Origin_Geo_y"] = df["Origin_Geo"].apply(lambda g: g.y)`
                # OR replaces with real column names in the emitted defs.yaml.
                "lat1_column": ((src_el.text or "Origin") + "_y") if src_el is not None and src_el.text else "Origin_y",
                "lng1_column": ((src_el.text or "Origin") + "_x") if src_el is not None and src_el.text else "Origin_x",
                "lat2_column": ((dst_el.text or "Destination") + "_y") if dst_el is not None and dst_el.text else "Destination_y",
                "lng2_column": ((dst_el.text or "Destination") + "_x") if dst_el is not None and dst_el.text else "Destination_x",
                # Alteryx Distance tool emits the distance column as `DistanceMiles`
                # or `DistanceKilometers` based on <OutputUnits> / <IsMetric>.
                # Match that naming so downstream Formula / Filter / Summarize
                # references resolve without manual rewiring.
                "output_column": (
                    "DistanceKilometers"
                    if (
                        (unit_el is not None and (unit_el.text or "").strip().lower() in ("kilometers", "km"))
                        or (metric_el is not None and (metric_el.attrib.get("value", "False").lower() == "true"))
                    )
                    else "DistanceMiles"
                ),
                "unit": (
                    "km"
                    if (
                        (unit_el is not None and (unit_el.text or "").strip().lower() in ("kilometers", "km"))
                        or (metric_el is not None and (metric_el.attrib.get("value", "False").lower() == "true"))
                    )
                    else "miles"
                ),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Distance on tool {node.tool_id}: Alteryx stored geometry "
                f"in {(src_el.text if src_el is not None and src_el.text else 'Origin_Geo')!r} + "
                f"{(dst_el.text if dst_el is not None and dst_el.text else 'Destination_Geo')!r} columns. "
                "distance_calculator needs lat/lng — added an upstream Formula "
                "that extracts geometry.x / geometry.y, OR replace the "
                "lat1_column / lng1_column / lat2_column / lng2_column attrs "
                "in defs.yaml with your data's real lat/lng column names."
            ],
        ))(
            node.config.find("SpatialObjSource"),
            node.config.find("SpatialObjDest"),
            node.config.find("OutputUnits"),
            node.config.find("IsMetric"),
        )
    ),
    "AlteryxSpatialPluginsGui.FindNearest.FindNearest": (
        # FindNearest: find K nearest points in target set per source point.
        # Alteryx config has <Target SpatialObj="X"/> and <Universe SpatialObj="Y"/>.
        # nearest_neighbors uses feature_columns (raw numeric columns). The
        # placeholder column scanner sees the geometry column name from the
        # XML — emit it as the single feature so placeholders include it.
        # For real workflows the user replaces with [Store_lat, Store_lng].
        lambda node, upstreams: (lambda tgt_el, uni_el: MappedTool(
            component_id="nearest_neighbors",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "feature_columns": [
                    tgt_el.attrib.get("SpatialObj", "Target") if tgt_el is not None else "Target",
                ],
                "n_neighbors": int((node.config.find("HowMany").attrib.get("value", "5")) if node.config.find("HowMany") is not None else 5),
                "metric": "euclidean",
                # Alteryx FindNearest emits the distance column as `DistanceMiles`
                # (and the matched-target join cols). Use that naming so downstream
                # Alteryx Formula/Filter expressions referencing DistanceMiles resolve.
                "distance_column_template": "DistanceMiles" if int((node.config.find("HowMany").attrib.get("value", "5")) if node.config.find("HowMany") is not None else 5) == 1 else "DistanceMiles_{i}",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"FindNearest on tool {node.tool_id}: target geometry "
                f"{(tgt_el.attrib.get('SpatialObj') if tgt_el is not None else 'Target')!r}, "
                f"universe geometry {(uni_el.attrib.get('SpatialObj') if uni_el is not None else 'Universe')!r}. "
                "feature_columns defaults to the geometry column name — "
                "replace with actual numeric columns to compare on "
                "(e.g. [Store_lat, Store_lng]) for meaningful distance. "
                "Consider metric='haversine' for geographic distance."
            ],
        ))(node.config.find("Target"), node.config.find("Universe"))
    ),
    "AlteryxSpatialPluginsGui.SpatialMatch.SpatialMatch": (
        # SpatialMatch: point-in-polygon test (or geometry intersection).
        # Maps to spatial_join (two-input: points + regions). Falls back to
        # point_in_polygon when there's only one upstream.
        # Read <Target SpatialObj="X"/> and <Universe SpatialObj="Y"/> to
        # detect when upstream emits a Shapely geom column directly (PolyBuild,
        # CreatePoints, geocoder) — pass it as points_geometry_column instead
        # of relying on lat/lng numerics.
        lambda node, upstreams: (lambda tgt_el, uni_el: MappedTool(
            component_id="spatial_join" if len(upstreams) > 1 else "point_in_polygon",
            asset_name=_asset_name_for(node),
            attributes=(
                {
                    "upstream_asset_key": upstreams[0],
                    "regions_asset_key": upstreams[1],
                    "lat_column": "latitude",
                    "lng_column": "longitude",
                    "points_geometry_column": (
                        tgt_el.attrib.get("SpatialObj") if tgt_el is not None else None
                    ),
                    "geometry_column": (
                        uni_el.attrib.get("SpatialObj", "geometry") if uni_el is not None else "geometry"
                    ),
                    "group_name": "alteryx_imported",
                }
                if len(upstreams) > 1
                else {
                    "upstream_asset_key": _single_upstream(upstreams),
                    "lat_column": "latitude",
                    "lng_column": "longitude",
                    "output_column": "region",
                    "group_name": "alteryx_imported",
                }
            ),
            notes=[
                f"SpatialMatch on tool {node.tool_id}: routed to "
                f"{('spatial_join' if len(upstreams) > 1 else 'point_in_polygon')}. "
                "Confirm lat_column / lng_column match your data. For "
                "point_in_polygon, set geojson_path or geojson_url to the "
                "polygon source."
            ],
        ))(node.config.find("Target"), node.config.find("Universe"))
    ),
    "AlteryxSpatialPluginsGui.PolyBuild.PolyBuild": (
        # PolyBuild → poly_build component. Alteryx config:
        #   <SpatialObj field="X"/>   geometry column (existing Point geoms)
        #   <GroupField field="Y"/>   group identifier
        #   <SequenceField field="Z"/> ordering within group (often blank → row order)
        #   <BuildType>SequencePolyline | SequencePolygon | etc.</BuildType>
        lambda node, upstreams: (lambda spatial_el, group_el, seq_el, type_el: MappedTool(
            component_id="poly_build",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_column": (group_el.attrib.get("field") if group_el is not None else "group") or "group",
                # SequenceField is often blank in Alteryx — None tells the
                # component to use row order, matching Alteryx semantics.
                "sequence_column": (
                    seq_el.attrib.get("field") or None
                ) if seq_el is not None else None,
                "input_geometry_column": (
                    spatial_el.attrib.get("field") if spatial_el is not None else "Centroid"
                ) or "Centroid",
                "output_type": (
                    "polygon" if (type_el is not None and type_el.text and "Polygon" in type_el.text)
                    else "line"
                ),
                "geometry_column": "geometry",  # output col, separate from input
                "group_name": "alteryx_imported",
            },
        ))(
            node.config.find("SpatialObj"),
            node.config.find("GroupField"),
            node.config.find("SequenceField"),
            node.config.find("BuildType"),
        )
    ),
    "AlteryxSpatialPluginsGui.SpatialInfo.SpatialInfo": (
        # SpatialInfo → spatial_info component: appends area / length / centroid
        # / bounds / geom_type columns.
        lambda node, upstreams: MappedTool(
            component_id="spatial_info",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "geometry_column": "Centroid",  # match Alteryx CreatePoints default
                "metrics": ["area", "length", "centroid", "bounds", "geom_type"],
                "area_unit": "sq_miles",
                "length_unit": "miles",
                "group_name": "alteryx_imported",
            },
        )
    ),
    "AlteryxSpatialPluginsGui.MapInput.MapInput": (
        # MapInput → file_ingestion. Alteryx MapInput accepts shapefiles,
        # GeoJSON, KML, etc. We default file_path to a .geojson stub name
        # so the auto-format detection picks GeoJSON; the stub generator
        # then writes a minimal GeoJSON-shaped file at that path.
        # User overrides with real path (cloud URL preferred) in defs.yaml.
        lambda node, upstreams: MappedTool(
            component_id="file_ingestion",
            asset_name=_asset_name_for(node),
            attributes={
                "file_path": f"./_alteryx_mapinput_{node.tool_id}.csv",
                "format": "csv",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"MapInput on tool {node.tool_id}: emitted with placeholder "
                "file_path (./_alteryx_mapinput_<id>.csv) — replace with the "
                "real geometry source (shapefile / GeoJSON / KML on disk or in "
                "S3/GCS). Alteryx's interactive map editor doesn't translate; "
                "the file path is the Dagster-deployable equivalent."
            ],
        )
    ),
    "AlteryxSpatialPluginsGui.TradeArea.TradeArea": (
        # TradeArea has two modes: straight-line buffer (geo_buffer) or
        # drive-time (drive_time). Default to drive_time since that's the
        # more common Alteryx use; user can swap to geo_buffer in defs.yaml.
        lambda node, upstreams: MappedTool(
            component_id="drive_time",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "drive_minutes": 15,
                "provider": "openrouteservice",
                "geometry_column": "Centroid",  # match Alteryx CreatePoints default
                "group_name": "alteryx_imported",
            },
            notes=[
                f"TradeArea on tool {node.tool_id}: defaulted to drive_time "
                "(15-min isochrone via openrouteservice). For straight-line "
                "TradeArea, swap to component_id=geo_buffer with distance + "
                "units in the emitted defs.yaml."
            ],
        )
    ),
    # Data Investigation
    "AlteryxBasePluginsGui.PearsonCorrelation.PearsonCorrelation": (
        lambda node, upstreams: MappedTool(
            component_id="pearson_correlation",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "output_shape": "long",
                "group_name": "alteryx_imported",
            },
        )
    ),
    # Predictive — sklearn-backed registry components
    "Linear_Regression": _make_predictive_mapper("linear_regression_model"),
    "Logistic_Regression": _make_predictive_mapper("logistic_regression_model"),
    "Decision_Tree": _make_predictive_mapper("decision_tree_model", task_type="classification"),
    "Random_Forest": _make_predictive_mapper("random_forest_model", task_type="classification"),
    "Naive_Bayes_Classifier": _make_predictive_mapper("naive_bayes_model", task_type="classification"),
    "Neural_Network": _make_predictive_mapper("neural_network_model", task_type="classification"),
    "Support_Vector_Machine": _make_predictive_mapper("svm", task_type="classification"),
    "Boosted_Model": _make_predictive_mapper("gradient_boosting_model", task_type="regression"),
    "Gradient_Boosted_Model": _make_predictive_mapper("gradient_boosting_model", task_type="regression"),
    "Principal_Components": _make_predictive_mapper("pca"),
    "Score": _map_score,
    # Predictive — statsmodels-backed registry components
    "Count_Regression": _make_predictive_mapper("count_regression"),
    "Gamma_Regression": _make_predictive_mapper("gamma_regression"),
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
    # Data Stream In — pandas DataFrame → warehouse staging table
    # (inverse of StreamOut). Plugin namespace varies; cover both forms.
    "AlteryxConnectorsGui.DataStreamIn.DataStreamIn": _map_data_stream_in,
    "AlteryxConnectorsGui.InDb.DataStreamIn": _map_data_stream_in,
    # Some versions use a different namespace prefix.
    "AlteryxConnectorsGui.InDb.InDbInput": _map_indb_input,
    "AlteryxConnectorsGui.InDb.InDbFilter": _map_indb_filter,
}


# Fuzzier match for In-DB tools whose plugin id varies across Alteryx versions
# (the canonical match goes through PLUGIN_REGISTRY first; this is a fallback).
def _fuzzy_indb_match(plugin: str) -> Optional[ToolMapping]:
    name = plugin.lower()
    if "indb" not in name and "datastreamin" not in name:
        return None
    if "datastreamin" in name:
        return _map_data_stream_in
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
    "AlteryxGuiToolkit.ToolContainer.ToolContainer": (
        "Tool Container — visual grouping only (no compute). Inner tools "
        "are spliced into the parent graph during import; the container "
        "itself emits no Dagster asset."
    ),
    "AlteryxGuiToolkit.HtmlBox.HtmlBox": (
        "HTML Box — documentation only. Drop; copy into the project README "
        "if the content is worth preserving."
    ),
    "AlteryxBasePluginsGui.AutoField.AutoField": (
        "Auto Field — runtime type inference. Dagster components handle "
        "dtype inference automatically (pandas read_csv / inline_dataframe "
        "use the configured dtypes; downstream tools coerce per-call). No "
        "emit needed."
    ),
    "AlteryxGuiToolkit.Detour.Detour": (
        "Detour — conditional bypass. Dagster equivalent: AutomationCondition "
        "on the downstream asset. No data flow; drop after wiring."
    ),
    "AlteryxGuiToolkit.Detour.DetourEnd": (
        "Detour End — companion to the Detour tool. No data flow; drop."
    ),
    # Alteryx ships Detour under both GuiToolkit (older) and BasePluginsGui
    # (newer) namespaces. Register both so neither leaks as truly unmapped.
    "AlteryxBasePluginsGui.Detour.Detour": (
        "Detour — conditional bypass. Dagster equivalent: AutomationCondition. "
        "No data flow; drop."
    ),
    "AlteryxBasePluginsGui.DetourEnd.DetourEnd": (
        "Detour End — companion to Detour. No data flow; drop."
    ),
    "AlteryxBasePluginsGui.Detour.DetourEnd": (
        "Detour End — companion to Detour. No data flow; drop."
    ),
    "AlteryxGuiToolkit.Action.Action": (
        "Action — Alteryx App machinery that mutates the workflow when a "
        "Question is answered. No runtime semantic outside the App; "
        "drop after wiring the equivalent Dagster config / partition logic."
    ),
    "AlteryxGuiToolkit.Questions.Tab.Tab": (
        "Tab — Alteryx App interface tool. No data flow; purely visual "
        "grouping inside the App's question pane. Drop."
    ),
    "AlteryxGuiToolkit.Questions.CheckBoxGroup.CheckBoxGroup": (
        "CheckBoxGroup — Alteryx App interface control. Becomes a Dagster "
        "config schema or partition-key list if you need user-driven runs. "
        "No data flow; drop and wire equivalent config."
    ),
    "AlteryxGuiToolkit.Questions.NumericUpDown.NumericUpDown": (
        "NumericUpDown — Alteryx App numeric input. No data emitted. To "
        "make the original chain's `df[\"Numeric Up Down Result\"]` reference "
        "resolve, declare a Dagster `Config` field on the downstream filter / "
        "formula asset and inject it via Run Config at launch. We do not "
        "auto-wire this — see the Alteryx App Inputs section below."
    ),
    "AlteryxGuiToolkit.Questions.DropDownListBox.DropDown": (
        "Drop Down (List Box) — Alteryx App dropdown input. No data emitted. "
        "Declare a Dagster `Config` field on the downstream filter / formula "
        "and inject the selected value via Run Config. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.DropDownListBox.ListBox": (
        "List Box — Alteryx App multi-select input. No data emitted. "
        "Declare a Dagster `Config` field on the downstream asset and "
        "inject the selected list via Run Config. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.TextBox.TextBox": (
        "Text Box (input) — Alteryx App free-text input. No data emitted. "
        "Declare a Dagster `Config` field on the downstream asset and inject "
        "the entered text via Run Config. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.FileBrowse.FileBrowse": (
        "File Browse — Alteryx App file-picker input. No data emitted. "
        "Declare a Dagster `Config` field for the file path and reference it "
        "from downstream input assets. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.FolderBrowse.FolderBrowse": (
        "Folder Browse — Alteryx App folder-picker input. No data emitted. "
        "Declare a Dagster `Config` field for the folder path and reference "
        "it from downstream input assets. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.DateInput.DateInput": (
        "Date Input — Alteryx App date picker. No data emitted. Declare a "
        "Dagster `Config` (or use a TimeWindowPartition for backfills) and "
        "inject the date value at run launch. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.DateTimeInput.DateTimeInput": (
        "DateTime Input — Alteryx App date+time picker. No data emitted. "
        "Declare a Dagster `Config` field and inject at run launch. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.RadioGroup.RadioGroup": (
        "Radio Group — Alteryx App single-choice input. No data emitted. "
        "Declare a Dagster `Config` field with the available choices and "
        "inject at run launch. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.ColorPicker.ColorPicker": (
        "Color Picker — Alteryx App color-selection input. No data emitted. "
        "Declare a Dagster `Config` field if the downstream needs the value."
    ),
    "AlteryxGuiToolkit.Questions.MapInput.MapInput": (
        "Map Input — Alteryx App spatial-input. No data emitted. Declare a "
        "Dagster `Config` field with the chosen geometry (WKT / GeoJSON) and "
        "inject at run launch. We do not auto-wire."
    ),
    "AlteryxGuiToolkit.Questions.Label.Label": (
        "Label — annotation only inside an Alteryx App. No data flow; drop."
    ),
    "AlteryxGuiToolkit.Questions.ControlParam.ControlParam": (
        "Control Parameter — Alteryx App input used by Macros. No data flow; "
        "becomes a Dagster `Config` field at the macro boundary. We do not auto-wire."
    ),
    "AlteryxBasePluginsGui.MacroOutput.MacroOutput": (
        "Macro Output — marks the output anchor of a custom .yxmc. The macro "
        "splicer wires this to the parent workflow; no standalone Dagster "
        "asset needed."
    ),
    "AlteryxReportChartGui.AlteryxReportChartGui": (
        "Interactive Chart — Alteryx's HTML-report-builder chart. No runtime "
        "data semantic; Dagster surfaces asset previews + materialization "
        "metadata natively. Drop, or move chart logic into a notebook asset."
    ),
    "AlteryxBasePluginsGui.ReportHeader.ReportHeader": (
        "Report Header — annotation only inside an Alteryx report. Drop."
    ),
    "AlteryxSpatialPluginsGui.Buffer.Buffer": (
        # Alteryx Buffer → geo_buffer. <BufferAmount value="N"/>
        lambda node, upstreams: (lambda amt_el, spatial_el, units_el: MappedTool(
            component_id="geo_buffer",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "geometry_column": (spatial_el.attrib.get("field") if spatial_el is not None else "geometry") or "geometry",
                "distance": float(amt_el.attrib.get("value", "0.001") if amt_el is not None else "0.001"),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Buffer on tool {node.tool_id}: amount={amt_el.attrib.get('value') if amt_el is not None else '0.001'} "
                f"units={(units_el.text or '').strip() if units_el is not None and units_el.text else 'degrees'}. "
                "Verify the distance is in the right CRS units; reproject upstream if you want meters/miles."
            ],
        ))(node.config.find("BufferAmount"), node.config.find("SpatialObj"), node.config.find("Units"))
    ),
    "AlteryxSpatialPluginsGui.Cass.Cass": (
        # Alteryx CASS (US-only address standardization). Route to free regex
        # fallback by default — user picks libpostal / geoapify / nominatim in
        # MIGRATION.md / defs.yaml for higher fidelity.
        lambda node, upstreams: (lambda addr_el: MappedTool(
            component_id="address_standardize",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "address_column": (
                    (addr_el.attrib.get("field") or addr_el.text or "Address").strip()
                    if addr_el is not None else "Address"
                ),
                "provider": "regex",  # free / no-dep default
                "group_name": "alteryx_imported",
            },
            notes=[
                f"CASS on tool {node.tool_id}: routed to address_standardize "
                "(provider=regex by default; switch to libpostal / geoapify / "
                "nominatim for higher fidelity). USPS CASS-certification is "
                "a paid product — use a commercial vendor if you need DPV."
            ],
        ))(node.config.find("AddressField") or node.config.find("Address") or node.config.find("Field"))
    ),
    "AlteryxSpatialPluginsGui.AddressVerification.AddressVerification": (
        lambda node, upstreams: (lambda addr_el: MappedTool(
            component_id="address_standardize",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "address_column": (
                    (addr_el.attrib.get("field") or addr_el.text or "Address").strip()
                    if addr_el is not None else "Address"
                ),
                "provider": "regex",
                "group_name": "alteryx_imported",
            },
            notes=[f"AddressVerification on tool {node.tool_id}: routed to address_standardize."],
        ))(node.config.find("AddressField") or node.config.find("Address") or node.config.find("Field"))
    ),
    "AlteryxSpatialPluginsGui.Geocoder.Geocoder": (
        # Alteryx Geocoder → geocoder (Nominatim default).
        lambda node, upstreams: (lambda addr_el: MappedTool(
            component_id="geocoder",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "address_column": (addr_el.text or "Address").strip() if addr_el is not None and addr_el.text else "Address",
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Geocoder on tool {node.tool_id}: defaults to Nominatim "
                "(free, ~1 req/sec rate limit). For higher volume switch to "
                "google / mapbox / geoapify and set the corresponding API key."
            ],
        ))(node.config.find("AddressField") or node.config.find("Address") or node.config.find("Field"))
    ),
    "AlteryxSpatialPluginsGui.MakeGroup.MakeGroup": (
        # Make Group bundles geometries into a single multi-geometry per group.
        # Maps to summarize with spatialobjcombine action.
        lambda node, upstreams: (lambda group_el, spatial_el: MappedTool(
            component_id="summarize",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_by": [
                    (group_el.attrib.get("field") or (group_el.text or "")).strip()
                    if group_el is not None else ""
                ],
                "aggregations": {
                    "Grouped_Geometry": {
                        "col": (spatial_el.attrib.get("field") or (spatial_el.text or "geometry")).strip()
                        if spatial_el is not None else "geometry",
                        "agg": "spatialobjcombine",
                    },
                },
                "group_name": "alteryx_imported",
            },
            notes=[
                f"MakeGroup on tool {node.tool_id}: routed to summarize w/ "
                "spatialobjcombine agg (unary_union per group)."
            ],
        ))(node.config.find("GroupField"), node.config.find("SpatialObj"))
    ),
    "AlteryxSpatialPluginsGui.Smooth.Smooth": (
        # Smooth simplifies vertex chains — maps to spatial_process.simplify.
        lambda node, upstreams: (lambda tol_el, spatial_el: MappedTool(
            component_id="spatial_process",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "method": "simplify",
                "geometry_column": (spatial_el.attrib.get("field") if spatial_el is not None else "geometry") or "geometry",
                "tolerance": float(tol_el.attrib.get("value", "0.0001") if tol_el is not None else "0.0001"),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Smooth on tool {node.tool_id}: routed to spatial_process.simplify."
            ],
        ))(node.config.find("Tolerance"), node.config.find("SpatialObj"))
    ),
    "AlteryxSpatialPluginsGui.Generalize.Generalize": (
        # Generalize is also a Douglas-Peucker pass; same target as Smooth.
        lambda node, upstreams: (lambda tol_el, spatial_el: MappedTool(
            component_id="spatial_process",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "method": "simplify",
                "geometry_column": (spatial_el.attrib.get("field") if spatial_el is not None else "geometry") or "geometry",
                "tolerance": float(tol_el.attrib.get("value", "0.0001") if tol_el is not None else "0.0001"),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Generalize on tool {node.tool_id}: routed to spatial_process.simplify."
            ],
        ))(node.config.find("Tolerance"), node.config.find("SpatialObj"))
    ),
    "AlteryxSpatialPluginsGui.HeatMap.HeatMap": (
        # HeatMap is a visualization — no Dagster-native runtime semantic.
        # Passthrough so downstream tools still get the upstream DataFrame.
        lambda node, upstreams: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"HeatMap on tool {node.tool_id}: passthrough; the heat-map "
                "rendering is a visual concern. Replace with a Folium / Plotly "
                "asset that consumes the same upstream."
            ],
        )
    ),
    "AlteryxSpatialPluginsGui.Demographic.Demographic": (
        lambda node, upstreams: MappedTool(
            component_id="select_columns",
            asset_name=_asset_name_for(node),
            attributes={
                "upstream_asset_key": _single_upstream(upstreams),
                "group_name": "alteryx_imported",
            },
            notes=[
                f"Demographic on tool {node.tool_id}: Alteryx data product (paid). "
                "Passthrough emitted; replace with the equivalent Census / "
                "OpenStreetMap / data-vendor source asset."
            ],
        )
    ),
    "AlteryxSpatialPluginsGui.ReportMap.ReportMap": (
        "Report Map — Alteryx's spatial map renderer for the HTML report. "
        "Drop, or replace with a Plotly / Folium map in a notebook asset."
    ),
    "PortfolioPluginsGui.ComposerLayout.PortfolioComposerLayout": (
        "Portfolio Composer Layout — page-layout grouping for a multi-section "
        "report. Children (Table / Text / Image) emit individual pdf_report "
        "assets today; combining them into one PDF is a future enhancement."
    ),
    "PortfolioPluginsGui.ComposerOverlay.Overlay": (
        "Portfolio Composer Overlay — stacks report sections. Drop or "
        "replace with the pdf_report template_html mode for full layout."
    ),
    "AlteryxBasePluginsGui.Message.Message": (
        "Message — emits a runtime notification. Use dagster's logger / "
        "Dagster+ alerting if the message was load-bearing; otherwise drop."
    ),
    "AlteryxGuiToolkit.Error.Error": (
        "Error — fails the workflow on a condition. Move to a Dagster "
        "AssetCheck or an op-level raise for the equivalent. Drop the "
        "Error tool itself."
    ),
    "AlteryxBasePluginsGui.MacroInput.MacroInput": (
        "Macro Input — boundary marker for a custom .yxmc. The macro "
        "splicer wires this into the parent workflow; standalone MacroInput "
        "nodes only appear when the macro itself was the imported file. "
        "Treat as a no-op input."
    ),
    # Additional App-interface Question types (radio buttons, dropdowns,
    # question-text-box, file-browse). All are App UI controls — no data flow.
    "AlteryxGuiToolkit.Questions.RadioButtonGroup.RadioButtonGroup": (
        "Radio Button Group — App interface control. No data flow; "
        "becomes a Dagster config field with choices. Drop."
    ),
    "AlteryxGuiToolkit.Questions.DropDownListBox.DropDown": (
        "DropDown — App interface control. No data flow; becomes a "
        "Dagster config field with an enum. Drop."
    ),
    "AlteryxGuiToolkit.Questions.DropDownListBox.DropDownListBox": (
        "DropDown List Box — App interface control. No data flow; becomes a "
        "Dagster config field with an enum. Drop."
    ),
    "AlteryxGuiToolkit.Questions.DropDownListBox.ListBox": (
        "List Box (DropDown family) — App interface control. No data flow; "
        "becomes a Dagster config list field. Drop."
    ),
    "ReportHeader": (
        "Report Header (bare-name alias) — annotation only. No data flow. Drop."
    ),
    "AlteryxGuiToolkit.Questions.TextBox.QuestionTextBox": (
        "Question Text Box — App interface free-text input. No data flow; "
        "becomes a Dagster config string field. Drop."
    ),
    "AlteryxGuiToolkit.Questions.FileBrowse.FileBrowse": (
        "File Browse — App interface file picker. No data flow; replace "
        "with a Dagster config field that takes a path string. Drop."
    ),
    "AlteryxGuiToolkit.Questions.FolderBrowse.FolderBrowse": (
        "Folder Browse — App interface folder picker. No data flow; "
        "becomes a Dagster config field with a path string. Drop."
    ),
    "AlteryxGuiToolkit.Questions.Date.Date": (
        "Date — App interface date picker. No data flow; becomes a Dagster "
        "partition key or config date field. Drop."
    ),
    "AlteryxGuiToolkit.Questions.ListBox.ListBox": (
        "List Box — App interface multi-select. No data flow; becomes a "
        "Dagster config list field. Drop."
    ),
    "AlteryxGuiToolkit.Questions.MapInput.MapInput": (
        "Map Input (Question) — App interface map-region selector. "
        "No data flow; replace with a Dagster config field. Drop."
    ),
    "AlteryxBasePluginsGui.ReportHeader.ReportHeader": (
        "Report Header — annotation only inside an HTML report. No data flow. Drop."
    ),
    # The Render terminal tool occasionally appears bare (rare). Treat as
    # a report sink — but if we got here it means we couldn't route to
    # pdf_report (e.g. no upstream); skip with note.
    "AlteryxGuiToolkit.Render.Render": (
        "Render — HTML report renderer. No runtime DataFrame output; "
        "route to pdf_report manually for PDF output."
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
