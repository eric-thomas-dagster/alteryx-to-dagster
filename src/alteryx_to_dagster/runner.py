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
from .indb_collapse import build_warehouse_pipelines
from .macro_splicer import splice_macros
from .mapper import MappedTool, UnmappedTool, _stock_macro_basenames, map_tool
from .parser import AlteryxEdge, AlteryxNode, AlteryxWorkflow, parse_workflow


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
# Tools like FindReplace store column refs as element TEXT, not attributes —
# e.g. <FieldFind>find</FieldFind> / <InputFieldName>Date</InputFieldName>.
# Match any single-line element whose tag contains "Field" or "Name" or
# "Column" — the text content is the column name.
_FIELD_ELEMENT_RE = __import__("re").compile(
    r"<([A-Za-z]*(?:Field|Name|Column)[A-Za-z]*)>([^<\n]{1,80})</",
)


def _rewrite_controlparam_filter(
    condition: str,
    *,
    field: str,
    values: List[str],
) -> str:
    r"""Rewrite a batch-macro-internal Filter's condition from a hardcoded
    ControlParam value to `context.partition_key`.

    The Alteryx batch macro encodes its iteration filter with a hardcoded
    operand (e.g. `df["Category"] == "Acceleration"`). At runtime the
    Alteryx engine substitutes this operand with each iteration value via
    an Action tool. We don't process Action tools, so we get the design-
    time hardcoded value at import. When that value is one of the known
    iteration values, swap it for `context.partition_key` so the
    partitioned Dagster asset filters to the current partition slice.

    Matches both pandas-eval and Python-eval forms:
        df["X"] == "value"       (both quote styles)
        df['X'] == 'value'
        X == "value"             (bare field, pandas-eval shortcut)
    """
    import re as _re
    _val_re = "|".join(_re.escape(v) for v in values if v)
    if not _val_re:
        return condition
    # Match `<field-ref> == '<known-value>'` (or `==`/`= `, single/double
    # quotes). The field-ref can be `df["X"]`, `df['X']`, or bare `X`.
    pat = _re.compile(
        rf"""(df\[\s*['"]\s*{_re.escape(field)}\s*['"]\s*\]|\b{_re.escape(field)}\b)"""
        rf"""(\s*==\s*)(['"])\s*(?:{_val_re})\s*\3""",
    )
    return pat.sub(lambda m: f"{m.group(1)}{m.group(2)}context.partition_key", condition)


def _columns_needed_downstream(wf: AlteryxWorkflow, origin_tool_id: str) -> tuple:
    """Scan downstream tools of `origin_tool_id` and return:

      (column_names: list[str], string_typed_columns: set[str],
       numeric_typed_columns: set[str])

    The second set lists columns used with string accessors / Alteryx string
    functions (Trim, Replace, Substring, UpperCase, LowerCase, Contains,
    etc.) — they MUST be string-typed in the placeholder stub or pandas
    raises 'Can only use .str accessor with string values'.

    The third set lists columns used in numeric contexts — summarize
    aggregations (Sum/Avg/Mean/Min/Max/StdDev/Median), tile, running total,
    arithmetic in formulas. They MUST be numeric or pd.agg() raises
    'agg function failed [how->mean,dtype->object]'.
    """
    import xml.etree.ElementTree as _ET
    by_id = wf.by_id()
    visited: set = set()
    stack = [e.dest_tool for e in wf.downstreams_of(origin_tool_id)]
    cols: set = set()
    str_cols: set = set()
    num_cols: set = set()

    _STRING_FNS = (
        "trim", "trimleft", "trimright", "uppercase", "lowercase", "titlecase",
        "left", "right", "substring", "length", "replace", "replacechar",
        "replacefirst", "contains", "startswith", "endswith", "regex_replace",
        "regex_match", "regex_countmatches", "findstring", "padleft",
        "padright", "tostring", "isempty",
    )
    _re = __import__("re")
    _STR_FN_RE = _re.compile(
        r"\b(?:" + "|".join(_STRING_FNS) + r")\s*\(\s*\[([^\[\]]+)\]",
        _re.IGNORECASE,
    )
    # Numeric Alteryx formula functions over a bracketed col arg.
    _NUMERIC_FNS = (
        "abs", "sqrt", "log", "ln", "exp", "ceil", "floor", "round",
        "min", "max", "mod", "average", "avg", "sum", "median", "stddev",
        "tonumber",
    )
    _NUM_FN_RE = _re.compile(
        r"\b(?:" + "|".join(_NUMERIC_FNS) + r")\s*\(\s*\[([^\[\]]+)\]",
        _re.IGNORECASE,
    )
    # `[Col] (+|-|*|/) ...` or `... (+|-|*|/) [Col]` — arithmetic on a bracketed
    # column implies the column is numeric.
    _ARITH_FIELD_RE = _re.compile(r"\[([^\[\]]+)\]\s*[+\-*/]|[+\-*/]\s*\[([^\[\]]+)\]")
    # Summarize aggregation action attrs that need numeric input.
    _NUM_AGG_ACTIONS = {
        "sum", "avg", "average", "mean", "median", "stddev",
        "stdev", "variance", "var", "min", "max",
    }

    while stack:
        tid = stack.pop()
        if tid in visited or tid not in by_id:
            continue
        visited.add(tid)
        node = by_id[tid]
        try:
            cfg_text = _ET.tostring(node.config, encoding="unicode")
        except Exception:
            cfg_text = ""
        cols.update(_BRACKETED_FIELD_RE.findall(cfg_text))
        cols.update(_FIELD_ATTR_RE.findall(cfg_text))
        cols.update(_FIELD2_ATTR_RE.findall(cfg_text))
        cols.update(_EXPRESSION_ATTR_RE.findall(cfg_text))
        cols.update(_RENAME_ATTR_RE.findall(cfg_text))
        cols.update(text for _tag, text in _FIELD_ELEMENT_RE.findall(cfg_text))
        for m in _STR_FN_RE.finditer(cfg_text):
            str_cols.add(m.group(1))
        for m in _NUM_FN_RE.finditer(cfg_text):
            num_cols.add(m.group(1))
        for m in _ARITH_FIELD_RE.finditer(cfg_text):
            for grp in m.groups():
                if grp:
                    num_cols.add(grp)
        # Summarize: <SummarizeField field="X" action="Sum"/>
        for sf in node.config.findall(".//SummarizeField"):
            action = (sf.attrib.get("action") or "").lower()
            field = sf.attrib.get("field") or ""
            if field and action in _NUM_AGG_ACTIONS:
                num_cols.add(field)
        # Tile / RunningTotal / NumericTransform: <Field>X</Field>
        plugin = (node.plugin or "").lower()
        if any(t in plugin for t in ("tile", "runningtotal", "running_total")):
            for fld in node.config.findall(".//Field"):
                fname = (fld.text or fld.attrib.get("field") or "").strip()
                if fname and fname != "*Unknown":
                    num_cols.add(fname)
        stack.extend(e.dest_tool for e in wf.downstreams_of(tid))
    out: list = []
    for c in cols:
        if not c or "*" in c or "{" in c or "(" in c or c.startswith("\\") or len(c) > 80:
            continue
        if any(s in c for s in [">", "<", "&", "|", "=", "+", "-", "/", "*", '"', "'", "\n"]):
            continue
        if c not in out:
            out.append(c)
    return out, str_cols, num_cols


_PATH_HINTS = ("path", "directory", "folder", "filename", "filepath", "url", "uri",
                "host", "server", "endpoint", "engine.")


def _stub_value_literal_for(col: str, force_string: bool = False, force_numeric: bool = False) -> str:
    """Pick a Python literal for a stub-row column based on the column name.

    `force_string=True` overrides the heuristic and always returns a
    string literal — used when downstream tools apply `.str.` accessors or
    string functions (Trim / Replace / Substring / etc.) to the column.

    `force_numeric=True` returns an int literal — used when downstream
    tools aggregate or arithmetic the column (sum / avg / mean / `+ - * /`
    in formulas), where a string value would crash with
    'agg function failed [how->mean,dtype->object]'.

    Mixed (default): columns whose names strongly suggest numeric use get
    an int (so arithmetic works); columns whose names suggest text get a
    string (so `.str.X` ops work). Date-like names get an ISO timestamp.
    """
    n = col.lower()
    # Path / URL / engine-var columns are ALWAYS string — override the
    # downstream scanner's numeric heuristic. Concatenation patterns
    # like `[Engine.WorkflowDirectory] + "style.yxdb"` look numeric to
    # the arithmetic regex but are actually string concat.
    if any(t in n for t in _PATH_HINTS):
        return '"/tmp/"'
    if force_numeric and not force_string:
        return "1"
    if "date" in n or "time" in n or "_at" in n:
        return '"2020-01-01"'
    if force_string:
        return '"1"'  # `"1"` parses to int via pd.to_numeric AND survives .str.X
    if "lat" in n:
        return "0.0"
    if "lon" in n or "lng" in n:
        return "0.0"
    numeric_hints = (
        "count", "qty", "quantity", "amount", "price", "_id", "id_",
        "num", "score", "rate", "total", "sum", "rank",
        "area", "sqmi", "sqkm", "sqft",
        "distance", "miles", "km", "feet", "pct", "percent", "ratio",
        "avg", "mean", "median", "stddev",
        "win", "lost", "loss", "size", "length",
        "year", "month", "day", "hour", "minute", "second",
        "salary", "revenue", "profit", "cost",
        "lock",
    )
    if any(t in n for t in numeric_hints):
        return "1"
    return '"x"'


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
        # Mixed pure-digit + prefixed tool_ids (after macro splicing) can't be
        # compared raw — bucket by isdigit, then by numeric/string within bucket.
        queue.sort(key=lambda n: (0, int(n.tool_id)) if n.tool_id.isdigit() else (1, n.tool_id))
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

    # Collapse connected In-DB subgraphs into a single warehouse_pipeline asset
    # per group. Subgraph detection runs BEFORE the per-tool loop so each
    # In-DB tool's tool_to_asset maps to the collapsed asset_name instead
    # of its own sql_transform CTAS. The actual YAML emit is DEFERRED until
    # after the per-tool loop so we can wire deps on external upstreams
    # (e.g. DataStreamIn assets) that are only mapped during the main loop.
    _indb_groups = build_warehouse_pipelines(wf)
    _collapsed_tool_ids: set = set()
    for _grp in _indb_groups:
        _collapsed_tool_ids |= _grp["tool_ids"]
        for _tid in _grp["tool_ids"]:
            tool_to_asset[_tid] = _grp["asset_name"]

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

        cols, string_typed, numeric_typed = _columns_needed_downstream(wf, origin_tool_id)
        if not cols:
            cols = ["col1", "col2", "col3"]  # safe non-empty default

        # Per-column placeholder values — keep them simple Python primitives
        # so the YAML round-trip is lossless (no repr() / eval() games).
        def _placeholder_value(c: str):
            if c in string_typed and c not in numeric_typed:
                return "x"
            if c in numeric_typed:
                return 1
            # Default: short non-empty string. NaN would survive `.str.` ops
            # only as object dtype, but pd.read_csv-style stubs already use "x".
            return "x"

        data_row = [_placeholder_value(c) for c in cols]

        # Detect a downstream Dynamic Rename in `first_row` mode. That mode
        # PROMOTES the first data row's VALUES into column names — consuming
        # the single row. When such a consumer exists, emit TWO rows: row 1 =
        # the literal column-name strings (which become headers after
        # promotion), row 2 = the per-column placeholder values.
        _dyn_rename_first_row = False
        for _e in wf.downstreams_of(origin_tool_id):
            _dn = wf.by_id().get(_e.dest_tool)
            if _dn is None:
                continue
            if "DynamicRename" not in (_dn.plugin or ""):
                continue
            _rm = _dn.config.find("RenameMode") or _dn.config.find(".//RenameMode")
            if _rm is not None and (_rm.text or "").strip().lower() == "firstrow":
                _dyn_rename_first_row = True
                break

        rows = [list(cols), data_row] if _dyn_rename_first_row else [data_row]
        emit_yaml(
            out_dir,
            pkg,
            "inline_dataframe",
            ph_name,
            {
                "columns": list(cols),
                "rows": rows,
                "group_name": "alteryx_unmapped",
                "description": (
                    f"Placeholder for unmapped Alteryx tool {origin_tool_id}. "
                    "Replace with the real upstream — see MIGRATION.md."
                ),
            },
        )
        return ph_name

    _BROWSE_PLUGINS = {
        "AlteryxBasePluginsGui.Browse.Browse",
        "AlteryxBasePluginsGui.BrowseV2.BrowseV2",
    }
    _by_id = wf.by_id()

    def _only_feeds_browse(tool_id: str) -> bool:
        downs = wf.downstreams_of(tool_id)
        if not downs:
            return False
        for e in downs:
            tgt = _by_id.get(e.dest_tool)
            if tgt is None or tgt.plugin not in _BROWSE_PLUGINS:
                return False
        return True

    for node in ordered:
        # Tools collapsed into a warehouse_pipeline group already have their
        # asset_name set in tool_to_asset; skip the per-tool sql_transform
        # mapping for them.
        if node.tool_id in _collapsed_tool_ids:
            continue

        # Sources whose only downstream is Alteryx Browse (a data-preview UI
        # tool we skip) become orphan assets in Dagster — Browse has no
        # Dagster equivalent. Drop them so the asset graph reflects the
        # real pipeline, not analyst scratch.
        if node.plugin == "AlteryxBasePluginsGui.TextInput.TextInput" and _only_feeds_browse(node.tool_id):
            unmapped_results.append((
                node.tool_id, node.plugin,
                "TextInput feeds only Browse (data-preview UI). Dropped — no "
                "Dagster equivalent (materialize the asset to see preview metadata).",
                "If you want to keep this as a reference table, manually emit "
                "an inline_dataframe component for it.",
            ))
            continue

        # Resolve upstreams in connection-anchor order so e.g. Join's Left/Right
        # arrive deterministically. Special-case Union-style `#N` anchors:
        # the plain `Input` anchor is anchor #1 (first input), then `#2`, `#3`,
        # etc. follow numerically. Default-key alphabetical sort would put
        # `Input` AFTER `#2`/`#3`/`#4` — wrong order for multi-input unions.
        def _anchor_sort_key(e):
            a = e.dest_anchor
            if a == "Input":
                return (0, 0, "")  # primary anchor first
            if a.startswith("#"):
                try:
                    return (0, int(a[1:]), "")
                except ValueError:
                    pass
            return (1, 0, a)  # everything else: alpha-sorted in third slot
        incoming_edges = sorted(
            wf.upstreams_of(node.tool_id),
            key=lambda e: (_anchor_sort_key(e), e.origin_tool),
        )
        # Rewire Filter's `.False` anchor for cascading-router patterns.
        # Alteryx Filter has two output anchors (True / False); chains
        # like `51.True → 37 (drivers); 37.False → 36 (bodies_karts);
        # 36.False → 38 (tires); …` route each row to exactly one of the
        # downstream Selects via the un-matched branch. Our `filter`
        # component only emits the TRUE rows, so the `.False` downstream
        # gets the wrong dataframe. Rewire each `<filter>.False → X.Input`
        # edge to instead read from `<filter>`'s OWN INPUT — X then
        # applies its own filter condition on the full source, producing
        # the same effective per-row routing without needing a multi-
        # output anchor.
        _by_id = wf.by_id()
        def _maybe_rewire_false_cascade(e):
            # Walk UP through every consecutive Filter `.False` link until
            # we hit either a non-Filter upstream or a non-`.False` anchor.
            # This collapses a long cascade `51.True → 37.F → 36.F → 38.F
            # → 39 (gliders)` so each cascaded filter reads from the ORIGINAL
            # source (51) — they then apply their own conditions on the full
            # slice, matching what Alteryx's per-row routing does
            # semantically without the multi-anchor TRUE/FALSE outputs.
            origin_tool = e.origin_tool
            origin_anchor = e.origin_anchor
            seen: set = set()
            while True:
                up_node = _by_id.get(origin_tool)
                if up_node is None or "Filter" not in (up_node.plugin or ""):
                    break
                if origin_anchor not in ("False", "F"):
                    break
                if origin_tool in seen:  # cycle guard
                    break
                seen.add(origin_tool)
                parent_in = next(
                    (pe for pe in wf.upstreams_of(origin_tool)
                     if pe.dest_anchor in ("Input", "")),
                    None,
                )
                if parent_in is None:
                    break
                origin_tool = parent_in.origin_tool
                origin_anchor = parent_in.origin_anchor
            if (origin_tool, origin_anchor) == (e.origin_tool, e.origin_anchor):
                return e
            return AlteryxEdge(
                origin_tool=origin_tool,
                origin_anchor=origin_anchor,
                dest_tool=e.dest_tool,
                dest_anchor=e.dest_anchor,
            )
        incoming_edges = [_maybe_rewire_false_cascade(e) for e in incoming_edges]

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

        # Resolve bundled file_path references. When the source workflow
        # was a .yxzp/.yxmz, the importer captured `file_path` as the
        # relative path inside the bundle (e.g. `Inputs/drivers.csv`,
        # `Not Your Average Mario Kart\Output.yxdb`). After import the
        # extraction tempdir is gone, so that relative path is dead.
        # Copy bundled files into `<out_dir>/data/` and rewrite to a
        # stable absolute path.
        _file_path = result.attributes.get("file_path") if isinstance(result.attributes, dict) else None
        if (
            isinstance(_file_path, str)
            and _file_path
            and not _file_path.startswith(("odbc:", "oledb:", "aka:", "Provider="))
            and wf.source_dir is not None
        ):
            # Normalize Windows-style backslashes from Alteryx XML.
            _normalized = _file_path.replace("\\", "/")
            _candidate = wf.source_dir / _normalized
            try:
                if _candidate.is_file():
                    _data_dir = out_dir / "data"
                    _data_dir.mkdir(parents=True, exist_ok=True)
                    _dest = _data_dir / _candidate.name
                    if not _dest.exists() or _dest.stat().st_size != _candidate.stat().st_size:
                        import shutil as _sh
                        _sh.copy2(_candidate, _dest)
                    # Emit RELATIVE path (`data/<basename>`) so the project
                    # stays portable — the path resolves against Dagster's
                    # CWD which is the project root. Absolute paths break
                    # the moment the project gets moved / combined into a
                    # larger workspace.
                    result.attributes["file_path"] = f"data/{_dest.name}"
            except OSError:
                pass

        # Batch-macro: when this node was spliced from an Alteryx batch
        # macro with known iteration values, add static-partition attrs
        # and (for the ControlParam-substituted Filter) rewrite the
        # hardcoded condition to read context.partition_key. Dagster
        # then runs the macro chain once per partition value, and the
        # unpartitioned downstream auto-unions via AllPartitionMapping.
        if (
            node.batch_macro_parent_id
            and node.batch_macro_control_values
            and result.component_id not in ("(inline_python)",)
        ):
            attrs = result.attributes
            attrs.setdefault("partition_type", "static")
            # Components expect partition_values as a comma-separated
            # string (Optional[str] on the model, parsed back into a list
            # at runtime). Don't emit a Python list — pydantic rejects.
            attrs.setdefault(
                "partition_values",
                ",".join(node.batch_macro_control_values),
            )
            # Filter substitution: if this filter's condition is
            # `df['<control_field>'] == '<value>'` where value is in the
            # control_values list, replace with context.partition_key.
            if result.component_id == "filter" and node.batch_macro_control_field:
                _cond = attrs.get("condition")
                if isinstance(_cond, str):
                    _new_cond = _rewrite_controlparam_filter(
                        _cond,
                        field=node.batch_macro_control_field,
                        values=node.batch_macro_control_values,
                    )
                    if _new_cond != _cond:
                        attrs["condition"] = _new_cond
                        result.notes.append(
                            f"Batch macro: filter condition rewired from "
                            f"hardcoded `{_cond}` to `context.partition_key` "
                            f"so it varies per-partition over "
                            f"{node.batch_macro_control_values}."
                        )

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

    # Deferred warehouse_pipeline emit — runs AFTER the per-tool loop so
    # we can resolve external upstreams (e.g. DataStreamIn tools) to
    # their emitted asset names via tool_to_asset, then wire them as
    # `deps` on the warehouse_pipeline asset.
    for _grp in _indb_groups:
        _attrs = dict(_grp["attrs"])
        _ext_ups = _grp.get("upstream_tool_ids") or []
        _dep_assets = [tool_to_asset[t] for t in _ext_ups if t in tool_to_asset]
        if _dep_assets:
            _attrs["deps"] = _dep_assets
        _path = emit_yaml(
            out_root=out_dir,
            pkg=pkg,
            component_id="warehouse_pipeline",
            asset_name=_grp["asset_name"],
            attributes=_attrs,
        )
        files_written.append(_path)
        component_ids_used.append("warehouse_pipeline")
        _note = f"Collapsed {len(_grp['tool_ids'])} In-DB tools into one CTE-chain asset."
        if _dep_assets:
            _note += (
                f" External upstreams (Data Stream In, etc.) wired via "
                f"deps={_dep_assets}; the first step reads from the "
                "warehouse staging table they uploaded to."
            )
        mapped_results.append((
            ",".join(sorted(_grp["tool_ids"])),
            "AlteryxConnectorsGui.InDb*",
            "warehouse_pipeline",
            _grp["asset_name"],
            [_note],
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

    # Alteryx workflows are eager by default — clicking Run executes every
    # tool unconditionally. Mirror that in Dagster by applying the eager
    # AutomationCondition to every imported asset via the
    # automation_condition_applicator infrastructure component. Users who
    # want different semantics (schedules / on_cron / freshness-based) can
    # remove this YAML or layer their own rules on top.
    if mapped_results:
        _eager_applicator_path = emit_yaml(
            out_dir, pkg,
            component_id="automation_condition_applicator",
            asset_name="alteryx_imported_eager_automation",
            attributes={
                "rules": [
                    {
                        "selection": "group:alteryx_imported",
                        "preset": "eager",
                    },
                ],
                "preserve_existing": True,
                "group_name": "alteryx_imported",
            },
        )
        files_written.append(_eager_applicator_path)
        if "automation_condition_applicator" not in component_ids_used:
            component_ids_used.append("automation_condition_applicator")
        # Emit a tiny definitions.py shim that wires the applicator helper.
        # Without this call the applicator is a no-op (the helper has to be
        # invoked from the project's definitions.py).
        _emit_definitions_shim(out_dir, pkg)

    report = emit_migration_report(
        out_dir,
        yxmd_source=str(yxmd_path),
        mapped=mapped_results,
        unmapped=unmapped_results,
        batch_macros=getattr(wf, "batch_macros", []) or [],
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


def _emit_definitions_shim(out_dir: Path, pkg: str) -> None:
    """Write a `definitions.py` in `src/<pkg>/` that wires the
    automation_condition_applicator into the project's load_from_defs_folder
    output. Without this call the applicator YAML is a no-op (the helper has
    to be invoked from the project's definitions.py — there's no automatic
    component post-processing hook in Dagster).

    Skip if the user already has a definitions.py — don't overwrite their work.
    """
    defs_py = out_dir / "src" / pkg / "definitions.py"
    if defs_py.exists():
        return
    defs_py.parent.mkdir(parents=True, exist_ok=True)
    defs_py.write_text(
        '"""Auto-generated by alteryx-to-dagster.\n\n'
        'Loads the components emitted under defs/ and applies the\n'
        'automation_condition_applicator rules so the imported assets\n'
        'run eagerly (matching Alteryx\'s default execution model).\n'
        '\n'
        'You can extend this file with your own assets, jobs, schedules,\n'
        'sensors — just keep the load_from_defs_folder + applicator wiring.\n'
        '"""\n'
        'from pathlib import Path\n'
        'from dagster import definitions, load_from_defs_folder\n'
        '\n'
        'try:\n'
        '    from dagster_community_components.helpers.automation_applicator import (\n'
        '        apply_automation_conditions_from_defs_folder,\n'
        '    )\n'
        '    _HAS_APPLICATOR = True\n'
        'except ImportError:\n'
        '    _HAS_APPLICATOR = False\n'
        '\n'
        '\n'
        '@definitions\n'
        'def defs():\n'
        '    base = load_from_defs_folder(path_within_project=Path(__file__).parent)\n'
        '    if _HAS_APPLICATOR:\n'
        '        return apply_automation_conditions_from_defs_folder(\n'
        '            base, Path(__file__).parent\n'
        '        )\n'
        '    return base\n'
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
        "# Cloud-portability notice — read before deploying to Dagster+ Cloud\n\n"
        "The imported project references **local filesystem paths** in one or "
        "more emitted assets. Local paths work for `dg dev` on your laptop, "
        "but break the moment the project deploys anywhere else.\n\n"
        "## What won't work in Dagster+ Cloud / Hybrid / Serverless\n\n"
        "These patterns FAIL the second the project leaves your laptop:\n\n"
        "1. **Absolute Windows paths** (`C:\\Users\\...`) — Dagster+ runners "
        "are Linux containers; the C:\\ drive doesn't exist.\n"
        "2. **Absolute POSIX paths to user / Downloads / Desktop dirs** "
        "(`/Users/<you>/Downloads/file.csv`, `~/data/file.xlsx`) — runner "
        "containers don't have your home directory mounted.\n"
        "3. **Relative paths assuming a current-working-dir** "
        "(`./data/file.csv`) — Dagster+ runs aren't guaranteed to start "
        "in your project root.\n"
        "4. **Local SQLite / DuckDB files** (`file.db`, `file.duckdb`) — "
        "ephemeral container filesystems get wiped between runs; data is lost.\n"
        "5. **Bundled .yxdb files** in your .yxzp / .yxmz package — the "
        "Alteryx binary format isn't accessible from the running container "
        "unless you copy it to cloud storage first.\n"
        "6. **Local Excel/CSV files referenced by their absolute laptop path** "
        "(this notice is firing because we detected one of these).\n\n"
        "## Recommended fix — cloud object storage URLs\n\n"
        "Copy each referenced file (or its converted equivalent — Parquet is "
        "usually the right swap for .yxdb / .csv on size+speed grounds) into "
        "cloud object storage, then update `file_path` in each defs.yaml:\n\n"
        "| Cloud | URL shape | Components that accept it |\n"
        "|---|---|---|\n"
        "| AWS S3 | `s3://my-bucket/alteryx-exports/customers.parquet` | `dataframe_from_*`, `file_ingestion` |\n"
        "| Google Cloud Storage | `gs://my-bucket/alteryx-exports/customers.parquet` | `dataframe_from_*`, `file_ingestion` |\n"
        "| Azure Blob | `abfs://container@account.dfs.core.windows.net/path/file.parquet` | `dataframe_from_*`, `file_ingestion` |\n"
        "| Snowflake stage | `@MY_STAGE/path/customers.parquet` | via `sql_transform` / `warehouse_pipeline` |\n"
        "| Databricks Unity Catalog | `abfss://container@account.dfs.core.windows.net/path` | `dataframe_from_*` |\n\n"
        "Most `dataframe_from_*` and `file_ingestion` components accept the "
        "same URL forms pandas / pyarrow / fsspec do, so the swap is usually "
        "one `file_path:` line per asset.\n\n"
        "## Dagster+ Cloud specifics\n\n"
        "- Use **Dagster+ Resources** (the `Resources` tab in the UI) to bind "
        "cloud credentials at deploy time — never hard-code AWS_ACCESS_KEY / "
        "GCP service-account JSON in defs.yaml.\n"
        "- For per-environment config (dev/staging/prod buckets), use "
        "`from_run_config:` on `file_ingestion` and supply the URL via a "
        "sensor / schedule's `RunConfig` at materialization time.\n"
        "- For files >100 MB, set the asset's `compute_kind: warehouse` and "
        "route through `sql_transform` / `warehouse_pipeline` against your "
        "Snowflake / BigQuery / Redshift cluster instead of pulling into "
        "pandas — Dagster+ Serverless workers have memory limits.\n\n"
        "## Local paths we detected in this import\n\n"
    )
    body += "\n".join(sorted(set(found_paths))) + "\n"
    md.write_text(body)
    files_written.append(md)
