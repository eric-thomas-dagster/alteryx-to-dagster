"""Collapse connected In-DB subgraphs into a single warehouse_pipeline asset.

Alteryx In-DB tools (`AlteryxConnectorsGui.InDb*`) push computation into the
warehouse — Filter / Formula / Select / Summarize / Join / Union / Sample
are SQL-pushdown ops that share one query plan in Alteryx. The per-tool
sql_transform mapping is correct (each tool CTASs an intermediate table)
but loses the single-query semantic: every step round-trips to the
warehouse as its own statement.

This module detects each connected component of In-DB tools and emits
ONE `warehouse_pipeline` asset per sink (Stream Out / Write Data) with
each upstream In-DB step compiled to a `steps:` entry that the pipeline
component compiles into a CTE chain.

Stream Out is the boundary where data leaves the warehouse — sinks with
`return_dataframe: true` make the asset return a pandas DataFrame so
downstream non-In-DB tools can consume it.
"""
import xml.etree.ElementTree as _ET
from typing import Dict, List, Optional, Set, Tuple

from .parser import AlteryxNode, AlteryxWorkflow


_INDB_PLUGIN_PREFIX = "AlteryxConnectorsGui.InDb"
_STREAM_OUT_PLUGINS = {
    "AlteryxConnectorsGui.InDbStreamOut.InDbStreamOut",
    "AlteryxConnectorsGui.InDb.InDbStreamOut",
}
_WRITE_DATA_PLUGINS = {
    "AlteryxConnectorsGui.InDbWriteData.InDbWriteData",
    "AlteryxConnectorsGui.InDb.InDbWriteData",
}
_CONNECTION_PLUGINS = {
    "AlteryxConnectorsGui.InDbConnectionManager.InDbConnectionManager",
}
_SINK_PLUGINS = _STREAM_OUT_PLUGINS | _WRITE_DATA_PLUGINS


def _is_indb(node: AlteryxNode) -> bool:
    return (
        node.plugin.startswith(_INDB_PLUGIN_PREFIX)
        and node.plugin not in _CONNECTION_PLUGINS
    )


def _strip_brackets(s: str) -> str:
    """Remove Alteryx [Col] brackets from a SQL-ish expression."""
    import re
    return re.sub(r"\[([^\[\]]+)\]", r"\1", s)


def _detect_subgraphs(wf: AlteryxWorkflow) -> List[Set[str]]:
    """Return a list of connected In-DB subgraphs (each is a set of tool_ids).
    Connections are followed through any edge between two In-DB tools.
    """
    by_id = wf.by_id()
    indb_ids = {n.tool_id for n in wf.nodes if _is_indb(n)}

    # Adjacency restricted to In-DB nodes.
    adj: Dict[str, Set[str]] = {tid: set() for tid in indb_ids}
    for e in wf.edges:
        if e.origin_tool in indb_ids and e.dest_tool in indb_ids:
            adj[e.origin_tool].add(e.dest_tool)
            adj[e.dest_tool].add(e.origin_tool)

    seen: Set[str] = set()
    groups: List[Set[str]] = []
    for tid in indb_ids:
        if tid in seen:
            continue
        # BFS
        comp: Set[str] = {tid}
        queue: List[str] = [tid]
        while queue:
            cur = queue.pop()
            for nxt in adj[cur]:
                if nxt not in comp:
                    comp.add(nxt)
                    queue.append(nxt)
        seen |= comp
        if len(comp) >= 2:  # singleton In-DB tool isn't worth collapsing
            groups.append(comp)
    return groups


def _topo_within(group: Set[str], wf: AlteryxWorkflow) -> List[str]:
    """Kahn's algorithm restricted to the group."""
    incoming: Dict[str, int] = {tid: 0 for tid in group}
    for e in wf.edges:
        if e.origin_tool in group and e.dest_tool in group:
            incoming[e.dest_tool] = incoming.get(e.dest_tool, 0) + 1
    queue = [tid for tid, n in incoming.items() if n == 0]
    out: List[str] = []
    while queue:
        queue.sort(key=lambda t: int(t) if t.isdigit() else t)
        cur = queue.pop(0)
        out.append(cur)
        for e in wf.downstreams_of(cur):
            if e.dest_tool in group:
                incoming[e.dest_tool] -= 1
                if incoming[e.dest_tool] == 0:
                    queue.append(e.dest_tool)
    return out


def _step_id_for(node: AlteryxNode) -> str:
    """Generate a stable step id for a node (used as `id:` in steps[])."""
    return f"step_{node.tool_id}"


def _step_source_for(node: AlteryxNode, upstream_step_ids: List[str]) -> Optional[Dict]:
    """For a step that takes one or two upstream steps, build the `source:` dict.
    Returns None for InDbInput (handled separately as a kind=table source)."""
    if len(upstream_step_ids) == 0:
        return None
    if len(upstream_step_ids) == 1:
        return {"kind": "ref", "ref": upstream_step_ids[0]}
    # Multi-upstream — handled at op level (join/union)
    return {"kind": "ref", "ref": upstream_step_ids[0]}


def _node_to_step(
    node: AlteryxNode,
    upstream_step_ids: List[str],
) -> Optional[Dict]:
    """Convert a single In-DB node into a `steps:` dict entry. Returns None
    for nodes that don't generate an intermediate step (e.g. ConnectionManager).
    """
    cfg = node.config
    plugin = node.plugin
    step_id = _step_id_for(node)

    if plugin.endswith(".InDbInput"):
        # ElementTree elements with text-only content are FALSY, so we can't
        # use `or` chains here — must check `is not None` explicitly.
        query_el = cfg.find("Query")
        if query_el is None:
            query_el = cfg.find("TableSelect")
        if query_el is None:
            query_el = cfg.find("Table")
        raw_sql = (query_el.text or "").strip() if query_el is not None and query_el.text else ""
        if raw_sql:
            return {
                "id": step_id,
                "source": {"kind": "sql", "sql": raw_sql},
                "operations": [],
            }
        table_el = cfg.find("TableName")
        if table_el is None:
            table_el = cfg.find("Source")
        table = (table_el.text or "TODO_source_table").strip() if table_el is not None and table_el.text else "TODO_source_table"
        return {
            "id": step_id,
            "source": {"kind": "table", "table": table},
            "operations": [],
        }

    if plugin.endswith(".InDbFilter"):
        expr_el = cfg.find("Expression")
        raw = (expr_el.text or "").strip() if expr_el is not None and expr_el.text else "TRUE"
        return {
            "id": step_id,
            "source": _step_source_for(node, upstream_step_ids),
            "operations": [{"op": "filter", "expression": _strip_brackets(raw)}],
        }

    if plugin.endswith(".InDbFormula"):
        ff_el = cfg.find("FormulaFields")
        new_cols: Dict[str, str] = {}
        if ff_el is not None:
            for f in ff_el.findall("FormulaField"):
                col = f.attrib.get("field", "")
                expr = _strip_brackets(f.attrib.get("expression", ""))
                if col:
                    new_cols[col] = expr
        return {
            "id": step_id,
            "source": _step_source_for(node, upstream_step_ids),
            "operations": [{"op": "with_columns", "expressions": new_cols}] if new_cols else [],
        }

    if plugin.endswith(".InDbSelect"):
        sf_el = cfg.find("SelectFields") or cfg.find("Fields")
        keep: List[str] = []
        renames: Dict[str, str] = {}
        if sf_el is not None:
            for f in sf_el.findall("SelectField") + sf_el.findall("Field"):
                fn = f.attrib.get("field")
                sel = f.attrib.get("selected", "True").lower() != "false"
                rn = f.attrib.get("rename")
                if fn and sel and fn != "*Unknown":
                    keep.append(rn or fn)
                    if rn and rn != fn:
                        renames[fn] = rn
        ops: List[Dict] = []
        if keep:
            ops.append({"op": "select", "columns": keep})
        return {
            "id": step_id,
            "source": _step_source_for(node, upstream_step_ids),
            "operations": ops,
        }

    if plugin.endswith(".InDbSummarize"):
        group_by: List[str] = []
        aggs: List[Dict[str, str]] = []
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
                    aggs.append({"col": field_name, "agg": action, "as": out_name})
        return {
            "id": step_id,
            "source": _step_source_for(node, upstream_step_ids),
            "operations": [{"op": "group_by", "by": group_by, "aggregations": aggs}],
        }

    if plugin.endswith(".InDbJoin"):
        join_left: List[str] = []
        join_right: List[str] = []
        jf_el = cfg.find("JoinInfo")
        if jf_el is not None:
            for f in jf_el.findall("Field"):
                l = f.attrib.get("field")
                r = f.attrib.get("field2") or l
                if l:
                    join_left.append(l)
                    join_right.append(r)
        if len(upstream_step_ids) < 2:
            return None
        return {
            "id": step_id,
            "source": {"kind": "ref", "ref": upstream_step_ids[0]},
            "operations": [{
                "op": "join",
                "with": {"ref": upstream_step_ids[1]},
                "how": "inner",
                "left_on": join_left,
                "right_on": join_right,
            }],
        }

    if plugin.endswith(".InDbUnion"):
        if not upstream_step_ids:
            return None
        return {
            "id": step_id,
            "source": {"kind": "ref", "ref": upstream_step_ids[0]},
            "operations": [{
                "op": "union",
                "with": [{"ref": u} for u in upstream_step_ids[1:]],
            }],
        }

    if plugin.endswith(".InDbSample"):
        n_el = cfg.find("N") or cfg.find("Records")
        n = int(n_el.text) if n_el is not None and n_el.text and n_el.text.isdigit() else 100
        return {
            "id": step_id,
            "source": _step_source_for(node, upstream_step_ids),
            "operations": [{"op": "limit", "n": n}],
        }

    # Sinks aren't steps — they're entries in `sinks:` (see _build_sinks).
    return None


def _find_connection_text(cfg) -> str:
    """Return the In-DB connection name text, walking through the two
    common element names. Avoids the ElementTree falsy-leaf trap with `or`."""
    cn = cfg.find("Connection")
    if cn is None:
        cn = cfg.find("ConnectionName")
    if cn is not None and cn.text:
        return cn.text.strip()
    return cfg.attrib.get("connection") or "DEFAULT"


def _connection_env_var(node: AlteryxNode) -> str:
    """Compute the env var name from the In-DB connection name (slugified)."""
    name = _find_connection_text(node.config)
    slug = "".join(c.upper() if c.isalnum() else "_" for c in name)
    slug = slug.strip("_").upper() or "DEFAULT"
    return f"{slug}_URL"


def _node_dialect_hint(node: AlteryxNode) -> str:
    """Infer SQL dialect from the connection name. Falls back to 'snowflake'."""
    name = _find_connection_text(node.config).lower()
    for k in ("snowflake", "bigquery", "redshift", "postgres", "mysql", "mssql", "duckdb", "databricks"):
        if k in name:
            return k
    return "snowflake"


def build_warehouse_pipelines(wf: AlteryxWorkflow) -> List[Dict]:
    """Detect each multi-tool In-DB subgraph and return a list of
    warehouse_pipeline asset definitions, one per group.

    Each returned dict has shape:
      {
        "asset_name": str,
        "tool_ids": Set[str],          # the In-DB tools subsumed
        "attrs": {                      # ready for emit_yaml
          "asset_name": ..., "dialect": ..., "database_url_env_var": ...,
          "steps": [...], "sinks": [...], "group_name": "alteryx_imported_indb",
        },
        "upstream_tool_ids": List[str], # external tools feeding this group
        "downstream_consumers": Dict[str, str],  # internal sink tool_id → external asset name OR sink table
      }
    """
    groups = _detect_subgraphs(wf)
    if not groups:
        return []

    by_id = wf.by_id()
    pipelines: List[Dict] = []

    for idx, group in enumerate(groups):
        ordered_ids = _topo_within(group, wf)
        # Build steps and sinks
        steps: List[Dict] = []
        sinks: List[Dict] = []
        # tool_id → list of upstream step_ids (only those in-group)
        upstream_step_map: Dict[str, List[str]] = {}
        for tid in ordered_ids:
            ups_in_group = [
                e.origin_tool
                for e in sorted(wf.upstreams_of(tid), key=lambda e: (e.dest_anchor, e.origin_tool))
                if e.origin_tool in group
            ]
            upstream_step_map[tid] = [_step_id_for(by_id[u]) for u in ups_in_group]

        # First pass: emit steps for non-sink nodes.
        sink_tool_ids: List[str] = []
        for tid in ordered_ids:
            node = by_id[tid]
            if node.plugin in _SINK_PLUGINS:
                sink_tool_ids.append(tid)
                continue
            step = _node_to_step(node, upstream_step_map[tid])
            if step is not None:
                steps.append(step)

        # Second pass: emit sinks.
        downstream_consumers: Dict[str, str] = {}
        for tid in sink_tool_ids:
            node = by_id[tid]
            # Pick the upstream step the sink consumes — it's the immediate
            # in-group upstream of the Stream Out / Write Data.
            ups = upstream_step_map[tid]
            from_step = ups[0] if ups else None
            if from_step is None:
                continue
            if node.plugin in _STREAM_OUT_PLUGINS:
                # Stream Out → return as DataFrame so downstream non-In-DB
                # tools can consume.
                sinks.append({
                    "id": f"sink_{node.tool_id}",
                    "from": from_step,
                    "return_dataframe": True,
                })
                downstream_consumers[node.tool_id] = f"sink_{node.tool_id}"
            elif node.plugin in _WRITE_DATA_PLUGINS:
                dest_el = node.config.find("TableName") or node.config.find("Destination") or node.config.find("OutputTable")
                dest = (dest_el.text or "TODO_dest_table").strip() if dest_el is not None and dest_el.text else "TODO_dest_table"
                sinks.append({
                    "id": f"sink_{node.tool_id}",
                    "from": from_step,
                    "table": dest,
                    "mode": "replace",
                })
                downstream_consumers[node.tool_id] = dest

        # Pick connection metadata from any node in the group.
        any_node = by_id[ordered_ids[0]]
        env_var = _connection_env_var(any_node)
        dialect = _node_dialect_hint(any_node)

        asset_name = f"warehouse_pipeline_{idx + 1}"
        attrs: Dict = {
            "asset_name": asset_name,
            "dialect": dialect,
            "database_url_env_var": env_var,
            "steps": steps,
            "sinks": sinks,
            "group_name": "alteryx_imported_indb",
        }
        pipelines.append({
            "asset_name": asset_name,
            "tool_ids": group,
            "attrs": attrs,
            "upstream_tool_ids": [],  # In-DB groups don't take external upstreams in current corpus
            "downstream_consumers": downstream_consumers,
        })

    return pipelines
