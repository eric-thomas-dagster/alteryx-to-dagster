"""Inline Alteryx macros (.yxmc) into the parent workflow's node/edge graph.

Alteryx macros are nested workflows. The parser flags any node whose
EngineSettings carries a `Macro="..."` attribute by synthesizing a
synthetic plugin string `AlteryxMacro::<filename>`. The splicer:

  1. Walks the parent workflow looking for those macro-reference nodes.
  2. Resolves the .yxmc on disk via `_resolve_macro_path`:
       - the literal path,
       - source_dir / relative-path,
       - source_dir / basename(macro),
       - source_dir / "macros" / basename(macro).
  3. Recursively parses the .yxmc (recursing through splice_macros so
     nested macros expand too — depth-limited to MAX_RECURSION_DEPTH=5).
  4. Renumbers every internal tool_id with a `m<parent_id>_` prefix so
     IDs don't collide with the parent's.
  5. Splices the renumbered nodes + edges into the parent.
  6. Re-routes Macro Input / Macro Output anchors:
       - Parent's incoming edge into the macro node → connects to the first
         downstream node of the macro's Macro Input tool.
       - Parent's outgoing edge from the macro node → connects from the
         upstream node of the macro's Macro Output tool.
  7. Drops the macro-reference node itself.

If we can't resolve the .yxmc, we leave the node in place — the mapper
will surface it as unmapped in MIGRATION.md.

Stock-macro routing (Cleanse, etc.) is handled separately in mapper.py via
`_STOCK_MACRO_COMPONENTS` — that lookup happens BEFORE splicing, so when a
parent references one of those macros we route to a registry component
instead of inlining.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Set

from .parser import AlteryxEdge, AlteryxNode, AlteryxWorkflow, parse_workflow


MAX_RECURSION_DEPTH = 5
MACRO_PLUGIN_PREFIX = "AlteryxMacro::"

# The plugin strings used by Alteryx's Macro I/O tools. The splicer treats
# these specially: they're the boundary between the macro's internals and
# the parent's connections, and get removed during splicing (rewriting
# adjacent edges to bypass them).
_MACRO_INPUT_PLUGINS = {
    "AlteryxBasePluginsGui.MacroInput.MacroInput",
    "AlteryxGuiToolkit.MacroInput.MacroInput",
}
_MACRO_OUTPUT_PLUGINS = {
    "AlteryxBasePluginsGui.MacroOutput.MacroOutput",
    "AlteryxGuiToolkit.MacroOutput.MacroOutput",
}


def splice_macros(
    wf: AlteryxWorkflow,
    source_dir: Path,
    *,
    stock_macro_basenames: Optional[Set[str]] = None,
    _depth: int = 0,
) -> AlteryxWorkflow:
    """Return a new AlteryxWorkflow with every macro node inlined.

    `stock_macro_basenames` — lowercase .yxmc basenames the mapper has a
    component routing for (Cleanse, etc.). Those are LEFT in place so the
    mapper can emit the stock component instead.

    `_depth` — internal recursion guard. Bumped each time we descend into
    a child macro.
    """
    if _depth >= MAX_RECURSION_DEPTH:
        # Bail: leave any remaining macro refs in place. Mapper will flag.
        return wf

    stock = stock_macro_basenames or set()

    macro_refs = [n for n in wf.nodes if n.plugin.startswith(MACRO_PLUGIN_PREFIX)]
    if not macro_refs:
        return wf

    spliced_nodes = list(wf.nodes)
    spliced_edges = list(wf.edges)

    for parent_node in macro_refs:
        macro_basename = parent_node.plugin[len(MACRO_PLUGIN_PREFIX):].lower()
        if macro_basename in stock:
            # Mapper handles this one via stock-macro routing.
            continue

        macro_path = _resolve_macro_path(parent_node, source_dir)
        if macro_path is None:
            # Couldn't find on disk — leave the node alone. The mapper will
            # mark it unmapped.
            continue

        try:
            child_wf = parse_workflow(macro_path)
        except (ET.ParseError, OSError):
            continue

        # Recurse so child's nested macros expand too.
        child_wf = splice_macros(
            child_wf,
            source_dir=macro_path.parent,
            stock_macro_basenames=stock,
            _depth=_depth + 1,
        )

        # Renumber child IDs with a parent-scoped prefix.
        prefix = f"m{parent_node.tool_id}_"
        renumbered_nodes = [
            AlteryxNode(
                tool_id=f"{prefix}{n.tool_id}",
                plugin=n.plugin,
                annotation=n.annotation,
                config=n.config,
                position=n.position,
            )
            for n in child_wf.nodes
        ]
        renumbered_edges = [
            AlteryxEdge(
                origin_tool=f"{prefix}{e.origin_tool}",
                origin_anchor=e.origin_anchor,
                dest_tool=f"{prefix}{e.dest_tool}",
                dest_anchor=e.dest_anchor,
            )
            for e in child_wf.edges
        ]

        # Identify Macro Input / Macro Output nodes (post-renumber).
        macro_inputs = [n for n in renumbered_nodes if n.plugin in _MACRO_INPUT_PLUGINS]
        macro_outputs = [n for n in renumbered_nodes if n.plugin in _MACRO_OUTPUT_PLUGINS]

        # Parent's existing in/out edges around this macro reference.
        parent_in_edges = [e for e in spliced_edges if e.dest_tool == parent_node.tool_id]
        parent_out_edges = [e for e in spliced_edges if e.origin_tool == parent_node.tool_id]

        # Wire parent-in → child's macro-input-downstreams.
        # Multi-anchor macros (batch macros with a Control anchor, or any
        # macro with multiple Input anchors) need anchor-aware matching —
        # otherwise the parent's Control edge gets cross-wired into the
        # data MacroInput and downstream filters consume the wrong upstream.
        # Match `parent_edge.dest_anchor` against the MacroInput's `<Name>`
        # (the anchor name the macro author assigned). Fall back to the
        # all-pairs cross-product when a MacroInput has no `<Name>` (older
        # single-anchor macros).
        def _macro_input_name(mi_node) -> str:
            name_el = mi_node.config.find("Name")
            return (name_el.text or "").strip() if name_el is not None and name_el.text else ""
        rewired_edges: List[AlteryxEdge] = []
        for mi in macro_inputs:
            mi_downstreams = [e for e in renumbered_edges if e.origin_tool == mi.tool_id]
            _mi_name = _macro_input_name(mi)
            # Pick parent edges whose destination anchor matches this
            # MacroInput's declared name. If none match (older single-anchor
            # macros where the parent edge anchor is "Input" by default),
            # fall back to all parent edges so we still wire SOMETHING.
            _matched = [p for p in parent_in_edges if p.dest_anchor == _mi_name]
            if not _matched and _mi_name:
                _matched = [p for p in parent_in_edges if not p.dest_anchor or p.dest_anchor == "Input"]
            if not _matched:
                _matched = list(parent_in_edges)
            for parent_in in _matched:
                for mi_e in mi_downstreams:
                    rewired_edges.append(AlteryxEdge(
                        origin_tool=parent_in.origin_tool,
                        origin_anchor=parent_in.origin_anchor,
                        dest_tool=mi_e.dest_tool,
                        dest_anchor=mi_e.dest_anchor,
                    ))

        # Wire child's macro-output-upstreams → parent-out.
        for mo in macro_outputs:
            mo_upstreams = [e for e in renumbered_edges if e.dest_tool == mo.tool_id]
            for parent_out in parent_out_edges:
                for mo_e in mo_upstreams:
                    rewired_edges.append(AlteryxEdge(
                        origin_tool=mo_e.origin_tool,
                        origin_anchor=mo_e.origin_anchor,
                        dest_tool=parent_out.dest_tool,
                        dest_anchor=parent_out.dest_anchor,
                    ))

        # Drop:
        #   - the parent node (it's now expanded inline)
        #   - the MacroInput / MacroOutput child nodes (their roles are now
        #     the rewired edges)
        #   - the renumbered edges touching MacroInput / MacroOutput
        #   - the parent's old in/out edges (replaced by rewired_edges)
        io_node_ids = {n.tool_id for n in macro_inputs + macro_outputs}
        spliced_nodes = (
            [n for n in spliced_nodes if n.tool_id != parent_node.tool_id]
            + [n for n in renumbered_nodes if n.tool_id not in io_node_ids]
        )
        spliced_edges = (
            [e for e in spliced_edges
             if e.dest_tool != parent_node.tool_id
             and e.origin_tool != parent_node.tool_id]
            + [e for e in renumbered_edges
               if e.origin_tool not in io_node_ids
               and e.dest_tool not in io_node_ids]
            + rewired_edges
        )

    return AlteryxWorkflow(
        yxmd_version=wf.yxmd_version,
        nodes=spliced_nodes,
        edges=spliced_edges,
        bundled_data_files=wf.bundled_data_files,
        bundled_macros=wf.bundled_macros,
        source_dir=getattr(wf, "source_dir", None),
    )


def _resolve_macro_path(node: AlteryxNode, source_dir: Path) -> Optional[Path]:
    """Find the .yxmc file referenced by a macro node.

    Looks up the raw macro reference from EngineSettings (the parser
    stashed it on the synthetic plugin string suffix), then tries:
      1. The literal path (if absolute and exists).
      2. source_dir / raw (relative refs alongside the workflow).
      3. source_dir / basename(raw) (when the .yxmc was relocated next to the .yxmd).
      4. source_dir / "macros" / basename(raw) (Alteryx user-macros default subdir).
      5. source_dir / "Macro" / basename(raw) (Alteryx .yxzp bundle convention).
      6. Recursive search anywhere under source_dir for a same-basename .yxmc
         (handles arbitrary nesting in bundled packages).

    Path normalization: Alteryx XML uses Windows backslashes (`Macro\X.yxmc`)
    even on macOS bundles, so we normalize separators before each candidate.
    """
    macro_ref = node.plugin[len(MACRO_PLUGIN_PREFIX):]  # raw filename, mixed case
    if not macro_ref:
        return None
    # Normalize Windows backslashes → forward slashes for cross-platform path math.
    macro_ref_norm = macro_ref.replace("\\", "/")
    basename = Path(macro_ref_norm).name
    candidates = [
        Path(macro_ref_norm),
        source_dir / macro_ref_norm,
        source_dir / basename,
        source_dir / "macros" / basename,
        source_dir / "Macro" / basename,
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    # Fallback: recursive search by basename. Works for .yxzp bundles whose
    # macros live in arbitrary subdirs.
    try:
        for found in source_dir.rglob(basename):
            if found.is_file() and found.suffix.lower() == ".yxmc":
                return found
    except OSError:
        pass
    return None
