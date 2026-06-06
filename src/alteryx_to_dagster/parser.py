"""Parse .yxmd (XML) and .yxmz (zip of XML) into a graph dataclass.

Alteryx workflow format (the parts we care about):

  <AlteryxDocument yxmdVer="...">
    <Nodes>
      <Node ToolID="1">
        <GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter">
          <Position x="..." y="..."/>
        </GuiSettings>
        <Properties>
          <Configuration>...tool-specific config...</Configuration>
        </Properties>
        <Annotation DefaultAnnotationText="optional_label"/>
      </Node>
      ...
    </Nodes>
    <Connections>
      <Connection>
        <Origin ToolID="1" Connection="Output"/>
        <Destination ToolID="2" Connection="Input"/>
      </Connection>
      ...
    </Connections>
  </AlteryxDocument>

Filter tool's True / False outputs are both legal Connection names — we
preserve that on the edge so the mapper can decide which downstream
asset depends on which branch.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AlteryxNode:
    tool_id: str               # XML's ToolID (string; numeric in practice)
    plugin: str                # e.g. "AlteryxBasePluginsGui.Filter.Filter"
    annotation: Optional[str]  # user-set tool label, often useful as asset_name
    config: ET.Element         # raw <Configuration> element for the mapper
    position: Dict[str, float] # {"x": ..., "y": ...} — kept for layout debugging

    @property
    def plugin_short(self) -> str:
        """e.g. 'Filter' from 'AlteryxBasePluginsGui.Filter.Filter'."""
        parts = self.plugin.split(".")
        return parts[-1] if parts else self.plugin


@dataclass
class AlteryxEdge:
    origin_tool: str           # upstream ToolID
    origin_anchor: str         # e.g. "Output" / "True" / "False" / "Left" / "Join"
    dest_tool: str             # downstream ToolID
    dest_anchor: str           # e.g. "Input" / "Left" / "Right"


@dataclass
class AlteryxWorkflow:
    yxmd_version: str
    nodes: List[AlteryxNode] = field(default_factory=list)
    edges: List[AlteryxEdge] = field(default_factory=list)
    # Populated when the source was a .yxzp / .yxmz bundle. Files we couldn't
    # convert (proprietary .yxdb data files, .yxmc macros) get listed here so
    # the importer can flag them in MIGRATION.md.
    bundled_data_files: List[str] = field(default_factory=list)
    bundled_macros: List[str] = field(default_factory=list)
    # The directory the source file lives in. macro_splicer.splice_macros()
    # uses this to resolve relative .yxmc references — Alteryx workflows
    # typically reference user macros sitting next to the .yxmd.
    source_dir: Optional[Path] = None

    def by_id(self) -> Dict[str, AlteryxNode]:
        return {n.tool_id: n for n in self.nodes}

    def upstreams_of(self, tool_id: str) -> List[AlteryxEdge]:
        return [e for e in self.edges if e.dest_tool == tool_id]

    def downstreams_of(self, tool_id: str) -> List[AlteryxEdge]:
        return [e for e in self.edges if e.origin_tool == tool_id]


def parse_workflow(path: str | Path) -> AlteryxWorkflow:
    """Read an Alteryx workflow. Supports:

      - `.yxmd` — raw XML.
      - `.yxmz` — zip containing exactly one .yxmd at the root.
      - `.yxzp` — Alteryx Package: zip bundling one or more .yxmd plus
        `.yxdb` data files, sample data, macros (`.yxmc`), and docs.
        We parse the first .yxmd at the shallowest depth (Alteryx convention)
        and list any `.yxdb` data files in `wf.bundled_data_files` so the
        importer can flag them in MIGRATION.md (the .yxdb binary format
        is proprietary and doesn't translate to a Dagster source).
    """
    p = Path(path)
    ext = p.suffix.lower()
    bundled_data_files: List[str] = []
    bundled_macros: List[str] = []

    if ext in (".yxmz", ".yxzp"):
        with zipfile.ZipFile(p) as z:
            members = z.namelist()
            yxmd_members = [n for n in members if n.lower().endswith(".yxmd")]
            if not yxmd_members:
                raise ValueError(f"{p} ({ext}) bundle contains no .yxmd inside.")
            # Prefer the shallowest .yxmd (Alteryx convention for the root workflow).
            yxmd_members.sort(key=lambda n: (n.count("/"), n))
            with z.open(yxmd_members[0]) as f:
                xml_bytes = f.read()
            # Inventory anything else worth flagging.
            bundled_data_files = [n for n in members if n.lower().endswith(".yxdb")]
            bundled_macros = [n for n in members if n.lower().endswith(".yxmc")]
        tree = ET.parse(io.BytesIO(xml_bytes))
    else:
        tree = ET.parse(p)

    root = tree.getroot()
    wf = _from_root(root)
    wf.bundled_data_files = bundled_data_files
    wf.bundled_macros = bundled_macros
    wf.source_dir = p.parent if p.exists() else None
    return wf


def _from_root(root: ET.Element) -> AlteryxWorkflow:
    wf = AlteryxWorkflow(yxmd_version=root.attrib.get("yxmdVer", "unknown"))

    nodes_el = root.find("Nodes")
    if nodes_el is not None:
        for node_el in nodes_el.findall("Node"):
            tool_id = node_el.attrib.get("ToolID", "")
            gui = node_el.find("GuiSettings")
            plugin = (gui.attrib.get("Plugin", "") if gui is not None else "")

            # Detect macro references — Alteryx stores them on
            # <EngineSettings Macro="path/to/something.yxmc"/>. The Plugin
            # attribute is typically empty for these, since the macro itself
            # provides the implementation. Synthesize a sentinel plugin
            # string so the mapper can route to either macro-splicing or
            # stock-macro components.
            if not plugin:
                eng = node_el.find("EngineSettings")
                if eng is not None:
                    macro_ref = eng.attrib.get("Macro") or ""
                    if macro_ref:
                        plugin = f"AlteryxMacro::{macro_ref}"

            pos_el = gui.find("Position") if gui is not None else None
            position = {
                "x": float(pos_el.attrib.get("x", 0)) if pos_el is not None else 0.0,
                "y": float(pos_el.attrib.get("y", 0)) if pos_el is not None else 0.0,
            }

            ann_el = node_el.find("Annotation")
            annotation = (
                ann_el.attrib.get("DefaultAnnotationText")
                if ann_el is not None
                else None
            )
            # Configuration lives under <Properties><Configuration>.
            props_el = node_el.find("Properties")
            cfg_el = props_el.find("Configuration") if props_el is not None else None
            if cfg_el is None:
                cfg_el = ET.Element("Configuration")  # empty placeholder

            wf.nodes.append(AlteryxNode(
                tool_id=tool_id,
                plugin=plugin,
                annotation=annotation,
                config=cfg_el,
                position=position,
            ))

    conns_el = root.find("Connections")
    if conns_el is not None:
        for conn_el in conns_el.findall("Connection"):
            origin = conn_el.find("Origin")
            dest = conn_el.find("Destination")
            if origin is None or dest is None:
                continue
            wf.edges.append(AlteryxEdge(
                origin_tool=origin.attrib.get("ToolID", ""),
                origin_anchor=origin.attrib.get("Connection", "Output"),
                dest_tool=dest.attrib.get("ToolID", ""),
                dest_anchor=dest.attrib.get("Connection", "Input"),
            ))

    return wf
