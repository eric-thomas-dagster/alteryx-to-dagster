"""Minimal combiner: copies per-workflow audit defs into /tmp/wc_combined.

Each /tmp/wc_audit/<wf>/src/<wf>/defs/<tool>/defs.yaml is copied to
/tmp/wc_combined/src/wc_combined/defs/<wf>/<tool>/defs.yaml after:
  - rewriting `type:` to reference wc_combined.components.<id>.component.<Cls>
  - prefixing asset_name + upstream_asset_key + dependent keys with `<wf>/`

Assumes components are already installed under wc_combined and the project
skeleton (definitions.py, pyproject.toml, .venv) is in place.
"""
import re
import shutil
import yaml
from pathlib import Path

AUDIT_ROOT = Path("/tmp/wc_audit")
COMBINED_ROOT = Path("/tmp/wc_combined")
COMBINED_DEFS = COMBINED_ROOT / "src" / "wc_combined" / "defs"
COMBINED_COMPONENTS = COMBINED_ROOT / "src" / "wc_combined" / "components"
TEMPLATES_ROOT = Path(
    "/Users/ericthomas/dagster_components/dagster-component-templates "
)


def overlay_local_templates() -> int:
    """Copy local component.py edits over installed registry components.

    `dagster-component add --auto-install` pulls components from the
    published `dagster-community-components` package. Local template
    edits don't propagate until republished. Overlay step lets us
    iterate on component code without bumping the package version.
    """
    if not COMBINED_COMPONENTS.is_dir() or not TEMPLATES_ROOT.is_dir():
        return 0
    overlaid = 0
    # Find each installed component dir's matching template.
    for installed in sorted(COMBINED_COMPONENTS.iterdir()):
        if not installed.is_dir():
            continue
        cid = installed.name
        # Search the templates tree for a sibling dir of the same name.
        matches = list(TEMPLATES_ROOT.rglob(f"{cid}/component.py"))
        # Filter out demo/test/install copies; prefer the canonical template.
        candidates = [
            m for m in matches
            if "/.venv/" not in str(m)
            and "/demo" not in str(m)
            and "/tests/" not in str(m)
            and "/__pycache__/" not in str(m)
        ]
        if not candidates:
            continue
        src = candidates[0]
        dst = installed / "component.py"
        if src.read_text() != (dst.read_text() if dst.exists() else ""):
            dst.write_text(src.read_text())
            overlaid += 1
    return overlaid


def rewrite_type(t: str) -> str:
    # Strip any project-specific prefix: <pkg>.components.<id>.component.<Cls>
    m = re.match(r"^[\w]+\.components\.(\w+)\.component\.(\w+)$", t)
    if m:
        return f"wc_combined.components.{m.group(1)}.component.{m.group(2)}"
    if t.startswith("dagster_community_components."):
        # legacy umbrella ref — leave for now; the umbrella alias is registered.
        return t
    return t


def prefix_key(key: str, wf: str) -> str:
    if not key or "/" in key:
        return key  # already hierarchical
    return f"{wf}/{key}"


def normalize_column_lists(data: dict) -> dict:
    """Stringify non-str non-int values in any list-typed attribute.

    Importer-generated stubs sometimes inherit Alteryx connection labels
    like `[0, 1]` (the batch-macro iteration tuple), which YAML parses
    as tuples or lists. Force them to strings so pydantic's
    Union[str, int] field validators accept them.
    """
    attrs = data.get("attributes") or {}
    def _safe(c):
        # Preserve dicts/lists (e.g. rules on AutomationConditionApplicator).
        if isinstance(c, (dict, list)):
            return c
        # Force everything else to str.
        s = c if isinstance(c, str) else str(c)
        # Dagster's Resolvable templating eval-resolves quoted YAML strings:
        # "0, 1" gets parsed as a Python tuple expression. Defuse by
        # replacing the offending comma-space with an underscore.
        if "," in s:
            s = s.replace(", ", "_").replace(",", "_")
        return s
    for k, v in list(attrs.items()):
        if isinstance(v, list):
            attrs[k] = [_safe(c) for c in v]
    return data


_FILE_LOADER_TYPES = (
    "dataframe_from_yxdb", "file_ingestion", "dataframe_from_csv",
    "dataframe_from_excel", "dataframe_from_parquet",
)


def _file_exists_locally(file_path: str) -> bool:
    """Return True if file_path resolves to a file under /tmp/wc_combined/.

    The runtime resolves relative paths against the project root, so
    `data/sales.xlsx` means `/tmp/wc_combined/data/sales.xlsx`.
    """
    import os
    if not file_path:
        return False
    expanded = os.path.expandvars(file_path)
    if os.path.isabs(expanded):
        return os.path.exists(expanded)
    return os.path.exists(os.path.join(str(COMBINED_ROOT), expanded))


def soften_yxdb_missing(data: dict, wf: str, consumers_by_full_key: dict) -> dict:
    """Replace file-loader defs pointing at missing files with a smart
    inline-dataframe stub (columns inferred from downstream refs).

    Covers dataframe_from_yxdb, file_ingestion, dataframe_from_csv,
    dataframe_from_excel — anything with a `file_path` that the runtime
    would try to read. Imported workflows commonly hardcode Windows-only
    paths (C:\\Users\\Kiran\\Downloads\\...) or reference data files
    that aren't bundled in the .yxzp. Swap to InlineDataframeComponent
    with downstream-inferred columns so transforms still see expected
    column names instead of an empty 0-col DataFrame.
    """
    t = data.get("type", "")
    if not any(loader in t for loader in _FILE_LOADER_TYPES):
        return data
    attrs = data.get("attributes") or {}
    file_path = attrs.get("file_path", "")
    if _file_exists_locally(file_path):
        # File is there at combine time — let the runtime load it.
        # Set if_missing='empty' as a safety net for yxdb.
        if "dataframe_from_yxdb" in t and "if_missing" not in attrs:
            attrs["if_missing"] = "empty"
        return data
    # Missing → swap to smart inline stub.
    asset_name = attrs.get("asset_name")
    if not isinstance(asset_name, str) or not asset_name:
        return data
    full_key = prefix_key(asset_name, wf)
    cols, row = infer_stub_schema(full_key, consumers_by_full_key)
    return {
        "type": "wc_combined.components.inline_dataframe.component.InlineDataframeComponent",
        "attributes": {
            "asset_name": asset_name,
            "columns": cols,
            "rows": [row],
            "group_name": attrs.get("group_name"),
            "description": (
                f"Auto-stub replacing missing data file {file_path!r}. "
                f"Columns inferred from {len(cols)} downstream refs."
            ),
        },
    }


def strip_unsupported_partitions(data: dict) -> dict:
    """InlineDataframeComponent doesn't support partition_* fields.
    The importer's batch-macro splicer adds them to every spliced asset
    including text inputs; strip them from inline-dataframe defs so
    pydantic doesn't reject the YAML at load time.
    """
    if "inline_dataframe" not in data.get("type", ""):
        return data
    attrs = data.get("attributes") or {}
    for k in (
        "partition_type", "partition_values", "partition_start",
        "partition_date_column", "dynamic_partition_name",
        "partition_dimensions", "partition_static_dim",
        "partition_static_column",
    ):
        attrs.pop(k, None)
    return data


# ---------------------------------------------------------------------------
# Smart auto-stub generator
# ---------------------------------------------------------------------------

# Patterns that extract column refs from expression-style fields.
# `condition: df['X'] > 5` and `condition: df["X"] > 5` are both common.
_DF_BRACKET_REF = re.compile(r"""df\s*\[\s*['"]([^'"]+?)['"]\s*\]""")
# `expression: row['X'] + row['Y']` or `[X]` Alteryx-style.
_ROW_BRACKET_REF = re.compile(r"""row\s*\[\s*['"]([^'"]+?)['"]\s*\]""")
_ALTERYX_BRACKET = re.compile(r"\[([A-Za-z_][\w\s\-\.]*)\]")


def _extract_column_refs(attrs: dict) -> set[str]:
    """Pull every column name a downstream tool references from one set of attrs.

    Handles the common cases. Misses things like raw pandas eval strings —
    those mostly fall under condition/expression which we DO scan.
    """
    cols: set[str] = set()
    # Scalar column-name fields.
    for k in (
        "column", "pivot_column", "value_column", "date_column",
        "lat_column", "lng_column", "geometry_column",
        "points_geometry_column", "weights_column", "sort_column",
        "key_column", "partition_date_column", "partition_static_column",
        "name_column", "id_column",
    ):
        v = attrs.get(k)
        if isinstance(v, (str, int)):
            cols.add(str(v))
    # List-of-column fields.
    for k in (
        "columns", "drop_columns", "keep_columns", "index_columns",
        "group_by", "key_columns", "join_columns", "left_on", "right_on",
        "output_columns", "passthrough_columns", "id_columns", "by", "on",
        "subset", "id_vars", "value_vars", "sort_by",
    ):
        v = attrs.get(k)
        if isinstance(v, list):
            for c in v:
                if isinstance(c, (str, int)):
                    cols.add(str(c))
                elif isinstance(c, dict) and "column" in c:
                    # sort_by can be [{column: X, ascending: bool}, ...]
                    cols.add(str(c["column"]))
    # Rename / aggregations dict keys = input column names.
    for k in ("rename", "renames", "aliases", "column_aliases"):
        v = attrs.get(k)
        if isinstance(v, dict):
            for kk in v.keys():
                cols.add(str(kk))
    # aggregations: keys are usually output names, but values like {col, agg}
    # name the input column. Cover both shapes.
    aggs = attrs.get("aggregations")
    if isinstance(aggs, dict):
        for av in aggs.values():
            if isinstance(av, dict) and av.get("col"):
                cols.add(str(av["col"]))
    # Free-form expressions.
    for k in ("condition", "expression"):
        v = attrs.get(k)
        if isinstance(v, str):
            cols.update(_DF_BRACKET_REF.findall(v))
            cols.update(_ROW_BRACKET_REF.findall(v))
            cols.update(_ALTERYX_BRACKET.findall(v))
    return {c for c in cols if c and c not in {"context", "pd", "np", "df", "row"}}


def _build_consumer_index(wf_yamls):
    """asset_key (full <wf>/name) → list of (tool_name, attrs) that consume it."""
    consumers: dict[str, list] = {}
    for tool_dir, data in wf_yamls:
        attrs = data.get("attributes") or {}
        for fk in ("upstream_asset_key", "left_asset_key",
                   "right_asset_key", "regions_asset_key"):
            v = attrs.get(fk)
            if isinstance(v, str) and v:
                consumers.setdefault(v, []).append((tool_dir.name, attrs))
        for fk in ("deps", "upstream_asset_keys"):
            v = attrs.get(fk)
            if isinstance(v, list):
                for kk in v:
                    if isinstance(kk, str) and kk:
                        consumers.setdefault(kk, []).append((tool_dir.name, attrs))
    return consumers


def _typed_sample(name: str):
    """Pick a sample value whose type matches what downstream tools likely
    expect from `name`. Date-ish → ISO string; numeric-ish → 1; spatial-ish
    → empty POINT; else 'x'."""
    n = name.lower()
    if any(k in n for k in ("date", "time", "timestamp")):
        return "2024-01-01"
    if any(k in n for k in (
        "count", "qty", "quantity", "amount", "price", "value", "total",
        "sum", "avg", "mean", "median", "num", "percent", "rate", "score",
        "sales", "revenue", "profit", "loss", "weight", "size", "age",
        "year", "month", "day", "id", "sequence", "rank", "areasq",
        "length", "distance", "lat", "lng", "longitude", "latitude",
    )):
        return 1
    if any(k in n for k in ("geom", "polygon", "polyline", "spatial",
                            "centroid", "tradearea", "point")):
        return "POINT(0 0)"
    return "x"


def infer_stub_schema(missing_key: str, consumers: dict, max_depth: int = 6):
    """BFS downstream from `missing_key` and aggregate every column ref.

    Returns (columns, row) tuple. `columns` is a list of names; `row` is a
    list of typed sample values matching column order.
    """
    seen: set[str] = set()
    cols: set[str] = set()
    queue: list[tuple[str, int]] = [(missing_key, 0)]
    while queue:
        ak, depth = queue.pop(0)
        if ak in seen or depth > max_depth:
            continue
        seen.add(ak)
        for _tool, attrs in consumers.get(ak, []):
            cols.update(_extract_column_refs(attrs))
            # Walk this tool's outputs further.
            an = attrs.get("asset_name")
            if isinstance(an, str) and an:
                queue.append((an, depth + 1))
    if not cols:
        cols = {"col_a", "col_b"}
    ordered = sorted(cols)
    row = [_typed_sample(c) for c in ordered]
    return ordered, row


def rewrite_attrs(attrs: dict, wf: str) -> dict:
    for k in ("asset_name", "upstream_asset_key", "left_asset_key",
              "right_asset_key", "regions_asset_key"):
        if k in attrs and isinstance(attrs[k], str):
            attrs[k] = prefix_key(attrs[k], wf)
    # list-valued: deps, upstream_asset_keys
    for k in ("deps", "upstream_asset_keys"):
        if k in attrs and isinstance(attrs[k], list):
            attrs[k] = [prefix_key(v, wf) if isinstance(v, str) else v for v in attrs[k]]
    # group_name override → wf, but only if the source already had one
    # (some components like AutomationConditionApplicator don't accept it).
    if "group_name" in attrs:
        attrs["group_name"] = wf
    return attrs


def main() -> None:
    if not AUDIT_ROOT.exists():
        raise SystemExit(f"audit root missing: {AUDIT_ROOT}")
    COMBINED_DEFS.mkdir(parents=True, exist_ok=True)
    # Wipe stale workflow dirs
    for child in COMBINED_DEFS.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    n_workflows = 0
    n_assets = 0
    skipped_wfs: list[str] = []
    for wf_dir in sorted(AUDIT_ROOT.iterdir()):
        if not wf_dir.is_dir():
            continue
        wf = wf_dir.name
        # CLI mangles pkg names (lowercase, _yxmd/_yxzp suffix). Find
        # whatever subdir of src/ actually exists rather than guessing.
        src_root = wf_dir / "src"
        if not src_root.is_dir():
            continue
        pkg_dirs = [p for p in src_root.iterdir() if p.is_dir()]
        if not pkg_dirs:
            continue
        src_defs = pkg_dirs[0] / "defs"
        if not src_defs.is_dir():
            continue
        # Collect all YAMLs. The previous version filtered workflows with
        # numeric-stringable column lists, but inline_dataframe and
        # text_to_columns now accept Union[str, int], so the filter is
        # no longer needed for those. Other components may still trip;
        # surface those failures so we can patch them too.
        wf_yamls: list[tuple[Path, dict]] = []
        for tool_dir in sorted(src_defs.iterdir()):
            if not tool_dir.is_dir():
                continue
            src_yaml = tool_dir / "defs.yaml"
            if not src_yaml.is_file():
                continue
            data = yaml.safe_load(src_yaml.read_text()) or {}
            wf_yamls.append((tool_dir, data))
        # Collect (produced, referenced) keys to detect missing upstreams.
        produced: set[str] = set()
        referenced: set[str] = set()
        for tool_dir, data in wf_yamls:
            attrs = data.get("attributes") or {}
            an = attrs.get("asset_name")
            if isinstance(an, str):
                produced.add(prefix_key(an, wf))
            for fk in ("upstream_asset_key", "left_asset_key",
                       "right_asset_key", "regions_asset_key"):
                v = attrs.get(fk)
                if isinstance(v, str) and v:
                    referenced.add(prefix_key(v, wf))
            for fk in ("deps", "upstream_asset_keys"):
                v = attrs.get(fk)
                if isinstance(v, list):
                    for k in v:
                        if isinstance(k, str) and k:
                            referenced.add(prefix_key(k, wf))
        missing_keys = referenced - produced
        # Build consumer index BEFORE rewrite_attrs mutates them in place.
        # (The walk uses POST-prefix keys to match `missing_keys`, which are
        # also post-prefix. Snapshot here while attrs still have the
        # original workflow-local upstream_asset_key values that the BFS
        # can rewrite to full keys on the fly via prefix_key.)
        consumers_by_full_key: dict[str, list] = {}
        for tool_dir, data in wf_yamls:
            attrs = data.get("attributes") or {}
            # Snapshot the attrs as-is, then pre-prefix asset_name so the
            # BFS can re-queue using full keys (consumers_by_full_key is
            # keyed by full keys too).
            attrs_snap = dict(attrs)
            an = attrs_snap.get("asset_name")
            if isinstance(an, str) and an:
                attrs_snap["asset_name"] = prefix_key(an, wf)
            for fk in ("upstream_asset_key", "left_asset_key",
                       "right_asset_key", "regions_asset_key"):
                v = attrs.get(fk)
                if isinstance(v, str) and v:
                    consumers_by_full_key.setdefault(
                        prefix_key(v, wf), []
                    ).append((tool_dir.name, attrs_snap))
            for fk in ("deps", "upstream_asset_keys"):
                v = attrs.get(fk)
                if isinstance(v, list):
                    for kk in v:
                        if isinstance(kk, str) and kk:
                            consumers_by_full_key.setdefault(
                                prefix_key(kk, wf), []
                            ).append((tool_dir.name, attrs_snap))
        for tool_dir, data in wf_yamls:
            data["type"] = rewrite_type(data.get("type", ""))
            data = normalize_column_lists(data)
            data = strip_unsupported_partitions(data)
            data = soften_yxdb_missing(data, wf, consumers_by_full_key)
            attrs = data.get("attributes") or {}
            data["attributes"] = rewrite_attrs(attrs, wf)
            out_dir = COMBINED_DEFS / wf / tool_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "defs.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            n_assets += 1
        # Emit smart InlineDataframeComponent stubs for every missing upstream.
        for mk in sorted(missing_keys):
            local_name = mk.split("/", 1)[1] if "/" in mk else mk
            cols, row = infer_stub_schema(mk, consumers_by_full_key)
            stub_dir = COMBINED_DEFS / wf / f"_stub_{local_name}"
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_data = {
                "type": "wc_combined.components.inline_dataframe.component.InlineDataframeComponent",
                "attributes": {
                    "asset_name": mk,
                    "columns": cols,
                    "rows": [row],
                    "group_name": wf,
                    "description": (
                        f"Auto-stub for missing upstream {mk!r}. "
                        f"Columns inferred from {len(cols)} downstream refs."
                    ),
                },
            }
            (stub_dir / "defs.yaml").write_text(yaml.safe_dump(stub_data, sort_keys=False))
            n_assets += 1
        n_workflows += 1
    if skipped_wfs:
        print(f"Skipped {len(skipped_wfs)} workflows with numeric-column refs:")
        for w in skipped_wfs[:10]:
            print(f"  - {w}")
    overlaid = overlay_local_templates()
    if overlaid:
        print(f"Overlaid {overlaid} installed component(s) with local edits.")
    print(f"Combined {n_workflows} workflows, {n_assets} assets → {COMBINED_DEFS}")


if __name__ == "__main__":
    main()
