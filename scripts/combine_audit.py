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


def soften_yxdb_missing(data: dict) -> dict:
    """Default dataframe_from_yxdb to if_missing='empty' for importer demos.

    Imported workflows often hardcode Windows-only paths (e.g.
    C:\\Users\\Kiran\\Downloads\\...) that the customer can't restore on
    macOS/Linux. Falling back to an empty DataFrame lets the rest of
    the chain materialize so the customer can see which transforms are
    wired up correctly. If they need strict behavior, they can override
    if_missing='raise' on the asset.
    """
    if "dataframe_from_yxdb" not in data.get("type", ""):
        return data
    attrs = data.get("attributes") or {}
    if "if_missing" not in attrs:
        attrs["if_missing"] = "empty"
    return data


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
        for tool_dir, data in wf_yamls:
            data["type"] = rewrite_type(data.get("type", ""))
            data = normalize_column_lists(data)
            data = strip_unsupported_partitions(data)
            data = soften_yxdb_missing(data)
            attrs = data.get("attributes") or {}
            data["attributes"] = rewrite_attrs(attrs, wf)
            out_dir = COMBINED_DEFS / wf / tool_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "defs.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            n_assets += 1
        # Emit minimal InlineDataframeComponent stubs for every missing upstream.
        for mk in sorted(missing_keys):
            # mk is "<wf>/<name>" — extract tool name.
            local_name = mk.split("/", 1)[1] if "/" in mk else mk
            stub_dir = COMBINED_DEFS / wf / f"_stub_{local_name}"
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_data = {
                "type": "wc_combined.components.inline_dataframe.component.InlineDataframeComponent",
                "attributes": {
                    "asset_name": mk,
                    "columns": ["col_a", "col_b"],
                    "rows": [["x", "x"]],
                    "group_name": wf,
                    "description": f"Auto-stub for missing upstream {mk!r}.",
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
