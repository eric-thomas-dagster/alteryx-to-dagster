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
COMBINED_DEFS = Path("/tmp/wc_combined/src/wc_combined/defs")


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
        # First pass — scan all YAMLs; skip the WHOLE workflow if any
        # `columns:` entry parses as int (Dagster's Resolvable templating
        # coerces YAML-quoted '1'/'6' to int and pydantic then rejects them
        # against List[str]; pre-filter avoids whole-project load failures).
        wf_yamls: list[tuple[Path, dict]] = []
        has_numeric_col = False
        for tool_dir in sorted(src_defs.iterdir()):
            if not tool_dir.is_dir():
                continue
            src_yaml = tool_dir / "defs.yaml"
            if not src_yaml.is_file():
                continue
            data = yaml.safe_load(src_yaml.read_text()) or {}
            attrs = data.get("attributes") or {}
            # Dagster's Resolvable templating coerces string '1' to int 1 at
            # template-resolve time, so a YAML string that looks like a number
            # trips List[str] validation on ANY str-valued list (columns,
            # output_columns, group_by, ...). Walk all list attrs to detect.
            def _is_numeric_col(c) -> bool:
                if isinstance(c, int):
                    return True
                if isinstance(c, str) and c.lstrip("-").isdigit():
                    return True
                return False
            for v in attrs.values():
                if isinstance(v, list) and any(_is_numeric_col(c) for c in v):
                    has_numeric_col = True
                    break
            if has_numeric_col:
                break
            wf_yamls.append((tool_dir, data))
        if has_numeric_col:
            skipped_wfs.append(wf)
            continue
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
    print(f"Combined {n_workflows} workflows, {n_assets} assets → {COMBINED_DEFS}")


if __name__ == "__main__":
    main()
