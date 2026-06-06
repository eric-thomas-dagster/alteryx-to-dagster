"""Emit per-tool defs.yaml (or inline .py) into a scaffolded Dagster project.

Layout produced under `<out_dir>/src/<pkg>/defs/`:
    <asset_name>/defs.yaml      ← one folder per Alteryx tool that mapped
    <asset_name>.py             ← inline Python for unmapped-but-emittable tools

`type:` lines use `<pkg>.components.<component_id>.component.<ClassName>`
so they work after `dagster-component add <component_id>` has been run
in the project. The runner takes care of running those adds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .mapper import MappedTool


# Map registry id → ComponentClassName for the `type:` line.
#
# ⚠️ Case matters AND the `*Component` suffix is NOT universal —
# `append_fields`, `dataframe_join`, `dataframe_union` drop it. When adding
# new entries, grep the installed source:
#   grep '^class .*Component\|^class [A-Za-z]\+(Component' .../component.py
_COMPONENT_CLASS_NAMES: Dict[str, str] = {
    "filter": "FilterComponent",
    "formula": "FormulaComponent",
    "multi_field_formula": "MultiFieldFormulaComponent",
    "summarize": "SummarizeComponent",
    "sort": "SortComponent",
    "unique_dedup": "UniqueDedupComponent",
    "select_columns": "SelectColumnsComponent",
    "sample": "SampleComponent",
    "record_id": "RecordIdComponent",
    "running_total": "RunningTotalComponent",
    "pivot": "PivotComponent",
    "unpivot": "UnpivotComponent",
    "append_fields": "AppendFields",       # no -Component suffix
    "dataframe_join": "DataframeJoin",     # no -Component suffix
    "dataframe_union": "DataframeUnion",   # no -Component suffix
    "dataframe_from_csv": "DataframeFromCsvComponent",
    "dataframe_from_parquet": "DataframeFromParquetComponent",
    "dataframe_from_excel": "DataframeFromExcelComponent",
    "dataframe_from_yxdb": "DataframeFromYxdbComponent",
    "dataframe_to_csv": "DataframeToCsvComponent",
    "dataframe_to_excel": "DataframeToExcelComponent",
    "dataframe_to_parquet": "DataframeToParquetComponent",
    "sql_transform": "SqlTransformComponent",
    # v0.5 additions
    "inline_dataframe": "InlineDataframeComponent",
    "file_ingestion": "FileIngestionComponent",
    "data_cleansing": "DataCleansingComponent",
    "generate_rows": "GenerateRowsComponent",
    # These five drop the -Component suffix (their class declarations are
    # `class FindReplace(...)` etc., not `class FindReplaceComponent`).
    "find_replace": "FindReplace",
    "regex_parser": "RegexParser",
    "datetime_parser": "DatetimeParser",
    "text_to_columns": "TextToColumns",
    "xml_parser": "XmlParser",
    "tile_binning": "TileBinningComponent",
    "json_flatten": "JsonFlattenComponent",
    "points_from_latlon": "PointsFromLatLonComponent",
    "drive_time": "DriveTimeComponent",
    "geo_buffer": "GeoBufferComponent",
    "geo_overlay": "GeoOverlayComponent",
    "geo_simplify": "GeoSimplifyComponent",
    "pearson_correlation": "PearsonCorrelationComponent",
    "per_row_http_fetcher": "PerRowHttpFetcherComponent",
    # New components from v0.5.14
    "blob_convert": "BlobConvertComponent",
    "pdf_report": "PdfReportComponent",
    "dynamic_rename": "DynamicRenameComponent",
    "window_calculation": "WindowCalculationComponent",
    "warehouse_pipeline": "WarehousePipelineComponent",
    # Spatial — new geo components from v0.5.13
    "poly_build": "PolyBuildComponent",
    "spatial_info": "SpatialInfoComponent",
    "distance_calculator": "DistanceCalculatorComponent",
    "nearest_neighbors": "NearestNeighborsComponent",
    "point_in_polygon": "PointInPolygonComponent",
    "spatial_join": "SpatialJoinComponent",
    # Predictive — sklearn-backed registry components (with model_path joblib save)
    "linear_regression_model": "LinearRegressionModelComponent",
    "logistic_regression_model": "LogisticRegressionModelComponent",
    "decision_tree_model": "DecisionTreeModelComponent",
    "random_forest_model": "RandomForestModelComponent",
    "naive_bayes_model": "NaiveBayesModelComponent",
    "neural_network_model": "NeuralNetworkModelComponent",
    "svm": "SVMComponent",
    "gradient_boosting_model": "GradientBoostingModelComponent",
    "pca": "PcaComponent",
    "model_score": "ModelScoreComponent",
    # Predictive — statsmodels-backed (with res.save() / sm.load() support)
    "count_regression": "CountRegressionComponent",
    "gamma_regression": "GammaRegressionComponent",
}


def emit_yaml(
    out_root: Path,
    pkg: str,
    component_id: str,
    asset_name: str,
    attributes: Dict[str, object],
    schema_url: str | None = None,
) -> Path:
    """Write src/<pkg>/defs/<asset_name>/defs.yaml. Returns the file path."""
    class_name = _COMPONENT_CLASS_NAMES.get(component_id)
    if class_name is None:
        raise ValueError(
            f"emitter has no class name for component_id={component_id!r}. "
            "Add an entry to _COMPONENT_CLASS_NAMES."
        )

    defs_dir = out_root / "src" / pkg / "defs" / asset_name
    defs_dir.mkdir(parents=True, exist_ok=True)
    defs_path = defs_dir / "defs.yaml"

    # v0.5 emitter: type lines use the lazy `dagster_community_components.<ClassName>`
    # form so the wheel install resolves directly (force-include puts each
    # component file under dagster_community_components/...). The earlier
    # `<pkg>.components.<id>.component.<ClassName>` form required
    # `dagster-component add` to wire each component into the project; this
    # form works out of the box once the wheel is on PYTHONPATH.
    type_line = f"dagster_community_components.{class_name}"
    # asset_name lives on MappedTool (and is what we used to pick the folder
    # name above) but most components also require it as an attribute. Merge
    # it in unless the caller already supplied one in `attributes`.
    merged_attrs = dict(attributes)
    merged_attrs.setdefault("asset_name", asset_name)
    # Preserve required-shape keys (upstream/left/right/asset_key etc.) even
    # when their value is "" — dropping them causes pydantic ValidationErrors
    # at component construction. An empty string is visible to the user as
    # "this upstream wasn't mapped" and the run still loads; dropping the
    # field entirely makes the whole project fail to load.
    _STRUCTURAL_KEYS = {
        "upstream_asset_key", "left_asset_key", "right_asset_key",
        "upstream_asset_key_target", "upstream_asset_key_source",
        "points_asset_key", "regions_asset_key",
        "asset_name",
    }
    body = {
        "type": type_line,
        "attributes": {
            k: v
            for k, v in merged_attrs.items()
            if k in _STRUCTURAL_KEYS or (v is not None and v != "")
        },
    }
    header = ""
    if schema_url:
        header = f"# yaml-language-server: $schema={schema_url}\n"
    defs_path.write_text(header + yaml.safe_dump(body, sort_keys=False))
    return defs_path


def emit_inline_python(out_root: Path, pkg: str, asset_name: str, py_source: str) -> Path:
    """Write src/<pkg>/defs/<asset_name>.py for tools we can't express as YAML."""
    defs_dir = out_root / "src" / pkg / "defs"
    defs_dir.mkdir(parents=True, exist_ok=True)
    py_path = defs_dir / f"{asset_name}.py"
    py_path.write_text(py_source)
    return py_path


def emit_migration_report(
    out_root: Path,
    *,
    yxmd_source: str,
    mapped: List[tuple],   # list of (tool_id, plugin_short, component_id, asset_name, notes)
    unmapped: List[tuple], # list of (tool_id, plugin, reason, suggestion)
) -> Path:
    """Write MIGRATION.md summarizing what was converted and what wasn't.

    Separates "control-flow" unmapped entries (Browse / TextBox / Tool
    Container — intentionally not mapped, NOT failures) from "real" unmapped
    (tools that NEED a mapper). The real-compute mapping rate is what users
    should evaluate the importer on.
    """
    # Bucket the unmapped list. Heuristic: control-flow entries' `reason`
    # contains phrases like "Drop" / "purely visual" / "Dagster's DAG already"
    # / "no compute" — i.e. an explanation rather than a defect.
    _CTRL_PHRASES = (
        "annotation only", "purely visual", "no compute", "no data flow",
        "drop after", "drop; if", "drop the comment", "data preview",
        "implicit in", "alteryx-specific data preview", "documentation only",
        "block until done", "cache dataset",
    )
    control_flow: List[tuple] = []
    real_unmapped: List[tuple] = []
    for entry in unmapped:
        reason = entry[2] if len(entry) > 2 else ""
        is_control = any(p in reason.lower() for p in _CTRL_PHRASES)
        if is_control:
            control_flow.append(entry)
        else:
            real_unmapped.append(entry)

    real_total = len(mapped) + len(real_unmapped)
    real_rate = (len(mapped) / real_total * 100) if real_total else 100.0

    out_root.mkdir(parents=True, exist_ok=True)
    md = out_root / "MIGRATION.md"
    lines = [
        f"# Alteryx → Dagster migration report",
        "",
        f"Source workflow: `{yxmd_source}`",
        "",
        f"## Coverage",
        "",
        f"- **Real-compute mapping rate: {real_rate:.1f}%** ({len(mapped)} / {real_total} real-compute tools)",
        f"- Mapped to Dagster components / inline-python:  **{len(mapped)}**",
        f"- Unmapped (need a mapper):                       **{len(real_unmapped)}**",
        f"- Control-flow (intentionally skipped):           **{len(control_flow)}**  (Browse / TextBox / Tool Container / etc. — purely visual)",
        "",
        f"- Tools mapped: **{len(mapped)}**",
        f"- Tools unmapped: **{len(unmapped)}**",
        "",
        "## Mapped tools",
        "",
        "| Tool ID | Alteryx plugin | Component id | Asset name |",
        "|---|---|---|---|",
    ]
    for tool_id, plugin, comp_id, asset_name, _notes in mapped:
        lines.append(f"| {tool_id} | `{plugin}` | `{comp_id}` | `{asset_name}` |")

    # Detect required env vars based on which components landed in `mapped`.
    # The matching `dagster-component-templates` components default to these
    # env var names, so the user just needs to `export` before running `dg dev`.
    _COMPONENT_ENV_VARS: Dict[str, List[tuple[str, str]]] = {
        "drive_time": [("OPENROUTESERVICE_API_KEY", "free key at https://openrouteservice.org/dev/#/signup (2000 req/day; or set provider=google/mapbox/osrm to use a different routing API)")],
        "geocoder": [("NOMINATIM_USER_AGENT", "any descriptive string for the OSM Nominatim user-agent header; defaults to 'alteryx-to-dagster' if unset")],
        "per_row_http_fetcher": [],
        "warehouse_pipeline": [("<CONNECTION_NAME>_URL", "SQLAlchemy URL for the warehouse the Alteryx In-DB Connection referenced — env var name is slugified from the connection (e.g. 'Snowflake Prod' → SNOWFLAKE_PROD_URL)")],
        "sql_transform": [("<CONNECTION_NAME>_URL", "same convention as warehouse_pipeline — SQLAlchemy URL for the In-DB connection")],
    }
    env_rows: List[tuple] = []
    used_comp_ids = sorted({ci for (_, _, ci, _, _) in mapped})
    for ci in used_comp_ids:
        for env_name, note in _COMPONENT_ENV_VARS.get(ci, []):
            env_rows.append((ci, env_name, note))
    if env_rows:
        lines += [
            "",
            "## Required environment variables",
            "",
            "Export these before running `dg dev` / materializing. Without "
            "them, the listed assets will fail with a clear OSError pointing "
            "you back here.",
            "",
            "| Component | Env var | What it's for |",
            "|---|---|---|",
        ]
        for ci, env_name, note in env_rows:
            lines.append(f"| `{ci}` | `{env_name}` | {note} |")

    note_rows = [(tid, plg, an, n) for (tid, plg, _ci, an, ns) in mapped for n in ns]
    if note_rows:
        lines += [
            "",
            "## Notes / TODOs from translation",
            "",
        ]
        for tool_id, plugin, asset_name, note in note_rows:
            lines.append(f"- **tool {tool_id}** (`{plugin}`, asset `{asset_name}`): {note}")

    if unmapped:
        lines += [
            "",
            "## Unmapped tools — manual conversion required",
            "",
            "| Tool ID | Alteryx plugin | Why | Suggestion |",
            "|---|---|---|---|",
        ]
        for tool_id, plugin, reason, suggestion in unmapped:
            lines.append(f"| {tool_id} | `{plugin}` | {reason} | {suggestion} |")

    lines += [
        "",
        "## Next steps",
        "",
        "1. Inspect each generated `defs.yaml` — bracket-stripped expressions",
        "   may need small tweaks for `pandas.eval` syntax.",
        "2. For unmapped tools, run `dagster-component search <keyword>` to find",
        "   the closest existing component, then write a defs.yaml by hand.",
        "3. Run `dg check defs` to validate every YAML loads.",
        "4. Run `dg dev` and visualize the imported asset graph at http://localhost:3000.",
        "5. v1.5 (LLM-assisted): re-run with `--llm-assist openai` to translate the",
        "   flagged Alteryx-only expressions automatically.",
        "",
    ]
    md.write_text("\n".join(lines) + "\n")
    return md
