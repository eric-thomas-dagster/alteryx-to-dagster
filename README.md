# alteryx-to-dagster

Convert Alteryx workflows (`.yxmd` / `.yxmz` / `.yxzp`) into runnable Dagster projects.

Each Alteryx tool becomes a Dagster asset, wired into the DAG via `deps`, using the [Dagster community components registry](https://dagster-component-ui.vercel.app/) as the transform vocabulary (`filter`, `summarize`, `dataframe_join`, `formula`, `sort`, `unique_dedup`, `pivot`, `unpivot`, `running_total`, `window_calculation`, the `dataframe_*` IO family, `sql_transform`, the spatial + predictive families, …).

Standalone CLI — **does not require an Alteryx Designer install or any Alteryx license**. Translation is deterministic; LLM is an opt-in fallback for the long tail.

## Install + use

```bash
uvx --from alteryx-to-dagster alteryx-to-dagster --help
```

End-to-end:

```bash
# 1. Scaffold a fresh Dagster project
uvx create-dagster@latest project my_project --no-uv-sync
cd my_project
uv add --dev dagster-dg-cli dagster-webserver
uv add pandas numpy

# 2. Convert the Alteryx workflow into it
uvx --from alteryx-to-dagster alteryx-to-dagster import \
    /path/to/workflow.yxmd \
    --out-dir . --pkg my_project --install

# 3. Validate + run
uv run dg check defs            # all generated YAML loads cleanly
uv run dg dev                   # asset graph at http://localhost:3000
```

The `--install` flag shells out to `dagster-component add <id> --auto-install` for every registry component the importer used.

## Tool coverage

| Category | Alteryx tools | Maps to |
|---|---|---|
| **Inputs** | Text Input · Input Data | `inline_dataframe` (preserves declared dtypes; auto-infers V_String numerics; leading-zero strings stay strings) · `dataframe_from_csv` / `_parquet` / `_excel` / **`dataframe_from_yxdb` (native binary)** — sniffed by file extension |
| **Outputs** | Output Data · Browse · Render · PortfolioComposer Image | `dataframe_to_csv` / `_excel` / `_parquet` · `pdf_report` · Browse → no-op (Dagster UI previews assets automatically) |
| **Row / column** | Filter · Select · AlteryxSelect · Sort · Unique · Sample · Date Filter · Tile · DynamicRename · BlobConvert | `filter` · `select_columns` (handles `*Unknown` wildcard) · `sort` · `unique_dedup` · `sample` · `filter` w/ between predicate · `tile_binning` · `dynamic_rename` · `blob_convert` |
| **Transforms** | Formula · Multi-Field Formula · Multi-Row Formula · Record ID · Running Total · Data Cleansing | `formula` (pandas-eval or PYTHON path) · `multi_field_formula` · pure `[Row±N:Col]` → `window_calculation` (lag/lead); compound IF/THEN/ELSE → `formula` with translated `df['Col'].shift(N)` · `record_id` · `running_total` · `data_cleansing` |
| **Aggregates / reshape** | Summarize · Count Records · CrossTab · Transpose | `summarize` (with named-agg rename, object-dtype coercion for arithmetic aggs, whole-frame fallback) · `summarize` size · `pivot` · `unpivot` |
| **Parse** | DateTime · Regex (Parse Simple / Parse Complex / Replace / Match / Tokenize) · JSON Parse · XML Parse · Text To Columns | `datetime_parser` (whitespace-tolerant) · `regex_parser` (ParseSimple → `extract` w/ `RootName1…N` output cols, auto-wraps groupless patterns) · `json_flatten` · `xml_parser` · `text_to_columns` |
| **Multi-input** | Join · Join Multiple · Union · Append · Append Fields · Find Replace | `dataframe_join` (forwards embedded `<SelectFields/>` as post-merge `rename` / `drop_columns` with fuzzy `Right_`/`Left_` prefix matching; coalesces mismatched join-key dtypes) · inline pandas chained `.merge()` · `dataframe_union` · alias · `append_fields` · `find_replace` |
| **GenerateRows** | per-row Expression_Init/Cond/Loop expansion | `generate_rows` mode `loop_expression` — emits one row per loop iteration; integer / date / datetime stepping all work |
| **Spatial** | Create Points · Geo Buffer · Geo Overlay · Geo Simplify · Drive Time · Poly Split · **Spatial Match** · **Distance** · **Find Nearest** · **Poly Build** · **Spatial Info** · **Map Input** · **Trade Area** | `points_from_latlon` · `geo_buffer` · `geo_overlay` · `geo_simplify` · `drive_time` (openrouteservice / google / mapbox / osrm) · `poly_split` · `spatial_join` (accepts pre-built geom column on points side) · `distance_calculator` · `nearest_neighbors` (auto-explodes Point feature cols; FindNearest's k=1 distance → `DistanceMiles`) · `poly_build` (geom-col input mode for `<SpatialObj/>`) · `spatial_info` · `file_ingestion` · `drive_time` w/ travel-mode profile |
| **Predictive** | Linear / Logistic / Decision Tree / Random Forest / Naive Bayes / Neural Network / SVM / Gradient Boosting / PCA / Score | sklearn-backed registry components w/ `model_path` joblib save; Score loads a saved model and predicts |
| **Time Series** | TS Forecast / TS Plot | `arima_forecast` · `ets_forecast` |
| **In-DB** (SQL pushdown) | Connect · Input · Filter · Formula · Select · Summarize · Join · Union · Sample · Stream Out · Write Data | each tool → one `sql_transform` asset that CTASs an intermediate table. Connection routed via an env var slugified from the Alteryx `<Connection>` name (e.g. `Snowflake_Prod` → `SNOWFLAKE_PROD_URL`). **Not yet:** collapsing connected In-DB subgraphs into a single `warehouse_pipeline` CTE chain — that preserves Alteryx's single-query pushdown and lets In-DB Stream Out route via `warehouse_pipeline.return_dataframe=True`. The corpus we test against has no In-DB workflows so this hasn't shipped yet. |
| **Macros (`.yxmc`)** | Custom macros · stock macros (Cleanse) | `macro_splicer.py` recursively inlines `.yxmc` (max depth 5), renumbers tool_ids with `m<parent>_` prefix, rewires Macro Input/Output anchors. Stock macros (Cleanse, etc.) route to dedicated registry components instead of inlining. |
| **Control flow** | Block Until Done · Cache Dataset · Browse · Message · Detour · Tool Container · Comment · Text Box · HTMLBox | Skipped (Dagster's DAG / IO manager / `AutomationCondition` already provides Block Until Done / Cache; visual-only tools have no runtime equivalent). Tool Container's INNER tools still get imported. |

## Formula translation — deterministic, no LLM needed

| Category | Functions |
|---|---|
| Conditional | `IIF` · `Switch` · `IF…THEN…ELSEIF…ELSE…ENDIF` (rewritten to nested IIF) |
| String | `Contains` · `StartsWith` · `EndsWith` · `Length` · `Trim` · `TrimLeft` · `TrimRight` · `UpperCase` · `LowerCase` · `TitleCase` · `Substring` · `Left` · `Right` · `ToString` · `ToNumber` · `Replace` · `ReplaceFirst` · `ReplaceChar` · `Regex_Replace` · `Regex_Match` · `Regex_CountMatches` · `FindString` · `PadLeft` · `PadRight` |
| Math | `Abs` · `Sqrt` · `Log` · `Round` · `Min` · `Max` · `Mod` · `Avg` · `Sum` · `Median` · `Count` · `Coalesce` |
| Null | `IsNull` · `IsEmpty` · `Null` |
| Date/Time | `DateTimeAdd` · `DateTimeDiff` · `DateTimeFormat` · `DateTimeParse` · `DateTimeNow` · `DateTimeToday` · `DateTimeYear` · `DateTimeMonth` · `DateTimeDay` · `DateTimeHour` · `DateTimeMinute` · `DateTimeSecond` · `ToDate` · `ToDateTime` |
| Operators | `AND` / `OR` / `NOT` / `!` → `&` / `\|` / `~`; `=` (Alteryx-style equality) → `==`; comparisons inside boolean ops are auto-paren-wrapped to fix pandas precedence |
| Field refs | `[Field]` → `Field` (pandas-eval) or `df["Field"]` (when the expression needs the PYTHON path) |

Arg parsing is paren-balanced and quote-aware: `IIF(Contains([s], "x"), 1, 0)` → `np.where(df["s"].str.contains("x", regex=False), 1, 0)`.

## `.yxzp` packages

Parses the workflow inside the zip; inventories bundled `.yxdb` data files and `.yxmc` macros. With the `dataframe_from_yxdb` registry component, bundled `.yxdb` files are readable natively.

## Output formats

- One `defs.yaml` per Alteryx tool, plus inline `@dg.asset` `.py` for the handful of cases that need pandas Series ops the pandas-eval engine can't compile (e.g. self-joins, rootless GenerateRows seeds).
- `MIGRATION.md` — every translation note, every unmapped tool with a suggestion, plus a **real-compute mapping rate** (excludes control-flow tools that are intentionally skipped).
- `CLOUD_PORTABILITY.md` — automatically emitted when any defs.yaml contains a local absolute path, warning that local paths break the moment the project deploys off a developer laptop and recommending S3 / GCS / ADLS / Snowflake-stage equivalents.

## LLM-assisted fallback

For Alteryx-only functions the deterministic translator doesn't recognize (vendor extensions, custom macros), opt in with `--llm-translate <model>`:

```bash
alteryx-to-dagster import workflow.yxmd \
    --out-dir . --pkg my_project \
    --llm-translate gpt-4o-mini --llm-api-key-env OPENAI_API_KEY
```

Two LiteLLM calls per flagged expression — translate + independent score — **at import time only**. Translations meeting `--llm-score-threshold` (default 0.8) get baked into the emitted YAML / `.py`. Below the threshold the expression stays flagged in MIGRATION.md.

**Runtime is 100% LLM-free regardless** — the resulting Dagster project just runs pandas / SQL.

Cost: ~$0.0004 per flagged expression at gpt-4o-mini. One-time per import.

## What it doesn't do today

Honest gaps — what won't work after import:

| Category | Status |
|---|---|
| **In-DB chain collapse** | Per-tool mapping works (each In-DB tool emits its own `sql_transform` with intermediate CTAS). The single-query collapse into `warehouse_pipeline` (CTE chain that preserves Alteryx's pushdown semantics) is **not yet wired** — needs to be rebuilt after the registry deletion that wiped the original implementation. |
| **Alteryx Apps** | Interface tools (Macro Input/Output / Control Parameter / Action) are skipped — these only exist for Alteryx's App/Macro UI and have no Dagster equivalent. |
| **Reporting (Render / Layout / Email / Charting)** | Render and PortfolioComposer route to `pdf_report` for the table-style output. Email/Layout/Charting (HTML report builders) are skipped — different paradigm. |
| **MultiRowFormula edge cases** | Pure `[Row±N:Col]` → `window_calculation`, compound IF/THEN/ELSE → `formula` with `df['Col'].shift(N)`. Expressions with nested `REGEX_Match` / function calls that don't fully deterministic-translate still need manual review (flagged in MIGRATION.md). |
| **Browse / Tool Container / Comment / Text Box / HTMLBox** | Skipped as control-flow — purely visual in Alteryx, no runtime equivalent. Tool Container's INNER tools still get imported. |
| **Custom data quality / lineage tools** | If your shop ships custom Alteryx tools (proprietary connectors, GIS-vendor specific tools, etc.), they land in MIGRATION.md as unmapped. The `--llm-translate` fallback can sometimes infer a translation for the formula bodies inside them. |
| **Alteryx engine-internal nuances** | A few Alteryx semantics aren't 1:1 in pandas — e.g., Alteryx's implicit type coercion when mixing dtypes in expressions. Workflows that rely on these may need targeted fixes in the generated `defs.yaml`. |

If you hit a gap not listed here, open an issue with the Alteryx XML and we'll add it.

## Validation corpus

Continuously tested against the [Alteryx Weekly Challenge](https://community.alteryx.com/categories/weeklychallenge-board) workflows — 83 .yxmd files covering most tool families. Current stats:

- Real-compute mapping rate: **100%** (every non-control-flow tool maps to a component)
- Static validation pass: **100%** (every emitted defs.yaml loads under `dg check`)
- Materialization (with auto-stubbed inputs): **~69%** and rising; remaining failures are split between Alteryx-internal workflow quirks (a few tools reference output cols before they're created) and component-internal long-tail edge cases.

## License

MIT.
