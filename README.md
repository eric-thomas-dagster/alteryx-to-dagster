# alteryx-to-dagster

Convert Alteryx workflows (`.yxmd` / `.yxmz` / `.yxzp`) into runnable Dagster projects.

Each Alteryx tool becomes a Dagster asset, wired into the DAG via `deps`, using the [Dagster community components registry](https://dagster-component-ui.vercel.app/) as the transform vocabulary (`filter`, `summarize`, `dataframe_join`, `formula`, `sort`, `unique_dedup`, `pivot`, `unpivot`, `running_total`, `window_calculation`, the `dataframe_*` IO family, `sql_transform`, the spatial + predictive families, …).

Standalone CLI — **does not require an Alteryx Designer install or any Alteryx license**. Translation is deterministic; LLM is an opt-in fallback for the long tail.

## Install + use

```bash
uvx --from git+https://github.com/eric-thomas-dagster/alteryx-to-dagster \
    alteryx-to-dagster --help
```

(PyPI publish pending — install from the git source for now.)

### One-command zero-config import

```bash
uvx --from git+https://github.com/eric-thomas-dagster/alteryx-to-dagster \
    alteryx-to-dagster import /path/to/workflow.yxmd
```

That single command:

- Scaffolds a fresh Dagster project at `./<workflow_stem>/` (via `uvx create-dagster project`)
- Imports the workflow into it
- Runs `dagster-component add <id> --auto-install` for each registry component used
- Adds `pandas`, `numpy`, `dagster-dg-cli`, `dagster-webserver` to `pyproject.toml` (the registry components need them at runtime; `dagster-component add` doesn't propagate their `requirements.txt`)

Then `cd <workflow_stem> && uv run dg dev` opens the asset graph at http://localhost:3000.

### Folder mode — many workflows at once

Point at a directory and every `.yxmd` / `.yxzp` / `.yxmz` inside it gets imported into the same Dagster project:

```bash
uvx --from git+https://github.com/eric-thomas-dagster/alteryx-to-dagster \
    alteryx-to-dagster import /path/to/workflows/
```

### Common overrides

```bash
# Custom output directory + package name
alteryx-to-dagster import workflow.yxmd --out-dir my_proj --pkg my_proj

# Skip the auto-install (you'll handle deps + components yourself)
alteryx-to-dagster import workflow.yxmd --no-install

# Pre-scaffold the Dagster project yourself (skip auto-bootstrap)
uvx create-dagster@latest project my_proj --uv-sync
alteryx-to-dagster import workflow.yxmd --out-dir my_proj
```

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
| **Spatial — full palette** | Create Points · Buffer · Geocoder · Geo Overlay · Geo Simplify · Drive Time · Poly Build · Poly Split · Spatial Info · Spatial Match · Spatial Process · Distance · Find Nearest · Make Group · Smooth / Generalize · Map Input · Trade Area · Heat Map · Demographic · Report Map | `points_from_latlon` · `geo_buffer` · `geocoder` (Nominatim default) · `geo_overlay` · `geo_simplify` · `drive_time` (openrouteservice / google / mapbox / osrm) · `poly_build` · inline polysplit · `spatial_info` · `spatial_join` · `spatial_process` (centroid / boundary / convex_hull / envelope / simplify / buffer / set_precision / polygon↔points / line↔polygon) · `distance_calculator` · `nearest_neighbors` (FindNearest's k=1 distance → `DistanceMiles`) · `summarize` w/ spatialobjcombine · `spatial_process.simplify` · `file_ingestion` · `drive_time` w/ travel-mode · passthrough · passthrough · passthrough |
| **CASS / Address standardization** | CASS · Address Verification | `address_standardize` (free, no API key required for `regex` mode; libpostal / Geoapify / Nominatim available; USPS CASS-certification is paid — use a commercial vendor for DPV) |
| **Predictive** | Linear / Logistic / Decision Tree / Random Forest / Naive Bayes / Neural Network / SVM / Gradient Boosting / PCA / Score · Find Nearest Neighbors · K-Centroids Cluster Analysis · K-Centroids Diagnostics · Append Cluster · MB Rules · MB Inspect · Model Comparison · Create Samples · Random Records · Frequency · IFS | sklearn-backed registry components w/ `model_path` joblib save; Score loads a saved model and predicts. **Alteryx Predictive Tools stock macros** (`Find_Nearest_Neighbors.yxmc`, `Forest_Model.yxmc`, `K-Centroids_Cluster_Analysis.yxmc`, `MB_Rules.yxmc`, `Model Comparison.yxmc`, etc.) route to their dedicated registry components with **`feature_columns` / `target_column` / `n_neighbors` / `n_estimators` / `max_depth` extracted automatically** from the macro CALL's `<Configuration>` block — no manual editing required for the standard cases. |
| **Time Series** | TS Forecast / TS Plot · ARIMA · ETS · Time Series Filler · Imputation | `arima_forecast` · `ets_forecast` · passthrough · `data_cleansing` (stock macros routed: `predictive_tools\arima.yxmc` etc.) |
| **R Tool / Jupyter Code** | R Tool · Jupyter Code | Inline `@dg.asset` stub with the original script preserved as a comment block. Port to Python or wrap with `rpy2` / `subprocess Rscript` / `dagstermill`. |
| **In-DB** (SQL pushdown — collapsed) | Connect · Input · Filter · Formula · Select · Summarize · Join · Union · Sample · Stream Out · Write Data · **Data Stream In** | Connected In-DB subgraphs collapse into ONE `warehouse_pipeline` asset (CTE chain, single warehouse round-trip). Stream Out sinks route via `return_dataframe: true` so downstream non-In-DB tools consume a DataFrame. **Data Stream In** (the inverse — pandas DataFrame → warehouse staging table) routes to `dataframe_to_table`; the downstream `warehouse_pipeline` declares the staging asset as a `deps` and its first step's `source: {kind: table, table: <staging>}` reads from it, so the entire round-trip runs in correct order. Connection env var is slugified from `<Connection>` (e.g. `Snowflake_Prod` → `SNOWFLAKE_PROD_URL`); SQL dialect auto-detected from the connection name. |
| **Reporting** | Render · Portfolio Composer (Image / Render / Table / Text / Layout / Overlay) | `pdf_report` for Render / Image. Composer Table / Text emit passthrough so downstream Joins keep working; combine with the terminal Render's `pdf_report` template_html for full styled layout. Layout / Overlay are visual-only — skipped. |
| **Apps / Interface** | Tab · CheckBoxGroup · NumericUpDown · Label · Control Parameter · Action · Macro Input/Output | Skipped as control-flow with notes — Alteryx App interface tools have no Dagster-runtime equivalent. The compute inside the App (under ToolContainers) still imports. |
| **Macros (`.yxmc`)** | Custom user macros · Alteryx stock macros · CReW community macros | `macro_splicer.py` recursively inlines custom `.yxmc` (max depth 5), renumbers tool_ids with `m<parent>_` prefix, rewires Macro Input/Output anchors. **Stock macros** (Cleanse, ARIMA, ETS, Imputation_v2, TimeSeriesFiller, OversampleField, Histogram, Field_Summary_Report, SelectRecords, CountRecords, WeightedAvg, RandomRecords, Frequency, IFS, Create_Samples, Append_Cluster, Find_Nearest_Neighbors, Forest_Model, Decision_Tree, Linear_Regression, Logistic_Regression, K-Centroids_Cluster_Analysis, K-Centroids_Diagnostics, MB_Rules, MB_Inspect, Model_Comparison) route to dedicated registry components, with **config fields like `feature_columns`, `target_column`, `n_neighbors`, `sample_size`, `seed`** auto-extracted from the macro CALL's `<Configuration>`. CReW macros (`crew_expectequal`, `crew_ensurefields`) → passthrough. |
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

| Category | Status |
|---|---|
| **USPS CASS-certification (DPV / ZIP+4 validation)** | Address parsing ships via `address_standardize` (regex / libpostal / Geoapify / Nominatim). USPS CASS-certification itself is a paid product — use a commercial vendor (Smarty / Loqate / USPS) when DPV is required. |
| **Multi-output Alteryx anchors** | DateTime's `B` (bad rows) / Filter's `T/F` / Join's `L/J/R` anchors aren't yet emitted as separate Dagster assets. Coerce-on-error gets us same-row-count correctness for most cases. Dagster's `@multi_asset(outs={...})` is the right shape; not yet wired. |
| **MultiRowFormula edge cases** | Cascading state that doesn't reduce to ffill / shift / cumcount (custom accumulators, conditional rolling state) is translated AND flagged with a warning — pandas `.shift(N)` returns the original prior value, not the previously-COMPUTED one, so the output diverges from Alteryx's row-by-row semantics. The warning in MIGRATION.md tells you to replace the emitted formula with a Python loop or `.cumsum()`/`.cummax()` equivalent. Expressions with deeply nested function calls *inside* the row-ref also need manual review. |
| **Visual-only tools (Heat Map / Plotly Charting / Report Map / Charting)** | Passthrough emitted so the data flows downstream — the visual rendering is dropped. Build a Plotly / Folium asset that consumes the same upstream for inline rendering. |
| **R Tool / Jupyter Code** | Emitted as inline `@dg.asset` stubs with the original script preserved as a comment. Port to Python or wrap with `rpy2` / `dagstermill` — we don't auto-translate R or notebook code. |
| **Custom vendor / shop-specific macros** | If your shop ships proprietary `.yxmc` macros that aren't in the imported `.yxzp` bundle, they land in MIGRATION.md as unmapped. The `--llm-translate` fallback can sometimes infer expressions inside them. |
| **Alteryx engine-internal nuances** | A handful of Alteryx semantics aren't 1:1 in pandas — e.g., implicit type coercion mixing dtypes in expressions, B-anchor error routing. Workflows depending on these may need targeted edits in the generated `defs.yaml`. |

If you hit a gap not listed here, open an issue with the Alteryx XML and we'll add it.

## Validation corpus

Continuously tested against **138 real Alteryx workflows** drawn from:

- [Alteryx Weekly Challenge community board](https://community.alteryx.com/categories/weeklychallenge-board) (atcodedog05 + sh0kat solution sets) — 83 .yxmd
- [Szymon-Czuszek Weekly Challenges](https://github.com/Szymon-Czuszek/Alteryx-Weekly-Challenges) — 19 .yxmd / .yxzp (Alteryx Apps included)
- [Szymon-Czuszek Superstore Reporting](https://github.com/Szymon-Czuszek/Superstore-Reporting) — 1 .yxzp (450+ tools across nested batch macros)
- [osabnis1776 iShares ETF Analysis](https://github.com/osabnis1776/iShares-IVV-ETF-Market-Risk-Analysis) — 1 .yxmd
- [OwenBData R-vs-Alteryx](https://github.com/OwenBData/RvsAlteryxBlog) — 1 .yxmd
- **Mario Kart challenge 498** (.yxzp w/ bundled macro) — 1 workflow
- [Alteryx Learnable Intro tutorials](https://github.com/learnable-content/alteryx-intro) — 30 .yxmd covering core transforms + the full Predictive Tools palette (KNN, K-Means, Random Forest, Decision Tree, Linear / Logistic Regression, Market Basket, Data Investigation)

Current stats:

- Real-compute mapping rate: **~99.8%** (1998 / 2003 non-control-flow tools across the corpus, and 100% on every workflow re-imported with the current dispatcher — the corpus-wide 5-tool gap is in saved reports that pre-date the latest control-flow classifier fixes; re-importing those workflows clears them).
- Static validation pass: **100%** (every emitted defs.yaml loads under `dg check`)
- Materialization rate (with auto-stubbed inputs): **~66%** and rising; remaining failures are split between Alteryx-internal workflow quirks (a few tools reference output cols before they're created) and stub-data dtype mismatches that don't bite real production data.

### Example: Find_Nearest_Neighbors stock macro

A KNN workflow's `Find_Nearest_Neighbors.yxmc` call XML:

```xml
<Value name="select.id">FruitID</Value>
<Value name="select.fields">FruitID=False,mass=True,width=True,height=True,color_score=True</Value>
<Value name="the_k">2</Value>
<Value name="standardize">True</Value>
<Value name="algo.kd_tree">True</Value>
```

Becomes:

```yaml
type: dagster_community_components.NearestNeighborsComponent
attributes:
  feature_columns: [mass, width, height, color_score]
  n_neighbors: 2
  normalize: true
  algorithm: kd_tree
  upstream_asset_key: join_6
  asset_name: yxmc_7
```

Materializable as-is — no placeholder editing required.

## License

MIT.
