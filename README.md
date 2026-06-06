# alteryx-to-dagster

Convert Alteryx workflows (`.yxmd` / `.yxmz` / `.yxzp`) into runnable Dagster projects.

Each Alteryx tool becomes a Dagster asset, wired into the DAG via `deps`, using the [Dagster community components registry](https://dagster-component-ui.vercel.app/) as the transform vocabulary (`filter`, `summarize`, `dataframe_join`, `formula`, `sort`, `unique_dedup`, `pivot`, `unpivot`, `running_total`, `record_id`, the `dataframe_*` IO family, `sql_transform`, …). Tools without a 1:1 mapping land in `MIGRATION.md` for manual review.

Standalone CLI — **does not require an Alteryx Designer install or any Alteryx license**. The deterministic translator covers ~40 Alteryx-only formula functions; LLM is an opt-in fallback for the long tail.

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

The `--install` flag shells out to `dagster-component add <id> --auto-install` for every registry component the importer used. Without it you'll see a printed list of `add` commands to run yourself.

## What it can do today

### Tool coverage

| Category | Alteryx tools handled | Maps to |
|---|---|---|
| **Inputs** | Text Input · Input Data | inline `@dg.asset` (Text Input keeps field-typed dtypes) · `dataframe_from_csv` / `_parquet` / `_excel` / **`dataframe_from_yxdb` (native binary)** — sniffed by file extension |
| **Outputs** | Output Data · Browse | `dataframe_to_csv` / `_excel` / `_parquet` — sniffed by extension; Browse is a no-op (Dagster UI previews assets automatically) |
| **Row / column** | Filter · Select · Sort · Unique · Sample · Date Filter · Tile | `filter` · `select_columns` · `sort` · `unique_dedup` · `sample` (random) or inline pandas `.head`/`.tail`/`.iloc` (FirstN/LastN/EveryNth) · `filter` with between predicate · `pd.qcut`/`pd.cut` |
| **Transforms** | Formula · Multi-Field Formula · Record ID · Running Total · Data Cleansing | `formula` (pandas-eval) or inline `np.where`/`.str.*`/`.dt.*` when needed · `multi_field_formula` · `record_id` · `running_total` · inline pandas |
| **Aggregates / reshape** | Summarize · Count Records · CrossTab · Transpose | `summarize` (with named-agg rename) · `summarize` size · `pivot` · `unpivot` |
| **Parse** | DateTime · Regex (Parse/Replace/Match/Tokenize) · JSON Parse · XML Parse · Text To Columns | inline pandas — `pd.to_datetime` / `.dt.strftime` · `.str.replace`/`extract`/`match`/`split` · `pd.json_normalize` · `xml.etree`-based parser · `.str.split(expand=True)` |
| **Multi-input** | Join · Join Multiple · Union · Append · Append Fields | `dataframe_join` · inline pandas chained `.merge()` · `dataframe_union` · alias · `append_fields` (cartesian) |
| **In-DB** (SQL pushdown) | Connect · Input · Filter · Formula · Select · Summarize · Join · Union · Sample · Stream Out · Write Data | each tool → one `sql_transform` asset that CTASs an intermediate table. Connection routed via an env var slugified from the Alteryx `<Connection>` name (e.g. `Snowflake_Prod` → `SNOWFLAKE_PROD_URL`). |
| **Control flow** | Block Until Done · Cache Dataset · Browse · Message · Detour | dropped with a MIGRATION.md note — Dagster's DAG / IO manager / `AutomationCondition` already provides each of these natively |

### Alteryx formula functions translated **deterministically** (no LLM needed)

| Category | Functions |
|---|---|
| Conditional | `IIF` · `Switch` |
| String | `Contains` · `StartsWith` · `EndsWith` · `Length` · `Trim` · `TrimLeft` · `TrimRight` · `UpperCase` · `LowerCase` · `TitleCase` · `Substring` · `Left` · `Right` · `ToString` · `ToNumber` · `Replace` · `ReplaceFirst` · `Regex_Replace` · `Regex_Match` · `Regex_CountMatches` · `FindString` · `PadLeft` · `PadRight` |
| Null | `IsNull` · `IsEmpty` · `Null` |
| Date/Time | `DateTimeAdd` · `DateTimeDiff` · `DateTimeFormat` · `DateTimeParse` · `DateTimeNow` · `DateTimeToday` · `DateTimeYear` · `DateTimeMonth` · `DateTimeDay` · `DateTimeHour` · `DateTimeMinute` · `DateTimeSecond` |
| Operators | `AND` / `OR` / `NOT` → `&` / `|` / `~`; arithmetic + comparisons passthrough |
| Field refs | `[Field]` → `Field` (pandas-eval) or `df["Field"]` (when the expression needs the PYTHON path) |

Arg parsing is paren-balanced and quote-aware, so nested calls work: `IIF(Contains([s], "x"), 1, 0)` → `np.where(df["s"].str.contains("x", regex=False), 1, 0)`.

### `.yxzp` packages

Parses the workflow inside the zip; inventories bundled `.yxdb` data files and `.yxmc` macros. With the `dataframe_from_yxdb` registry component, bundled `.yxdb` files are readable natively — no conversion required.

### Output formats

- One `defs.yaml` per Alteryx tool (or inline `@dg.asset` `.py` for tools that need pandas Series ops that pandas-eval can't compile, like `np.where`/`.str.contains`/`.dt.strftime`).
- `MIGRATION.md` — every translation note + every unmapped tool with a suggestion.
- `CLOUD_PORTABILITY.md` — automatically emitted when any defs.yaml contains a local absolute path (e.g. `/data/customers.csv`), warning that local paths break the moment the project deploys off a developer laptop and recommending S3 / GCS / ADLS / Snowflake-stage equivalents.

### LLM-assisted fallback

For Alteryx-only functions the deterministic translator doesn't recognize (custom macros, vendor extensions), opt in with `--llm-translate <model>`:

```bash
alteryx-to-dagster import workflow.yxmd \
    --out-dir . --pkg my_project \
    --llm-translate gpt-4o-mini --llm-api-key-env OPENAI_API_KEY
```

Two LiteLLM calls per flagged expression — translate + independent score — at **import time only**. Translations meeting `--llm-score-threshold` (default 0.8) get baked into the emitted YAML / `.py`. Below the threshold the expression stays flagged in MIGRATION.md.

**Runtime is 100% LLM-free regardless** — the resulting Dagster project just runs pandas / SQL.

Cost: ~$0.0004 per flagged expression at gpt-4o-mini. One-time per import.

## What it can't do today

| Category | Status |
|---|---|
| Reporting tools (Render / Layout / Email / Charting) | Skipped — different paradigm; reports don't have a clean Dagster equivalent |
| Interface tools (Macro Input/Output / Control Parameter / Action) | Skipped — these only exist for Alteryx Apps/Macros UI |
| Documentation tools (Comment / Tool Container / Browse / HTMLBox / Text Box) | Skipped as control-flow — purely visual. Tool Container's INNER tools still get imported (the container itself is a no-op wrapper). |
| Auto Field (runtime dtype inference) | Skipped — Dagster components handle dtype inference at read time (`pd.read_csv`, `inline_dataframe`'s `dtypes` field). |
| Data Investigation tools (Field Summary / Pearson Correlation / Frequency Table) | **Pearson Correlation** ✓ via `pearson_correlation` component. Field Summary maps via `summarize` + `dataframe_describe`. Frequency Table not yet wired. |
| Basic spatial (Create Points / Geo Buffer / Geo Overlay / Geo Simplify / Drive Time / Poly Split) | **Done** — `points_from_latlon`, `geo_buffer`, `geo_overlay`, `geo_simplify`, `drive_time` (openrouteservice / google / mapbox / osrm), `poly_split`. |
| Advanced spatial (Spatial Match / Distance / Trade Area / Find Nearest / Poly Build / Map Input / Spatial Info) | Partial — Find Nearest can route to `nearest_neighbors`. Spatial Match / Distance / Trade Area / Poly Build mappers not yet wired. |
| Predictive — sklearn-backed (Linear/Logistic Regression / Decision Tree / Random Forest / Naive Bayes / Neural Network / SVM / Gradient Boosting / PCA / Score) | Registry components exist with `model_path` joblib save; Alteryx-plugin → mapper entries not yet wired. |
| Predictive — statsmodels-backed (Count Regression / Gamma Regression) | Registry components exist with `.save()` / `sm.load()`; mapper entries not yet wired. |
| Time Series (TS Forecast / TS Plot) | Registry has `arima_forecast`, `ets_forecast`; mapper entries not yet wired. |
| Custom macros (`.yxmc`) | **Done** — `macro_splicer.py` recursively inlines macros (max depth 5), renumbers tool_ids with `m<parent>_` prefix, rewires Macro Input/Output anchors. Stock macros like `Cleanse.yxmc` route to dedicated registry components (`data_cleansing`) instead of inlining. |
| In-DB tools (Connect/Input/Filter/Formula/Select/Summarize/Join/Union/Sample/StreamOut/WriteData) | Mapped 1:1 to `sql_transform` per-tool today. **Pending:** collapse In-DB subgraphs into a single `warehouse_pipeline` CTE chain (preserves Alteryx's pushdown semantics; In-DB Stream Out routes via `warehouse_pipeline.return_dataframe=True`). |
| Multi-Row Formula (window-style) | Not yet — would need a `multi_row_formula` component (windowed `df.shift()` / rolling pattern). |
| Dynamic Rename (pattern-based) | Not yet — maps roughly to `select_columns` with `rename`; needs the rename-pattern → column-pair expansion. |
| Alteryx Apps (Interface tools / Action / Control Parameter) | Not supported — Interface tools have no Dagster equivalent. |

For In-DB tools: each Alteryx In-DB tool becomes its own `sql_transform` asset that materializes an intermediate table. This is the simplest correct mapping but loses Alteryx's single-query pushdown. Future versions can detect connected In-DB subgraphs and collapse them into one `sql_transform` with CTEs.

## End-to-end validated

Sample at `samples/sample_with_iif.yxmd` — 5 tools (Text Input → Formula with IIF + Contains → Sample FirstN=5 → Record ID → CSV) imports → installs → materializes:

```
region,product,quantity,unit_price,revenue,bulk_tier,is_widget,row_num
North,Widget,10,5.5,55.0,standard,True,1
North,Gadget,3,12.0,36.0,standard,False,2
South,Widget,7,5.5,38.5,standard,True,3
South,Gadget,15,12.0,180.0,bulk,False,4
East,Widget,20,5.5,110.0,bulk,True,5
```

IIF + Contains translated deterministically — no LLM call made.

## Why an external tool, not a sub-command of `dagster-component`?

Migration is a one-shot affair — you run it once per Alteryx workflow, get a Dagster project, then iterate from there. The community-components CLI is a daily-driver tool (search / info / add / schema). Different cadence, different scope.

The importer **uses** the community registry — it shells out to `dagster-component add` to install the components it maps to. No tight coupling either way.

## License

MIT.
