# alteryx-to-dagster

Convert Alteryx workflows (`.yxmd` / `.yxmz` / `.yxzp`) into runnable Dagster projects.

Each Alteryx tool becomes a Dagster asset, wired into the DAG via `deps`, using the [Dagster community components registry](https://dagster-component-ui.vercel.app/) as the transform vocabulary (`filter`, `summarize`, `dataframe_join`, `formula`, `sort`, `unique_dedup`, `pivot`, `unpivot`, `running_total`, `record_id`, the `dataframe_*` IO family, …). Tools without a 1:1 mapping land in `MIGRATION.md` for manual review.

Standalone CLI — does **not** require a running Alteryx Designer or any Alteryx license.

## Install + use

```bash
uvx --from alteryx-to-dagster alteryx-to-dagster --help
```

End-to-end flow against a `.yxmd`:

```bash
# 1. Scaffold a fresh Dagster project
uvx create-dagster@latest project my_project --no-uv-sync
cd my_project
uv add --dev dagster-dg-cli dagster-webserver
uv add pandas

# 2. Convert the Alteryx workflow into it
uvx --from alteryx-to-dagster alteryx-to-dagster import \
    /path/to/workflow.yxmd \
    --out-dir . --pkg my_project --install

# 3. Validate + run
uv run dg check defs             # all generated YAML loads cleanly
uv run dg dev                    # interactive UI at http://localhost:3000
```

The `--install` flag shells out to `dagster-component add <id> --auto-install` for every registry component the importer used. Without it you'll see a printed list of `add` commands to run yourself.

## Coverage (v0.2 — 18 Alteryx tools mapped)

| Alteryx tool | Dagster target |
|---|---|
| Text Input | inline `@dg.asset` Python (field types preserved from Alteryx Field defs) |
| Input Data (delimited file) | `dataframe_from_csv` |
| Output Data | `dataframe_to_csv` / `_excel` / `_parquet` (sniffed by extension) |
| Filter | `filter` |
| Formula | `formula` (with v1.5 LLM translation for IIF / Contains / DateTimeAdd / etc.) |
| Multi-Field Formula | `multi_field_formula` (same LLM path) |
| Select | `select_columns` |
| Sort | `sort` |
| Unique | `unique_dedup` |
| Sample | `sample` (Random / 1-in-N modes) **or** inline pandas `.head/.tail/.iloc[::n]` (FirstN / LastN / EveryNth — faithful, deterministic) |
| Record ID | `record_id` |
| Running Total | `running_total` |
| Summarize | `summarize` |
| Count Records | `summarize` (trivial Count) |
| CrossTab | `pivot` |
| Transpose | `unpivot` |
| Join | `dataframe_join` |
| Union / Append | `dataframe_union` |
| Append Fields | `append_fields` (cartesian product) |

**What gets translated automatically (v1):**
- `[Field]` bracket-stripping → bare field name (pandas eval style)
- Arithmetic and comparison expressions: `[a] * 2`, `[qty] > 5`, etc.
- Summarize `GroupBy` / `Sum` / `Count` / `Avg` actions with rename
- Field types from Text Input `<Field type="..."/>` preserved (Int32 → Int64, Double → float64, etc.)
- `.yxzp` bundles parsed; bundled `.yxdb` data files and `.yxmc` macros inventoried in `MIGRATION.md`

**What gets flagged for manual review:**
- Alteryx-only functions: `IIF`, `Switch`, `Contains`, `StartsWith`, `EndsWith`, `DateTimeAdd`, `DateTimeDiff`, `DateTimeFormat`, `DateTimeParse`, `Substring`, `Regex`, `Trim`, `UpperCase`/`LowerCase`, `ToString`/`ToNumber`, `Null`/`IsNull`/`IsEmpty`, `FindString`, `PadLeft`/`PadRight`
- Custom macros (`.yxmc`)
- Proprietary `.yxdb` data files
- Tools with no mapping yet

**v1.5 (shipped in v0.2):** LLM-assisted translation of Alteryx-only formula expressions. Opt in with `--llm-translate <model>`. Two LiteLLM calls per flagged expression — translate + independent score — at **import time only**. The resulting Dagster project carries zero LLM dependency at materialization.

```bash
uvx --from alteryx-to-dagster alteryx-to-dagster import workflow.yxmd \
    --out-dir . --pkg my_project \
    --llm-translate gpt-4o-mini --llm-api-key-env OPENAI_API_KEY
```

Translations that score ≥ 0.8 (combined translator self-confidence + independent scorer) get baked into the emitted YAML or — when the translation needs pandas Series ops that `pandas.eval` can't compile — into an inline `@dg.asset` `.py` file using `np.where` / `.str.*` / `.dt.*`. Below the threshold, the expression stays flagged in `MIGRATION.md` for human review (no silent emission of low-confidence translations).

Live-validated end-to-end against `samples/sample_with_iif.yxmd`:
- `IIF([quantity] > 10, "bulk", "standard")` → `np.where(df["quantity"] > 10, "bulk", "standard")` (PYTHON path, score 0.85)
- `Contains([product], "Widget")` → `df["product"].str.contains("Widget")` (PYTHON path, score 0.95)
- Confused / wrong translations (e.g. Alteryx Switch's value/default arg-order confusion) get caught by the scorer and dropped.

Cost: ~$0.0004 per flagged expression at `gpt-4o-mini`. One-time per import.

## Live-validated end-to-end

5-tool sample workflow shipped with the package (`alteryx_to_dagster/samples/sample_workflow.yxmd`):

```
TextInput(4 rows) → Filter(qty>5) → Formula(rev = qty*price) → Summarize(by region, sum) → CSV
```

Run on a fresh Dagster project, all 5 assets materialize via `dg launch --assets '*'`:

```
region,total_revenue,total_quantity
North,55.0,10
South,218.5,22
```

## Why an external tool, not a sub-command of `dagster-component`?

The migration tool is a one-shot affair — you run it once per Alteryx workflow, get a Dagster project, then iterate from there. The community-components CLI is a daily-driver tool (search / info / add / schema). Different cadence, different scope. Keeping the importer as its own package lets it move quickly on Alteryx-format coverage without bloating the registry CLI.

The importer **uses** the community registry — it shells out to `dagster-component add` to install the components it maps to. No tight coupling either way.

## License

MIT.
