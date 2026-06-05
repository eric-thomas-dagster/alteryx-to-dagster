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

## Coverage (v1 — deterministic mapping)

| Alteryx tool | Dagster component |
|---|---|
| Text Input | inline `@dg.asset` Python (TODO: replace with future `inline_dataframe` component) |
| Filter | `filter` |
| Formula | `formula` (untranslatable Alteryx-only functions like `IIF` are dropped + flagged) |
| Summarize | `summarize` |
| Join | `dataframe_join` |
| Union | `dataframe_union` |
| Sort | `sort` |
| Unique | `unique_dedup` |
| Select | `select_columns` |
| Input Data (delimited file) | `dataframe_from_csv` |
| Output Data | `dataframe_to_csv` / `_excel` / `_parquet` (sniffed by extension) |

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

**v1.5 (planned):** LLM-assisted translation of Alteryx-only formula expressions to pandas eval — pair the `formula` flagging step with a `litellm_agent` call. Opt-in via `--llm-assist openai` / `--llm-assist anthropic`.

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
