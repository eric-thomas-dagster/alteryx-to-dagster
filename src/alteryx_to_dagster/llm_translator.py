"""v1.5 LLM-assisted Alteryx-formula translation.

Used **at import time only** to translate Alteryx-specific functions
(IIF / Switch / Contains / DateTimeAdd / etc.) that the v1 deterministic
mapper drops. The translated pandas expression gets baked into the
emitted defs.yaml (or an inline .py asset when pandas eval is
insufficient) — the resulting Dagster project carries **zero** LLM
dependency at materialization time.

Two calls per flagged expression:
  1. translate_expression(expr) → {pandas_expr, is_python, self_confidence, reasoning}
  2. score_translation(expr, translation) → {score 0..1, reason}

A second model call as an independent scorer matters: it doesn't see
the translator's reasoning, so its score isn't biased by the
translator's self-report. Below the threshold the importer keeps the
v1 behavior — flag in MIGRATION.md and don't emit.

Cost: ~$0.0002 per expression × N expressions × one-time, at gpt-4o-mini.
A 50-tool Alteryx workflow with 8 flagged formulas ≈ $0.0032. Once.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional


_TRANSLATE_SYSTEM_PROMPT = """\
You translate Alteryx formula expressions into pandas-compatible expressions for a Dagster project.

The target is the `formula` Dagster component, which uses pandas DataFrame.eval().
For things pandas eval can't handle (string ops, date math, .dt accessor, etc.),
return a pandas Series expression instead and prefix with "PYTHON:".

PANDAS-EVAL RULES (preferred — return without PYTHON: prefix):
  - Field references like [FieldName] → bare FieldName
  - Logical operators: AND → &, OR → |, NOT → ~  (always parenthesize the operands)
  - Arithmetic / comparison / numeric Min/Max/Abs map straight through

PYTHON PATH (prefix output with "PYTHON:" — for things pandas.eval cannot compile).
**Conditionals always go PYTHON-path** — pandas.eval does not support np.where
as a function and Python ternaries don't vectorize over Series.

  - IIF(cond, a, b)             → PYTHON: np.where(<cond>, <a>, <b>)
  - Switch(val, default, k1, v1, k2, v2, …)
                                → PYTHON: np.select([df["val"] == k1, df["val"] == k2, ...], [v1, v2, ...], default=default)
  - Contains([s], "x")           → PYTHON: df["s"].str.contains("x")
  - StartsWith / EndsWith        → PYTHON: df["s"].str.startswith / .str.endswith
  - Length([s])                  → PYTHON: df["s"].str.len()
  - UpperCase / LowerCase / Trim → PYTHON: df["s"].str.upper / .str.lower / .str.strip()
  - Substring([s], start, n)     → PYTHON: df["s"].str.slice(start - 1, start - 1 + n)   (Alteryx is 1-indexed!)
  - ToString / ToNumber          → PYTHON: df["col"].astype(str / "float64")
  - DateTimeAdd(d, n, "days")    → PYTHON: df["d"] + pd.Timedelta(days=n)
  - DateTimeDiff(a, b, "days")   → PYTHON: (df["a"] - df["b"]).dt.days
  - DateTimeFormat(d, "%Y-%m-%d")→ PYTHON: df["d"].dt.strftime("%Y-%m-%d")
  - DateTimeParse(s, "%Y-%m-%d") → PYTHON: pd.to_datetime(df["s"], format="%Y-%m-%d")
  - IsNull / IsEmpty             → PYTHON: df["col"].isna()
  - FindString([s], "x")         → PYTHON: df["s"].str.find("x")   (returns -1 if not found; Alteryx returns 0)

When in doubt, prefer the PYTHON path with a clear pandas Series expression
over a clever pandas-eval string. The PYTHON path becomes an inline @dg.asset
that runs deterministically with NO LLM dependency at materialization time —
that's the whole point.

⚠️ CRITICAL: in PYTHON path, *every* field reference must be wrapped as
`df["<field>"]`. The bare `field` form is ONLY for pandas-eval. Examples:

  - IIF([quantity] > 10, "bulk", "standard")
    → PYTHON: np.where(df["quantity"] > 10, "bulk", "standard")
                       ^^^^^^^^^^^^^^^^^^^  (NOT bare `quantity`)

  - Contains([name], "x") AND [score] > 0.5
    → PYTHON: df["name"].str.contains("x") & (df["score"] > 0.5)
              ^^^^^^^^^^                       ^^^^^^^^^^^

  - Switch([region], "Other", "N", "North", "S", "South")
    → PYTHON: np.select(
                  [df["region"] == "N", df["region"] == "S"],
                  ["North", "South"],
                  default="Other",
              )
    (Alteryx Switch arg order is value, default, k1, v1, k2, v2, …)

If the expression mixes columns and constants, all columns become df["…"].
If it's a pure column-name comparison without bracket markers, still use df["…"]
in PYTHON path output.

OUTPUT — STRICT JSON, no other text, no markdown fences:

{
  "pandas_expr": "...",         // the translated expression. Prefix with "PYTHON:" if Series-based.
  "is_python": true|false,      // true iff pandas_expr starts with "PYTHON:"
  "self_confidence": 0.0-1.0,   // how confident you are the semantics are preserved
  "reasoning": "..."            // ≤ 1 sentence
}
"""


_SCORE_SYSTEM_PROMPT = """\
You evaluate whether a pandas expression preserves the semantics of an Alteryx formula expression.

Score 0 to 10:
  10 = exact match for all valid inputs
  7-9 = correct for typical inputs; edge cases may differ (null handling, 0- vs 1-indexing, etc.)
  4-6 = correct shape but wrong in important cases
  0-3 = wrong or unsafe

Output in exactly this format (no other text):
SCORE: <integer 0-10>
REASON: <one sentence>
"""


@dataclass
class TranslationResult:
    original: str               # the raw Alteryx expression
    pandas_expr: str            # without the "PYTHON:" prefix
    is_python: bool             # True ⇒ pandas_expr is a Series expression for df["col"] = …
    self_confidence: float      # 0.0 – 1.0 from the translator's own self-report
    reasoning: str              # one-sentence rationale
    independent_score: float    # 0.0 – 1.0 from the scorer (separate call)
    score_reason: str           # one-sentence justification from the scorer

    @property
    def combined_score(self) -> float:
        """Average of the two scores. We bias slightly to the independent one
        because the translator can be confidently wrong, but the independent
        scorer doesn't know what alternative would have been better."""
        return 0.4 * self.self_confidence + 0.6 * self.independent_score


class LLMTranslator:
    """Holds the LiteLLM-backed translator + scorer.

    Usage:
        t = LLMTranslator(model="gpt-4o-mini", api_key_env_var="OPENAI_API_KEY")
        result = t.translate_and_score('IIF([qty] > 10, "bulk", "standard")')
        if result.combined_score >= 0.8:
            emit(result.pandas_expr)
        else:
            flag_for_manual_review(result)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env_var: Optional[str] = None,
        score_threshold: float = 0.8,
    ):
        self.model = model
        self.api_key_env_var = api_key_env_var
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------ public

    def translate_and_score(self, expr: str) -> TranslationResult:
        translation = self._translate(expr)
        independent_score, score_reason = self._score(expr, translation)
        return TranslationResult(
            original=expr,
            pandas_expr=translation["pandas_expr"],
            is_python=bool(translation.get("is_python", False)),
            self_confidence=float(translation.get("self_confidence", 0.0)),
            reasoning=str(translation.get("reasoning", "")),
            independent_score=independent_score,
            score_reason=score_reason,
        )

    # ------------------------------------------------------------------ internal

    def _completion_kwargs(self) -> dict:
        try:
            import litellm  # noqa: F401  (verify the dep is present)
        except ImportError as e:
            raise ImportError(
                "v1.5 LLM-assisted translation needs LiteLLM. "
                "Install with: pip install 'alteryx-to-dagster[llm]' "
                "or pip install 'litellm>=1.30.0'"
            ) from e

        kwargs: dict = {"model": self.model, "temperature": 0.0, "max_tokens": 400}
        if self.api_key_env_var:
            key = os.environ.get(self.api_key_env_var)
            if not key:
                raise RuntimeError(
                    f"LLM translator: env var {self.api_key_env_var!r} is not set."
                )
            kwargs["api_key"] = key
        return kwargs

    def _translate(self, expr: str) -> dict:
        import litellm

        kwargs = self._completion_kwargs()
        messages = [
            {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Alteryx expression: {expr}"},
        ]
        resp = litellm.completion(messages=messages, **kwargs)
        raw = resp.choices[0].message.content or ""
        body = _strip_codefences(raw).strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # The model sometimes adds prose around the JSON. Pull the first {…} block.
            match = re.search(r"\{[\s\S]*\}", body)
            if not match:
                raise RuntimeError(
                    f"LLM translator returned non-JSON for {expr!r}: {body[:400]!r}"
                )
            data = json.loads(match.group(0))

        pandas_expr = str(data.get("pandas_expr", "")).strip()
        is_python = bool(data.get("is_python", False))
        # Normalize: strip the PYTHON: prefix when present, regardless of is_python flag.
        if pandas_expr.upper().startswith("PYTHON:"):
            pandas_expr = pandas_expr.split(":", 1)[1].strip()
            is_python = True
        return {
            "pandas_expr": pandas_expr,
            "is_python": is_python,
            "self_confidence": data.get("self_confidence", 0.0),
            "reasoning": data.get("reasoning", ""),
        }

    def _score(self, original_expr: str, translation: dict) -> tuple[float, str]:
        import litellm

        kwargs = self._completion_kwargs()
        kwargs["max_tokens"] = 200
        path = "PYTHON path (pandas Series expression)" if translation["is_python"] else "pandas eval expression"
        user_msg = (
            f"Alteryx expression: {original_expr}\n"
            f"Pandas translation ({path}): {translation['pandas_expr']}"
        )
        messages = [
            {"role": "system", "content": _SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        resp = litellm.completion(messages=messages, **kwargs)
        text = resp.choices[0].message.content or ""

        score_m = re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        reason_m = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        if not score_m:
            return 0.0, f"(score parse failed; raw={text[:200]!r})"
        raw_score = float(score_m.group(1))
        raw_score = max(0.0, min(10.0, raw_score))
        return raw_score / 10.0, (reason_m.group(1).strip() if reason_m else "")


def _strip_codefences(s: str) -> str:
    """Models sometimes wrap JSON in ```json…``` despite instructions."""
    s = s.strip()
    if s.startswith("```"):
        # drop opening fence (with or without lang tag)
        s = re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
