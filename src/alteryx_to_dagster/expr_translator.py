"""Deterministic Alteryx-formula → pandas translation.

The goal: skip the LLM for everything we can mechanically translate.
LLM stays as the v1.5 fallback for the long tail (nested macros,
custom UDFs, the genuinely ambiguous stuff).

Public surface:

    translate(alteryx_expr: str) -> ExprTranslation

ExprTranslation carries:
  - pandas_expr  the translated expression
  - is_python    True ⇒ pandas Series expression (df["col"].str.contains(…), np.where(…),
                 pd.to_datetime(…)); False ⇒ pandas-eval-compatible string
  - fully        True ⇒ every token was recognized + translated deterministically.
                 False ⇒ at least one Alteryx function was left as-is (LLM should retry).
  - notes        ≥0 caveats per translation (1-indexed → 0-indexed conversions,
                 Alteryx-vs-pandas null semantics, etc.) — surfaced in MIGRATION.md.

Function coverage (all deterministic):

  Conditional:    IIF, Switch
  String:         Contains, StartsWith, EndsWith, Length, Trim, TrimLeft, TrimRight,
                  UpperCase, LowerCase, TitleCase, Substring, Left, Right,
                  ToString, ToNumber, Replace, ReplaceFirst,
                  Regex_Replace, Regex_Match, Regex_CountMatches, FindString,
                  PadLeft, PadRight
  Null:           IsNull, IsEmpty, Null
  DateTime:       DateTimeAdd, DateTimeDiff, DateTimeFormat, DateTimeParse,
                  DateTimeNow, DateTimeToday, DateTimeYear, DateTimeMonth,
                  DateTimeDay, DateTimeHour, DateTimeMinute, DateTimeSecond
  Bracket fields: [Field] → df["Field"] in PYTHON path, bare Field in eval

Argument parsing is paren-balanced + quote-aware — so nested calls like
`IIF(Contains([s], ",x"), a, b)` split correctly. We also recursively
translate the args of each function before assembling the result, so
chained Alteryx calls translate fully in one pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ExprTranslation:
    pandas_expr: str
    is_python: bool
    fully: bool                              # True iff no Alteryx-only tokens remain
    notes: List[str] = field(default_factory=list)


# Alteryx-function names we know how to translate. Match is case-INSENSITIVE,
# but we preserve the canonical name in the registry.
_KNOWN_FUNCTIONS: List[str] = []  # populated below by @register


# ---------------------------------------------------------------- registry

_TRANSLATORS: Dict[str, Callable[[List[str]], Tuple[str, bool, List[str]]]] = {}


def register(name: str):
    """Decorator to add a function translator.

    Each translator takes the already-translated argument expressions
    (Python/pandas-ready) and returns (pandas_expr, is_python, notes).
    """
    def deco(fn):
        _TRANSLATORS[name.lower()] = fn
        _KNOWN_FUNCTIONS.append(name)
        return fn
    return deco


# ---------------------------------------------------------------- public

def translate(alteryx_expr: str) -> ExprTranslation:
    """Translate an Alteryx formula expression to a pandas-compatible
    expression. Returns ExprTranslation with `fully=True` when every
    token translated deterministically."""
    expr = alteryx_expr.strip()
    notes: List[str] = []

    # First pass: if no function calls at all, this is bracket-stripping +
    # operator passthrough. Doesn't need PYTHON path.
    if not _has_function_call(expr):
        return ExprTranslation(
            pandas_expr=_strip_brackets_for_eval(expr),
            is_python=False,
            fully=True,
            notes=[],
        )

    # Otherwise: walk the tree, translating each function as we hit it.
    # Output of any translated function call is PYTHON-path (uses
    # np./pd./df["…"].* idioms that pandas eval can't compile).
    translated, fully, sub_notes = _translate_expr(expr)
    notes.extend(sub_notes)
    return ExprTranslation(
        pandas_expr=translated,
        is_python=True,
        fully=fully,
        notes=notes,
    )


# ---------------------------------------------------------------- parsing

_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_BRACKET_FIELD_RE = re.compile(r"\[([^\[\]]+)\]")
_OPERATOR_REPLACEMENTS = [
    # Alteryx logical operators → pandas / Python equivalents.
    # Be careful: replace as whole-word, not as substrings.
    (re.compile(r"\bAND\b", re.IGNORECASE), "&"),
    (re.compile(r"\bOR\b", re.IGNORECASE), "|"),
    (re.compile(r"\bNOT\b", re.IGNORECASE), "~"),
]


def _has_function_call(expr: str) -> bool:
    """True iff there's a `Name(...)` somewhere outside string literals."""
    return _FUNCTION_CALL_RE.search(_strip_string_literals(expr)) is not None


def _strip_string_literals(s: str) -> str:
    """Replace string-literal contents with X's so regex matches on names
    don't fire inside them."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in ('"', "'"):
            quote = c
            out.append(quote)
            i += 1
            while i < n and s[i] != quote:
                # Handle Alteryx escape: "" inside "…" represents a literal quote.
                if s[i] == quote and i + 1 < n and s[i+1] == quote:
                    out.append("X"); out.append("X")
                    i += 2
                    continue
                out.append("X")
                i += 1
            if i < n:
                out.append(quote)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _strip_brackets_for_eval(expr: str) -> str:
    """`[Field]` → `Field` for pandas-eval expressions."""
    return _BRACKET_FIELD_RE.sub(r"\1", expr)


def _wrap_brackets_for_python(expr: str) -> str:
    """`[Field]` → `df["Field"]` for PYTHON-path expressions, but only
    outside string literals."""
    # Walk char-by-char so we don't touch [ inside "..." strings.
    out = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c in ('"', "'"):
            quote = c
            out.append(quote)
            i += 1
            while i < n:
                out.append(expr[i])
                if expr[i] == quote and not (i + 1 < n and expr[i+1] == quote):
                    i += 1
                    break
                i += 1
            continue
        if c == "[":
            close = expr.find("]", i + 1)
            if close > i:
                name = expr[i + 1: close]
                out.append(f'df["{name}"]')
                i = close + 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _apply_operator_replacements(expr: str) -> str:
    """AND/OR/NOT → &/|/~. Skip replacements inside string literals."""
    # Easier approach: walk strings, only apply replacements to non-string segments.
    out = []
    buf = []
    in_str: Optional[str] = None
    for ch in expr:
        if in_str is None:
            if ch in ('"', "'"):
                # Flush buf with replacements, then enter string.
                seg = "".join(buf)
                for pat, repl in _OPERATOR_REPLACEMENTS:
                    seg = pat.sub(repl, seg)
                out.append(seg)
                buf = []
                in_str = ch
                out.append(ch)
            else:
                buf.append(ch)
        else:
            out.append(ch)
            if ch == in_str:
                in_str = None
    seg = "".join(buf)
    for pat, repl in _OPERATOR_REPLACEMENTS:
        seg = pat.sub(repl, seg)
    out.append(seg)
    return "".join(out)


def _split_args(args_str: str) -> List[str]:
    """Split a function-arg list on top-level commas, paren+quote aware."""
    args = []
    depth = 0
    in_str: Optional[str] = None
    buf = []
    i, n = 0, len(args_str)
    while i < n:
        ch = args_str[i]
        if in_str is None:
            if ch in ('"', "'"):
                in_str = ch
                buf.append(ch)
            elif ch in "([":
                depth += 1
                buf.append(ch)
            elif ch in ")]":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                args.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
            if ch == in_str:
                # Alteryx escape: "" inside a "…" string is a literal quote.
                if i + 1 < n and args_str[i+1] == in_str:
                    buf.append(args_str[i+1])
                    i += 2
                    continue
                in_str = None
        i += 1
    if buf:
        args.append("".join(buf).strip())
    return args


_PLACEHOLDER_FMT = "\x00ALTX_{}\x00"   # NUL-bracketed so no real expr text matches


def _translate_expr(expr: str) -> Tuple[str, bool, List[str]]:
    """Translate an Alteryx expression to a pandas (PYTHON-path) Series
    expression. Returns (translated_expr, fully, notes).

    Strategy: substitute each translated function call with an opaque
    placeholder so we never re-scan our OWN output (`np.where(…)`, etc.)
    as an "unknown function". Walk innermost-first by re-detecting candidates
    each iteration.
    """
    notes: List[str] = []
    fully = True

    # 1. Apply operator replacements outside strings.
    expr = _apply_operator_replacements(expr)

    # 2. Translate every Alteryx-named function call to a placeholder.
    #    Continue until no recognized calls remain; track unknowns so we
    #    don't loop on them.
    placeholders: Dict[str, str] = {}
    next_placeholder_id = 0
    seen_unknown_keys: set = set()

    while True:
        cand = _find_next_alteryx_call(expr, exclude=seen_unknown_keys)
        if cand is None:
            break
        name, start, args_start, args_end, after_end = cand
        args_str = expr[args_start: args_end]
        raw_args = _split_args(args_str)

        # Recursively translate each arg first (innermost-first via recursion).
        translated_args = []
        sub_fully_all = True
        for a in raw_args:
            t, sub_fully, sub_notes = _translate_expr(a)
            translated_args.append(t)
            sub_fully_all = sub_fully_all and sub_fully
            notes.extend(sub_notes)

        translator = _TRANSLATORS.get(name.lower())
        if translator is None:
            # Unknown — leave the call literally in place; record its
            # position-string so we don't try again.
            seen_unknown_keys.add(f"{name}:{start}")
            fully = False
            notes.append(
                f"Unknown Alteryx function `{name}(…)` left as-is. The LLM "
                f"translator can take a shot, or you can map it manually."
            )
            continue

        replacement, _is_python, fn_notes = translator(translated_args)
        notes.extend(fn_notes)
        # Don't propagate fully=False from sub-args here — the sub-arg
        # notes already speak for themselves; this function call itself
        # was translated successfully.
        if not sub_fully_all:
            fully = False

        # Stash the replacement behind a placeholder so we never re-scan
        # our own output (`np.where(…)`, `df["x"].str.contains(…)`, etc.)
        # as a candidate function call.
        ph_key = _PLACEHOLDER_FMT.format(next_placeholder_id)
        next_placeholder_id += 1
        placeholders[ph_key] = replacement
        expr = expr[:start] + ph_key + expr[after_end:]

    # 3. Wrap any leftover [Field] refs as df["Field"] (PYTHON path).
    expr = _wrap_brackets_for_python(expr)

    # 4. Substitute the placeholders back. Do this AFTER bracket-wrapping
    #    so the wrap pass doesn't touch the pre-translated `df["x"]` /
    #    `pd.…` / `np.…` strings hiding inside the placeholders.
    #
    #    Substitute highest-id first because outer translations reference
    #    inner placeholders in their replacement strings (e.g. ALTX_1's
    #    value contains ALTX_0). Forward iteration would replace ALTX_0
    #    in the original expr, then later replace ALTX_1 with a value that
    #    contains an UN-substituted ALTX_0 reference. Reverse-order keeps
    #    a single pass correct.
    for ph_key in sorted(
        placeholders.keys(),
        key=lambda k: -int(k.strip("\x00").split("_")[-1]),
    ):
        expr = expr.replace(ph_key, placeholders[ph_key])

    return expr, fully, notes


def _find_next_alteryx_call(expr: str, *, exclude: set) -> Optional[Tuple[str, int, int, int, int]]:
    """Find the next Alteryx-NAMED function call (one whose name is in
    `_TRANSLATORS` OR is an unknown name we haven't already marked as
    skip-able). Skips placeholders (which contain NUL bytes).

    Returns (name, name_start, args_start, args_end, after_end) or None.
    Prefers innermost first.
    """
    stripped = _strip_string_literals(expr)

    candidates = []
    for m in _FUNCTION_CALL_RE.finditer(stripped):
        start = m.start()
        name = m.group(1)
        # Skip placeholders' NUL bytes by definition (they aren't function calls).
        # Skip already-seen unknown-name positions.
        if f"{name}:{start}" in exclude:
            continue
        open_paren = m.end() - 1
        depth = 0
        end = None
        for i in range(open_paren, len(expr)):
            ch = expr[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        candidates.append((name, start, open_paren + 1, end, end + 1))

    # Prefer a call whose args contain no further alteryx-style function
    # calls (innermost-first).
    for cand in candidates:
        _, start, args_start, args_end, _ = cand
        inner = _strip_string_literals(expr[args_start: args_end])
        if not _FUNCTION_CALL_RE.search(inner):
            return cand
    return candidates[0] if candidates else None


# ============================================================ translators

@register("IIF")
def _t_iif(args: List[str]):
    if len(args) != 3:
        return f"IIF({', '.join(args)})", True, ["IIF expects 3 args; left as-is."]
    cond, a, b = args
    return f"np.where({cond}, {a}, {b})", True, []


@register("Switch")
def _t_switch(args: List[str]):
    """Alteryx Switch(val, default, k1, v1, k2, v2, …) — note the value/
    default come FIRST (Alteryx-specific arg order)."""
    if len(args) < 4 or (len(args) - 2) % 2 != 0:
        return f"Switch({', '.join(args)})", True, [
            "Switch needs (val, default, k1, v1, …) with paired keys/values; left as-is."
        ]
    val = args[0]
    default = args[1]
    pairs = args[2:]
    conds = []
    values = []
    for i in range(0, len(pairs), 2):
        k, v = pairs[i], pairs[i + 1]
        conds.append(f"({val} == {k})")
        values.append(v)
    return (
        f"np.select([{', '.join(conds)}], [{', '.join(values)}], default={default})",
        True,
        [],
    )


# ----------------------------------------------------------------- strings

@register("Contains")
def _t_contains(args: List[str]):
    if len(args) < 2:
        return f"Contains({', '.join(args)})", True, []
    s, sub = args[0], args[1]
    case_sensitive = "True" if len(args) < 3 else args[2]
    if case_sensitive == "1" or case_sensitive == "True":
        return f"{s}.str.contains({sub}, regex=False)", True, []
    return f"{s}.str.contains({sub}, regex=False, case=False)", True, []


@register("StartsWith")
def _t_startswith(args: List[str]):
    s, sub = args[0], args[1]
    return f"{s}.str.startswith({sub})", True, []


@register("EndsWith")
def _t_endswith(args: List[str]):
    s, sub = args[0], args[1]
    return f"{s}.str.endswith({sub})", True, []


@register("Length")
def _t_length(args: List[str]):
    return f"{args[0]}.str.len()", True, []


@register("Trim")
def _t_trim(args: List[str]):
    if len(args) == 1:
        return f"{args[0]}.str.strip()", True, []
    # Alteryx Trim(s, "chars") strips the specified characters.
    return f"{args[0]}.str.strip({args[1]})", True, []


@register("TrimLeft")
def _t_trim_left(args: List[str]):
    return f"{args[0]}.str.lstrip(" + (args[1] if len(args) > 1 else "") + ")", True, []


@register("TrimRight")
def _t_trim_right(args: List[str]):
    return f"{args[0]}.str.rstrip(" + (args[1] if len(args) > 1 else "") + ")", True, []


@register("UpperCase")
def _t_upper(args: List[str]):
    return f"{args[0]}.str.upper()", True, []


@register("LowerCase")
def _t_lower(args: List[str]):
    return f"{args[0]}.str.lower()", True, []


@register("TitleCase")
def _t_title(args: List[str]):
    return f"{args[0]}.str.title()", True, []


@register("Substring")
def _t_substring(args: List[str]):
    """Alteryx Substring(s, start) or Substring(s, start, len). 1-indexed.

    Pandas .str.slice() is 0-indexed end-exclusive. We translate:
       Substring(s, start)      → s.str.slice(start - 1)
       Substring(s, start, n)   → s.str.slice(start - 1, start - 1 + n)
    """
    s = args[0]
    start = args[1]
    notes = [
        "Alteryx Substring is 1-indexed; emitted pandas .str.slice() uses 0-indexed "
        "args (start - 1) so semantics are preserved for fixed integer starts."
    ]
    if len(args) == 2:
        return f"{s}.str.slice(({start}) - 1)", True, notes
    n = args[2]
    return f"{s}.str.slice(({start}) - 1, ({start}) - 1 + ({n}))", True, notes


@register("Left")
def _t_left(args: List[str]):
    s, n = args[0], args[1]
    return f"{s}.str.slice(0, {n})", True, []


@register("Right")
def _t_right(args: List[str]):
    s, n = args[0], args[1]
    return f"{s}.str.slice(-({n}))", True, []


@register("Replace")
def _t_replace(args: List[str]):
    s, find, repl = args[0], args[1], args[2]
    return f"{s}.str.replace({find}, {repl}, regex=False)", True, []


@register("ReplaceFirst")
def _t_replace_first(args: List[str]):
    s, find, repl = args[0], args[1], args[2]
    return f"{s}.str.replace({find}, {repl}, n=1, regex=False)", True, []


@register("Regex_Replace")
def _t_regex_replace(args: List[str]):
    s, pattern, repl = args[0], args[1], args[2]
    return f"{s}.str.replace({pattern}, {repl}, regex=True)", True, []


@register("REGEX_Replace")
def _t_regex_replace_alt(args: List[str]):
    return _t_regex_replace(args)


@register("Regex_Match")
def _t_regex_match(args: List[str]):
    s, pattern = args[0], args[1]
    return f"{s}.str.match({pattern})", True, []


@register("Regex_CountMatches")
def _t_regex_count(args: List[str]):
    s, pattern = args[0], args[1]
    return f"{s}.str.count({pattern})", True, []


@register("FindString")
def _t_find_string(args: List[str]):
    """Alteryx FindString returns the 0-indexed position of the first occurrence,
    or -1 if not found. pandas Series.str.find() has matching semantics."""
    s, sub = args[0], args[1]
    return f"{s}.str.find({sub})", True, []


@register("PadLeft")
def _t_pad_left(args: List[str]):
    s, width, ch = args[0], args[1], args[2]
    return f"{s}.str.rjust({width}, {ch})", True, []


@register("PadRight")
def _t_pad_right(args: List[str]):
    s, width, ch = args[0], args[1], args[2]
    return f"{s}.str.ljust({width}, {ch})", True, []


@register("ToString")
def _t_to_string(args: List[str]):
    if len(args) == 1:
        return f"{args[0]}.astype(str)", True, []
    # Alteryx ToString(x, n) → numeric x to string with n decimal places.
    return (
        f'{args[0]}.round({args[1]}).astype(str)',
        True,
        ["Alteryx ToString(x, n) — emitted as round(n).astype(str). May differ "
         "from Alteryx on trailing-zero formatting; tweak the format-spec by "
         "hand if you rely on it (e.g. `.map(lambda v: f\"{{v:.{{n}}f}}\")`)."],
    )


@register("ToNumber")
def _t_to_number(args: List[str]):
    return f'pd.to_numeric({args[0]}, errors="coerce")', True, []


# ----------------------------------------------------------------- nulls

@register("IsNull")
def _t_is_null(args: List[str]):
    return f"{args[0]}.isna()", True, []


@register("IsEmpty")
def _t_is_empty(args: List[str]):
    s = args[0]
    return f'({s}.isna() | ({s} == ""))', True, []


@register("Null")
def _t_null(args: List[str]):
    # Alteryx Null() is the literal null. Numpy nan covers both numeric + object-dtype.
    return "np.nan", True, []


# ------------------------------------------------------------- datetime

_DATETIME_UNIT_MAP = {
    "years": "years", "year": "years",
    "months": "months", "month": "months",
    "days": "days", "day": "days",
    "hours": "hours", "hour": "hours",
    "minutes": "minutes", "minute": "minutes",
    "seconds": "seconds", "second": "seconds",
}


def _normalize_unit(unit_str: str) -> Optional[str]:
    """Pull a unit name out of an arg that's a quoted string like '"days"'."""
    cleaned = unit_str.strip().strip('"').strip("'").lower()
    return _DATETIME_UNIT_MAP.get(cleaned)


@register("DateTimeAdd")
def _t_dt_add(args: List[str]):
    d, n, unit_raw = args[0], args[1], args[2]
    unit = _normalize_unit(unit_raw)
    if unit is None:
        return f"DateTimeAdd({', '.join(args)})", False, [
            f"DateTimeAdd: unknown unit {unit_raw!r}; left as-is."
        ]
    if unit in ("years", "months"):
        # Months/years aren't fixed-length — use pd.DateOffset.
        return f"({d} + pd.DateOffset({unit}={n}))", True, []
    return f"({d} + pd.Timedelta({unit}={n}))", True, []


@register("DateTimeDiff")
def _t_dt_diff(args: List[str]):
    a, b, unit_raw = args[0], args[1], args[2]
    unit = _normalize_unit(unit_raw)
    if unit is None:
        return f"DateTimeDiff({', '.join(args)})", False, [
            f"DateTimeDiff: unknown unit {unit_raw!r}; left as-is."
        ]
    if unit == "days":
        return f"(({a}) - ({b})).dt.days", True, []
    if unit == "hours":
        return f"(({a}) - ({b})).dt.total_seconds() / 3600", True, []
    if unit == "minutes":
        return f"(({a}) - ({b})).dt.total_seconds() / 60", True, []
    if unit == "seconds":
        return f"(({a}) - ({b})).dt.total_seconds()", True, []
    return f"DateTimeDiff({', '.join(args)})", False, [
        f"DateTimeDiff unit {unit!r} not implemented for now; left as-is."
    ]


@register("DateTimeFormat")
def _t_dt_format(args: List[str]):
    d, fmt = args[0], args[1]
    return f"{d}.dt.strftime({fmt})", True, [
        "Alteryx and Python strftime format codes differ in a few places "
        "(e.g. Alteryx `yyyy` ≈ Python `%Y`). Spot-check the format string."
    ]


@register("DateTimeParse")
def _t_dt_parse(args: List[str]):
    s, fmt = args[0], args[1]
    return f"pd.to_datetime({s}, format={fmt}, errors='coerce')", True, []


@register("DateTimeNow")
def _t_dt_now(args: List[str]):
    return "pd.Timestamp.now()", True, []


@register("DateTimeToday")
def _t_dt_today(args: List[str]):
    return "pd.Timestamp.today().normalize()", True, []


@register("DateTimeYear")
def _t_dt_year(args: List[str]):
    return f"{args[0]}.dt.year", True, []


@register("DateTimeMonth")
def _t_dt_month(args: List[str]):
    return f"{args[0]}.dt.month", True, []


@register("DateTimeDay")
def _t_dt_day(args: List[str]):
    return f"{args[0]}.dt.day", True, []


@register("DateTimeHour")
def _t_dt_hour(args: List[str]):
    return f"{args[0]}.dt.hour", True, []


@register("DateTimeMinute")
def _t_dt_minute(args: List[str]):
    return f"{args[0]}.dt.minute", True, []


@register("DateTimeSecond")
def _t_dt_second(args: List[str]):
    return f"{args[0]}.dt.second", True, []
