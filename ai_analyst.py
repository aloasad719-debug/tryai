"""
ai_analyst.py
Natural-language data analysis with Gemini.

Flow for every question:
  1. Gemini reads the dataset schema + a small sample and writes pandas code.
  2. The code is checked (no imports, no file/network access) and executed on a COPY of the data.
  3. Gemini turns the real output into a short, plain-language answer.
If the code fails, Gemini gets the error and tries once more.
"""

import ast
import builtins
import io
import os
import re
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.genai import types

# --------------------------------------------------
# Safety checks for AI-written code
# --------------------------------------------------

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None",
]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "memoryview", "os", "sys", "subprocess",
}
BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_json",
    "to_clipboard", "to_feather", "to_hdf", "savefig", "system", "popen",
}


def validate_code(code: str) -> None:
    """Raise ValueError if the code tries to do anything unsafe."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are not allowed.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal are not allowed.")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("Private attributes are not allowed.")
            if node.attr.startswith("read_") or node.attr in BLOCKED_ATTRS:
                raise ValueError(f"Use of '.{node.attr}' is not allowed.")


# --------------------------------------------------
# Prompt helpers
# --------------------------------------------------

def build_context(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()[:80]
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}", "Columns (name: dtype, non-null, unique):"]
    for c in cols:
        lines.append(
            f"- {c!r}: {df[c].dtype}, {int(df[c].notna().sum())} non-null, {int(df[c].nunique())} unique"
        )
    sample = df.head(5).to_string(max_cols=30, max_colwidth=30)
    lines.append("\nFirst 5 rows:\n" + sample)
    return "\n".join(lines)


def _history_text(history, limit=6) -> str:
    if not history:
        return "(none)"
    recent = history[-limit:]
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)


CODE_RULES = """You are a careful data analyst who works in Python with pandas.
A DataFrame named `df` is already loaded. Write Python code that answers the user's question.

Rules:
- pandas as `pd`, numpy as `np` and matplotlib.pyplot as `plt` are already imported. NEVER write import statements.
- Do not modify `df` in place. Work on copies.
- Store the final answer in a variable named `result` (a number, string, Series or DataFrame).
  Keep tables to at most 50 rows unless the user asks for more.
- If a chart was requested or would clearly help, build it with matplotlib, give it a title and axis labels,
  and assign the figure to `fig`.
- Column names must match EXACTLY. Convert text to dates with pd.to_datetime(errors="coerce") when needed.
  Convert text numbers with pd.to_numeric(errors="coerce") when needed.
- Be tolerant of typos in the user's wording and map it to the closest real column.
- If the question is NOT about the dataset (small talk, general knowledge, coding help),
  write only:  result = "NOT_DATA"
- Reply with exactly ONE ```python code block and nothing else."""


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


_MODEL_REMAP = {"gemini-2.5-flash": "gemini-3.8-flash", "gemini-2.5-flash-lite": "gemini-3.5-flash-lite"}
FALLBACK_MODEL = _MODEL_REMAP.get(os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"), os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"))
BUSY_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "overloaded")


def generate_with_retry(client, model, contents, config=None) -> str:
    """Call Gemini; if it is busy, wait and retry, then try a backup model."""
    models = [model] if model == FALLBACK_MODEL else [model, FALLBACK_MODEL]
    first_error = None
    for current in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=current, contents=contents, config=config
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001
                first_error = first_error or exc
                text = str(exc)
                if any(marker in text for marker in BUSY_MARKERS):
                    time.sleep(2 * (attempt + 1))
                    continue
                if "404" in text or "NOT_FOUND" in text:
                    break  # this model name is not available: try the next one
                raise
    raise first_error


def _generate(client, model, prompt, temperature=0.1) -> str:
    return generate_with_retry(
        client,
        model,
        prompt,
        types.GenerateContentConfig(temperature=temperature),
    )


# --------------------------------------------------
# Run code
# --------------------------------------------------

def run_code(code: str, df: pd.DataFrame):
    validate_code(code)
    plt.close("all")

    env = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    exec(compile(code, "<analysis>", "exec"), env)

    result = env.get("result")
    fig = env.get("fig")
    if fig is None and plt.get_fignums():
        fig = plt.gcf()

    png = None
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        png = buf.getvalue()
    plt.close("all")

    return result, png


def result_to_text(result, limit=6000) -> str:
    if result is None:
        return "(no result variable)"
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.head(40).to_string()
    else:
        text = str(result)
    return text[:limit]


def result_to_table(result):
    """Return a DataFrame to display, or None for scalars/strings."""
    if isinstance(result, pd.DataFrame):
        return result.head(100)
    if isinstance(result, pd.Series):
        return result.head(100).rename("value").to_frame()
    return None


# --------------------------------------------------
# Public entry point
# --------------------------------------------------

def analyze_question(client, model, question, df, history=None):
    """
    Returns a dict:
      status: "ok" | "not_data" | "error"
      answer, code, table (DataFrame|None), png (bytes|None)
    """
    context = build_context(df)
    base_prompt = (
        f"{CODE_RULES}\n\nDATASET INFO:\n{context}\n\n"
        f"RECENT CONVERSATION:\n{_history_text(history)}\n\n"
        f"USER QUESTION: {question}"
    )

    prompt = base_prompt
    code, result, png, last_error = "", None, None, None

    for _attempt in range(2):
        code = _extract_code(_generate(client, model, prompt))
        try:
            result, png = run_code(code, df)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            prompt = (
                f"{base_prompt}\n\nYour previous code:\n```python\n{code}\n```\n"
                f"failed with: {last_error}\nFix it and return one corrected code block."
            )

    if last_error:
        return {
            "status": "error",
            "answer": (
                "I couldn't work that out from the data. Try rephrasing the question, "
                f"or mention the column names you care about.\n\n(Technical detail: {last_error})"
            ),
            "code": code,
            "table": None,
            "png": None,
        }

    if isinstance(result, str) and result.strip() == "NOT_DATA":
        return {"status": "not_data", "answer": "", "code": "", "table": None, "png": None}

    explain_prompt = (
        "You are a friendly data analyst talking to a non-technical person.\n"
        f"Their question: {question}\n\n"
        f"The analysis produced this real output:\n{result_to_text(result)}\n"
        f"{'A chart was also created and is shown below your answer.' if png else ''}\n\n"
        "Write a clear, concise answer in plain language using the exact numbers from the output. "
        "Add one short insight or caveat if it is useful. Do not mention code, variables or pandas. "
        "If the output is a table, summarize the key takeaway; the table is shown separately."
    )
    try:
        answer = _generate(client, model, explain_prompt, temperature=0.3).strip()
    except Exception:  # noqa: BLE001
        answer = result_to_text(result)

    return {
        "status": "ok",
        "answer": answer or result_to_text(result),
        "code": code,
        "table": result_to_table(result),
        "png": png,
    }