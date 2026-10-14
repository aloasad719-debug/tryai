"""
ai_agent.py
Lets Gemini USE YOUR PYTHON CODE AS TOOLS.

How it works:
  1. The user types a normal request in the chat.
  2. Gemini decides which tool(s) below to call (summary, clean, export, chart, PDF, dashboard...).
  3. Each tool is a normal Python method in the Toolbox class. It runs on the real data.
  4. Gemini reads the tool results and writes the final answer.
Files, charts and tables the tools create are collected in `toolbox.artifacts`
so the app can show them and offer downloads.
"""

import difflib
import functools
import io
import os
import re
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from google.genai import types

from ai_analyst import (
    BUSY_MARKERS,
    FALLBACK_MODEL,
    build_context,
    result_to_table,
    result_to_text,
    run_code,
)
from builders import (
    build_dashboard_zip,
    build_report_pdf,
    categorical_columns,
    chart_bar,
    chart_box,
    chart_heatmap,
    chart_histogram,
    chart_line,
    chart_pie,
    chart_scatter,
    detect_date_columns,
    fmt_number,
    measure_columns,
    numeric_columns,
)
from data_engine import cap_iqr_outliers, detect_iqr_outliers
from data_engine import remove_duplicates as _remove_duplicates

AGGREGATIONS = {"sum", "mean", "count", "median", "min", "max"}
CHART_ALIASES = {
    "bar chart": "bar", "column": "bar", "bars": "bar", "barchart": "bar",
    "hist": "histogram", "distribution": "histogram",
    "trend": "line", "line chart": "line", "time series": "line", "linechart": "line",
    "donut": "pie", "pie chart": "pie",
    "scatter plot": "scatter", "scatterplot": "scatter",
    "boxplot": "box", "box plot": "box",
    "correlation": "heatmap", "correlation heatmap": "heatmap",
}
ALLOWED_CODE_FILES = (".py", ".dax", ".m", ".pq", ".sql", ".md", ".txt", ".json", ".html", ".css", ".js", ".csv")


def _tool(label):
    """Record that a tool was used, and turn any crash into a message Gemini can react to."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            self.tools_used.append(label)
            try:
                return function(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                return f"ERROR in {function.__name__}: {type(exc).__name__}: {exc}"

        return wrapper

    return decorator


class Toolbox:
    """The tools Gemini can call. All data work happens here, in Python."""

    def __init__(self, df, name, original=None):
        self.df = df
        self.original = original if original is not None else df
        self.name = name or "dataset"
        self.cleaning_log = []
        self.report_charts = []
        self.artifacts = []
        self.tools_used = []
        self.codes = []

    # ---------- bookkeeping ----------

    def begin_turn(self):
        self.artifacts = []
        self.tools_used = []
        self.codes = []

    def snapshot(self):
        return (self.df, list(self.cleaning_log), list(self.report_charts))

    def restore(self, snap):
        self.df, log, charts = snap
        self.cleaning_log = list(log)
        self.report_charts = list(charts)
        self.begin_turn()

    def tool_functions(self):
        return [
            self.get_dataset_summary,
            self.run_analysis,
            self.make_chart,
            self.clean_data,
            self.export_cleaned_data,
            self.reset_to_original_data,
            self.generate_pdf_report,
            self.generate_dashboard_package,
            self.save_code_file,
        ]

    def _add_file(self, name, data, mime):
        self.artifacts.append({"kind": "file", "name": name, "data": data, "mime": mime})

    def _add_chart(self, caption, png):
        self.artifacts.append({"kind": "chart", "caption": caption, "data": png})
        self.report_charts.append((caption, png))

    def _col(self, name, numeric=False):
        wanted = str(name).strip()
        if not wanted:
            raise ValueError("A column name is required.")
        columns = list(self.df.columns)
        for column in columns:
            if str(column) == wanted:
                found = column
                break
        else:
            lowered = [c for c in columns if str(c).strip().lower() == wanted.lower()]
            if lowered:
                found = lowered[0]
            else:
                close = difflib.get_close_matches(wanted, [str(c) for c in columns], n=1, cutoff=0.6)
                if not close:
                    raise ValueError(f"Column '{name}' not found. Available columns: {[str(c) for c in columns][:40]}")
                found = next(c for c in columns if str(c) == close[0])
        if numeric and not (pd.api.types.is_numeric_dtype(self.df[found]) and not pd.api.types.is_bool_dtype(self.df[found])):
            raise ValueError(f"Column '{found}' is not numeric.")
        return found

    # ---------- TOOL 1: summary ----------

    @_tool("Dataset summary")
    def get_dataset_summary(self) -> str:
        """Read the loaded dataset and return a full overview: number of rows and columns, every column with its type,
        missing values and unique counts, duplicate rows, statistics of numeric columns, most common values of text
        columns, date ranges and outlier counts. Use this first whenever the user asks what the data contains,
        asks for a summary or overview, or before writing a report."""
        df = self.df
        numeric = numeric_columns(df)
        dates = detect_date_columns(df)
        categories = categorical_columns(df)
        lines = [
            f"Dataset: {self.name}",
            f"Rows: {len(df):,} | Columns: {len(df.columns)}",
            f"Duplicate rows: {int(df.duplicated().sum()):,}",
            f"Missing cells: {int(df.isna().sum().sum()):,}",
            f"Numeric columns: {[str(c) for c in numeric]}",
            f"Text/category columns: {[str(c) for c in categories]}",
            f"Date columns: {[str(c) for c in dates]}",
            "",
            "Columns (type | non-null | missing | unique):",
        ]
        for column in list(df.columns)[:80]:
            series = df[column]
            lines.append(f"- {column}: {series.dtype} | {int(series.notna().sum()):,} | {int(series.isna().sum()):,} | {int(series.nunique()):,}")
        if numeric:
            lines += ["", "Numeric statistics:", df[numeric[:15]].describe().round(3).transpose().to_string()]
        for column in categories[:8]:
            top = df[column].value_counts().head(5)
            lines.append(f"Top values of {column}: " + ", ".join(f"{k} ({v:,})" for k, v in top.items()))
        for column in dates[:3]:
            parsed = pd.to_datetime(df[column], errors="coerce").dropna()
            if not parsed.empty:
                lines.append(f"Date range of {column}: {parsed.min().date()} to {parsed.max().date()}")
        outliers = detect_iqr_outliers(df)
        if not outliers.empty:
            outliers = outliers[outliers["outlier_count"] > 0].sort_values("outlier_count", ascending=False).head(10)
            if not outliers.empty:
                lines += ["", "Columns with IQR outliers:", outliers.to_string(index=False)]
        return "\n".join(lines)[:9000]

    # ---------- TOOL 2: free-form analysis ----------

    @_tool("Custom analysis")
    def run_analysis(self, python_code: str, description: str = "") -> str:
        """Answer any specific question about the data by running pandas code on the dataset. The DataFrame is
        named `df`; `pd`, `np` and `plt` (matplotlib) are already imported - do NOT write imports. Put the final
        answer (a number, text, Series or DataFrame) in a variable named `result`. If a chart helps, build it with
        matplotlib (title and axis labels required) and assign the figure to `fig`. Column names must match exactly.
        Use this for calculations, filters, rankings, group-bys, percentages, comparisons, trends and any chart
        that make_chart cannot draw. The code cannot modify the dataset or write files.
        description: a short title for the result or chart, for example 'Revenue by region'."""
        result, png = run_code(python_code, self.df)
        self.codes.append(python_code)
        if result is None and png is None:
            return "The code ran but did not set a `result` variable. Assign the answer to `result` and try again."
        table = result_to_table(result)
        if table is not None:
            self.artifacts.append({"kind": "table", "data": table})
        if png is not None:
            self._add_chart(description or "Analysis chart", png)
        text = result_to_text(result)
        extras = []
        if table is not None:
            extras.append("The table is shown to the user.")
        if png is not None:
            extras.append("The chart is shown to the user.")
        return text + ("\n\n" + " ".join(extras) if extras else "")

    # ---------- TOOL 3: charts ----------

    @_tool("Chart")
    def make_chart(self, chart_type: str, x_column: str = "", y_column: str = "", aggregation: str = "sum",
                   top_n: int = 10, title: str = "") -> str:
        """Create a chart image and show it to the user. chart_type is one of: bar, line, pie, histogram, scatter,
        box, heatmap.
        bar: x_column is a category, y_column an optional numeric column (without it, rows are counted).
        line: x_column is a date, year or any ordered column; y_column is an optional numeric column.
        pie: x_column is a category, y_column an optional numeric column.
        histogram: x_column is a numeric column.
        scatter: x_column and y_column are both numeric.
        box: y_column is numeric; x_column is an optional category to compare.
        heatmap: shows correlations between all numeric columns (no columns needed).
        aggregation is one of sum, mean, count, median, min, max. top_n limits the number of bars or slices.
        title is optional."""
        kind = CHART_ALIASES.get(chart_type.strip().lower(), chart_type.strip().lower())
        agg = aggregation.strip().lower() if aggregation.strip().lower() in AGGREGATIONS else "sum"
        top_n = max(1, min(int(top_n), 30))
        df = self.df

        def grouped(x, y):
            if y:
                return df.groupby(x)[y].agg(agg)
            return df.groupby(x).size()

        if kind in ("bar", "pie"):
            x = self._col(x_column)
            y = self._col(y_column, numeric=True) if y_column else None
            series = grouped(x, y).sort_values(ascending=False)
            top = series.head(top_n)
            label = f"{agg.title()} of {y}" if y else "Number of rows"
            chart_title = title or f"{label} by {x}"
            if kind == "bar":
                png = chart_bar(top.index, top.values, chart_title, str(x), label)
            else:
                values, labels = list(top.values), list(top.index)
                rest = series.iloc[top_n:].sum()
                if len(series) > top_n and rest > 0:
                    values.append(rest)
                    labels.append("Other")
                if any(v < 0 for v in values):
                    raise ValueError("A pie chart cannot show negative values. Use a bar chart instead.")
                png = chart_pie(labels, values, chart_title)
            self._add_chart(chart_title, png)
            pairs = ", ".join(f"{k}={fmt_number(v)}" for k, v in top.items())
            return f"{kind.title()} chart created: {chart_title}. Values: {pairs}."

        if kind == "line":
            x = self._col(x_column)
            y = self._col(y_column, numeric=True) if y_column else None
            data = df[[x] + ([y] if y else [])].copy()
            if x in detect_date_columns(df):
                data[x] = pd.to_datetime(data[x], errors="coerce")
                data = data.dropna(subset=[x])
                if data[x].nunique() > 60:
                    data[x] = data[x].dt.to_period("M").dt.to_timestamp()
            series = (data.groupby(x)[y].agg(agg) if y else data.groupby(x).size()).sort_index()
            label = f"{agg.title()} of {y}" if y else "Number of rows"
            chart_title = title or f"{label} by {x}"
            self._add_chart(chart_title, chart_line(series.index, series.values, chart_title, str(x), label))
            pairs = ", ".join(f"{k}={fmt_number(v)}" for k, v in list(series.items())[:25])
            return f"Line chart created: {chart_title}. Points: {pairs}."

        if kind == "histogram":
            x = self._col(x_column, numeric=True)
            chart_title = title or f"Distribution of {x}"
            self._add_chart(chart_title, chart_histogram(df[x], chart_title, str(x)))
            values = df[x].dropna()
            return (f"Histogram created: {chart_title}. min={fmt_number(values.min())}, median={fmt_number(values.median())}, "
                    f"mean={fmt_number(values.mean())}, max={fmt_number(values.max())}.")

        if kind == "scatter":
            x = self._col(x_column, numeric=True)
            y = self._col(y_column, numeric=True)
            chart_title = title or f"{y} vs {x}"
            self._add_chart(chart_title, chart_scatter(df[x], df[y], chart_title, str(x), str(y)))
            return f"Scatter plot created: {chart_title}. Correlation = {df[x].corr(df[y]):.3f}."

        if kind == "box":
            y = self._col(y_column, numeric=True)
            if x_column:
                x = self._col(x_column)
                top = df[x].value_counts().head(8).index
                groups = {k: df.loc[df[x] == k, y] for k in top}
                chart_title = title or f"{y} by {x}"
                png = chart_box(groups, chart_title, str(x), str(y))
            else:
                chart_title = title or f"Spread of {y}"
                png = chart_box({y: df[y]}, chart_title, "", str(y))
            self._add_chart(chart_title, png)
            return f"Box plot created: {chart_title}."

        if kind == "heatmap":
            numeric = [c for c in measure_columns(df) if df[c].nunique() > 1][:15]
            if len(numeric) < 2:
                raise ValueError("At least two numeric columns are needed.")
            chart_title = title or "Correlation between numeric columns"
            self._add_chart(chart_title, chart_heatmap(df[numeric], chart_title))
            corr = df[numeric].corr().abs().where(~np.eye(len(numeric), dtype=bool)).stack().sort_values(ascending=False)
            top_pairs = ", ".join(f"{a} & {b}={df[a].corr(df[b]):.2f}" for (a, b), _ in corr.head(6).items())
            return f"Correlation heatmap created. Strongest relationships: {top_pairs}."

        raise ValueError(f"Unknown chart type '{chart_type}'. Use bar, line, pie, histogram, scatter, box or heatmap.")

    # ---------- TOOL 4: cleaning ----------

    @_tool("Data cleaning")
    def clean_data(self, remove_duplicates: bool = True, fill_missing: bool = False, cap_outliers: bool = False,
                   drop_empty_columns: bool = False, strip_text: bool = False) -> str:
        """Clean the dataset. The cleaned version replaces the working data, so later charts, exports and reports
        use it. Options: remove_duplicates removes repeated rows; fill_missing fills empty numeric cells with the
        median and empty text cells with the most common value; cap_outliers limits extreme numeric values to the
        IQR bounds; drop_empty_columns removes columns that are completely empty; strip_text removes extra spaces
        around text. Only enable the options the user asked for. Returns what changed."""
        df = self.df.copy()
        rows_before, cols_before = df.shape
        missing_before = int(df.isna().sum().sum())
        notes = []

        if strip_text:
            for column in categorical_columns(df):
                if pd.api.types.is_bool_dtype(df[column]) or isinstance(df[column].dtype, pd.CategoricalDtype):
                    continue
                df[column] = df[column].where(df[column].isna(), df[column].astype(str).str.strip())
            notes.append("Trimmed extra spaces in text columns")

        if drop_empty_columns:
            empty = [c for c in df.columns if df[c].isna().all()]
            df = df.drop(columns=empty)
            notes.append(f"Dropped {len(empty)} completely empty column(s)" + (f": {empty}" if empty else ""))

        if remove_duplicates:
            df, removed = _remove_duplicates(df)
            notes.append(f"Removed {removed:,} duplicate row(s)")

        if fill_missing:
            filled = 0
            for column in df.columns:
                gaps = int(df[column].isna().sum())
                if not gaps:
                    continue
                if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                    value = df[column].median()
                else:
                    modes = df[column].mode()
                    value = modes.iloc[0] if not modes.empty else None
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    df[column] = df[column].fillna(value)
                    filled += gaps
            notes.append(f"Filled {filled:,} missing value(s) (median for numbers, most common value for text)")

        if cap_outliers:
            before = df.copy()
            df = cap_iqr_outliers(df)
            changed = 0
            for column in numeric_columns(df):
                changed += int((before[column] != df[column]).sum())
            notes.append(f"Capped {changed:,} outlier value(s) to the IQR limits")

        df = df.reset_index(drop=True)
        self.df = df
        self.cleaning_log.extend(notes)
        missing_after = int(df.isna().sum().sum())
        return (
            f"Cleaning finished. Rows: {rows_before:,} -> {len(df):,}. Columns: {cols_before} -> {len(df.columns)}. "
            f"Missing cells: {missing_before:,} -> {missing_after:,}. Actions: " + "; ".join(notes) + "."
        )

    @_tool("Reset data")
    def reset_to_original_data(self) -> str:
        """Undo all cleaning and go back to the dataset exactly as it was uploaded. Use only when the user asks to
        undo, reset or restore the original data."""
        self.df = self.original
        self.cleaning_log = []
        return f"Restored the original data: {len(self.df):,} rows, {len(self.df.columns)} columns."

    # ---------- TOOL 5: export ----------

    @_tool("Export data")
    def export_cleaned_data(self, file_format: str = "csv") -> str:
        """Prepare the current (cleaned) dataset as a downloadable file for the user. file_format is 'csv' or
        'excel'. The download button appears under the chat answer. Use when the user wants to download, export or
        save the data."""
        fmt = file_format.strip().lower()
        if fmt in ("excel", "xlsx", "xls"):
            if len(self.df) > 1_000_000:
                return "The data has more than 1,000,000 rows, which Excel cannot hold. Export as CSV instead."
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                self.df.to_excel(writer, index=False, sheet_name="Cleaned Data")
            self._add_file("cleaned_dataset.xlsx", buffer.getvalue(),
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            kind = "Excel"
        else:
            self._add_file("cleaned_dataset.csv", self.df.to_csv(index=False).encode("utf-8-sig"), "text/csv")
            kind = "CSV"
        return f"{kind} file prepared with {len(self.df):,} rows and {len(self.df.columns)} columns. The download button is shown to the user."

    # ---------- TOOL 6: PDF report ----------

    @_tool("PDF report")
    def generate_pdf_report(self, title: str = "Data Analysis Report", executive_summary: str = "",
                            key_findings: str = "", recommendations: str = "", include_auto_charts: bool = True) -> str:
        """Create a detailed, professional PDF report with the dataset overview, data quality, cleaning actions,
        column details, statistics and charts, and offer it for download. BEFORE calling this, gather real facts with
        get_dataset_summary, run_analysis and make_chart (charts you already made are included automatically).
        Then write the text yourself from those facts:
        executive_summary: 3-6 sentences describing the data and the main story.
        key_findings: one specific finding per line, with real numbers.
        recommendations: one practical recommendation per line.
        include_auto_charts: also add automatic charts (distributions, correlations, top categories, trends)."""
        pdf = build_report_pdf(
            self.df, self.name, title, executive_summary, key_findings, recommendations,
            self.cleaning_log, self.report_charts[-10:], include_auto_charts,
        )
        self._add_file("data_analysis_report.pdf", pdf, "application/pdf")
        return f"PDF report created ({len(pdf) / 1024:.0f} KB). The download button is shown to the user."

    # ---------- TOOL 7: dashboard ----------

    @_tool("Dashboard code")
    def generate_dashboard_package(self, title: str = "Data Dashboard") -> str:
        """Create a ready-to-run dashboard package for the current data, as a downloadable ZIP. It contains
        dashboard.py (an interactive Power BI style dashboard built with Streamlit and Plotly: filters, KPI cards,
        bar, pie, line, histogram, box, scatter and correlation charts), cleaned_data.csv, requirements.txt, a README
        and powerbi_guide.md (step-by-step Power BI instructions with ready DAX measures). Use when the user asks for
        a dashboard, Power BI, or dashboard code. After calling it, explain in plain words how to run it."""
        data, _code = build_dashboard_zip(self.df, title)
        self._add_file("dashboard_package.zip", data, "application/zip")
        return ("Dashboard package created with dashboard.py, cleaned_data.csv, requirements.txt, README.md and "
                "powerbi_guide.md. Run it with: pip install -r requirements.txt, then streamlit run dashboard.py. "
                "The download button is shown to the user.")

    @_tool("Code file")
    def save_code_file(self, filename: str, code: str, description: str = "") -> str:
        """Save code that YOU wrote as a downloadable file. Use this when the user asks for code (for example Power BI
        DAX measures, Power Query M code, a custom dashboard script, or a Python script). Write complete, working,
        well-commented code tailored to the real column names. filename must end in .py, .dax, .m, .pq, .sql, .md,
        .txt, .json, .html, .css, .js or .csv."""
        name = os.path.basename(filename.strip()) or "code.txt"
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        if not name.lower().endswith(ALLOWED_CODE_FILES):
            name += ".txt"
        self._add_file(name, code.encode("utf-8"), "text/plain")
        self.codes.append(f"# {name}\n{code}")
        return f"Saved {name} ({len(code):,} characters). The download button is shown to the user."


# --------------------------------------------------
# The agent loop
# --------------------------------------------------

SYSTEM_PROMPT = """You are an AI data analyst inside a chat app. The user has uploaded a dataset and talks to you in plain language (English or Bangla - answer in the language they use). You work by calling your tools; the tools are real Python functions that run on the real data.

What you can do with your tools:
- get_dataset_summary: read the file and summarize it.
- run_analysis: answer any specific question with pandas code (calculations, rankings, filters, group-bys, custom charts).
- make_chart: draw bar, line, pie, histogram, scatter, box and heatmap charts.
- clean_data: remove duplicates, fill missing values, cap outliers, drop empty columns, trim text.
- export_cleaned_data: give the user a CSV or Excel download.
- generate_pdf_report: build a detailed PDF report with charts.
- generate_dashboard_package: build a Power BI style dashboard (code + Power BI guide) as a ZIP.
- save_code_file: give the user code you wrote (DAX, Power Query M, Python) as a download.
- reset_to_original_data: undo cleaning.

Rules:
1. Never guess or invent numbers. Every number in your answer must come from a tool result.
2. Only clean the data when the user asks for it. Say exactly what changed (rows before and after).
3. When a tool creates a file, chart or table, it is shown to the user automatically under your answer. Say "you can download it below" - do not paste links.
4. For a report: first call get_dataset_summary, then make 3-6 useful analyses or charts, then write a real executive summary, specific findings and recommendations and call generate_pdf_report.
5. For dashboard or Power BI requests: call generate_dashboard_package and explain simply how to run it. If they want specific DAX / Power Query / custom code, write it yourself with the real column names and save it with save_code_file.
6. If a tool returns an ERROR, read it, fix your call and try again.
7. Keep answers short, friendly and clear for a non-technical person. Lead with the answer, then one useful insight. Do not mention pandas, variables or tool names unless asked.
8. If the message is not about the data, just answer normally without tools.
"""


def _to_contents(messages):
    contents = []
    for message in messages[-12:]:
        role = "user" if message["role"] == "user" else "model"
        text = message.get("content", "")
        if not text:
            continue
        if not contents and role == "model":
            continue
        if contents and contents[-1].role == role:
            contents[-1].parts.append(types.Part(text=text))
        else:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def run_agent(client, model, messages, toolbox):
    """
    Let Gemini answer the last message in `messages`, using the toolbox.
    Returns: {"answer", "artifacts", "tools_used", "code"}
    """
    contents = _to_contents(messages)
    system = SYSTEM_PROMPT + "\nCURRENT DATASET:\n" + build_context(toolbox.df)
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=toolbox.tool_functions(),
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=20),
    )

    models = [model] if model == FALLBACK_MODEL else [model, FALLBACK_MODEL]
    snapshot = toolbox.snapshot()
    text, last_error, success = "", None, False

    for current in models:
        for attempt in range(3):
            toolbox.restore(snapshot)
            try:
                response = client.models.generate_content(model=current, contents=contents, config=config)
                text = response.text or ""
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if any(marker in str(exc) for marker in BUSY_MARKERS):
                    time.sleep(2 * (attempt + 1))
                    continue
                if "404" in str(exc) or "NOT_FOUND" in str(exc):
                    break  # this model name is not available: try the next one
                raise
        if success:
            break
    if not success:
        raise last_error

    tools_used = list(dict.fromkeys(toolbox.tools_used))
    if not text.strip():
        text = "Done." if toolbox.artifacts else "I couldn't produce an answer. Please try asking in a different way."
        if toolbox.artifacts:
            text += " Your results are shown below."

    return {
        "answer": text.strip(),
        "artifacts": list(toolbox.artifacts),
        "tools_used": tools_used,
        "code": "\n\n".join(toolbox.codes),
    }