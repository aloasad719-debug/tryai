"""
builders.py
Everything that CREATES something:
  - chart images (bar, line, pie, histogram, scatter, box, heatmap)
  - the detailed PDF report
  - the dashboard package (Streamlit dashboard + Power BI guide, as a .zip)
"""

import io
import warnings
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from data_engine import detect_iqr_outliers

BLUE = "#2F6DB5"
DARK_BLUE = "#174A7E"
PALETTE = ["#2F6DB5", "#E07B39", "#3E9B6E", "#B5546B", "#7A6BB5", "#C9A227", "#4AA3B5", "#8C8C8C"]


# --------------------------------------------------
# Column helpers (work with old and new pandas versions)
# --------------------------------------------------

def numeric_columns(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def is_year_like(df, column):
    """True for columns such as Year (2014, 2015 ...) that should be grouped on, not summed."""
    series = df[column]
    if not (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)):
        return False
    values = series.dropna()
    if values.empty or not (values == values.round()).all():
        return False
    if not values.between(1900, 2100).all():
        return False
    return "year" in str(column).lower() or values.nunique() <= 60


def measure_columns(df):
    """Numeric columns that make sense to sum / average (not years, not unique IDs)."""
    result = []
    for column in numeric_columns(df):
        if is_year_like(df, column):
            continue
        values = df[column].dropna()
        if len(values) > 20 and values.nunique() == len(values) and (values == values.round()).all():
            continue
        result.append(column)
    return result


def year_columns(df):
    return [c for c in numeric_columns(df) if is_year_like(df, c)]


def detect_date_columns(df):
    """Columns that are real dates, or text columns that look like dates."""
    found = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        name = str(column).lower()
        if not any(word in name for word in ("date", "time", "day", "month", "created", "updated")):
            continue
        sample = series.dropna().head(300)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample.astype(str), errors="coerce")
        if parsed.notna().mean() >= 0.8:
            found.append(column)
    return found


def categorical_columns(df, max_unique=None):
    """Text / category / bool columns (not dates, not numbers)."""
    dates = set(detect_date_columns(df))
    found = []
    for column in df.columns:
        series = df[column]
        if column in dates:
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            continue
        unique = int(series.nunique())
        if unique < 2:
            continue
        if max_unique is not None and unique > max_unique:
            continue
        found.append(column)
    return found


def fmt_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return ""
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.1f}B"
    if a >= 1e6:
        return f"{value / 1e6:.1f}M"
    if a >= 1e4:
        return f"{value / 1e3:.1f}K"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


# --------------------------------------------------
# Charts (each returns PNG bytes)
# --------------------------------------------------

def _new_axes(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return fig, ax


def _png(fig):
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def chart_histogram(values, title, xlabel):
    values = pd.to_numeric(values, errors="coerce").dropna()
    fig, ax = _new_axes()
    bins = int(min(60, max(10, np.sqrt(len(values)))))
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    return _png(fig)


def chart_bar(labels, values, title, xlabel, ylabel):
    labels = [str(label)[:28] for label in labels]
    values = [float(v) for v in values]
    horizontal = len(labels) > 7 or max((len(l) for l in labels), default=0) > 12
    fig, ax = _new_axes((9, max(4.5, 0.42 * len(labels) + 1.5)) if horizontal else (9, 5))
    if horizontal:
        positions = range(len(labels))
        bars = ax.barh(list(positions), values, color=BLUE)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt_number(value),
                    va="center", fontsize=8)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.margins(x=0.12)
    else:
        bars = ax.bar(labels, values, color=BLUE)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_number(value),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.12)
    ax.set_title(title, fontweight="bold")
    return _png(fig)


def chart_line(x, y, title, xlabel, ylabel):
    fig, ax = _new_axes()
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o" if len(x) <= 40 else None, markersize=4)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.15)
    fig.autofmt_xdate() if len(x) > 12 else None
    return _png(fig)


def chart_scatter(x, y, title, xlabel, ylabel):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) > 5000:
        data = data.sample(5000, random_state=1)
    fig, ax = _new_axes()
    ax.scatter(data["x"], data["y"], alpha=0.4, s=14, color=BLUE)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.15)
    return _png(fig)


def chart_pie(labels, values, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.pie(
        values,
        labels=[str(l)[:24] for l in labels],
        autopct="%1.0f%%",
        startangle=90,
        colors=(PALETTE * 4)[: len(values)],
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax.set_title(title, fontweight="bold")
    ax.axis("equal")
    return _png(fig)


def chart_box(groups, title, xlabel, ylabel):
    fig, ax = _new_axes()
    labels = [str(k)[:20] for k in groups.keys()]
    ax.boxplot(
        [pd.to_numeric(v, errors="coerce").dropna() for v in groups.values()],
        patch_artist=True,
        boxprops={"facecolor": "#CFE0F3", "edgecolor": BLUE},
        medianprops={"color": "#E07B39"},
    )
    ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(labels) > 6:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _png(fig)


def chart_heatmap(df_numeric, title="Correlation between numeric columns"):
    corr = df_numeric.corr()
    size = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(6, size * 0.8), max(5, size * 0.7)))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([str(c)[:18] for c in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(c)[:18] for c in corr.columns])
    if size <= 12:
        for i in range(size):
            for j in range(size):
                value = corr.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(value) > 0.6 else "black")
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title(title, fontweight="bold")
    return _png(fig)


def auto_report_charts(df):
    """Pick sensible charts automatically. Returns a list of (caption, png)."""
    charts = []
    numeric = [c for c in measure_columns(df) if df[c].nunique() > 1]
    categories = [c for c in categorical_columns(df) if df[c].nunique() <= max(50, 0.5 * len(df)) and df[c].nunique() <= 1000]
    dates = detect_date_columns(df)

    for column in numeric[:4]:
        charts.append((f"Distribution of {column}", chart_histogram(df[column], f"Distribution of {column}", str(column))))

    if len(numeric) >= 2:
        charts.append(("Correlation between numeric columns", chart_heatmap(df[numeric[:12]])))

    for column in categories[:3]:
        counts = df[column].value_counts().head(10)
        charts.append((
            f"Top values of {column}",
            chart_bar(counts.index, counts.values, f"Top {len(counts)} values of {column}", str(column), "Rows"),
        ))

    if dates:
        column = dates[0]
        series = pd.to_datetime(df[column], errors="coerce").dropna()
        if not series.empty:
            per_month = series.dt.to_period("M").value_counts().sort_index()
            if len(per_month) >= 2:
                charts.append((
                    f"Records over time ({column})",
                    chart_line(per_month.index.to_timestamp(), per_month.values,
                               f"Records per month ({column})", str(column), "Rows"),
                ))
    return charts


# --------------------------------------------------
# PDF report
# --------------------------------------------------

PAGE_WIDTH = A4[0] - 3.6 * cm


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("RTitle", parent=base["Title"], textColor=colors.HexColor(DARK_BLUE),
                                fontSize=24, leading=28, alignment=0, spaceAfter=4),
        "subtitle": ParagraphStyle("RSub", parent=base["Normal"], textColor=colors.HexColor("#666666"),
                                   fontSize=10, spaceAfter=14),
        "h1": ParagraphStyle("RH1", parent=base["Heading2"], textColor=colors.HexColor(DARK_BLUE),
                             fontSize=15, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("RH2", parent=base["Heading3"], textColor=colors.HexColor("#333333"),
                             fontSize=11, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("RBody", parent=base["Normal"], fontSize=10, leading=14, spaceAfter=6),
        "bullet": ParagraphStyle("RBullet", parent=base["Normal"], fontSize=10, leading=14,
                                 leftIndent=14, bulletIndent=2, spaceAfter=3),
        "cell": ParagraphStyle("RCell", parent=base["Normal"], fontSize=8, leading=10),
        "cellhead": ParagraphStyle("RCellHead", parent=base["Normal"], fontSize=8, leading=10,
                                   textColor=colors.white, fontName="Helvetica-Bold"),
        "caption": ParagraphStyle("RCaption", parent=base["Normal"], fontSize=8.5, leading=11,
                                  textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=10),
    }


def _table(rows, widths=None):
    styles = _styles()
    data = []
    for r, row in enumerate(rows):
        style = styles["cellhead"] if r == 0 else styles["cell"]
        data.append([Paragraph(escape(str(cell))[:300], style) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(DARK_BLUE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6FB")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8D3DF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _image(png, max_width=PAGE_WIDTH, max_height=10.5 * cm):
    width, height = ImageReader(io.BytesIO(png)).getSize()
    scale = min(max_width / width, max_height / height)
    return Image(io.BytesIO(png), width=width * scale, height=height * scale)


def _lines(text):
    """Split free text into clean lines (removes bullets and numbering)."""
    result = []
    for raw in str(text or "").splitlines():
        line = raw.strip().lstrip("-•*").strip()
        if line[:3].rstrip(".) ").isdigit():
            line = line.split(" ", 1)[-1] if " " in line else line
        if line:
            result.append(line)
    return result


def build_report_pdf(df, dataset_name, title, executive_summary="", key_findings="",
                     recommendations="", cleaning_log=None, extra_charts=None,
                     include_auto_charts=True):
    styles = _styles()
    cleaning_log = cleaning_log or []
    extra_charts = extra_charts or []
    story = []

    numeric = numeric_columns(df)
    dates = detect_date_columns(df)
    categories = categorical_columns(df)

    story.append(Paragraph(escape(title), styles["title"]))
    story.append(Paragraph(
        f"Dataset: {escape(str(dataset_name))} &nbsp;|&nbsp; Generated {datetime.now():%d %B %Y, %H:%M} "
        f"&nbsp;|&nbsp; {len(df):,} rows × {len(df.columns):,} columns",
        styles["subtitle"],
    ))

    if str(executive_summary).strip():
        story.append(Paragraph("Executive summary", styles["h1"]))
        for paragraph in str(executive_summary).strip().split("\n\n"):
            story.append(Paragraph(escape(" ".join(paragraph.split())), styles["body"]))

    findings = _lines(key_findings)
    if findings:
        story.append(Paragraph("Key findings", styles["h1"]))
        for finding in findings:
            story.append(Paragraph(escape(finding), styles["bullet"], bulletText="•"))

    # 1. Overview
    story.append(Paragraph("1. Dataset overview", styles["h1"]))
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    overview = [
        ["Metric", "Value"],
        ["Rows", f"{len(df):,}"],
        ["Columns", f"{len(df.columns):,}"],
        ["Numeric columns", f"{len(numeric)}"],
        ["Text / category columns", f"{len(categories)}"],
        ["Date columns", f"{len(dates)}"],
        ["Duplicate rows", f"{int(df.duplicated().sum()):,}"],
        ["Missing cells", f"{int(df.isna().sum().sum()):,} ({df.isna().sum().sum() / max(df.size, 1) * 100:.2f}%)"],
        ["Memory used", f"{memory_mb:.1f} MB"],
    ]
    story.append(_table(overview, [7 * cm, PAGE_WIDTH - 7 * cm]))

    # 2. Data quality
    story.append(Paragraph("2. Data quality", styles["h1"]))
    story.append(Paragraph("Missing values", styles["h2"]))
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    if missing.empty:
        story.append(Paragraph("No missing values were found.", styles["body"]))
    else:
        rows = [["Column", "Missing", "% of rows"]]
        for column, count in missing.items():
            rows.append([column, f"{int(count):,}", f"{count / max(len(df), 1) * 100:.2f}%"])
        story.append(_table(rows, [9 * cm, 4 * cm, PAGE_WIDTH - 13 * cm]))

    story.append(Paragraph("Possible outliers (IQR method)", styles["h2"]))
    try:
        outliers = detect_iqr_outliers(df)
        outliers = outliers[outliers["outlier_count"] > 0].sort_values("outlier_count", ascending=False).head(15)
    except Exception:  # noqa: BLE001
        outliers = pd.DataFrame()
    if outliers.empty:
        story.append(Paragraph("No outliers were detected in the numeric columns.", styles["body"]))
    else:
        rows = [["Column", "Outliers", "Lower bound", "Upper bound"]]
        for _, r in outliers.iterrows():
            rows.append([r["column"], f"{int(r['outlier_count']):,}", fmt_number(r["lower_bound"]), fmt_number(r["upper_bound"])])
        story.append(_table(rows, [7 * cm, 3.4 * cm, 3.5 * cm, PAGE_WIDTH - 13.9 * cm]))

    story.append(Paragraph("Cleaning actions applied", styles["h2"]))
    if cleaning_log:
        for entry in cleaning_log:
            story.append(Paragraph(escape(entry), styles["bullet"], bulletText="•"))
    else:
        story.append(Paragraph("No cleaning was applied. This report describes the data as uploaded.", styles["body"]))

    # 3. Columns
    story.append(Paragraph("3. Column information", styles["h1"]))
    rows = [["Column", "Type", "Non-null", "Unique", "Example"]]
    for column in list(df.columns)[:60]:
        series = df[column]
        non_null = series.dropna()
        example = str(non_null.iloc[0])[:30] if not non_null.empty else ""
        rows.append([column, str(series.dtype), f"{len(non_null):,}", f"{int(series.nunique()):,}", example])
    story.append(_table(rows, [5 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, PAGE_WIDTH - 12.6 * cm]))
    if len(df.columns) > 60:
        story.append(Paragraph(f"Showing the first 60 of {len(df.columns)} columns.", styles["caption"]))

    # 4. Statistics
    story.append(Paragraph("4. Statistical summary", styles["h1"]))
    if numeric:
        describe = df[numeric].describe().round(3)
        order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for start in range(0, min(len(numeric), 18), 6):
            chunk = numeric[start:start + 6]
            rows = [["Statistic"] + [str(c)[:16] for c in chunk]]
            for stat in order:
                rows.append([stat] + [fmt_number(describe.loc[stat, c]) for c in chunk])
            first = 2.2 * cm
            story.append(_table(rows, [first] + [(PAGE_WIDTH - first) / len(chunk)] * len(chunk)))
            story.append(Spacer(1, 0.3 * cm))
        if len(numeric) > 18:
            story.append(Paragraph(f"Showing 18 of {len(numeric)} numeric columns.", styles["caption"]))
    else:
        story.append(Paragraph("This dataset has no numeric columns.", styles["body"]))

    if categories:
        story.append(Paragraph("Text / category columns", styles["h2"]))
        rows = [["Column", "Unique values", "Most common", "Rows", "Share"]]
        for column in categories[:12]:
            counts = df[column].value_counts()
            rows.append([column, f"{len(counts):,}", str(counts.index[0])[:30], f"{int(counts.iloc[0]):,}",
                         f"{counts.iloc[0] / max(len(df), 1) * 100:.1f}%"])
        story.append(_table(rows, [4.6 * cm, 2.8 * cm, 5 * cm, 2.4 * cm, PAGE_WIDTH - 14.8 * cm]))

    # 5. Charts
    story.append(Paragraph("5. Visual analysis", styles["h1"]))
    charts = list(extra_charts)
    if include_auto_charts:
        try:
            charts += auto_report_charts(df)
        except Exception:  # noqa: BLE001
            pass
    seen, unique_charts = set(), []
    for caption, png in charts:
        if caption not in seen:
            seen.add(caption)
            unique_charts.append((caption, png))
    charts = unique_charts
    if not charts:
        story.append(Paragraph("No charts could be created for this dataset.", styles["body"]))
    for number, (caption, png) in enumerate(charts, start=1):
        story.append(KeepTogether([
            _image(png),
            Paragraph(f"Figure {number}. {escape(str(caption))}", styles["caption"]),
        ]))

    # 6. Recommendations
    recs = _lines(recommendations)
    if recs:
        story.append(Paragraph("6. Recommendations", styles["h1"]))
        for rec in recs:
            story.append(Paragraph(escape(rec), styles["bullet"], bulletText="•"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(1.8 * cm, 1.0 * cm, str(title)[:80])
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.8 * cm, title=str(title),
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# --------------------------------------------------
# Dashboard package (Streamlit dashboard + Power BI guide)
# --------------------------------------------------

DASHBOARD_TEMPLATE = '''import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

TITLE = __TITLE__
DATA_FILE = Path(__file__).parent / "cleaned_data.csv"
NUMERIC = __NUMERIC__
CATEGORICAL = __CATEGORICAL__
DATES = __DATES__
TREND_COLUMNS = __TREND__

st.set_page_config(page_title=TITLE, page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: #F4F7FB; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    for column in DATES:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


df = load_data()

st.title("📊 " + TITLE)
st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

# ---------------- Filters (sidebar) ----------------
st.sidebar.header("Filters")
filtered = df

for column in CATEGORICAL[:5]:
    options = sorted(df[column].dropna().astype(str).unique().tolist())[:300]
    picked = st.sidebar.multiselect(column, options)
    if picked:
        filtered = filtered[filtered[column].astype(str).isin(picked)]

if DATES:
    date_column = DATES[0]
    valid = df[date_column].dropna()
    if not valid.empty:
        chosen = st.sidebar.date_input(
            date_column, (valid.min().date(), valid.max().date())
        )
        if isinstance(chosen, tuple) and len(chosen) == 2:
            start = pd.Timestamp(chosen[0])
            end = pd.Timestamp(chosen[1]) + pd.Timedelta(days=1)
            filtered = filtered[(filtered[date_column] >= start) & (filtered[date_column] < end)]

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ---------------- KPI cards ----------------
slots = st.columns(min(4, 1 + len(NUMERIC)))
slots[0].metric("Rows", f"{len(filtered):,}")
for slot, column in zip(slots[1:], NUMERIC[:3]):
    slot.metric(
        f"Total {column}",
        f"{filtered[column].sum():,.0f}",
        help=f"Average: {filtered[column].mean():,.2f}",
    )

overview_tab, distribution_tab, relation_tab, data_tab = st.tabs(
    ["Overview", "Distributions", "Relationships", "Data"]
)

# ---------------- Overview ----------------
with overview_tab:
    if CATEGORICAL and NUMERIC:
        c1, c2, c3 = st.columns(3)
        group = c1.selectbox("Group by", CATEGORICAL, key="group")
        measure = c2.selectbox("Measure", NUMERIC, key="measure")
        how = c3.selectbox("Calculation", ["sum", "mean", "count", "max", "min"], key="how")

        grouped = (
            filtered.groupby(group)[measure].agg(how)
            .sort_values(ascending=False).head(15).reset_index()
        )
        left, right = st.columns(2)
        left.plotly_chart(
            px.bar(grouped, x=group, y=measure, title=f"{how.title()} of {measure} by {group}"),
            use_container_width=True,
        )
        if (grouped[measure] >= 0).all():
            right.plotly_chart(
                px.pie(grouped, names=group, values=measure, title=f"Share of {measure}"),
                use_container_width=True,
            )

    if TREND_COLUMNS and NUMERIC:
        t1, t2 = st.columns(2)
        trend_x = t1.selectbox("Trend over", TREND_COLUMNS, key="trend_x")
        trend_y = t2.selectbox("Trend measure", NUMERIC, key="trend_y")
        if trend_x in DATES:
            trend = (
                filtered.set_index(trend_x)[trend_y]
                .resample("MS").sum().reset_index()
            )
        else:
            trend = filtered.groupby(trend_x)[trend_y].sum().reset_index().sort_values(trend_x)
        st.plotly_chart(
            px.line(trend, x=trend_x, y=trend_y, markers=True, title=f"{trend_y} over {trend_x}"),
            use_container_width=True,
        )

# ---------------- Distributions ----------------
with distribution_tab:
    if NUMERIC:
        number = st.selectbox("Numeric column", NUMERIC, key="dist_number")
        st.plotly_chart(
            px.histogram(filtered, x=number, nbins=40, title=f"Distribution of {number}"),
            use_container_width=True,
        )
        if CATEGORICAL:
            category = st.selectbox("Compare across", CATEGORICAL, key="dist_category")
            top = filtered[category].value_counts().head(10).index
            st.plotly_chart(
                px.box(filtered[filtered[category].isin(top)], x=category, y=number,
                       title=f"{number} by {category}"),
                use_container_width=True,
            )
    else:
        st.info("No numeric columns to show.")

# ---------------- Relationships ----------------
with relation_tab:
    if len(NUMERIC) >= 2:
        r1, r2 = st.columns(2)
        x_axis = r1.selectbox("X axis", NUMERIC, key="scatter_x")
        y_axis = r2.selectbox("Y axis", NUMERIC, index=1, key="scatter_y")
        sample = filtered.sample(min(5000, len(filtered)), random_state=1)
        st.plotly_chart(
            px.scatter(sample, x=x_axis, y=y_axis, opacity=0.5, title=f"{y_axis} vs {x_axis}"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.imshow(filtered[NUMERIC[:12]].corr(), text_auto=".2f",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      title="Correlation between numeric columns"),
            use_container_width=True,
        )
    else:
        st.info("At least two numeric columns are needed.")

# ---------------- Data ----------------
with data_tab:
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button(
        "⬇ Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )
'''


def _dax_name(column):
    return str(column).replace("]", "]]")


def _powerbi_guide(title, numeric, categories, dates, years=()):
    lines = [
        f"# Power BI guide: {title}",
        "",
        "## 1. Load the data",
        "1. Open Power BI Desktop → **Get data** → **Text/CSV** → choose `cleaned_data.csv` → **Load**.",
        "2. In the Fields pane the table will be called `cleaned_data`.",
        "",
        "## 2. Create these measures",
        "Right-click the table → **New measure**, then paste one line at a time:",
        "",
        "```dax",
        "Total Rows = COUNTROWS(cleaned_data)",
    ]
    for column in numeric[:6]:
        name = _dax_name(column)
        lines.append(f"Total {column} = SUM(cleaned_data[{name}])")
        lines.append(f"Average {column} = AVERAGE(cleaned_data[{name}])")
    if dates:
        lines.append("")
        lines.append(f"Calendar = CALENDAR(MIN(cleaned_data[{_dax_name(dates[0])}]), MAX(cleaned_data[{_dax_name(dates[0])}]))")
    lines += ["```", "", "## 3. Build the report page", "Add these visuals from the Visualizations pane:", ""]
    lines.append("- **Cards**: `Total Rows`" + "".join(f", `Total {c}`" for c in numeric[:3]))
    if categories and numeric:
        lines.append(f"- **Clustered bar chart**: Y-axis `{categories[0]}`, X-axis `Total {numeric[0]}`")
        lines.append(f"- **Donut chart**: Legend `{categories[0]}`, Values `Total {numeric[0]}`")
    if dates and numeric:
        lines.append(f"- **Line chart**: X-axis `{dates[0]}`, Y-axis `Total {numeric[0]}`")
    elif years and numeric:
        lines.append(f"- **Line chart**: X-axis `{years[0]}` (set it to *Don't summarize*), Y-axis `Total {numeric[0]}`")
    elif numeric:
        lines.append(f"- **Histogram-style column chart**: use a Bin on `{numeric[0]}` (right-click field → New group → Bins)")
    if len(numeric) >= 2:
        lines.append(f"- **Scatter chart**: X `{numeric[0]}`, Y `{numeric[1]}`")
    for column in categories[:3]:
        lines.append(f"- **Slicer**: `{column}`")
    if dates:
        lines.append(f"- **Date slicer** (Between): `{dates[0]}`")
    lines += [
        "- **Table**: add the columns you want to show detail for",
        "",
        "## 4. Make it look good",
        "- View → Themes → choose a theme, align visuals with Format → Align.",
        "- Put cards in a row on top, charts in the middle, slicers on the left.",
        "",
    ]
    return "\n".join(lines)


def build_dashboard_zip(df, title="Data Dashboard"):
    numeric = measure_columns(df)
    dates = detect_date_columns(df)
    years = year_columns(df)
    categories = categorical_columns(df, max_unique=300)
    trend = list(dates) + years

    code = (
        DASHBOARD_TEMPLATE
        .replace("__TITLE__", repr(str(title)))
        .replace("__NUMERIC__", repr([str(c) for c in numeric]))
        .replace("__CATEGORICAL__", repr([str(c) for c in categories]))
        .replace("__DATES__", repr([str(c) for c in dates]))
        .replace("__TREND__", repr([str(c) for c in trend]))
    )

    readme = (
        f"# {title}\n\n"
        "## Option A: Python dashboard (works right now)\n"
        "1. Open a terminal in this folder.\n"
        "2. `pip install -r requirements.txt`\n"
        "3. `streamlit run dashboard.py`\n\n"
        "## Option B: Power BI\n"
        "Open `powerbi_guide.md` and follow the steps.\n"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dashboard.py", code)
        archive.writestr("cleaned_data.csv", df.to_csv(index=False))
        archive.writestr("requirements.txt", "streamlit\nplotly\npandas\n")
        archive.writestr("README.md", readme)
        archive.writestr("powerbi_guide.md", _powerbi_guide(title, numeric, categories, dates, years))
    return buffer.getvalue(), code