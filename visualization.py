"""Compatibility shim: chart helpers built on top of builders.py."""

import pandas as pd

from builders import (
    chart_bar,
    chart_heatmap,
    chart_histogram,
    chart_line,
    chart_scatter,
    measure_columns,
    numeric_columns,
)


def get_numeric_columns(df):
    return numeric_columns(df)


def create_histogram(df, column):
    return chart_histogram(df[column], f"Distribution of {column}", str(column))


def create_scatter_plot(df, x_column, y_column):
    return chart_scatter(
        df[x_column], df[y_column], f"{y_column} vs {x_column}", str(x_column), str(y_column)
    )


def create_correlation_heatmap(df):
    numeric = [c for c in measure_columns(df) if df[c].nunique() > 1][:12]
    if len(numeric) < 2:
        return None
    return chart_heatmap(df[numeric], "Correlation between numeric columns")


def create_line_chart(df, x_column, y_column):
    data = df[[x_column] + ([y_column] if y_column else [])].copy()
    if y_column:
        grouped = data.groupby(x_column)[y_column].mean().sort_index()
        x, y = list(grouped.index), list(grouped.values)
        title = f"Average of {y_column} by {x_column}"
        ylabel = str(y_column)
    else:
        grouped = data.groupby(x_column).size().sort_index()
        x, y = list(grouped.index), list(grouped.values)
        title = f"Number of rows by {x_column}"
        ylabel = "Rows"
    return chart_line(x, y, title, str(x_column), ylabel)
