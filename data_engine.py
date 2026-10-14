
import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(uploaded_file):
    """Load an uploaded CSV or Excel file into a DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")


def inspect_dataset(df):
    """Return a useful, automatic dataset summary."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.astype(str).tolist(),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_kb": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        ),
        "preview": df.head(5),
        "statistics": df.describe(include="all").transpose(),
    }


def remove_duplicates(df):
    """Return a copy with duplicate rows removed."""
    cleaned = df.drop_duplicates().copy()
    removed = len(df) - len(cleaned)
    return cleaned, removed


def missing_value_report(df):
    """Summarize missing values in each column."""
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (
            df.isnull().mean() * 100
        ).round(2),
    })
    return report.sort_values("missing_count", ascending=False)


def detect_iqr_outliers(df):
    """Detect possible outliers in numeric columns using IQR."""
    results = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        values = df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((values < lower) | (values > upper)).sum())

        results.append({
            "column": str(column),
            "outlier_count": count,
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        })

    return pd.DataFrame(results)


def cap_iqr_outliers(df, columns=None):
    """Return a copy with selected numeric columns capped to IQR bounds."""
    cleaned = df.copy()

    if columns is None:
        columns = cleaned.select_dtypes(
            include=np.number
        ).columns.tolist()

    for column in columns:
        if column not in cleaned.columns:
            continue

        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            continue

        values = cleaned[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        cleaned[column] = cleaned[column].clip(
            lower=lower, upper=upper
        )

    return cleaned
