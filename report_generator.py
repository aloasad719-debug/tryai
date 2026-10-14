"""Compatibility shim around builders.build_report_pdf."""

from builders import build_report_pdf


def generate_pdf_report(df, chart=None, title="Data Analysis Report", dataset_name="uploaded_dataset"):
    extra_charts = [("Chart", chart)] if chart is not None else []
    return build_report_pdf(
        df,
        dataset_name,
        title,
        include_auto_charts=True,
        extra_charts=extra_charts,
    )
