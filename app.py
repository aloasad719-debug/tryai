import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai_agent import Toolbox, run_agent
from ai_analyst import generate_with_retry
from builders import categorical_columns
from data_engine import (
    cap_iqr_outliers,
    detect_iqr_outliers,
    inspect_dataset,
    load_dataset,
    missing_value_report,
    remove_duplicates,
)
from report_generator import generate_pdf_report
from visualization import (
    create_correlation_heatmap,
    create_histogram,
    create_line_chart,
    create_scatter_plot,
    get_numeric_columns,
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Google has retired these older names for newer accounts; remap to working equivalents.
MODEL_REMAP = {
    "gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
}
MODEL_NAME = MODEL_REMAP.get(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

st.set_page_config(page_title="My AI Chat", page_icon="🤖", layout="wide")

HISTORY_FILE = Path(__file__).parent / "chat_history.json"

for key, default in {
    "dataset": None,
    "original_dataset": None,
    "dataset_name": None,
    "dataset_sig": None,
    "artifacts": {},
    "toolbox": None,
    "excel_ready": None,
    "pdf_ready": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# 2. CHAT HISTORY (load / save)
# --------------------------------------------------

def load_chats():
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_chats():
    temporary_file = HISTORY_FILE.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(st.session_state.chats, file, ensure_ascii=False, indent=2)
    temporary_file.replace(HISTORY_FILE)


def create_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    st.session_state.active_chat_id = chat_id
    save_chats()


if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if not st.session_state.chats:
    create_chat()

if (
    "active_chat_id" not in st.session_state
    or st.session_state.active_chat_id not in st.session_state.chats
):
    st.session_state.active_chat_id = next(iter(st.session_state.chats))

active_chat = st.session_state.chats[st.session_state.active_chat_id]


# --------------------------------------------------
# 3. STYLE
# --------------------------------------------------

st.markdown(
    """
<style>
.stApp { background: #ffffff; }
.block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 7rem; }
h1 { color: #202123; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stChatMessage"] {
    background: #f7f7f8; border: 1px solid #eeeeef;
    border-radius: 18px; padding: 1rem; margin-bottom: 12px;
}
[data-testid="stChatInput"] { border: 1px solid #dedee3; border-radius: 28px; background: #ffffff; }
.stButton button, [data-testid="stDownloadButton"] button { border-radius: 18px; }
[data-testid="stFileUploader"] { border-radius: 16px; background: #fafafa; padding: 10px; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. CONNECT TO GEMINI
# --------------------------------------------------

if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🤖 My AI Chat")

    if st.button("＋ New chat", use_container_width=True):
        create_chat()
        st.rerun()

    chat_ids = list(st.session_state.chats.keys())
    selected_chat_id = st.selectbox(
        "Chat history",
        options=chat_ids,
        index=chat_ids.index(st.session_state.active_chat_id),
        format_func=lambda cid: st.session_state.chats[cid].get("title", "New chat"),
    )
    if selected_chat_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_chat_id
        st.rerun()

    if st.button("🗑 Delete this chat", use_container_width=True):
        del st.session_state.chats[st.session_state.active_chat_id]
        save_chats()
        st.session_state.pop("active_chat_id", None)
        st.rerun()

    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    st.divider()
    st.subheader("📊 Your data")

    uploaded_dataset = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="dataset_uploader",
    )

    if uploaded_dataset is not None:
        signature = (uploaded_dataset.name, uploaded_dataset.size)
        if st.session_state.dataset_sig != signature:
            try:
                loaded = load_dataset(uploaded_dataset)
                st.session_state.dataset = loaded
                st.session_state.original_dataset = loaded
                st.session_state.dataset_name = uploaded_dataset.name
                st.session_state.dataset_sig = signature
                st.session_state.excel_ready = None
                st.session_state.pdf_ready = None
            except Exception as error:
                st.error(f"Could not read this file: {error}")
    elif st.session_state.dataset_sig is not None:
        st.session_state.dataset = None
        st.session_state.original_dataset = None
        st.session_state.dataset_name = None
        st.session_state.dataset_sig = None

    dataset = st.session_state.dataset

    if dataset is not None:
        st.success(
            f"{st.session_state.dataset_name}: "
            f"{len(dataset):,} rows × {len(dataset.columns):,} columns"
        )
        view = st.radio("View", ["💬 Chat", "📊 Data tools"], horizontal=True)
    else:
        view = "💬 Chat"
        st.caption("Upload a file, then just ask questions about it in the chat.")

    st.divider()
    with st.expander("＋ Attach an image"):
        uploaded_image = st.file_uploader(
            "PNG, JPG, JPEG or WebP",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"image_{st.session_state.active_chat_id}",
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Image attached", width=200)
            st.caption("While an image is attached, the chat answers about the image.")

    st.divider()
    st.subheader("Download history")

    export_json = json.dumps(active_chat, ensure_ascii=False, indent=2)
    export_text = "\n\n".join(
        f"{m['role'].title()}:\n{m['content']}" for m in active_chat["messages"]
    )
    st.download_button(
        "⬇ Download as TXT",
        data=export_text or "This conversation is empty.",
        file_name="my_ai_chat.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download as JSON",
        data=export_json,
        file_name="my_ai_chat.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption("Your history is saved locally in chat_history.json.")


# --------------------------------------------------
# 6. HELPERS FOR CHAT
# --------------------------------------------------

def figure_key(index):
    return f"{st.session_state.active_chat_id}:{index}"


def render_message_body(message, index):
    """Show one message: text, tools used, charts, tables, download buttons, code."""
    st.markdown(message["content"])

    tools_used = message.get("tools_used")
    if tools_used:
        st.caption("🔧 Tools used: " + " → ".join(tools_used))

    for number, item in enumerate(st.session_state.artifacts.get(figure_key(index), [])):
        if item["kind"] == "table":
            st.dataframe(item["data"], use_container_width=True)
        elif item["kind"] == "chart":
            st.image(item["data"], caption=item.get("caption") or None)
        elif item["kind"] == "file":
            st.download_button(
                f"⬇ Download {item['name']}",
                data=item["data"],
                file_name=item["name"],
                mime=item["mime"],
                key=f"download_{figure_key(index)}_{number}",
            )

    if message.get("code"):
        with st.expander("🔍 How I calculated this"):
            st.code(message["code"], language="python")


def render_message(message, index):
    with st.chat_message(message["role"]):
        render_message_body(message, index)


def get_toolbox():
    """The toolbox holds the data the AI works on. A new one is made when the data changes."""
    toolbox = st.session_state.toolbox
    if toolbox is None or toolbox.df is not st.session_state.dataset:
        toolbox = Toolbox(
            st.session_state.dataset,
            st.session_state.dataset_name,
            original=st.session_state.original_dataset,
        )
        st.session_state.toolbox = toolbox
    return toolbox


def build_contents(messages, image):
    contents = []
    for m in messages[-12:]:
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    if image is not None and contents:
        contents[-1].parts.append(
            types.Part.from_bytes(data=image.getvalue(), mime_type=image.type)
        )
    return contents


def plain_chat_reply(messages, image):
    text = generate_with_retry(client, MODEL_NAME, build_contents(messages, image))
    return text or "I couldn't generate a response."


def fill_missing(df):
    cleaned = df.copy()
    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median_value = cleaned[column].median()
            if pd.notna(median_value):
                cleaned[column] = cleaned[column].fillna(median_value)
        else:
            mode_values = cleaned[column].mode()
            if not mode_values.empty:
                cleaned[column] = cleaned[column].fillna(mode_values.iloc[0])
    return cleaned


# --------------------------------------------------
# 7. DATA TOOLS VIEW
# --------------------------------------------------

def render_data_tools(df):
    import matplotlib.pyplot as plt

    st.title("📊 Data tools")
    st.caption(f"Working on: {st.session_state.dataset_name}")

    tab_overview, tab_charts, tab_clean, tab_compare, tab_report = st.tabs(
        ["Overview", "Charts", "Cleaning", "Compare", "PDF report"]
    )
    numeric_columns = get_numeric_columns(df)

    # ---------- Overview ----------
    with tab_overview:
        summary = inspect_dataset(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{summary['rows']:,}")
        c2.metric("Columns", f"{summary['columns']:,}")
        c3.metric("Duplicate rows", f"{summary['duplicate_rows']:,}")

        with st.expander("👀 Preview dataset", expanded=True):
            st.dataframe(summary["preview"], use_container_width=True)

        with st.expander("📋 Column names and data types"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": summary["column_names"],
                        "Data type": [
                            summary["data_types"][c] for c in summary["column_names"]
                        ],
                    }
                ),
                use_container_width=True,
            )

        with st.expander("❗ Missing-value report"):
            st.dataframe(missing_value_report(df), use_container_width=True)

        with st.expander("📈 Statistical summary"):
            st.dataframe(summary["statistics"].astype(str), use_container_width=True)

        with st.expander("🔎 Possible outliers"):
            outliers = detect_iqr_outliers(df)
            if outliers.empty:
                st.info("No numeric columns are available for outlier analysis.")
            else:
                st.dataframe(outliers, use_container_width=True)

    # ---------- Charts ----------
    with tab_charts:
        if not numeric_columns:
            st.info("No numeric columns are available for charting.")
        else:
            chart_type = st.selectbox(
                "Choose chart type",
                ["Histogram", "Line chart", "Scatter plot", "Correlation heatmap"],
                key="analysis_chart_type",
            )
            try:
                fig = None
                if chart_type == "Histogram":
                    column = st.selectbox("Numeric column", numeric_columns, key="histogram_column")
                    fig = create_histogram(df, column)
                elif chart_type == "Line chart":
                    x_col = st.selectbox("X-axis", df.columns.tolist(), key="line_x")
                    y_col = st.selectbox("Y-axis", numeric_columns, key="line_y")
                    fig = create_line_chart(df, x_col, y_col)
                elif chart_type == "Scatter plot":
                    if len(numeric_columns) < 2:
                        st.warning("Scatter plots require two numeric columns.")
                    else:
                        x_col = st.selectbox("X-axis", numeric_columns, key="scatter_x")
                        y_col = st.selectbox("Y-axis", numeric_columns, index=1, key="scatter_y")
                        fig = create_scatter_plot(df, x_col, y_col)
                else:
                    fig = create_correlation_heatmap(df)

                if fig is not None:
                    st.image(fig)
            except Exception as chart_error:
                st.warning(f"Could not create chart: {chart_error}")

    # ---------- Cleaning ----------
    with tab_clean:
        remove_dup = st.checkbox("Remove duplicate rows", value=True, key="clean_duplicates")
        fill_miss = st.checkbox("Fill missing values (median / most common)", value=False, key="clean_missing")
        cap_out = st.checkbox("Cap numeric outliers using IQR", value=False, key="clean_outliers")

        cleaned = df.copy()
        if remove_dup:
            cleaned, _ = remove_duplicates(cleaned)
        if fill_miss:
            cleaned = fill_missing(cleaned)
        if cap_out:
            cleaned = cap_iqr_outliers(cleaned)

        st.session_state["cleaned_dataset"] = cleaned

        r1, r2 = st.columns(2)
        r1.metric("Original rows", f"{len(df):,}")
        r2.metric("Cleaned rows", f"{len(cleaned):,}")
        st.dataframe(cleaned.head(20), use_container_width=True)

        b1, b2 = st.columns(2)
        if b1.button("✅ Use cleaned data in chat", use_container_width=True):
            st.session_state.dataset = cleaned
            st.session_state.excel_ready = None
            st.session_state.pdf_ready = None
            st.success("The chat will now answer using the cleaned data.")
        if b2.button("↩ Back to original data", use_container_width=True):
            st.session_state.dataset = st.session_state.original_dataset
            st.session_state.excel_ready = None
            st.session_state.pdf_ready = None
            st.rerun()

        st.download_button(
            "⬇️ Download cleaned CSV",
            data=cleaned.to_csv(index=False).encode("utf-8-sig"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_cleaned_csv",
        )

        excel_sig = (remove_dup, fill_miss, cap_out, len(cleaned), st.session_state.dataset_name)
        if st.button("Prepare Excel file", key="prepare_excel"):
            with st.spinner("Building Excel file..."):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
                st.session_state.excel_ready = (excel_sig, buffer.getvalue())

        ready = st.session_state.excel_ready
        if ready and ready[0] == excel_sig:
            st.download_button(
                "⬇️ Download cleaned Excel",
                data=ready[1],
                file_name="cleaned_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_cleaned_excel",
            )

    # ---------- Compare by category ----------
    with tab_compare:
        category_options = categorical_columns(df)

        if category_options and numeric_columns:
            group_column = st.selectbox("Category", category_options, key="group_analysis_category")
            value_column = st.selectbox("Numeric column", numeric_columns, key="group_analysis_value")
            calculation = st.selectbox("Calculation", ["Sum", "Average", "Count"], key="group_analysis_calculation")

            if st.button("Run group analysis", key="run_group_analysis"):
                grouped = df.groupby(group_column, dropna=False)[value_column]
                if calculation == "Sum":
                    result = grouped.sum()
                elif calculation == "Average":
                    result = grouped.mean()
                else:
                    result = grouped.count()
                result = result.sort_values(ascending=False)

                st.write(f"**{calculation} of {value_column} by {group_column}**")
                st.dataframe(result.rename(calculation).reset_index(), use_container_width=True)
                st.bar_chart(result.head(10))
        else:
            st.info("This needs at least one category column and one numeric column.")

    # ---------- PDF report ----------
    with tab_report:
        report_chart = None
        report_df = st.session_state.get("cleaned_dataset", df)

        if numeric_columns:
            report_column = st.selectbox(
                "Numeric column for the report chart",
                numeric_columns,
                key="pdf_report_column",
            )
        else:
            report_column = None

        if st.button("Generate PDF report", key="make_pdf"):
            with st.spinner("Building PDF..."):
                try:
                    if report_column:
                        report_chart = create_histogram(df, report_column)
                    pdf_bytes = generate_pdf_report(report_df, chart=report_chart)
                    st.session_state.pdf_ready = pdf_bytes
                except Exception as pdf_error:
                    st.error(f"Could not generate the PDF report: {pdf_error}")

        if st.session_state.pdf_ready:
            st.download_button(
                "📥 Download PDF Report",
                data=st.session_state.pdf_ready,
                file_name="ai_data_analysis_report.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )


# --------------------------------------------------
# 8. MAIN AREA
# --------------------------------------------------

if view == "📊 Data tools" and dataset is not None:
    render_data_tools(dataset)
    st.stop()

# ---------- Chat view ----------

st.title(active_chat.get("title", "My AI Chat"))

if not active_chat["messages"]:
    subtitle = (
        f"Ask anything about <b>{st.session_state.dataset_name}</b>, or chat about anything else."
        if dataset is not None
        else "Ask a question, upload an image, or upload a CSV/Excel file in the sidebar."
    )
    st.markdown(
        f"""
        <div style="text-align:center; padding:35px 0 25px 0;">
            <div style="font-size:42px;">✳️</div>
            <h2>What can I help you with?</h2>
            <p style="color:#777;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataset is not None:
        numeric_cols = get_numeric_columns(dataset)
        suggestions = [
            "Give me a summary of this dataset",
            "Clean the data and let me download it as CSV",
            "Make a detailed PDF report with charts",
            "Build a Power BI style dashboard for this data",
        ]
        if numeric_cols:
            suggestions.insert(2, f"Show me a chart of {numeric_cols[0]}")

        for row_start in range(0, len(suggestions), 2):
            row_cols = st.columns(2)
            for col, text in zip(row_cols, suggestions[row_start:row_start + 2]):
                if col.button(text, use_container_width=True, key=f"sugg_{text}"):
                    st.session_state.pending_prompt = text
                    st.rerun()

for index, message in enumerate(active_chat["messages"]):
    render_message(message, index)

pending = st.session_state.pop("pending_prompt", None)
typed = st.chat_input(
    "Ask about your data…" if dataset is not None else "Ask me anything…"
)
prompt = typed or pending

if prompt:
    active_chat["messages"].append({"role": "user", "content": prompt})

    if active_chat["title"] == "New chat":
        active_chat["title"] = prompt[:40].strip() + ("..." if len(prompt) > 40 else "")

    save_chats()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if dataset is not None and uploaded_image is None:
                # The AI uses the Python tools (summary, cleaning, charts, PDF, dashboard...)
                toolbox = get_toolbox()
                with st.spinner("Working on your data…"):
                    outcome = run_agent(client, MODEL_NAME, active_chat["messages"], toolbox)

                if toolbox.df is not st.session_state.dataset:
                    st.session_state.dataset = toolbox.df  # cleaning changed the data
                    st.session_state.excel_ready = None
                    st.session_state.pdf_ready = None

                assistant_message = {"role": "assistant", "content": outcome["answer"]}
                if outcome["tools_used"]:
                    assistant_message["tools_used"] = outcome["tools_used"]
                if outcome["code"]:
                    assistant_message["code"] = outcome["code"]
                if outcome["artifacts"]:
                    st.session_state.artifacts[
                        f"{st.session_state.active_chat_id}:{len(active_chat['messages'])}"
                    ] = outcome["artifacts"]
            else:
                with st.spinner("Thinking…"):
                    answer = plain_chat_reply(active_chat["messages"], uploaded_image)
                assistant_message = {"role": "assistant", "content": answer}

            active_chat["messages"].append(assistant_message)
            save_chats()
            render_message_body(assistant_message, len(active_chat["messages"]) - 1)

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.caption(
                "If this mentions a model not found (404), set GEMINI_MODEL in your .env "
                "file to a model your key can use, e.g. gemini-3.8-flash (or gemini-3.5-flash-lite)."
            )
