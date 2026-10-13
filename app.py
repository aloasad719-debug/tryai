
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="My AI Chat",
    page_icon="🤖",
    layout="wide",
)

HISTORY_FILE = Path(__file__).parent / "chat_history.json"


# --------------------------------------------------
# 2. LOAD AND SAVE CHAT HISTORY
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


# --------------------------------------------------
# 3. INITIALIZE SESSION
# --------------------------------------------------

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "active_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.active_chat_id = next(
            iter(st.session_state.chats)
        )
    else:
        create_chat()

if not st.session_state.chats:
    create_chat()

if st.session_state.active_chat_id not in st.session_state.chats:
    st.session_state.active_chat_id = next(iter(st.session_state.chats))

active_chat = st.session_state.chats[
    st.session_state.active_chat_id
]


# --------------------------------------------------
# 4. SIDEBAR: HISTORY AND DOWNLOADS
# --------------------------------------------------

with st.sidebar:
    st.title("🤖 My AI Chat")
    st.caption("Your conversations")

    if st.button("＋ New chat", use_container_width=True):
        create_chat()
        st.rerun()

    st.divider()

    st.subheader("Chat history")

    chat_ids = list(st.session_state.chats.keys())

    def chat_label(chat_id):
        chat = st.session_state.chats[chat_id]
        return chat.get("title", "New chat")

    selected_chat_id = st.selectbox(
        "Choose a conversation",
        options=chat_ids,
        index=chat_ids.index(st.session_state.active_chat_id),
        format_func=chat_label,
        label_visibility="collapsed",
    )

    if selected_chat_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_chat_id
        st.rerun()

    active_chat = st.session_state.chats[
        st.session_state.active_chat_id
    ]

    st.divider()
    st.subheader("Download history")

    export_json = json.dumps(
        active_chat,
        ensure_ascii=False,
        indent=2,
    )

    export_text = "\n\n".join(
        f"{message['role'].title()}:\n{message['content']}"
        for message in active_chat["messages"]
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

    st.divider()

    st.caption(
        "Your history is saved locally in chat_history.json."
    )


# --------------------------------------------------
# 5. CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #ffffff;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

h1 {
    color: #202123;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

[data-testid="stChatMessage"] {
    background: #f7f7f8;
    border: 1px solid #eeeeef;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border: 1px solid #dedee3;
    border-radius: 28px;
    background: #ffffff;
}

.stButton button, [data-testid="stDownloadButton"] button {
    border-radius: 18px;
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
    background: #fafafa;
    padding: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. CONNECT TO GEMINI
# --------------------------------------------------

if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 7. DISPLAY THE ACTIVE CONVERSATION
# --------------------------------------------------

st.title(active_chat.get("title", "My AI Chat"))

if not active_chat["messages"]:
    st.markdown(
        """
        <div style="text-align:center; padding:35px 0 25px 0;">
            <div style="font-size:42px;">✳️</div>
            <h2>What can I help you with?</h2>
            <p style="color:#777;">
                Ask a question or upload an image.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 8. IMAGE UPLOAD
# --------------------------------------------------

with st.expander("＋ Attach an image", expanded=False):
    uploaded_image = st.file_uploader(
        "Choose a PNG, JPG, JPEG, or WebP image",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"image_{st.session_state.active_chat_id}",
    )

if uploaded_image is not None:
    st.image(uploaded_image, caption="Image attached", width=300)


# --------------------------------------------------
# 9. SEND MESSAGE AND SAVE THE RESPONSE
# --------------------------------------------------

prompt = st.chat_input("Ask me anything...")

if prompt:
    active_chat = st.session_state.chats[
        st.session_state.active_chat_id
    ]

    user_message = {
        "role": "user",
        "content": prompt,
    }

    active_chat["messages"].append(user_message)

    if active_chat["title"] == "New chat":
        active_chat["title"] = prompt[:40].strip()
        if len(prompt) > 40:
            active_chat["title"] += "..."

    save_chats()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if uploaded_image is not None:
                    image_part = types.Part.from_bytes(
                        data=uploaded_image.getvalue(),
                        mime_type=uploaded_image.type,
                    )
                    contents = [prompt, image_part]
                else:
                    contents = prompt

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                )

                answer = response.text or (
                    "I couldn't generate a response."
                )

                st.markdown(answer)

                active_chat["messages"].append({
                    "role": "assistant",
                    "content": answer,
                })

                save_chats()

            except Exception as error:
                st.error(f"Something went wrong: {error}")

                error_message = (
                    f"Request failed: {error}"
                )

                active_chat["messages"].append({
                    "role": "assistant",
                    "content": error_message,
                })

                save_chats()
