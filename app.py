
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="My AI Chat",
    page_icon="🤖",
    layout="centered"
)


st.markdown("""
<style>
/* Main app background and spacing */
.stApp {
    background: #ffffff;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

/* Main heading */
h1 {
    color: #202123;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: #f7f7f8;
    border: 1px solid #eeeeef;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 12px;
}

/* Rounded chat input */
[data-testid="stChatInput"] {
    border: 1px solid #dedee3;
    border-radius: 28px;
    background: #ffffff;
    box-shadow: 0 3px 15px rgba(0, 0, 0, 0.05);
    padding: 5px 10px;
}

/* Input focus effect */
[data-testid="stChatInput"]:focus-within {
    border-color: #a5a5b5;
    box-shadow: 0 0 0 2px rgba(100, 100, 120, 0.08);
}

/* Buttons */
.stButton button {
    border-radius: 24px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    background: #fafafa;
    padding: 10px;
}

/* Remove excess decoration */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)



st.markdown(
    """
    <div style="text-align:center; padding: 35px 0 25px 0;">
        <div style="font-size:42px;">✳️</div>
        <h1 style="font-size:32px; margin-bottom:8px;">
            What can I help you with?
        </h1>
        <p style="color:#777; font-size:16px;">
            Ask questions, explore ideas, or upload an image.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.expander("＋ Attach an image", expanded=False):
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_upload"
    )

if uploaded_image is not None:
    st.image(uploaded_image, caption="Image attached", width=300)


if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                
                if uploaded_image is not None:
                    image_part = types.Part.from_bytes(
                        data=uploaded_image.getvalue(),
                        mime_type=uploaded_image.type
                    )
                    contents = [prompt, image_part]
                else:
                    contents = prompt

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents
                )

                answer = response.text or "I couldn't generate a response."
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
