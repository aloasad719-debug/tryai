import asyncio
import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-3.8-flash"  # understands text, images and PDFs

client = genai.Client(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = (
    "You are a helpful multimodal AI assistant. You can understand images and "
    "documents (PDF). Always reply in the same language the user writes in "
    "(Bangla, English, etc.). Be clear and easy to understand."
)

ALLOWED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only: accept any port (5173, 5174, ...)
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "ok", "api_key_loaded": bool(API_KEY)}


@app.post("/chat")
async def chat(
    message: str = Form(""),
    history: str = Form("[]"),
    files: List[UploadFile] = File(default=[]),
):
    if client is None:
        return JSONResponse(
            {"error": "API key not found. Add GEMINI_API_KEY to the .env file."},
            status_code=500,
        )

    try:
        # 1) Earlier messages (so the AI remembers the conversation)
        contents = []
        for item in json.loads(history or "[]"):
            role = "model" if item.get("role") == "assistant" else "user"
            text = item.get("text", "")
            if text:
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=text)])
                )

        # 2) The new message: text + any images/PDFs
        parts = []
        for upload in files:
            data = await upload.read()
            mime = upload.content_type or "application/octet-stream"
            if mime not in ALLOWED_TYPES:
                return JSONResponse(
                    {"error": f"File type not supported: {mime}"}, status_code=400
                )
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))

        parts.append(
            types.Part.from_text(
                text=message or "Please describe and explain the attached file(s)."
            )
        )
        contents.append(types.Content(role="user", parts=parts))

        # 3) Ask Gemini
        # If Google is busy (503) or limit hit (429), wait and try again
        last_error = None
        for attempt in range(4):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT
                    ),
                )
                return {"reply": response.text or "(No answer returned.)"}
            except Exception as err:
                last_error = err
                busy = any(code in str(err) for code in ("503", "429", "UNAVAILABLE"))
                if not busy or attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))  # wait 2s, 4s, 6s
        raise last_error

    except Exception as error:  # show the real problem in the chat
        return JSONResponse({"error": str(error)}, status_code=500)