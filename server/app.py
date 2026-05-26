# server/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import socketio
import uvicorn
import time

from abp_voice.rag.pipeline import RAGPipeline
from abp_voice.rag.prompts import scrub

# Socket.IO
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

# FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG pipeline
pipeline = RAGPipeline()

print("Warming up pipeline...")
pipeline.warmup()
print("Pipeline ready.")

# ASGI wrapper
socket_app = socketio.ASGIApp(
    sio,
    other_asgi_app=app
)

# Health route
@app.get("/")
async def root():
    return {
        "status": "running"
    }

# Client connect
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

# Client disconnect
@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Detect language from Unicode
def detect_lang(text: str) -> str:

    lower = text.lower()

    # Explicit language requests
    if "in bengali" in lower or "বাংলায়" in lower:
        return "bn"

    if "in hindi" in lower or "हिंदी" in lower:
        return "hi"

    # Bengali Unicode
    if any("\u0980" <= c <= "\u09FF" for c in text):
        return "bn"

    # Hindi Unicode
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "hi"

    return "en"

# Chat event
@sio.event
async def message(sid, data):
    try:
        total_start = time.time()

        user_message = data["message"]

        lang = detect_lang(user_message)

        print(f"\nUser: {user_message}")
        print(f"Detected language: {lang}")

        # Retrieval + stream setup
        retrieval_start = time.time()

        stream, hits = pipeline.stream_generate(
            question=user_message,
            lang=lang
        )

        print("\n========== RETRIEVED CONTEXT ==========")

        for i, h in enumerate(hits, 1):
            print(f"\n[{i}]")
            print(h.text[:500])

        print("\n=======================================\n")

        retrieval_time = time.time() - retrieval_start

        # Streaming
        llm_start = time.time()

        full_response = ""

        for token in stream:
            full_response += token

            await sio.emit(
                "token",
                {
                    "token": token
                },
                to=sid
            )

        llm_time = time.time() - llm_start

        # Cleanup
        clean_response = scrub(full_response)

        print(f"Assistant: {clean_response}")

        # Sources
        await sio.emit(
            "sources",
            {
                "sources": [h.source for h in hits]
            },
            to=sid
        )

        # Done
        await sio.emit(
            "done",
            {
                "status": "complete"
            },
            to=sid
        )

        total_time = time.time() - total_start

        # Logs
        print("\n========== PERFORMANCE ==========")
        print(f"Retrieval : {retrieval_time:.2f}s")
        print(f"LLM       : {llm_time:.2f}s")
        print(f"Total     : {total_time:.2f}s")
        print("=================================\n")

    except Exception as e:
        print("Error:", str(e))

        await sio.emit(
            "error",
            {
                "message": str(e)
            },
            to=sid
        )

# Run server
if __name__ == "__main__":
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000
    )