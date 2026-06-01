# server/app.py

import time

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from abp_voice.languages import detect_primary_language
from abp_voice.logging_setup import get_logger
from abp_voice.rag.pipeline import RAGPipeline
from abp_voice.rag.prompts import scrub

log = get_logger(__name__)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

log.info("warming up pipeline...")
pipeline.warmup()
log.info("pipeline ready")

socket_app = socketio.ASGIApp(sio, other_asgi_app=app)


@app.get("/")
async def root():
    return {"status": "running"}


@sio.event
async def connect(sid, environ):
    log.info("client connected: %s", sid)


@sio.event
async def disconnect(sid):
    log.info("client disconnected: %s", sid)


@sio.event
async def message(sid, data):
    try:
        total_start = time.time()

        user_message = data["message"]
        lang = detect_primary_language(user_message)

        log.info("user (%s): %s", lang, user_message)

        retrieval_start = time.time()
        stream, hits = pipeline.stream_generate(question=user_message, lang=lang)
        retrieval_time = time.time() - retrieval_start

        for i, h in enumerate(hits, 1):
            log.debug("hit [%d]: %s", i, h.text[:200])

        llm_start = time.time()
        full_response = ""

        for token in stream:
            full_response += token
            await sio.emit("token", {"token": token}, to=sid)

        llm_time = time.time() - llm_start
        clean_response = scrub(full_response)

        log.info("assistant: %s", clean_response)

        await sio.emit("sources", {"sources": [h.source for h in hits]}, to=sid)
        await sio.emit("done", {"status": "complete"}, to=sid)

        log.info(
            "retrieval=%.2fs  llm=%.2fs  total=%.2fs",
            retrieval_time, llm_time, time.time() - total_start,
        )

    except Exception as e:
        log.exception("error handling message from %s", sid)
        await sio.emit("error", {"message": str(e)}, to=sid)


if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
