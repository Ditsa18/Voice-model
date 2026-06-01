"""Ollama LLM client wrapper."""

from __future__ import annotations

from collections.abc import Iterator

import ollama

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_STOP_TOKENS = ["Question:", "Q (", "Context:", "\nSystem:"]


class OllamaLLM:
    """Thin wrapper around the Ollama chat API."""

    def __init__(self) -> None:
        s = get_settings()

        self._model = s.llm_model

        self._client = ollama.Client(
            host=s.ollama_host
        )

        self._options = {
            "temperature": s.llm_temperature,
            "num_ctx": s.llm_num_ctx,
            "num_predict": s.llm_num_predict,
            "stop": _STOP_TOKENS,
        }

        self._keep_alive = s.llm_keep_alive

    @property
    def model(self) -> str:
        return self._model

    def warmup(self) -> None:
        try:
            self._client.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": "ok"
                    }
                ],
                options={
                    "num_predict": 1,
                    "num_ctx": self._options["num_ctx"],
                },
                keep_alive=self._keep_alive,
                stream=False,
            )

        except Exception as e:
            log.warning("LLM warmup skipped: %s", e)

    # Normal full response
    def chat(
        self,
        messages: list[dict[str, str]]
    ) -> str:

        resp = self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
            keep_alive=self._keep_alive,
            stream=False,
        )

        return (
            resp.get("message", {})
            .get("content") or ""
        ).strip()

    # Real streaming
    def stream_chat(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:

        stream = self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
            keep_alive=self._keep_alive,
            stream=True,
        )

        for chunk in stream:
            token = (
                chunk.get("message", {})
                .get("content", "")
            )

            if token:
                yield token