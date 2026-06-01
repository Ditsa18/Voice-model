"""Central configuration. Settings are loaded from environment variables with sensible defaults.

Override any setting via a .env file at the project root or by exporting the env var, e.g.:

    ABP_LLM_MODEL=gemma3:4b
    ABP_WHISPER_DEVICE=cuda
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ABP_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── paths ─────────────────────────────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    audio_cache: Path = PROJECT_ROOT / "audio_cache"

    # ── LLM (Ollama) ──────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    llm_num_ctx: int = 1024
    llm_num_predict: int = 80
    llm_temperature: float = 0.2
    llm_keep_alive: str = "30m"

    # ── Embeddings + Vector store ─────────────────────────────────────────
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    collection_name: str = "abp_multilingual"
    chunk_size: int = 300
    chunk_overlap: int = 40
    top_k: int = 2

    # ── STT (faster-whisper) ──────────────────────────────────────────────
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    whisper_use_vad: bool = True
    whisper_no_speech_threshold: float = 0.65
    whisper_vad_threshold: float = 0.6
    whisper_vad_min_speech_ms: int = 250
    whisper_vad_min_silence_ms: int = 400

    # ── Audio I/O ─────────────────────────────────────────────────────────
    sample_rate: int = 16000
    silence_duration: float = 1.2
    noise_calibration_seconds: float = 1.5
    noise_multiplier: float = 4.5
    min_speech_threshold: float = 0.005

    # ── Misc ──────────────────────────────────────────────────────────────
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.chroma_dir, self.audio_cache):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
