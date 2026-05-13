# abp-voice — Multilingual Voice-RAG (English / Hindi / Bengali)

A local, fully-offline-capable voice agent that:

1. Listens in **English / Hindi / Bengali** (auto-detected by Whisper).
2. Retrieves relevant chunks from your documents using multilingual embeddings + ChromaDB.
3. Generates an answer with **Gemma 3** via **Ollama**, locked to the speaker's language.
4. Speaks the answer back in the same language.

## Stack

| Layer | Component |
|---|---|
| Speech-to-text | `faster-whisper` (medium, multilingual) |
| LLM | `gemma3:1b` / `gemma3:4b` via Ollama |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Vector store | ChromaDB (persistent) |
| Text-to-speech | `edge-tts` (Aria EN / Swara HI / Tanishaa BN) with `gTTS` fallback |
| Audio I/O | `sounddevice` + `soundfile` + ffmpeg |

## Project layout

```
abp-task/
├── voice_agent.py / test_call.py / ingest.py / mic_check.py   # CLI shims
├── pyproject.toml         # installable package + console scripts
├── requirements.txt
├── setup.sh
├── .env.example / .gitignore
├── data/                  # source documents (drop your PDFs / MDs / DOCXs here)
├── chroma_db/             # persistent vector store (gitignored)
├── audio_cache/           # temp TTS files (gitignored)
├── abp_voice/             # the package
│   ├── config.py          # pydantic-settings, env-driven
│   ├── languages.py       # codes, names, voices, script hints, exit phrases
│   ├── logging_setup.py   # rich-based logger
│   ├── agent.py           # VoiceAgent orchestrator (mic → STT → RAG → TTS)
│   ├── audio/             # MicRecorder + playback
│   ├── stt/               # Transcriber (faster-whisper wrapper)
│   ├── tts/               # Synthesizer (edge-tts + gTTS)
│   ├── rag/               # embeddings · store · chunking · loaders · prompts · llm · pipeline
│   └── cli/               # ingest · chat · test_call · mic_check
└── tests/                 # pytest unit tests (chunking · languages · prompts)
```

Each module has a single responsibility, type hints, and a docstring at the top.

## Quick start

```bash
bash setup.sh
```
This installs portaudio/ffmpeg, creates `.venv`, installs the package in editable mode, and pulls `gemma3:1b` + `gemma3:4b`.

> **Nushell users:** the script uses the venv binaries directly so it works under any shell.

### Add documents and ingest

```bash
# drop .pdf / .docx / .txt / .md files into ./data/
python -m abp_voice ingest
# or
./ingest.py
```

### Talk to the agent

```bash
# voice conversation (best with --seconds 6 for clearer Bengali/Hindi capture)
python -m abp_voice chat --seconds 6

# text-only
python -m abp_voice chat --mode text

# one-shot test, bypass mic
python -m abp_voice test --say "এবিপি কোথায় অবস্থিত?" --lang bn
```

After `pip install -e .` (done by `setup.sh`), four console scripts are also registered:

```bash
abp-chat --seconds 6
abp-ingest
abp-test --say "When was ABP founded?"
abp-mic
```

## Configuration

All settings are env-driven via `pydantic-settings`. Copy `.env.example` → `.env` and uncomment lines you want to override. Common ones:

```bash
ABP_LLM_MODEL=gemma3:4b      # bigger model — needs GPU for fast replies
ABP_WHISPER_DEVICE=cuda      # once nvidia driver is loaded
ABP_WHISPER_COMPUTE=float16
```

Or just `export ABP_LLM_MODEL=gemma3:4b` in your shell — no `.env` required.

## Performance notes

| Setup | Per-turn latency |
|---|---|
| `gemma3:4b` on CPU | ~40–70 s ❌ |
| `gemma3:1b` on CPU, warm | ~5–15 s ✅ |
| `gemma3:1b` on GPU | ~1–2 s ⚡ |
| `gemma3:4b` on GPU (best quality) | ~2–4 s ⚡ |

If `nvidia-smi` errors, the NVIDIA driver isn't loaded — reboot to fix. The code auto-falls back from CUDA → CPU if needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PortAudio library not found` | `sudo apt install -y portaudio19-dev` |
| `No module named 'audioop'` (Py 3.13+) | already handled (pydub removed in favor of ffmpeg subprocess) |
| Mic doesn't trigger | `python -m abp_voice mic` to pick a device; or use `--seconds 6` to bypass silence detection |
| Bengali/Hindi transcribed as English gibberish | force the language: `--lang bn` or `--lang hi` |
| Bengali reply has Hindi characters mixed in | switch to `gemma3:4b` — the 1B model isn't strong enough for clean Bengali |
| `cuda failed: no CUDA-capable device` | reboot to load nvidia driver, or stay on `ABP_WHISPER_DEVICE=cpu` |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
# gemma-voice-agent
