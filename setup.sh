#!/usr/bin/env bash
# Bootstrap the abp-voice environment.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing system audio deps (portaudio, ffmpeg, libsndfile)"
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y python3-venv python3-pip portaudio19-dev ffmpeg libsndfile1
fi

echo "==> Creating virtualenv (.venv)"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Use the venv's binaries directly so this works under any shell (bash/zsh/nu).
PIP=".venv/bin/pip"
PY=".venv/bin/python"

echo "==> Upgrading pip / installing project (editable mode)"
"$PIP" install --upgrade pip wheel
"$PIP" install -e .

echo "==> Checking Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active --quiet ollama; then
  echo "Starting ollama service..."
  sudo systemctl start ollama || true
fi

echo "==> Pulling Gemma 3 (1B = fast on CPU, 4B = better, needs GPU for speed)"
ollama pull gemma3:1b
ollama pull gemma3:4b

cat <<EOF

Setup complete.

Next steps:
  1) Drop documents (pdf/docx/txt/md) into ./data
  2) python -m abp_voice ingest                    (or:  ./ingest.py)
  3) python -m abp_voice chat --seconds 6          (or:  ./voice_agent.py --seconds 6)
     python -m abp_voice chat --mode text          (text in / text out)
     python -m abp_voice test --say "..." --lang bn
     python -m abp_voice mic                       (list audio devices)

After installing in editable mode you also have console scripts:
  abp-chat / abp-ingest / abp-test / abp-mic
EOF
