"""Audio file playback. Prefers soundfile's native MP3 decode; falls back to ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import sounddevice as sd

from ..logging_setup import get_logger

log = get_logger(__name__)


def _mp3_to_wav(mp3_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install with: sudo apt install -y ffmpeg"
        )
    wav_path = mp3_path.with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-i", str(mp3_path),
            "-ac", "1",
            "-ar", "24000",
            str(wav_path),
        ],
        check=True,
    )
    return wav_path


def play_audio_file(path: Path) -> None:
    """Play an MP3 or WAV file via the default output device (blocking)."""
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        log.debug("soundfile direct read failed (%s); decoding via ffmpeg", e)
        wav_path = _mp3_to_wav(path)
        try:
            data, sr = sf.read(str(wav_path), dtype="float32")
        finally:
            wav_path.unlink(missing_ok=True)
    sd.play(data, sr)
    sd.wait()
