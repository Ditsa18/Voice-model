"""Microphone recording with adaptive silence detection."""

from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_BLOCK_DURATION = 0.1
_BAR_WIDTH = 30
_BAR_PEAK = 0.2


def _bar(level: float, width: int = _BAR_WIDTH) -> str:
    n = int(min(1.0, level / _BAR_PEAK) * width)
    return "█" * n + "·" * (width - n)


@dataclass(slots=True)
class RecordingResult:
    audio: np.ndarray
    peak_rms: float
    voiced: bool

    @property
    def is_empty(self) -> bool:
        return self.audio.size == 0


class MicRecorder:
    """Capture audio from the default (or selected) input device."""

    def __init__(self, sample_rate: int | None = None) -> None:
        s = get_settings()
        self.sample_rate = sample_rate or s.sample_rate
        self.silence_duration = s.silence_duration
        self.calib_seconds = s.noise_calibration_seconds
        self.noise_mult = s.noise_multiplier
        self.min_threshold = s.min_speech_threshold

    # ── public API ────────────────────────────────────────────────────────
    def record_until_silence(
        self, max_seconds: int = 30, show_meter: bool = True
    ) -> RecordingResult:
        threshold = self._calibrate_noise()
        return self._stream_until_silence(threshold, max_seconds, show_meter)

    def record_fixed(self, seconds: float) -> RecordingResult:
        log.info("recording for %.1fs — speak now", seconds)
        rec = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = rec.flatten().astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        log.info("recording done (peak abs amplitude=%.4f)", peak)
        return RecordingResult(audio=audio, peak_rms=peak, voiced=True)

    # ── internals ─────────────────────────────────────────────────────────
    def _calibrate_noise(self) -> float:
        log.info(
            "calibrating ambient noise for %.1fs — stay quiet", self.calib_seconds
        )
        rec = sd.rec(
            int(self.calib_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        rms = float(np.sqrt(np.mean(rec.astype(np.float32) ** 2)))
        threshold = max(rms * self.noise_mult, self.min_threshold)
        log.info("ambient rms=%.4f -> speech threshold=%.4f", rms, threshold)
        return threshold

    def _stream_until_silence(
        self, threshold: float, max_seconds: int, show_meter: bool
    ) -> RecordingResult:
        q: queue.Queue[np.ndarray] = queue.Queue()

        def cb(indata, frames, t, status):  # noqa: ARG001 — sounddevice signature
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        chunks: list[np.ndarray] = []
        silent_for = 0.0
        voiced = False
        peak_seen = 0.0
        start = time.time()
        blocksize = int(self.sample_rate * _BLOCK_DURATION)

        print("[mic] listening... (speak now; pause ~1s to stop)")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=cb,
        ):
            while True:
                try:
                    block = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                chunks.append(block)
                rms = float(np.sqrt(np.mean(block**2)))
                peak_seen = max(peak_seen, rms)

                if show_meter:
                    state = "SPEAK" if rms > threshold else "....."
                    sys.stdout.write(
                        f"\r[{state}] {_bar(rms)}  rms={rms:.4f} thr={threshold:.4f}  "
                    )
                    sys.stdout.flush()

                if rms > threshold:
                    voiced = True
                    silent_for = 0.0
                else:
                    silent_for += _BLOCK_DURATION

                if voiced and silent_for >= self.silence_duration:
                    break
                if time.time() - start > max_seconds:
                    break

        if show_meter:
            sys.stdout.write("\n")

        if not voiced:
            log.warning(
                "no speech detected (peak rms=%.4f, threshold=%.4f). "
                "Try louder/closer mic, run mic_check, or use --seconds N.",
                peak_seen,
                threshold,
            )
            return RecordingResult(np.zeros(0, dtype=np.float32), peak_seen, False)

        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        return RecordingResult(audio, peak_seen, True)
