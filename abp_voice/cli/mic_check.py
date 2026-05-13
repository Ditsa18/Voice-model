"""Microphone diagnostics.

  python -m abp_voice mic                  # list all input devices
  python -m abp_voice mic --device 4       # live VU on device #4
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

_VU_WIDTH = 40
_VU_PEAK = 0.2


def _bar(level: float) -> str:
    n = int(min(1.0, level / _VU_PEAK) * _VU_WIDTH)
    return "█" * n + "·" * (_VU_WIDTH - n)


def _list_devices() -> None:
    default_in = sd.default.device[0]
    print("\nAvailable audio devices (input channels > 0 = usable as mic):\n")
    for i, d in enumerate(sd.query_devices()):
        ch_in = d.get("max_input_channels", 0)
        if ch_in <= 0:
            continue
        marker = "  <-- DEFAULT INPUT" if i == default_in else ""
        print(
            f"  #{i:2d}  {d['name']:50s}  in={ch_in}  "
            f"sr={int(d['default_samplerate'])}{marker}"
        )
    print(
        "\nUse:  python -m abp_voice mic --device <N>   to test that mic.\n"
        "Then: python -m abp_voice chat --device <N>    to use it in the agent.\n"
    )


def _vu(device: int, seconds: int) -> None:
    sr = 16000
    print(
        f"\n[live VU on device #{device}] speak normally — Ctrl+C to stop.\n"
    )
    peak_seen = 0.0
    start = time.time()

    def cb(indata, frames, t, status):  # noqa: ARG001 — sounddevice signature
        nonlocal peak_seen
        if status:
            print(status, file=sys.stderr)
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        peak_seen = max(peak_seen, rms)
        sys.stdout.write(f"\rrms={rms:.4f}  peak={peak_seen:.4f}  {_bar(rms)}")
        sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=int(sr * 0.1), callback=cb, device=device,
        ):
            while time.time() - start < seconds:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n")
        if peak_seen < 0.01:
            print(
                f"WARNING: peak RMS only {peak_seen:.4f} — this mic is very quiet "
                "or muted. Pick another device or raise input gain (pavucontrol)."
            )
        elif peak_seen < 0.03:
            print(
                f"OK-ish: peak RMS {peak_seen:.4f}. Try closer mic position, or "
                "use `--seconds 5` to bypass silence detection."
            )
        else:
            print(f"GOOD: peak RMS {peak_seen:.4f}. Mic should work fine.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Microphone diagnostics.")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--seconds", type=int, default=30, help="VU meter duration")
    args = ap.parse_args()

    if args.device is None:
        _list_devices()
    else:
        _vu(args.device, args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
