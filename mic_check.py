#!/usr/bin/env python3
"""Compatibility shim — forwards to abp_voice.cli.mic_check."""

from abp_voice.cli.mic_check import main

if __name__ == "__main__":
    raise SystemExit(main())
