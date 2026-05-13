#!/usr/bin/env python3
"""Compatibility shim — forwards to abp_voice.cli.chat."""

from abp_voice.cli.chat import main

if __name__ == "__main__":
    raise SystemExit(main())
