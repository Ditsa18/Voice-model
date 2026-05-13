#!/usr/bin/env python3
"""Compatibility shim — forwards to abp_voice.cli.test_call."""

from abp_voice.cli.test_call import main

if __name__ == "__main__":
    raise SystemExit(main())
