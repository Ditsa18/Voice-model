#!/usr/bin/env python3
"""Compatibility shim — forwards to abp_voice.cli.ingest."""

from abp_voice.cli.ingest import main

if __name__ == "__main__":
    raise SystemExit(main())
