#!/usr/bin/env python3
"""Render a self-contained report without granting any verification gate."""
import sys
from rl_consistency.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
