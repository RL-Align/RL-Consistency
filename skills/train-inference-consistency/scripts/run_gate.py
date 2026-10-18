#!/usr/bin/env python3
"""Use after installing RL-Consistency; accepts the same flags as `gate`."""
import sys
from rl_consistency.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["gate", *sys.argv[1:]]))
