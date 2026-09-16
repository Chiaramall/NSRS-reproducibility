#!/usr/bin/env python3
"""Compatibility launcher for baselines.models.heteroembed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baselines.models.heteroembed import *  # noqa: F401,F403,E402
if __name__ == "__main__":
    main()
