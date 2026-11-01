"""Generate synthetic demand dataset and events calendar."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# allow running as `python scripts/generate_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import cmd_generate
from src.config import ensure_dirs, get_settings

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    ensure_dirs()
    raise SystemExit(cmd_generate(None))  # type: ignore[arg-type]
