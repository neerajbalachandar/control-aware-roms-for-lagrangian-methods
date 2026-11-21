"""Small helpers shared by command-line scripts (logging, YAML configs, overrides)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(verbose: bool = False, logfile: Optional[Path] = None) -> None:
    handlers = [logging.StreamHandler()]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    for noisy in ("matplotlib", "PIL", "h5py"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def parse_ranks(spec: Optional[str]) -> Optional[dict]:
    """'fluid=12,structure=4,structure_rate=4' -> dict."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out
