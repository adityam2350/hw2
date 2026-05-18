"""
Read the C++ harness's defaults via ``./bin/conv --print-defaults``.

``main.cpp`` is the single source of truth
for default ``ny``, ``nx``, ``ky``, ``kx``, ``ni``, ``nn``,
``chunk_ni`` (fixed at 64), and default
tile sizes. Python consumers invoke the binary rather than redeclaring these
so the two sides cannot drift.
"""

from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path
from typing import Dict

from build import REPO_ROOT, default_binary


REQUIRED_KEYS = (
    "ny", "nx", "ky", "kx",
    "ni", "nn", "chunk_ni",
    "tile_x", "tile_y",
)


@functools.lru_cache(maxsize=8)
def read_defaults(binary: Path | None = None) -> Dict[str, int]:
    """Invoke ``binary --print-defaults`` and return the parsed JSON.

    ``ny`` / ``nx`` defaults come from ``./bin/conv --print-defaults`` and can
    be overridden at runtime via the binary CLI.

    Falls back to ``bin/conv`` if no binary is given. Building that binary
    is the caller's responsibility (Makefile's default target).
    """
    bin_path = Path(binary) if binary is not None else default_binary()
    if not bin_path.exists():
        raise FileNotFoundError(
            f"binary {bin_path} not found. Run `make` to build it before "
            f"reading defaults."
        )
    proc = subprocess.run(
        [str(bin_path), "--print-defaults"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise RuntimeError(
            f"--print-defaults JSON missing required keys: {missing}. "
            f"Output was: {proc.stdout!r}"
        )
    return {k: int(data[k]) for k in REQUIRED_KEYS}
