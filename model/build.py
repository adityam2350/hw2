"""
Helpers for locating and building per-tile binaries.

The Makefile's `tiled` target emits ``bin/conv_t<TX>_<TY>_c<C>`` per tile
config; this module is the single place that knows that path convention so
the single-point harness (run.py)
agree on it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def tile_binary_path(tile_x: int, tile_y: int, chunk_ni: int = 64) -> Path:
    """Path the Makefile writes to for `make tiled TILE_X=... TILE_Y=... CHUNK_NI=...`."""
    return REPO_ROOT / "bin" / f"conv_t{tile_x}_{tile_y}_c{chunk_ni}"


def build_tile(tile_x: int, tile_y: int, chunk_ni: int = 64) -> Path:
    """Invoke `make tiled` for the given tile config; return the binary path.

    No-op if the binary already exists.
    """
    binary = tile_binary_path(tile_x, tile_y, chunk_ni)
    if binary.exists():
        return binary
    print(f"[build] {binary.name} "
          f"(TILE_X={tile_x} TILE_Y={tile_y} CHUNK_NI={chunk_ni})",
          file=sys.stderr, flush=True)
    subprocess.run(
        ["make", "tiled",
         f"TILE_X={tile_x}", f"TILE_Y={tile_y}", f"CHUNK_NI={chunk_ni}"],
        cwd=REPO_ROOT, check=True,
        stdout=sys.stderr, stderr=sys.stderr,
    )
    return binary


def default_binary() -> Path:
    """The unflagged `make` target, suitable for cheap CLI calls like --print-defaults."""
    return REPO_ROOT / "bin" / "conv"
