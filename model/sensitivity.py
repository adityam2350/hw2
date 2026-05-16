#!/usr/bin/env python3
"""
Sensitivity of ``predict_v1`` / roofline to one hardware knob (no GPU).

Example::

  python3 model/sensitivity.py --param dram_bandwidth_GBps \\
    --values 400 500 652.8 800 --ni 64 --nn 64 --tile-x 8 --tile-y 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from defaults import read_defaults
from perf_model import hardware_titan_v_v1, predict_v1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        required=True,
        choices=tuple(sorted(hardware_titan_v_v1.keys())),
        help="hardware dict key to sweep",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument("--ni", type=int, default=None)
    parser.add_argument("--nn", type=int, default=None)
    parser.add_argument("--ky", type=int, default=None)
    parser.add_argument("--kx", type=int, default=None)
    parser.add_argument("--tile-x", type=int, default=None)
    parser.add_argument("--tile-y", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: stdout)",
    )
    args = parser.parse_args()

    d0 = read_defaults()
    ny, nx = d0["ny"], d0["nx"]
    ni = args.ni if args.ni is not None else d0["ni"]
    nn = args.nn if args.nn is not None else d0["nn"]
    ky = args.ky if args.ky is not None else d0["ky"]
    kx = args.kx if args.kx is not None else d0["kx"]
    tx = args.tile_x if args.tile_x is not None else d0["tile_x"]
    ty = args.tile_y if args.tile_y is not None else d0["tile_y"]

    problem = {
        "Ny": ny,
        "Nx": nx,
        "Ky": ky,
        "Kx": kx,
        "Ni": ni,
        "Nn": nn,
        "NXPAD": nx + kx - 1,
        "NXSCL": nx,
    }
    tiling = {
        "TILE_Y": ty,
        "TILE_X": tx,
        "CHUNK_NI": d0["chunk_ni"],
        "MAX_N_PER_THREAD": 2,
        "threads_per_block": min(nn, 256),
    }

    stream = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        w = csv.DictWriter(
            stream,
            fieldnames=[
                args.param,
                "valid",
                "predicted_time_seconds",
                "roofline_time_seconds",
                "predicted_tflops",
                "roofline_tflops",
                "bottleneck",
                "roofline_bottleneck",
            ],
        )
        w.writeheader()
        for v in args.values:
            hw = copy(hardware_titan_v_v1)
            hw[args.param] = float(v)
            r = predict_v1(problem, tiling, hw)
            w.writerow(
                {
                    args.param: v,
                    "valid": r.valid,
                    "predicted_time_seconds": r.predicted_time_seconds,
                    "roofline_time_seconds": r.roofline_time_seconds,
                    "predicted_tflops": r.predicted_tflops,
                    "roofline_tflops": r.roofline_tflops,
                    "bottleneck": r.bottleneck,
                    "roofline_bottleneck": r.roofline_bottleneck,
                }
            )
    finally:
        if args.out:
            stream.close()


if __name__ == "__main__":
    main()
