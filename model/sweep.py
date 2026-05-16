#!/usr/bin/env python3
"""
Sweep convolution configs: NCU + V1 + roofline per point, append Extended CSV.

Typical usage (from repo root, GPU + ncu on PATH, ``make`` done):

  python3 model/sweep.py --out sweep.csv \\
    --square-channels 32 64 128 \\
    --tile-xs 8 16 --tile-ys 8 16

Channels default to a single (ni, nn) from ``bin/conv --print-defaults`` if
omitted; tiles default similarly if omitted.

Extra columns vs ``run.py`` single row:
  mape_time_v1_pct, mape_time_roofline_pct, rel_err_time_v1, rel_err_time_roofline
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import fields
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_DIR))

import run as conv_run  # noqa: E402


def _rel_err(pred: float, meas: float) -> float:
    if meas <= 0 or not math.isfinite(meas) or not math.isfinite(pred):
        return float("nan")
    return pred / meas - 1.0


def _mape_pct(pred: float, meas: float) -> float:
    if meas <= 0 or not math.isfinite(meas) or not math.isfinite(pred):
        return float("nan")
    return abs(pred - meas) / meas * 100.0


def _channel_configs(
    square: list[int] | None,
    ni_list: list[int] | None,
    nn_list: list[int] | None,
    defaults: dict[str, int],
) -> list[tuple[int, int]]:
    if square and (ni_list or nn_list):
        raise ValueError("use either --square-channels or --ni-list/--nn-list, not both")
    if square:
        return [(v, v) for v in square]
    if ni_list and nn_list:
        return list(itertools.product(ni_list, nn_list))
    if ni_list or nn_list:
        raise ValueError("--ni-list and --nn-list must both be set for a product sweep")
    return [(defaults["ni"], defaults["nn"])]


def _tile_configs(
    tx: list[int] | None,
    ty: list[int] | None,
    defaults: dict[str, int],
) -> list[tuple[int, int]]:
    if tx and not ty:
        raise ValueError("--tile-ys required when --tile-xs is set")
    if ty and not tx:
        raise ValueError("--tile-xs required when --tile-ys is set")
    if tx and ty:
        return list(itertools.product(tx, ty))
    return [(defaults["tile_x"], defaults["tile_y"])]


def _extended_row(r: conv_run.RunResult) -> dict[str, object]:
    t_m = r.t_measured_s
    base = {f.name: getattr(r, f.name) for f in fields(r)}
    pred = r.predicted_time_seconds
    roof = r.roofline_time_seconds
    base["mape_time_v1_pct"] = _mape_pct(pred, t_m)
    base["mape_time_roofline_pct"] = _mape_pct(roof, t_m)
    base["rel_err_time_v1"] = _rel_err(pred, t_m)
    base["rel_err_time_roofline"] = _rel_err(roof, t_m)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write CSV here (default: stdout)",
    )
    parser.add_argument(
        "--square-channels",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="sweep (ni,nn)=(N,N) for each N",
    )
    parser.add_argument(
        "--ni-list",
        type=int,
        nargs="*",
        default=None,
        metavar="NI",
        help="Cartesian product with --nn-list",
    )
    parser.add_argument(
        "--nn-list",
        type=int,
        nargs="*",
        default=None,
        metavar="NN",
    )
    parser.add_argument(
        "--tile-xs",
        type=int,
        nargs="*",
        default=None,
        metavar="TX",
    )
    parser.add_argument(
        "--tile-ys",
        type=int,
        nargs="*",
        default=None,
        metavar="TY",
    )
    parser.add_argument("--ky", type=int, default=None)
    parser.add_argument("--kx", type=int, default=None)
    args = parser.parse_args()

    if not conv_run.default_binary().exists():
        raise FileNotFoundError(
            f"{conv_run.default_binary()} not found. Run `make` from repo root."
        )

    defaults = conv_run.read_defaults()
    ky = args.ky if args.ky is not None else defaults["ky"]
    kx = args.kx if args.kx is not None else defaults["kx"]

    ch_cfgs = _channel_configs(
        args.square_channels, args.ni_list, args.nn_list, defaults
    )
    til_cfgs = _tile_configs(args.tile_xs, args.tile_ys, defaults)

    out_stream = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer: csv.DictWriter | None = None
        n_run = 0
        any_fallback = False
        for (ni, nn), (tile_x, tile_y) in itertools.product(ch_cfgs, til_cfgs):
            n_run += 1
            print(
                f"[sweep] run {n_run}: ni={ni} nn={nn} tile={tile_x}x{tile_y}",
                file=sys.stderr,
                flush=True,
            )
            result, flags = conv_run.measure_and_model(
                ni, nn, ky, kx, tile_x, tile_y
            )
            if flags:
                any_fallback = True
                print(
                    "[sweep] measurement_fallbacks: " + "; ".join(flags),
                    file=sys.stderr,
                    flush=True,
                )

            row = _extended_row(result)
            if writer is None:
                writer = csv.DictWriter(out_stream, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)

        print(f"[sweep] completed {n_run} runs", file=sys.stderr, flush=True)
        if any_fallback:
            print(
                "[sweep] warning: some runs used NCU fallbacks; "
                "check stderr above per run.",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if args.out:
            out_stream.close()


if __name__ == "__main__":
    main()
