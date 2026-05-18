#!/usr/bin/env python3
"""
Model-only hardware sensitivity utilities (no GPU required).

Mode 1 (legacy single-parameter sweep)::

  python3 model/sensitivity.py --param dram_bandwidth_GBps \\
    --values 400 500 652.8 800 --ny 112 --nx 112 \\
    --ni 64 --nn 64 --tile-x 8 --tile-y 8

Mode 2 (recommended multi-parameter normalized sensitivity)::

  python3 model/sensitivity.py --out sensitivity.csv --plot-prefix fig_sens \\
    --shapes 56x56 112x112 224x224 \\
    --params peak_fp32_tflops compute_efficiency_base dram_bandwidth_GBps l2_bandwidth_GBps num_sms
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from defaults import read_defaults
from perf_model import hardware_titan_v_v1, predict_v1


def _parse_shape(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def _base_problem_tiling(
    ny: int,
    nx: int,
    ni: int,
    nn: int,
    ky: int,
    kx: int,
    tx: int,
    ty: int,
    chunk_ni: int,
) -> tuple[dict[str, int], dict[str, int]]:
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
        "CHUNK_NI": chunk_ni,
        "MAX_N_PER_THREAD": 2,
        "threads_per_block": min(nn, 256),
    }
    return problem, tiling


def _predict_seconds(problem: dict[str, int], tiling: dict[str, int], hw: dict[str, float]) -> float:
    r = predict_v1(problem, tiling, hw)
    return float(r.predicted_time_seconds)


def _emit_legacy_single_param(
    args: argparse.Namespace,
    d0: dict[str, int],
) -> None:
    ny = args.ny if args.ny is not None else d0["ny"]
    nx = args.nx if args.nx is not None else d0["nx"]
    ni = args.ni if args.ni is not None else d0["ni"]
    nn = args.nn if args.nn is not None else d0["nn"]
    ky = args.ky if args.ky is not None else d0["ky"]
    kx = args.kx if args.kx is not None else d0["kx"]
    tx = args.tile_x if args.tile_x is not None else d0["tile_x"]
    ty = args.tile_y if args.tile_y is not None else d0["tile_y"]

    problem, tiling = _base_problem_tiling(ny, nx, ni, nn, ky, kx, tx, ty, d0["chunk_ni"])

    stream = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        w = csv.DictWriter(
            stream,
            fieldnames=[
                args.param,
                "ny",
                "nx",
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
                    "ny": ny,
                    "nx": nx,
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


def _emit_multivar_sensitivity(
    args: argparse.Namespace,
    d0: dict[str, int],
) -> None:
    ni = args.ni if args.ni is not None else d0["ni"]
    nn = args.nn if args.nn is not None else d0["nn"]
    ky = args.ky if args.ky is not None else d0["ky"]
    kx = args.kx if args.kx is not None else d0["kx"]
    tx = args.tile_x if args.tile_x is not None else d0["tile_x"]
    ty = args.tile_y if args.tile_y is not None else d0["tile_y"]
    if args.ny is not None and args.nx is not None:
        shapes = [(int(args.ny), int(args.nx))]
    else:
        shapes = [_parse_shape(s) for s in args.shapes]
    if len(shapes) != 1:
        raise ValueError(
            "multi-parameter mode now supports one fixed Ny/Nx for plotting. "
            "Pass --ny and --nx (recommended), or a single --shapes value."
        )
    multipliers = [float(x) for x in args.multipliers]
    params = list(args.params)

    # Local elasticity around baseline using (1-delta, 1+delta).
    delta = float(args.elasticity_delta)
    if not (0.0 < delta < 1.0):
        raise ValueError("--elasticity-delta must be between 0 and 1")

    rows: list[dict[str, object]] = []
    elastic_rows: list[dict[str, object]] = []

    for ny, nx in shapes:
        problem, tiling = _base_problem_tiling(ny, nx, ni, nn, ky, kx, tx, ty, d0["chunk_ni"])
        hw0 = dict(hardware_titan_v_v1)
        base_r = predict_v1(problem, tiling, hw0)
        base_t = float(base_r.predicted_time_seconds)

        for p in params:
            base_val = float(hw0[p])
            for m in multipliers:
                hw = dict(hw0)
                hw[p] = base_val * m
                r = predict_v1(problem, tiling, hw)
                t = float(r.predicted_time_seconds)
                rows.append(
                    {
                        "ny": ny,
                        "nx": nx,
                        "param": p,
                        "multiplier": m,
                        "param_value": hw[p],
                        "predicted_time_seconds": t,
                        "normalized_time": (t / base_t) if base_t > 0 else math.nan,
                        "predicted_tflops": r.predicted_tflops,
                        "bottleneck": r.bottleneck,
                        "base_time_seconds": base_t,
                    }
                )

            hw_lo = dict(hw0)
            hw_hi = dict(hw0)
            hw_lo[p] = base_val * (1.0 - delta)
            hw_hi[p] = base_val * (1.0 + delta)
            t_lo = _predict_seconds(problem, tiling, hw_lo)
            t_hi = _predict_seconds(problem, tiling, hw_hi)
            if t_lo > 0 and t_hi > 0:
                e = -(
                    (math.log(t_hi) - math.log(t_lo))
                    / (math.log(1.0 + delta) - math.log(1.0 - delta))
                )
            else:
                e = math.nan
            elastic_rows.append(
                {
                    "ny": ny,
                    "nx": nx,
                    "param": p,
                    "elasticity": e,
                    "abs_elasticity": abs(e) if math.isfinite(e) else math.nan,
                }
            )

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if args.elasticity_out:
        with open(args.elasticity_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(elastic_rows[0].keys()))
            w.writeheader()
            w.writerows(elastic_rows)

    # Optional plots for quick architecture-insight reading.
    if args.plot_prefix:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise SystemExit("matplotlib required for --plot-prefix") from e

        # Curves: one subplot per parameter (single fixed Ny/Nx), color by parameter.
        n_params = len(params)
        ncols = 2
        nrows = 2
        nslots = nrows * ncols
        if n_params > nslots:
            raise ValueError(
                f"2x2 grid supports up to {nslots} params, got {n_params}. "
                "Pass fewer --params."
            )
        param_colors = {p: f"C{i % 10}" for i, p in enumerate(params)}
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(6.2 * ncols, 4.2 * nrows),
            sharex=True,
            sharey=True,
        )
        if hasattr(axs, "ravel"):
            axs_list = list(axs.ravel())
        else:
            axs_list = [axs]
        for i, p in enumerate(params):
            ax = axs_list[i]
            ny, nx = shapes[0]
            pts = [
                (float(r["multiplier"]), float(r["normalized_time"]))
                for r in rows
                if r["param"] == p and r["ny"] == ny and r["nx"] == nx
            ]
            pts.sort(key=lambda t: t[0])
            if pts:
                xs = [x for x, _ in pts]
                ys = [y for _, y in pts]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=param_colors[p],
                    label=f"{ny}x{nx}",
                )
            ax.set_title(p)
            ax.set_xlabel("Hardware multiplier")
            ax.grid(True, alpha=0.3)
        for j in range(n_params, len(axs_list)):
            axs_list[j].set_visible(False)
        axs_list[0].set_ylabel("Normalized runtime (t / t_base)")
        axs_list[min(n_params, len(axs_list)) - 1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{args.plot_prefix}_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Elasticity bars: one subplot per parameter, one bar at fixed Ny/Nx.
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(6.2 * ncols, 4.2 * nrows),
            sharey=True,
        )
        if hasattr(axs, "ravel"):
            axs_list = list(axs.ravel())
        else:
            axs_list = [axs]
        for i, p in enumerate(params):
            ax = axs_list[i]
            grp = [r for r in elastic_rows if r["param"] == p]
            by_shape = {(int(r["ny"]), int(r["nx"])): float(r["elasticity"]) for r in grp}
            labels = [f"{shapes[0][0]}x{shapes[0][1]}"]
            vals = [by_shape.get(shapes[0], math.nan)]
            colors = [param_colors[p]]
            ax.bar(range(len(labels)), vals, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title(p)
            ax.grid(True, axis="y", alpha=0.3)
        for j in range(n_params, len(axs_list)):
            axs_list[j].set_visible(False)
        axs_list[0].set_ylabel("Local elasticity: -dln(t)/dln(param)")
        fig.tight_layout()
        fig.savefig(f"{args.plot_prefix}_elasticity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"[sensitivity] wrote {args.plot_prefix}_curves.png and "
            f"{args.plot_prefix}_elasticity.png",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        default=None,
        choices=tuple(sorted(hardware_titan_v_v1.keys())),
        help="legacy mode: single hardware key to sweep (requires --values)",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="legacy mode: explicit values for --param",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=[
            "peak_fp32_tflops",
            "compute_efficiency_base",
            "dram_bandwidth_GBps",
            "l2_bandwidth_GBps",
            "num_sms",
        ],
        choices=tuple(sorted(hardware_titan_v_v1.keys())),
        help="multi-parameter mode: list of hardware params",
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="multi-parameter mode multipliers relative to baseline hardware value",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["56x56", "112x112", "224x224"],
        help="multi-parameter mode shapes as NyxNx",
    )
    parser.add_argument(
        "--elasticity-delta",
        type=float,
        default=0.10,
        help="local elasticity delta for +/- perturbation (default 0.10)",
    )
    parser.add_argument(
        "--elasticity-out",
        type=Path,
        default=None,
        help="write elasticity summary CSV (multi-parameter mode)",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="multi-parameter mode: write <prefix>_curves.png and <prefix>_elasticity.png",
    )
    parser.add_argument("--ni", type=int, default=None)
    parser.add_argument("--nn", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
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
    legacy_mode = args.param is not None or args.values is not None
    if legacy_mode:
        if args.param is None or args.values is None:
            raise SystemExit("legacy mode requires both --param and --values")
        _emit_legacy_single_param(args, d0)
    else:
        _emit_multivar_sensitivity(args, d0)


if __name__ == "__main__":
    main()
