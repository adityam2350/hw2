#!/usr/bin/env python3
"""
Roofline chart (TFLOP/s vs arithmetic intensity) with sweep points overlaid.

X-axis: AI measured as FLOP per byte of **NCU DRAM traffic**
        (``total_flops / dram_bytes_measured`` — same as sweep column
        ``AI_dram_measured``).

Y-axis: **achieved** TFLOP/s from hardware (``t_flops_measured``).

The roof is ``min(Peak_fp32, AI * dram_BW)`` with BW in GB/s (Volta-style:
``TFLOP/s = AI[FLOP/B] * BW_GBps / 1000``).

Optional second series: V1 **predicted** TF/s at the **same** x
(``AI_dram_measured``) as the hardware points (hollow), so each config pairs
vertically—offset in y is time/throughput error, not a different AI definition.

Example::

  python3 model/plot_roofline.py sweep.csv -o roofline.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _f(x: str) -> float:
    return float(x)


def _i(x: str) -> int:
    return int(x)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_roofline(
    rows: list[dict[str, str]],
    out_path: str,
    peak_tflops: float,
    dram_bw_gbps: float,
    show_model_points: bool,
) -> None:
    import matplotlib.pyplot as plt

    # TFLOP/s from memory: AI [FLOP/byte] * BW [10^9 byte/s] / 10^12
    slope = dram_bw_gbps / 1000.0
    ridge_ai = peak_tflops / slope if slope > 0 else math.inf

    pts_ai: list[float] = []
    pts_perf: list[float] = []
    pts_label: list[str] = []
    colors: list[float] = []

    m_ai: list[float] = []
    m_perf: list[float] = []

    for r in rows:
        ai_m = _f(r["AI_dram_measured"])
        perf = _f(r["t_flops_measured"])
        if (
            not math.isfinite(ai_m)
            or not math.isfinite(perf)
            or ai_m <= 0
            or perf <= 0
        ):
            continue
        pts_ai.append(ai_m)
        pts_perf.append(perf)
        ni, nn = _i(r["ni"]), _i(r["nn"])
        tx, ty = _i(r["tile_x"]), _i(r["tile_y"])
        pts_label.append(f"{ni}x{nn} {tx}×{ty}")
        colors.append(float(ni))
        if show_model_points:
            p_t = _f(r["predicted_tflops"])
            if math.isfinite(p_t) and p_t > 0:
                m_ai.append(ai_m)
                m_perf.append(p_t)

    if not pts_ai:
        raise SystemExit("no valid (AI_dram_measured, t_flops_measured) rows in CSV")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # x-axis from measured AI and ridge (model points share the same x)
    ai_max = max(max(pts_ai), ridge_ai * 1.05, 10.0)
    ai_grid = [i * ai_max / 400 for i in range(401)]
    roof = [min(peak_tflops, x * slope) for x in ai_grid]
    ax.plot(ai_grid, roof, "k-", lw=2, label=f"Roof (peak {peak_tflops} TF/s, {dram_bw_gbps} GB/s)")

    ax.axvline(ridge_ai, color="0.5", ls=":", lw=1)
    ax.plot(
        ridge_ai,
        peak_tflops,
        "kv",
        ms=8,
        label=f"Ridge AI ≈ {ridge_ai:.1f} FLOP/byte",
    )

    sc = ax.scatter(
        pts_ai,
        pts_perf,
        c=colors,
        s=55,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.4,
        zorder=5,
        label="Measured (NCU DRAM AI)",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ni (= nn in square sweeps)")

    if show_model_points and m_ai:
        ax.scatter(
            m_ai,
            m_perf,
            s=80,
            facecolors="none",
            edgecolors="C3",
            linewidths=1.2,
            marker="o",
            zorder=6,
            label="V1 model (predicted TF/s, same NCU DRAM AI)",
        )

    ax.set_xlabel("Arithmetic intensity (FLOP / byte, NCU DRAM R+W)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("Roofline with measured conv points")
    ax.set_xlim(0, ai_max * 1.02)
    perf_for_lim = list(pts_perf)
    if show_model_points and m_perf:
        perf_for_lim.extend(m_perf)
    y_hi = max(max(perf_for_lim), peak_tflops * 1.08)
    ax.set_ylim(0, y_hi * 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="e.g. sweep.csv from model/sweep.py")
    parser.add_argument("-o", "--output", required=True, help="PNG path")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="FP32 peak TFLOP/s (default: from perf_model hardware table)",
    )
    parser.add_argument(
        "--dram-gbps",
        type=float,
        default=None,
        help="DRAM bandwidth GB/s (default: from perf_model hardware table)",
    )
    parser.add_argument(
        "--no-model-points",
        action="store_true",
        help="do not overlay V1 predicted TF/s markers (same AI as measured)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from perf_model import hardware_titan_v_v1

    peak = args.peak_tflops
    if peak is None:
        peak = float(hardware_titan_v_v1["peak_fp32_tflops"])
    dram = args.dram_gbps
    if dram is None:
        dram = float(hardware_titan_v_v1["dram_bandwidth_GBps"])

    rows = load_rows(args.csv)
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    plot_roofline(
        rows,
        args.output,
        peak,
        dram,
        show_model_points=not args.no_model_points,
    )
    print(f"[plot_roofline] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
