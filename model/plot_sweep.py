#!/usr/bin/env python3
"""Build validation figures from ``model/sweep.py`` CSV output."""

from __future__ import annotations

import argparse
import csv
import sys


def _f(x: str) -> float:
    return float(x)


def _i(x: str) -> int:
    return int(x)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _style_xticks(ax, x_num: list[int], xs: list[str]) -> None:
    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=35, ha="right")


def _plot_mape_split_panels(
    suptitle: str,
    xlabel: str,
    xs: list[str],
    m_v1: list[float],
    m_rf: list[float],
    out_path: str,
    single_axis: bool,
) -> None:
    import matplotlib.pyplot as plt

    x_num = list(range(len(xs)))
    if single_axis:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_num, m_v1, "o-", label="Analytic model MAPE (%)")
        ax.plot(x_num, m_rf, "s--", label="Roofline MAPE (%)")
        _style_xticks(ax, x_num, xs)
        ax.set_ylabel("MAPE vs measured time (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(suptitle)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(8, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.12},
        )
        fig.suptitle(suptitle, fontsize=12, y=1.02)

        ax_top.plot(x_num, m_v1, "o-", color="C0", label="Analytic model")
        ax_top.set_ylabel("MAPE (%)")
        ax_top.set_title("Analytic model (calibrated bottleneck)")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")
        top_max = max(m_v1) if m_v1 else 1.0
        ax_top.set_ylim(0, max(top_max * 1.25 + 0.5, 4.0))

        ax_bot.plot(x_num, m_rf, "s--", color="C1", label="Roofline baseline")
        ax_bot.set_ylabel("MAPE (%)")
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_title("Roofline (ideal DRAM reuse vs peak compute)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper left")
        bot_max = max(m_rf) if m_rf else 10.0
        ax_bot.set_ylim(0, min(bot_max * 1.08 + 2.0, 100.0))

        plt.setp(ax_top.get_xticklabels(), visible=False)
        _style_xticks(ax_bot, x_num, xs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_channels(
    rows: list[dict[str, str]],
    tile_x: int,
    tile_y: int,
    out_path: str,
    single_axis: bool = False,
) -> None:
    subset = [
        r
        for r in rows
        if _i(r["tile_x"]) == tile_x and _i(r["tile_y"]) == tile_y
    ]
    if not subset:
        raise SystemExit(f"no rows with tile {tile_x}x{tile_y}")

    subset.sort(key=lambda r: (_i(r["ni"]), _i(r["nn"])))
    xs = [f"{_i(r['ni'])}/{_i(r['nn'])}" for r in subset]
    m_v1 = [_f(r["mape_time_v1_pct"]) for r in subset]
    m_rf = [_f(r["mape_time_roofline_pct"]) for r in subset]

    _plot_mape_split_panels(
        f"Channel sweep (tile {tile_x}×{tile_y})",
        "ni / nn",
        xs,
        m_v1,
        m_rf,
        out_path,
        single_axis,
    )


def plot_by_tiles(
    rows: list[dict[str, str]],
    ni: int,
    nn: int,
    out_path: str,
    single_axis: bool = False,
) -> None:
    subset = [r for r in rows if _i(r["ni"]) == ni and _i(r["nn"]) == nn]
    if not subset:
        raise SystemExit(f"no rows with ni={ni} nn={nn}")

    def tile_key(r: dict[str, str]) -> tuple[int, int]:
        return _i(r["tile_x"]), _i(r["tile_y"])

    subset.sort(key=tile_key)
    xs = [f"{_i(r['tile_x'])}×{_i(r['tile_y'])}" for r in subset]
    m_v1 = [_f(r["mape_time_v1_pct"]) for r in subset]
    m_rf = [_f(r["mape_time_roofline_pct"]) for r in subset]

    _plot_mape_split_panels(
        f"Tile sweep (ni={ni}, nn={nn})",
        "Tile (Tx×Ty)",
        xs,
        m_v1,
        m_rf,
        out_path,
        single_axis,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="sweep output CSV")
    parser.add_argument(
        "--by-channels",
        action="store_true",
        help="x-axis: ni/nn (hold tile fixed)",
    )
    parser.add_argument(
        "--by-tiles",
        action="store_true",
        help="x-axis: tile (hold channels fixed)",
    )
    parser.add_argument(
        "--tile-x",
        type=int,
        default=8,
        help="for --by-channels: filter rows (default 8)",
    )
    parser.add_argument(
        "--tile-y",
        type=int,
        default=8,
        help="for --by-channels: filter rows (default 8)",
    )
    parser.add_argument(
        "--ni",
        type=int,
        default=64,
        help="for --by-tiles: filter rows (default 64)",
    )
    parser.add_argument(
        "--nn",
        type=int,
        default=64,
        help="for --by-tiles: filter rows (default 64)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output PNG path",
    )
    parser.add_argument(
        "--single-axis",
        action="store_true",
        help="one plot for both series (V1 looks ~0% if roofline dominates scale)",
    )
    args = parser.parse_args()

    if args.by_channels == args.by_tiles:
        raise SystemExit("pass exactly one of --by-channels / --by-tiles")

    rows = load_rows(args.csv)
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    if args.by_channels:
        plot_by_channels(
            rows, args.tile_x, args.tile_y, args.output, args.single_axis
        )
    else:
        plot_by_tiles(rows, args.ni, args.nn, args.output, args.single_axis)
    print(f"[plot_sweep] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
