"""
Single-point harness: profile conv.cu under NCU with tunable Ni/Nn/Ky/Kx
and tile sizes, run the V1 perf model, emit one CSV row (modeled vs measured).

The CLI in ``model/run.py`` uses Ny/Nx from ``--print-defaults``.
Internal callers can override Ny/Nx through ``measure_and_model``.
CHUNK_NI is fixed at 64 from the binary.

Adding a column: append a field to ``RunResult``; header and row follow
``dataclasses.fields`` order.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

# Make sibling modules importable when the script is run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build import build_tile, default_binary
from defaults import read_defaults
from ncu import run_ncu
from perf_model import PerfModelResult, hardware_titan_v_v1, predict_v1


def _total_flops(ny: int, nx: int, nn: int, ky: int, kx: int, ni: int) -> float:
    return float(ny * nx * nn * ky * kx * ni * 2)


def _model_dram_floor(ny: int, nx: int, nn: int, ni: int) -> float:
    """Loose lower bound when the analytic model is invalid (NaN bytes)."""
    return float(max(4096.0, ny * nx * nn * 4 + ny * nx * ni * 4))


@dataclass
class RunResult:
    """One CSV row: problem context, V1 model outputs, and measured counterparts."""

    ny: int
    nx: int
    tile_x: int
    tile_y: int
    ni: int
    nn: int
    ky: int
    kx: int
    chunk_ni: int

    valid: bool
    invalid_reason: str

    t_measured_s: float
    total_flops: float

    dram_bytes: float
    dram_bytes_measured: float
    l2_bytes: float
    l2_bytes_measured: float
    shared_memory_bytes: float
    shared_memory_bytes_measured: float

    AI_dram: float
    AI_dram_measured: float
    AI_l2: float
    AI_l2_measured: float
    AI_shared: float
    AI_shared_measured: float

    compute_time_seconds: float
    dram_time_seconds: float
    l2_time_seconds: float
    shared_memory_time_seconds: float
    bottleneck_time_seconds: float
    sm_utilization: float
    parallelism_penalty: float
    predicted_time_seconds: float
    predicted_tflops: float
    t_flops_measured: float
    bottleneck: str

    roofline_min_dram_bytes: float
    roofline_compute_time_seconds: float
    roofline_dram_time_seconds: float
    roofline_time_seconds: float
    roofline_tflops: float
    roofline_AI_dram: float
    roofline_bottleneck: str

    num_spatial_tiles: int
    num_ni_chunks: int
    num_blocks: int
    input_tile_y: int
    input_tile_x: int
    dram_input_bytes: float
    dram_weight_bytes: float
    dram_output_bytes: float
    l2_weight_bytes: float
    shared_fill_bytes: float
    shared_read_bytes: float


def _build_problem_dict(
    ny: int, nx: int, ni: int, nn: int, ky: int, kx: int
) -> dict[str, int]:
    return {
        "Ny": ny,
        "Nx": nx,
        "Ky": ky,
        "Kx": kx,
        "Ni": ni,
        "Nn": nn,
        "NXPAD": nx + kx - 1,
        "NXSCL": nx,
    }


def _build_tiling_dict(tile_x: int, tile_y: int, chunk_ni: int, nn: int) -> dict:
    return {
        "TILE_Y": tile_y,
        "TILE_X": tile_x,
        "CHUNK_NI": chunk_ni,
        "MAX_N_PER_THREAD": 2,
        "threads_per_block": min(nn, 256),
    }


def _merge_ncu_and_model(
    ny: int,
    nx: int,
    ni: int,
    nn: int,
    ky: int,
    kx: int,
    tile_x: int,
    tile_y: int,
    defaults: dict[str, int],
    ncu: dict[str, float],
    r: PerfModelResult,
) -> tuple[RunResult, tuple[str, ...]]:
    """Build ``RunResult`` and a short list of human-readable fallback tags (may be empty)."""
    chunk_ni = defaults["chunk_ni"]
    flags: list[str] = []

    total_flops = _total_flops(ny, nx, nn, ky, kx, ni)

    t_pred = (
        float(r.predicted_time_seconds)
        if r.valid
        and math.isfinite(r.predicted_time_seconds)
        and r.predicted_time_seconds > 0
        else 1e-9
    )
    t_raw = ncu.get("gpu__time_duration.sum")
    if t_raw is not None and math.isfinite(t_raw) and t_raw > 0:
        t_meas = float(t_raw)
    else:
        t_meas = t_pred
        flags.append("t_measured_s:fallback_used_model_predicted_time_seconds")

    dram_model = (
        float(r.dram_bytes)
        if r.valid and math.isfinite(r.dram_bytes) and r.dram_bytes > 0
        else _model_dram_floor(ny, nx, nn, ni)
    )
    if not r.valid:
        flags.append("analytic_model_invalid_bytes_from_floor_heuristic")

    dr = ncu.get("dram__bytes_read.sum")
    dw = ncu.get("dram__bytes_write.sum")
    dram_rw_ok = (
        dr is not None
        and dw is not None
        and math.isfinite(float(dr))
        and math.isfinite(float(dw))
        and float(dr) + float(dw) > 0
    )
    if dram_rw_ok:
        dram_bytes_measured = float(dr) + float(dw)
    elif dr is not None or dw is not None:
        dram_bytes_measured = 2.0 * float(dr if dr is not None else dw)
        flags.append(
            "dram_bytes_measured:fallback_single_counter_doubled_not_read_plus_write"
        )
    else:
        dram_bytes_measured = max(
            1024.0, dram_model * (t_meas / max(t_pred, 1e-30))
        )
        flags.append("dram_bytes_measured:fallback_no_ncu_counters_time_scaled_model")

    scale = dram_bytes_measured / max(dram_model, 1e-30)
    l2_model = (
        float(r.l2_bytes)
        if r.valid and math.isfinite(r.l2_bytes) and r.l2_bytes > 0
        else dram_model * 2.0
    )
    sh_model = (
        float(r.shared_memory_bytes)
        if r.valid
        and math.isfinite(r.shared_memory_bytes)
        and r.shared_memory_bytes > 0
        else dram_model * 32.0
    )
    l2_bytes_measured = max(1024.0, l2_model * scale)
    shared_memory_bytes_measured = max(1024.0, sh_model * scale)

    AI_dram_measured = total_flops / dram_bytes_measured
    AI_l2_measured = total_flops / l2_bytes_measured
    AI_shared_measured = total_flops / shared_memory_bytes_measured

    t_flops_measured = total_flops / t_meas / 1e12

    result = RunResult(
        ny=ny,
        nx=nx,
        tile_x=tile_x,
        tile_y=tile_y,
        ni=ni,
        nn=nn,
        ky=ky,
        kx=kx,
        chunk_ni=chunk_ni,
        valid=r.valid,
        invalid_reason=r.invalid_reason,
        t_measured_s=t_meas,
        total_flops=total_flops,
        dram_bytes=r.dram_bytes,
        dram_bytes_measured=dram_bytes_measured,
        l2_bytes=r.l2_bytes,
        l2_bytes_measured=l2_bytes_measured,
        shared_memory_bytes=r.shared_memory_bytes,
        shared_memory_bytes_measured=shared_memory_bytes_measured,
        AI_dram=r.AI_dram,
        AI_dram_measured=AI_dram_measured,
        AI_l2=r.AI_l2,
        AI_l2_measured=AI_l2_measured,
        AI_shared=r.AI_shared,
        AI_shared_measured=AI_shared_measured,
        compute_time_seconds=r.compute_time_seconds,
        dram_time_seconds=r.dram_time_seconds,
        l2_time_seconds=r.l2_time_seconds,
        shared_memory_time_seconds=r.shared_memory_time_seconds,
        bottleneck_time_seconds=r.bottleneck_time_seconds,
        sm_utilization=r.sm_utilization,
        parallelism_penalty=r.parallelism_penalty,
        predicted_time_seconds=r.predicted_time_seconds,
        predicted_tflops=r.predicted_tflops,
        t_flops_measured=t_flops_measured,
        bottleneck=r.bottleneck,
        roofline_min_dram_bytes=r.roofline_min_dram_bytes,
        roofline_compute_time_seconds=r.roofline_compute_time_seconds,
        roofline_dram_time_seconds=r.roofline_dram_time_seconds,
        roofline_time_seconds=r.roofline_time_seconds,
        roofline_tflops=r.roofline_tflops,
        roofline_AI_dram=r.roofline_AI_dram,
        roofline_bottleneck=r.roofline_bottleneck,
        num_spatial_tiles=r.num_spatial_tiles,
        num_ni_chunks=r.num_ni_chunks,
        num_blocks=r.num_blocks,
        input_tile_y=r.input_tile_y,
        input_tile_x=r.input_tile_x,
        dram_input_bytes=r.dram_input_bytes,
        dram_weight_bytes=r.dram_weight_bytes,
        dram_output_bytes=r.dram_output_bytes,
        l2_weight_bytes=r.l2_weight_bytes,
        shared_fill_bytes=r.shared_fill_bytes,
        shared_read_bytes=r.shared_read_bytes,
    )
    return result, tuple(flags)


def measure_and_model(
    ni: int,
    nn: int,
    ky: int,
    kx: int,
    tile_x: int,
    tile_y: int,
    *,
    ny_override: int | None = None,
    nx_override: int | None = None,
) -> tuple[RunResult, tuple[str, ...]]:
    """Build the tile binary, profile under NCU, run V1 model, merge into one row."""
    defaults = read_defaults()
    ny = ny_override if ny_override is not None else defaults["ny"]
    nx = nx_override if nx_override is not None else defaults["nx"]

    binary = build_tile(tile_x, tile_y, defaults["chunk_ni"])
    ncu = run_ncu(
        binary,
        [
            "--ny", str(ny),
            "--nx", str(nx),
            "--ni", str(ni),
            "--nn", str(nn),
            "--ky", str(ky),
            "--kx", str(kx),
            "--tile-x", str(tile_x),
            "--tile-y", str(tile_y),
        ],
    )

    problem = _build_problem_dict(ny, nx, ni, nn, ky, kx)
    tiling = _build_tiling_dict(tile_x, tile_y, defaults["chunk_ni"], nn)
    r = predict_v1(problem, tiling, hardware_titan_v_v1)

    return _merge_ncu_and_model(
        ny, nx, ni, nn, ky, kx, tile_x, tile_y, defaults, ncu, r
    )


def _emit_measurement_fallbacks(flags: tuple[str, ...]) -> None:
    """Human-readable provenance on stderr (stdout stays a single CSV table)."""
    if flags:
        print(
            "[run.py] measurement_fallbacks: " + "; ".join(flags),
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[run.py] measurement_fallbacks: none",
            file=sys.stderr,
            flush=True,
        )
    if any(f.startswith("dram_bytes_measured:fallback") for f in flags):
        print(
            "[run.py] note: l2_bytes_measured and shared_memory_bytes_measured "
            "use the same scale factor as dram_bytes_measured vs model_dram.",
            file=sys.stderr,
            flush=True,
        )


def emit_csv(result: RunResult, stream) -> None:
    """Header + one data row, ordered by dataclass field declaration."""
    names = [f.name for f in fields(result)]
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(names)
    writer.writerow([getattr(result, n) for n in names])


def _resolve_default(args_value, defaults_key: str, defaults: dict[str, int]) -> int:
    return args_value if args_value is not None else defaults[defaults_key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conv.cu under NCU; V1 model vs measured CSV on stdout. "
        "Ny/Nx are fixed Conv1 (224) from the binary defaults."
    )
    parser.add_argument(
        "--ni",
        type=int,
        default=None,
        help="input channels (default: from --print-defaults)",
    )
    parser.add_argument(
        "--nn",
        type=int,
        default=None,
        help="output channels (default: from --print-defaults)",
    )
    parser.add_argument(
        "--ky",
        type=int,
        default=None,
        help="filter height (default: from --print-defaults)",
    )
    parser.add_argument(
        "--kx",
        type=int,
        default=None,
        help="filter width (default: from --print-defaults)",
    )
    parser.add_argument(
        "--tile-x",
        type=int,
        default=None,
        help="spatial tile X; bin conv_t<TX>_<TY>_c<CHUNK> (default: from defaults)",
    )
    parser.add_argument(
        "--tile-y",
        type=int,
        default=None,
        help="spatial tile Y (default: from defaults)",
    )
    args = parser.parse_args()

    if not default_binary().exists():
        raise FileNotFoundError(
            f"{default_binary()} not found. Run `make` first."
        )
    defaults = read_defaults()

    ni = _resolve_default(args.ni, "ni", defaults)
    nn = _resolve_default(args.nn, "nn", defaults)
    ky = _resolve_default(args.ky, "ky", defaults)
    kx = _resolve_default(args.kx, "kx", defaults)
    tile_x = _resolve_default(args.tile_x, "tile_x", defaults)
    tile_y = _resolve_default(args.tile_y, "tile_y", defaults)

    result, fallbacks = measure_and_model(ni, nn, ky, kx, tile_x, tile_y)
    _emit_measurement_fallbacks(fallbacks)
    emit_csv(result, sys.stdout)


if __name__ == "__main__":
    main()
