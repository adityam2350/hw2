"""
Convolution performance model (roofline baseline + hierarchical bottleneck).

- **Analytic core:** DRAM / L2 / compute with optional traffic calibration and
  Ni-dependent sustained-compute efficiency (Volta conv behavior).
- The naive ``shared_read`` byte volume is kept for AI_shared reporting only;
  the **bottleneck** ``max()`` uses staging-oriented shared time that does not
  duplicate the MAC-unroll volume story from the original V1.

Entry point: ``predict_v1(problem, tiling, hardware) -> PerfModelResult``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

BYTES_PER_FLOAT = 4

# Titan V–style table. Extra keys use ``hardware.get`` in ``predict_v1`` with
# defaults so callers can override or strip for ablations.
hardware_titan_v_v1: dict[str, float] = {
    "peak_fp32_tflops": 15.0,
    "dram_bandwidth_GBps": 652.8,
    "l2_bandwidth_GBps": 2000.0,
    "shared_memory_bandwidth_GBps": 12000.0,
    "num_sms": 80.0,
    "shared_memory_capacity_per_sm": float(96 * 1024),
    "l2_capacity_bytes": float(4608 * 1024),
    # Effective DRAM bytes vs ideal tiling count (sectors, evictions, RMW, etc.).
    "dram_traffic_scale": 2.9,
    "l2_traffic_scale": 2.9,
    # Sustained FP32 vs peak: scales with channel count (better util at larger Ni/Nn).
    "compute_efficiency_base": 0.202,
    "compute_efficiency_ni_ref": 64.0,
    "compute_efficiency_ni_exp": 0.58,
    "compute_efficiency_cap": 0.72,
    "compute_efficiency_floor": 0.162,
    # Shared: only staging traffic limits this leg (fraction of peak BW).
    "shared_staging_bw_fraction": 0.35,
    # Register / occupancy drag for large spatial tiles (penalty on final time).
    "tile_area_reference": 64.0,
    "tile_area_penalty_per_unit": 0.0145,
    # Non-square-large tiles: effective FLOP/s grows with tile area vs 8×8 baseline.
    "rectangular_tile_compute_gamma": 0.88,
    # Extra boost when Tx≠Ty (better wave mix vs square blocks at same area).
    "rectangular_aspect_exp": 0.72,
    # Scale 10×10 penalty magnitude with channels (weaker drag at large Ni).
    "square_tile_penalty_ni_ref": 64.0,
    "square_tile_penalty_ni_exp": 0.85,
}


def _chunks(ni: int, chunk_ni: int) -> Iterator[int]:
    for ni_base in range(0, ni, chunk_ni):
        yield min(chunk_ni, ni - ni_base)


def _roofline_min_dram_bytes(Ny: int, Nx: int, Ky: int, Kx: int, Ni: int, Nn: int) -> float:
    """One logical pass: padded input + full weights + output write (float32)."""
    nypad = Ny + Ky - 1
    nxpad = Nx + Kx - 1
    inp_b = float(nypad * nxpad * Ni * BYTES_PER_FLOAT)
    w_b = float(Ky * Kx * Nn * Ni * BYTES_PER_FLOAT)
    out_b = float(Ny * Nx * Nn * BYTES_PER_FLOAT)
    return inp_b + w_b + out_b


def _roofline_from_work(
    total_flops: float,
    roofline_min_dram_bytes: float,
    peak_tflops: float,
    dram_bandwidth_GBps: float,
    parallelism_penalty: float,
) -> tuple[float, float, float, float, str, float]:
    """Classic two-corner roof: time = max(compute, DRAM) * parallelism_penalty."""
    peak_flops_s = peak_tflops * 1e12
    compute_t = total_flops / peak_flops_s if peak_flops_s > 0 else math.inf
    dram_bps = dram_bandwidth_GBps * 1e9
    dram_t = (
        roofline_min_dram_bytes / dram_bps if dram_bps > 0 else math.inf
    )
    bottleneck_t = max(compute_t, dram_t)
    roofline_t = bottleneck_t * parallelism_penalty if math.isfinite(bottleneck_t) else math.inf
    if math.isclose(compute_t, dram_t, rel_tol=1e-12, abs_tol=1e-30):
        roofline_bn = "compute_dram"
    elif compute_t > dram_t:
        roofline_bn = "compute"
    else:
        roofline_bn = "dram"
    roofline_tflops = (
        total_flops / roofline_t / 1e12
        if roofline_t > 0 and math.isfinite(roofline_t)
        else math.nan
    )
    ai = (
        total_flops / roofline_min_dram_bytes
        if roofline_min_dram_bytes > 0
        else math.inf
    )
    return compute_t, dram_t, roofline_t, roofline_tflops, roofline_bn, ai


@dataclass(frozen=True)
class PerfModelResult:
    valid: bool
    invalid_reason: str

    num_spatial_tiles: int
    num_ni_chunks: int
    num_blocks: int
    input_tile_y: int
    input_tile_x: int
    sm_utilization: float
    parallelism_penalty: float

    total_flops: float
    dram_input_bytes: float
    dram_weight_bytes: float
    dram_output_bytes: float
    dram_bytes: float
    l2_weight_bytes: float
    l2_bytes: float
    shared_fill_bytes: float
    shared_read_bytes: float
    shared_memory_bytes: float

    compute_time_seconds: float
    dram_time_seconds: float
    l2_time_seconds: float
    shared_memory_time_seconds: float
    bottleneck_time_seconds: float
    predicted_time_seconds: float

    AI_dram: float
    AI_l2: float
    AI_shared: float
    predicted_tflops: float
    bottleneck: str

    roofline_min_dram_bytes: float
    roofline_compute_time_seconds: float
    roofline_dram_time_seconds: float
    roofline_time_seconds: float
    roofline_tflops: float
    roofline_AI_dram: float
    roofline_bottleneck: str


@dataclass(frozen=True)
class Workload:
    Ny: int
    Nx: int
    Ni: int
    Nn: int
    Ky: int = 3
    Kx: int = 3


@dataclass(frozen=True)
class Tile:
    Tx: int = 8
    Ty: int = 8


def predict_v1(
    problem: Mapping[str, Any],
    tiling: Mapping[str, Any],
    hardware: Mapping[str, float],
) -> PerfModelResult:
    """Hierarchical roofline-style model with calibrated DRAM/L2 traffic and Ni-scaled compute."""
    Ny = int(problem["Ny"])
    Nx = int(problem["Nx"])
    Ky = int(problem["Ky"])
    Kx = int(problem["Kx"])
    Ni = int(problem["Ni"])
    Nn = int(problem["Nn"])

    tile_y = int(tiling["TILE_Y"])
    tile_x = int(tiling["TILE_X"])
    chunk_ni = int(tiling["CHUNK_NI"])
    _threads_per_block = tiling.get("threads_per_block")
    if _threads_per_block is None:
        _threads_per_block = min(Nn, 256)
    else:
        _threads_per_block = int(_threads_per_block)
    _ = (_threads_per_block, tiling.get("MAX_N_PER_THREAD", 2))  # V1 unused

    peak_tflops = float(hardware["peak_fp32_tflops"])
    dram_bw = float(hardware["dram_bandwidth_GBps"])
    l2_bw = float(hardware["l2_bandwidth_GBps"])
    shmem_bw = float(hardware["shared_memory_bandwidth_GBps"])
    num_sms = float(hardware["num_sms"])
    shmem_cap = float(hardware["shared_memory_capacity_per_sm"])
    l2_cap = float(hardware["l2_capacity_bytes"])
    dram_scale = float(hardware.get("dram_traffic_scale", 1.0))
    l2_scale = float(hardware.get("l2_traffic_scale", dram_scale))
    sh_staging_frac = float(hardware.get("shared_staging_bw_fraction", 0.35))
    tile_area_ref = float(hardware.get("tile_area_reference", 64.0))
    tile_penalty_lam = float(hardware.get("tile_area_penalty_per_unit", 0.0))
    rect_gamma = float(hardware.get("rectangular_tile_compute_gamma", 0.0))
    rect_aspect_exp = float(hardware.get("rectangular_aspect_exp", 0.0))
    sq_pen_ni_ref = float(hardware.get("square_tile_penalty_ni_ref", 64.0))
    sq_pen_ni_exp = float(hardware.get("square_tile_penalty_ni_exp", 0.85))

    _eff_base = hardware.get("compute_efficiency_base")
    if _eff_base is not None:
        mn_ch = min(Ni, Nn)
        eff_ref = float(hardware.get("compute_efficiency_ni_ref", 64.0))
        eff_exp = float(hardware.get("compute_efficiency_ni_exp", 0.58))
        eff_cap = float(hardware.get("compute_efficiency_cap", 0.72))
        eff_floor = float(hardware.get("compute_efficiency_floor", 0.0))
        compute_efficiency = max(
            eff_floor,
            min(
                eff_cap,
                float(_eff_base) * (mn_ch / eff_ref) ** eff_exp,
            ),
        )
    else:
        compute_efficiency = 1.0

    num_tiles_y = math.ceil(Ny / tile_y)
    num_tiles_x = math.ceil(Nx / tile_x)
    num_spatial_tiles = num_tiles_y * num_tiles_x
    num_blocks = num_spatial_tiles
    num_ni_chunks = math.ceil(Ni / chunk_ni) if Ni > 0 else 0

    input_tile_y = tile_y + Ky - 1
    input_tile_x = tile_x + Kx - 1

    sm_utilization = min(1.0, num_blocks / num_sms) if num_sms > 0 else 1.0
    parallelism_penalty = 1.0 / sm_utilization if sm_utilization > 0 else math.inf

    total_flops = float(Ny * Nx * Nn * Ky * Kx * Ni * 2)
    roofline_min_dram_bytes = _roofline_min_dram_bytes(Ny, Nx, Ky, Kx, Ni, Nn)
    (
        roofline_compute_time_seconds,
        roofline_dram_time_seconds,
        roofline_time_seconds,
        roofline_tflops,
        roofline_bottleneck,
        roofline_AI_dram,
    ) = _roofline_from_work(
        total_flops,
        roofline_min_dram_bytes,
        peak_tflops,
        dram_bw,
        parallelism_penalty,
    )

    shmem_bytes = input_tile_y * input_tile_x * min(chunk_ni, Ni) * BYTES_PER_FLOAT
    if shmem_bytes > shmem_cap:
        nan = math.nan
        return PerfModelResult(
            valid=False,
            invalid_reason=(
                "shared_memory_per_block exceeds shared_memory_capacity_per_sm: "
                f"{shmem_bytes} > {shmem_cap}"
            ),
            num_spatial_tiles=num_spatial_tiles,
            num_ni_chunks=num_ni_chunks,
            num_blocks=num_blocks,
            input_tile_y=input_tile_y,
            input_tile_x=input_tile_x,
            sm_utilization=sm_utilization,
            parallelism_penalty=parallelism_penalty,
            total_flops=total_flops,
            dram_input_bytes=nan,
            dram_weight_bytes=nan,
            dram_output_bytes=nan,
            dram_bytes=nan,
            l2_weight_bytes=nan,
            l2_bytes=nan,
            shared_fill_bytes=nan,
            shared_read_bytes=nan,
            shared_memory_bytes=nan,
            compute_time_seconds=nan,
            dram_time_seconds=nan,
            l2_time_seconds=nan,
            shared_memory_time_seconds=nan,
            bottleneck_time_seconds=nan,
            predicted_time_seconds=nan,
            AI_dram=nan,
            AI_l2=nan,
            AI_shared=nan,
            predicted_tflops=nan,
            bottleneck="invalid",
            roofline_min_dram_bytes=roofline_min_dram_bytes,
            roofline_compute_time_seconds=roofline_compute_time_seconds,
            roofline_dram_time_seconds=roofline_dram_time_seconds,
            roofline_time_seconds=roofline_time_seconds,
            roofline_tflops=roofline_tflops,
            roofline_AI_dram=roofline_AI_dram,
            roofline_bottleneck=roofline_bottleneck,
        )

    dram_input_bytes = 0.0
    dram_weight_no_l2_bytes = 0.0
    l2_weight_bytes = 0.0
    shared_read_bytes = 0.0

    for chunk_i in _chunks(Ni, chunk_ni):
        tile_bytes = (
            num_spatial_tiles * input_tile_y * input_tile_x * chunk_i * BYTES_PER_FLOAT
        )
        dram_input_bytes += tile_bytes

        w_tile = num_spatial_tiles * Ky * Kx * Nn * chunk_i * BYTES_PER_FLOAT
        dram_weight_no_l2_bytes += w_tile
        l2_weight_bytes += w_tile

        shared_read_bytes += (
            num_spatial_tiles
            * tile_y
            * tile_x
            * Ky
            * Kx
            * Nn
            * chunk_i
            * BYTES_PER_FLOAT
        )

    full_weight_bytes = Ky * Kx * Nn * Ni * BYTES_PER_FLOAT
    if full_weight_bytes <= l2_cap:
        dram_weight_bytes = full_weight_bytes
    else:
        dram_weight_bytes = dram_weight_no_l2_bytes

    dram_output_bytes = float(Ny * Nx * Nn * BYTES_PER_FLOAT)
    dram_bytes = dram_input_bytes + dram_weight_bytes + dram_output_bytes

    l2_bytes = dram_input_bytes + l2_weight_bytes + dram_output_bytes

    shared_fill_bytes = dram_input_bytes
    shared_memory_bytes = shared_fill_bytes + shared_read_bytes

    peak_flops_s = peak_tflops * 1e12 * max(compute_efficiency, 1e-9)
    compute_time_seconds = (
        total_flops / peak_flops_s if peak_flops_s > 0 else math.inf
    )

    is_square_large_tile = tile_x >= 10 and tile_y >= 10
    tile_area = float(tile_x * tile_y)
    if (
        not is_square_large_tile
        and rect_gamma > 0.0
        and tile_area_ref > 0.0
    ):
        ar = float(max(tile_x, tile_y)) / float(max(1, min(tile_x, tile_y)))
        exp_scale = (sq_pen_ni_ref / float(max(min(Ni, Nn), 1))) ** 0.22
        eff_aspect_exp = rect_aspect_exp * exp_scale
        aspect_boost = max(1.0, ar**eff_aspect_exp) if eff_aspect_exp > 0.0 else 1.0
        compute_time_seconds /= max(
            (tile_area / tile_area_ref) ** rect_gamma * aspect_boost,
            0.55,
        )

    dram_bps = dram_bw * 1e9
    dram_bytes_scaled = dram_bytes * dram_scale
    dram_time_seconds = (
        dram_bytes_scaled / dram_bps if dram_bps > 0 else math.inf
    )

    l2_bps = l2_bw * 1e9
    l2_bytes_scaled = l2_bytes * l2_scale
    l2_time_seconds = l2_bytes_scaled / l2_bps if l2_bps > 0 else math.inf

    # Staging: move input tiles into shared (overlap with compute captured by BW fraction).
    sh_bps = shmem_bw * 1e9
    shared_memory_time_seconds = (
        (shared_fill_bytes / max(sh_bps * sh_staging_frac, 1e-30))
        if sh_bps > 0
        else math.inf
    )

    candidates = [
        ("compute", compute_time_seconds),
        ("dram", dram_time_seconds),
        ("l2", l2_time_seconds),
        ("shared_memory", shared_memory_time_seconds),
    ]
    bottleneck_time_seconds = max(t for _, t in candidates)
    bottleneck = next(
        n
        for n, t in candidates
        if math.isclose(t, bottleneck_time_seconds, rel_tol=0.0, abs_tol=1e-30)
    )

    # Penalty for square ≥10×10 only; damp at large Ni (measured drag fades vs 8×8).
    if is_square_large_tile:
        tile_excess = max(0.0, tile_area - tile_area_ref)
        ni_damp = (
            (sq_pen_ni_ref / float(max(min(Ni, Nn), 1))) ** sq_pen_ni_exp
        )
        ni_damp = min(1.0, max(0.4, ni_damp))
        tiling_slowdown = 1.0 + tile_penalty_lam * tile_excess * ni_damp
    else:
        tiling_slowdown = 1.0
    predicted_time_seconds = (
        bottleneck_time_seconds * parallelism_penalty * tiling_slowdown
    )

    AI_dram = total_flops / dram_bytes if dram_bytes > 0 else math.inf
    AI_l2 = total_flops / l2_bytes if l2_bytes > 0 else math.inf
    AI_shared = (
        total_flops / shared_memory_bytes if shared_memory_bytes > 0 else math.inf
    )
    predicted_tflops = (
        total_flops / predicted_time_seconds / 1e12
        if predicted_time_seconds > 0 and math.isfinite(predicted_time_seconds)
        else math.nan
    )

    return PerfModelResult(
        valid=True,
        invalid_reason="",
        num_spatial_tiles=num_spatial_tiles,
        num_ni_chunks=num_ni_chunks,
        num_blocks=num_blocks,
        input_tile_y=input_tile_y,
        input_tile_x=input_tile_x,
        sm_utilization=sm_utilization,
        parallelism_penalty=parallelism_penalty,
        total_flops=total_flops,
        dram_input_bytes=dram_input_bytes,
        dram_weight_bytes=dram_weight_bytes,
        dram_output_bytes=dram_output_bytes,
        dram_bytes=dram_bytes,
        l2_weight_bytes=l2_weight_bytes,
        l2_bytes=l2_bytes,
        shared_fill_bytes=shared_fill_bytes,
        shared_read_bytes=shared_read_bytes,
        shared_memory_bytes=shared_memory_bytes,
        compute_time_seconds=compute_time_seconds,
        dram_time_seconds=dram_time_seconds,
        l2_time_seconds=l2_time_seconds,
        shared_memory_time_seconds=shared_memory_time_seconds,
        bottleneck_time_seconds=bottleneck_time_seconds,
        predicted_time_seconds=predicted_time_seconds,
        AI_dram=AI_dram,
        AI_l2=AI_l2,
        AI_shared=AI_shared,
        predicted_tflops=predicted_tflops,
        bottleneck=bottleneck,
        roofline_min_dram_bytes=roofline_min_dram_bytes,
        roofline_compute_time_seconds=roofline_compute_time_seconds,
        roofline_dram_time_seconds=roofline_dram_time_seconds,
        roofline_time_seconds=roofline_time_seconds,
        roofline_tflops=roofline_tflops,
        roofline_AI_dram=roofline_AI_dram,
        roofline_bottleneck=roofline_bottleneck,
    )


def predict(w: Workload, t: Tile) -> float:
    """Modeled kernel time in seconds; NaN if tile exceeds shared memory cap."""
    problem = {
        "Ny": w.Ny,
        "Nx": w.Nx,
        "Ky": w.Ky,
        "Kx": w.Kx,
        "Ni": w.Ni,
        "Nn": w.Nn,
    }
    tiling = {
        "TILE_Y": t.Ty,
        "TILE_X": t.Tx,
        "CHUNK_NI": 64,
        "MAX_N_PER_THREAD": 2,
        "threads_per_block": min(w.Nn, 256),
    }
    r = predict_v1(problem, tiling, hardware_titan_v_v1)
    if not r.valid:
        return float("nan")
    return r.predicted_time_seconds
