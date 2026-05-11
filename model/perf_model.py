"""
V1 bottleneck performance model for conv.cu (DRAM / L2 / shared / compute).

Entry point: ``predict_v1(problem, tiling, hardware) -> PerfModelResult``.

``Workload`` / ``Tile`` / ``predict()`` remain for callers that only need
seconds using defaults aligned with the harness (CHUNK_NI=64, Titan V V1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

BYTES_PER_FLOAT = 4

hardware_titan_v_v1: dict[str, float] = {
    "peak_fp32_tflops": 15.0,
    "dram_bandwidth_GBps": 652.8,
    "l2_bandwidth_GBps": 2000.0,
    "shared_memory_bandwidth_GBps": 12000.0,
    "num_sms": 80.0,
    "shared_memory_capacity_per_sm": float(96 * 1024),
    "l2_capacity_bytes": float(4608 * 1024),
}


def _chunks(ni: int, chunk_ni: int) -> Iterator[int]:
    for ni_base in range(0, ni, chunk_ni):
        yield min(chunk_ni, ni - ni_base)


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
    """V1 convolution roofline + bottleneck model (spec CS251A hw1)."""
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

    num_tiles_y = math.ceil(Ny / tile_y)
    num_tiles_x = math.ceil(Nx / tile_x)
    num_spatial_tiles = num_tiles_y * num_tiles_x
    num_blocks = num_spatial_tiles
    num_ni_chunks = math.ceil(Ni / chunk_ni) if Ni > 0 else 0

    input_tile_y = tile_y + Ky - 1
    input_tile_x = tile_x + Kx - 1

    sm_utilization = min(1.0, num_blocks / num_sms) if num_sms > 0 else 1.0
    parallelism_penalty = 1.0 / sm_utilization if sm_utilization > 0 else math.inf

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
            total_flops=nan,
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
        )

    total_flops = float(Ny * Nx * Nn * Ky * Kx * Ni * 2)

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

    peak_flops_s = peak_tflops * 1e12
    compute_time_seconds = total_flops / peak_flops_s if peak_flops_s > 0 else math.inf

    dram_bps = dram_bw * 1e9
    dram_time_seconds = dram_bytes / dram_bps if dram_bps > 0 else math.inf

    l2_bps = l2_bw * 1e9
    l2_time_seconds = l2_bytes / l2_bps if l2_bps > 0 else math.inf

    sh_bps = shmem_bw * 1e9
    shared_memory_time_seconds = (
        shared_memory_bytes / sh_bps if sh_bps > 0 else math.inf
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

    predicted_time_seconds = bottleneck_time_seconds * parallelism_penalty

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
    )


def predict(w: Workload, t: Tile) -> float:
    """Modeled kernel time in seconds (V1); NaN if tile config exceeds shared mem."""
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
