# CUDA convolution kernel + performance model

This repo contains a CUDA convolution (`conv.cu`), a C++ harness (`main.cpp`), and a **Python performance layer** under `model/`. You need a built `./bin/conv`, and for profiling **`ncu` on `PATH`** plus a GPU.

---

## Python layout

| Module | Role |
|--------|------|
| [model/perf_model.py](model/perf_model.py) | **Analytic model:** `predict_v1(problem, tiling, hardware) -> PerfModelResult` (hierarchical bottleneck + optional roofline baseline columns) |
| [model/run.py](model/run.py) | **CLI + NCU harness:** profiles the CUDA binary, calls `predict_v1`, prints one CSV row on stdout |
| [model/sweep.py](model/sweep.py) | **Batch sweep:** Cartesian product of channel / tile settings; NCU per point; CSV with MAPE columns |
| [model/plot_sweep.py](model/plot_sweep.py) | **Figures** from sweep CSV (needs **matplotlib**; use project `venv` if needed — see below) |
| [model/plot_roofline.py](model/plot_roofline.py) | **Roofline** plot from sweep CSV (same matplotlib / venv note) |
| [model/sensitivity.py](model/sensitivity.py) | **Hardware sensitivity** on the analytic model only (no GPU): sweep one `hardware` knob |
| [model/defaults.py](model/defaults.py) | Reads `./bin/conv --print-defaults` (JSON) for defaults (`ny`, `nx`, `ni`/`nn`/tiles, `chunk_ni`, …) |
| [model/build.py](model/build.py) | Locates/builds `bin/conv_t<TX>_<TY>_c<CHUNK>` via `make tiled` |
| [model/ncu.py](model/ncu.py) | Runs Nsight Compute (`ncu --csv`) with **`--launch-skip 0 --launch-count 1`** (needed for reliable CSV on NCU 2024+) and parses metrics |

Add `sys.path.insert(0, "model")` (or run from repo root) if you import from outside `model/`.

**Plots / matplotlib:** If the system Python has no `pip`/`venv` packages, create a local env (example):

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python /tmp/get-pip.py
.venv/bin/pip install matplotlib
.venv/bin/python model/plot_sweep.py sweep.csv --by-channels --tile-x 8 --tile-y 8 -o fig_channels.png
.venv/bin/python model/plot_sweep.py sweep.csv --by-tiles --ni 64 --nn 64 -o fig_tiles.png
.venv/bin/python model/plot_roofline.py sweep.csv -o roofline.png
```

`.venv/` is gitignored.

---

## Calling the model in Python (`predict_v1`)

The model is **pure**: three mappings in, one `PerfModelResult` out. It does not run CUDA or NCU.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("model").resolve()))

from perf_model import predict_v1, hardware_titan_v_v1

problem = {
    "Ny": 224,
    "Nx": 224,
    "Ky": 3,
    "Kx": 3,
    "Ni": 64,
    "Nn": 64,
}
tiling = {
    "TILE_Y": 8,
    "TILE_X": 8,
    "CHUNK_NI": 64,
    "MAX_N_PER_THREAD": 2,
    "threads_per_block": min(64, 256),  # optional; default min(Nn, 256)
}
hardware = dict(hardware_titan_v_v1)  # or your own dict (see below)

r = predict_v1(problem, tiling, hardware)
print(r.valid, r.predicted_time_seconds, r.predicted_tflops, r.bottleneck)
print(r.roofline_time_seconds, r.roofline_bottleneck)
```

### Model inputs (three dicts)

**1. `problem` — convolution shape**

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `Ny` | int | yes | Output height |
| `Nx` | int | yes | Output width |
| `Ky` | int | yes | Filter height |
| `Kx` | int | yes | Filter width |
| `Ni` | int | yes | Input channels |
| `Nn` | int | yes | Output channels |
| `NXPAD`, `NXSCL` | int | no | Extra keys are ignored by the math; include for parity with the C++ harness if you like |

**2. `tiling` — kernel / blocking parameters**

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `TILE_Y` | int | yes | Output tile height (CUDA block tile) |
| `TILE_X` | int | yes | Output tile width |
| `CHUNK_NI` | int | yes | Input-channel chunk staged in shared memory (harness uses 64) |
| `threads_per_block` | int | no | Default `min(Nn, 256)` to match the launcher |
| `MAX_N_PER_THREAD` | int | no | Used by CUDA kernel; not in the analytic time breakdown |

**3. `hardware` — rooflines, capacities, and calibration**

Core keys (all **floats**):

| Key | Meaning |
|-----|---------|
| `peak_fp32_tflops` | Peak FP32 TFLOP/s |
| `dram_bandwidth_GBps` | DRAM GB/s |
| `l2_bandwidth_GBps` | Effective L2 GB/s |
| `shared_memory_bandwidth_GBps` | Peak-class shared BW (used for staging term) |
| `num_sms` | SM count (underfill penalty when `num_blocks < num_sms`) |
| `shared_memory_capacity_per_sm` | Shared capacity (reject tile if patch exceeds) |
| `l2_capacity_bytes` | L2 size (DRAM weight traffic heuristic) |

Calibration / behavior keys (defaults on **`hardware_titan_v_v1`**; omit from a custom dict to fall back to safe defaults in code):

| Key | Role |
|-----|------|
| `dram_traffic_scale`, `l2_traffic_scale` | Effective traffic vs naïve byte count (NCU alignment / evictions) |
| `compute_efficiency_base`, `compute_efficiency_ni_ref`, `compute_efficiency_ni_exp`, `compute_efficiency_cap`, `compute_efficiency_floor` | Ni-dependent sustained FP32 vs peak |
| `shared_staging_bw_fraction` | Staging into shared vs peak BW |
| `tile_area_reference`, `tile_area_penalty_per_unit` | Extra cost for large **square** tiles (e.g. 10×10) |
| `rectangular_tile_compute_gamma`, `rectangular_aspect_exp` | Faster effective compute for non-square large rectangles (8×10, …) |
| `square_tile_penalty_ni_ref`, `square_tile_penalty_ni_exp` | Weaken square-tile penalty at large `Ni`/`Nn` |

Pass a **copy** (`dict(hardware_titan_v_v1)`) if you mutate the dict.

### Model outputs (`PerfModelResult`)

Frozen dataclass from `predict_v1`. If **`valid` is `False`** (shared memory over capacity for the declared tile), detailed byte/timing fields are **`nan`**, but **roofline** fields derived from `(Ny,Nx,Ni,Nn,Ky,Kx)` remain defined for baseline comparisons.

| Field group | Fields |
|-------------|--------|
| Validity | `valid`, `invalid_reason` |
| Geometry / scheduling | `num_spatial_tiles`, `num_ni_chunks`, `num_blocks`, `input_tile_y`, `input_tile_x`, `sm_utilization`, `parallelism_penalty` |
| Work | `total_flops` |
| Bytes (modeled) | `dram_*`, `l2_*`, `shared_fill_bytes`, `shared_read_bytes`, `shared_memory_bytes` |
| Times (seconds) | `compute_time_seconds`, `dram_time_seconds`, `l2_time_seconds`, `shared_memory_time_seconds`, `bottleneck_time_seconds` (= `max` of those four legs), then **`predicted_time_seconds`** = that max × `parallelism_penalty` × **tiling slowdown** (large square tiles, when applicable) |
| Roofline (ideal minimal DRAM + peak) | `roofline_min_dram_bytes`, `roofline_compute_time_seconds`, `roofline_dram_time_seconds`, `roofline_time_seconds`, `roofline_tflops`, `roofline_AI_dram`, `roofline_bottleneck` |
| Summary | `AI_dram`, `AI_l2`, `AI_shared`, `predicted_tflops`, `bottleneck` (`"compute"` \| `"dram"` \| `"l2"` \| `"shared_memory"` \| `"invalid"`) |

**Legacy:** **`predict(Workload, Tile) -> float`** uses `CHUNK_NI=64` and `hardware_titan_v_v1`, or **`nan`** if the tile fails the shared-memory check.

---

## CLI harness (`model/run.py`)

Runs NCU on the built binary, merges metrics with `predict_v1`, prints **one CSV** (header + row) on stdout. **Stderr:** `measurement_fallbacks: ...` when NCU fields are missing or approximated.

```bash
python3 model/run.py
python3 model/run.py --nn 128 --tile-x 8 --tile-y 8
python3 model/run.py --ni 64 --nn 64 --tile-x 8 --tile-y 8 > result.csv
```

**Batch experiments:**

```bash
python3 model/sweep.py --out sweep.csv --square-channels 32 64 128 --tile-xs 8 10 --tile-ys 8 10
```

**Kernel / tile note:** Default dynamic shared memory is **48 KiB/block** on many Volta configs. A tile whose padded patch `(Ty+Ky-1)(Tx+Kx-1)·CHUNK_NI·4` bytes exceeds that will **fail to launch** (CUDA invalid argument). Prefer **8×8 / 8×10 / 10×10** etc. over **12×12** with `CHUNK_NI=64` unless you configure larger dynamic shared memory in the launcher.

### Harness output (`RunResult` → CSV)

Same columns as `PerfModelResult` where applicable, plus measured/derived **`t_measured_s`**, **`dram_bytes_measured`**, MAPE-friendly scalings, etc. (see `run.py` / `sweep.py`).

**Measured semantics:** `dram_bytes_measured` prefers NCU `dram__bytes_read.sum` + `dram__bytes_write.sum`. **`l2_bytes_measured` / `shared_memory_bytes_measured`** are scaled from the model when NCU does not expose direct counters.

---

## How to update the model

Work in **[model/perf_model.py](model/perf_model.py)**.

1. **Formulas** — Edit **`predict_v1`**. Keep the three-dict API unless you fork (e.g. `predict_v2`).
2. **Constants** — Edit **`hardware_titan_v_v1`** or pass a custom `hardware` dict.
3. **New outputs** — Extend **`PerfModelResult`** and, if needed, **`RunResult`** + **`_merge_ncu_and_model`** in `run.py`.
4. **NCU** — Add metrics to **`NCU_METRICS`** in [model/ncu.py](model/ncu.py).
5. **Defaults JSON** — If `--print-defaults` changes, update [model/defaults.py](model/defaults.py).

Run **`python3 -m py_compile model/*.py`** after edits.

---

## Minimal non-Python notes (kernel + build)

- **Build:** `make` → `bin/conv`; `make tiled TILE_X=TX TILE_Y=TY CHUNK_NI=64` → `bin/conv_t<TX>_<TY>_c64`.
- **C++ CLI:** `./bin/conv --help`; spatial size fixed at **224×224**; tunables `--ni`, `--nn`, `--ky`, `--kx`, `--tile-x`, `--tile-y`.

## Project layout

```
.
├── conv.cu
├── main.cpp
├── Makefile
├── conv1.md
├── model/
│   ├── perf_model.py
│   ├── run.py
│   ├── sweep.py
│   ├── plot_sweep.py
│   ├── plot_roofline.py
│   ├── sensitivity.py
│   ├── build.py
│   ├── ncu.py
│   └── defaults.py
└── bin/                 # built binaries (gitignored)
```
