# CUDA convolution kernel + performance model

This repo contains a CUDA convolution (`conv.cu`), a C++ harness (`main.cpp`), and a **Python performance layer** under `model/`. The sections below focus on **how to use and change the Python code**. You only need a built `./bin/conv` and, for `model/run.py`, `ncu` on `PATH` plus a GPU.

---

## Python layout

| Module | Role |
|--------|------|
| [model/perf_model.py](model/perf_model.py) | **Analytic V1 model**: `predict_v1(problem, tiling, hardware) -> PerfModelResult` |
| [model/run.py](model/run.py) | **CLI + NCU harness**: profiles the CUDA binary, calls `predict_v1`, prints one CSV row on stdout |
| [model/defaults.py](model/defaults.py) | Reads `./bin/conv --print-defaults` (JSON) for fixed defaults (`ny`, `nx`, default `ni`/`nn`/tiles, `chunk_ni`, …) |
| [model/build.py](model/build.py) | Locates/builds `bin/conv_t<TX>_<TY>_c<CHUNK>` via `make tiled` |
| [model/ncu.py](model/ncu.py) | Runs Nsight Compute (`ncu --csv`) and parses metrics |

Add `sys.path.insert(0, "model")` (or run from repo root with `python3 -c` adjusting path) if you import from outside `model/`.

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
    "MAX_N_PER_THREAD": 2,       # reserved for future models; V1 ignores
    "threads_per_block": min(64, 256),  # optional; default min(Nn, 256)
}
hardware = dict(hardware_titan_v_v1)  # or your own dict (see below)

r = predict_v1(problem, tiling, hardware)
print(r.valid, r.predicted_time_seconds, r.predicted_tflops, r.bottleneck)
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
| `NXPAD`, `NXSCL` | int | no | Ignored by **V1** math today; extra keys in `problem` are always ignored. You may include them for documentation parity with the C++ harness. |

**2. `tiling` — kernel / blocking parameters**

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `TILE_Y` | int | yes | Output tile height (CUDA block tile) |
| `TILE_X` | int | yes | Output tile width |
| `CHUNK_NI` | int | yes | Input-channel chunk staged in shared memory (harness uses 64) |
| `threads_per_block` | int | no | Default `min(Nn, 256)` to match the launcher |
| `MAX_N_PER_THREAD` | int | no | Ignored by V1 math; kept for API compatibility |

**3. `hardware` — machine rooflines and capacities**

All values are **floats** in the units implied by the names:

| Key | Typical meaning |
|-----|-----------------|
| `peak_fp32_tflops` | Peak FP32 TFLOP/s (compute bound) |
| `dram_bandwidth_GBps` | DRAM GB/s |
| `l2_bandwidth_GBps` | L2 “effective” GB/s (tunable) |
| `shared_memory_bandwidth_GBps` | Shared-memory effective GB/s (tunable) |
| `num_sms` | Number of SMs (for block-level utilization) |
| `shared_memory_capacity_per_sm` | Bytes of shared memory per SM (capacity check) |
| `l2_capacity_bytes` | L2 size in bytes (DRAM weight reuse heuristic) |

The module exposes a default Titan V–style table as **`hardware_titan_v_v1`**. Pass a **copy** (`dict(hardware_titan_v_v1)`) if you will mutate it.

### Model outputs (`PerfModelResult`)

Frozen dataclass returned by `predict_v1`. When **`valid` is `False`** (shared memory over capacity), numeric fields that depend on the roofline are **`nan`**, `bottleneck` is `"invalid"`, and **`invalid_reason`** is a short explanation.

| Field group | Fields |
|-------------|--------|
| Validity | `valid`, `invalid_reason` |
| Geometry / scheduling | `num_spatial_tiles`, `num_ni_chunks`, `num_blocks`, `input_tile_y`, `input_tile_x`, `sm_utilization`, `parallelism_penalty` |
| Work | `total_flops` |
| Bytes (modeled) | `dram_input_bytes`, `dram_weight_bytes`, `dram_output_bytes`, `dram_bytes`, `l2_weight_bytes`, `l2_bytes`, `shared_fill_bytes`, `shared_read_bytes`, `shared_memory_bytes` |
| Times (modeled, seconds) | `compute_time_seconds`, `dram_time_seconds`, `l2_time_seconds`, `shared_memory_time_seconds`, `bottleneck_time_seconds` (= max of the four), `predicted_time_seconds` (= that max × `parallelism_penalty`) |
| Roofline summary | `AI_dram`, `AI_l2`, `AI_shared`, `predicted_tflops` (= `total_flops / predicted_time_seconds / 1e12`), `bottleneck` (`"compute"` \| `"dram"` \| `"l2"` \| `"shared_memory"` \| `"invalid"`) |

Legacy helper: **`predict(Workload, Tile) -> float`** returns modeled seconds with `CHUNK_NI=64` and `hardware_titan_v_v1`, or **`nan`** if the tile fails the shared-memory check.

---

## CLI harness (`model/run.py`)

Runs NCU on the built convolution binary, merges metrics with `predict_v1`, prints **one CSV table on stdout**. **Stderr** prints `measurement_fallbacks: ...` when NCU fields are missing or approximated (see below).

**Prerequisites:** `make` has produced `./bin/conv`; `ncu` on `PATH`; GPU available.

```bash
python3 model/run.py
python3 model/run.py --nn 128 --tile-x 8 --tile-y 8
python3 model/run.py --ni 64 --nn 64 --ky 3 --kx 3 --tile-x 8 --tile-y 8 > result.csv
```

**CLI flags** (defaults from `./bin/conv --print-defaults` when omitted):

| Flag | Meaning |
|------|---------|
| `--ni`, `--nn` | Input / output channels |
| `--ky`, `--kx` | Filter height / width |
| `--tile-x`, `--tile-y` | Must match compile-time `TILE_X` / `TILE_Y` of the binary used |

**Spatial size** `Ny`, `Nx` are **not** CLI flags; they always come from `read_defaults()` (Conv1 224×224 in the default binary).

### Harness output (`RunResult` → CSV)

The CSV header is the `RunResult` dataclass field order. It contains:

- **Problem echo:** `ny`, `nx`, `tile_x`, `tile_y`, `ni`, `nn`, `ky`, `kx`, `chunk_ni`
- **Model fields:** same analytic quantities as `PerfModelResult` (times, bytes, AIs, `predicted_tflops`, `bottleneck`, …) plus modeled byte breakdown columns
- **Measured / estimated columns:** `t_measured_s`, `dram_bytes_measured`, `l2_bytes_measured`, `shared_memory_bytes_measured`, `AI_*_measured`, `t_flops_measured`
- **Validity:** `valid`, `invalid_reason`

**Measured semantics (important):**

- **`t_measured_s`**: NCU `gpu__time_duration.sum` when present; otherwise falls back to modeled `predicted_time_seconds` (flagged on stderr).
- **`dram_bytes_measured`**: Prefer `dram__bytes_read.sum` + `dram__bytes_write.sum`; if only one exists, doubled; if neither, time-scaled model DRAM (flagged).
- **`l2_bytes_measured` / `shared_memory_bytes_measured`**: No direct NCU byte pair in this harness; values are **model L2 / model shared × (`dram_bytes_measured` / `dram_bytes`)**. If DRAM used a fallback, stderr adds a one-line note.
- **`t_flops_measured`**: `total_flops / t_measured_s / 1e12`.

After each run, stderr shows **`[run.py] measurement_fallbacks: none`** or a `;`-separated list of tags when a fallback path was used.

---

## How to update the model

Work primarily in **[model/perf_model.py](model/perf_model.py)**.

1. **Change formulas or add bottlenecks**  
   Edit the body of **`predict_v1`**. Keep the three-dict input contract unless you intentionally version the API (e.g. add `predict_v2`). The roofline steps are: derive tile counts and chunk sums → `total_flops` → byte models (`dram_*`, `l2_*`, `shared_*`) → per-resource times from `hardware` bandwidths and peak FLOPs → `bottleneck_time_seconds` → apply `parallelism_penalty` → `predicted_time_seconds` → `AI_*` and `predicted_tflops`.

2. **Change GPU constants**  
   Edit **`hardware_titan_v_v1`** or pass a custom `hardware` dict into `predict_v1`. Keys must match what `predict_v1` reads (`peak_fp32_tflops`, `dram_bandwidth_GBps`, …).

3. **Add or rename model outputs**  
   - Add fields to **`PerfModelResult`** in `perf_model.py` and set them in `predict_v1` (including the invalid early-return branch so callers never see missing attributes).  
   - If the harness should emit them: add matching fields to **`RunResult`** in `run.py`, populate them in **`_merge_ncu_and_model`**, and extend **`_build_problem_dict` / `_build_tiling_dict`** if new inputs are needed from the CLI or defaults.

4. **Wire new NCU metrics into “measured” columns**  
   Append metric names to **`NCU_METRICS`** in [model/ncu.py](model/ncu.py), then read them in **`_merge_ncu_and_model`** in `run.py` and map them into existing or new `RunResult` fields. Keep stderr fallback tagging honest when you still approximate.

5. **New CLI knobs**  
   Add `argparse` entries in **`main()`** in `run.py`, thread values into **`measure_and_model`** and into the `ncu` argv list so the profiled kernel matches the modeled problem.

6. **Defaults JSON shape**  
   If `./bin/conv --print-defaults` changes, update **`REQUIRED_KEYS`** and docstrings in [model/defaults.py](model/defaults.py).

Run **`python3 -m py_compile model/*.py`** after edits.

---

## Minimal non-Python notes (kernel + build)

- **Build:** `make` → `bin/conv`; `make tiled TILE_X=TX TILE_Y=TY CHUNK_NI=64` → `bin/conv_t<TX>_<TY>_c64`. `CHUNK_NI` is fixed at 64 for this assignment.
- **C++ CLI:** `./bin/conv --help`; spatial size is fixed at 224×224; tunables are `--ni`, `--nn`, `--ky`, `--kx`, `--tile-x`, `--tile-y`.

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
│   ├── build.py
│   ├── ncu.py
│   └── defaults.py
└── bin/                 # built binaries (gitignored)
```
