"""
Nsight Compute (`ncu`) wrapper: run + CSV parse, normalized to base units.

NCU prints values in whatever unit it picks ("nsecond", "Kbyte", etc.); the
parser here normalizes everything to seconds / bytes / counts so callers
never have to think about unit prefixes.

The NCU lock file lives in another user's /tmp on this shared machine, so
we always set TMPDIR to a project-local directory (mirrors what the
Makefile does).
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
from pathlib import Path
from typing import Dict, Sequence

from build import REPO_ROOT

NCU_TMP = REPO_ROOT / ".ncu-tmp"

# `dram__bytes_*` are summed in model/run.py as measured DRAM traffic (vs modeled bytes).
# L2 / shared-memory byte metrics are architecture- and NCU-version specific; extend this
# list when you have validated metric names, then plumb through RunResult.
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
]


def run_ncu(binary: Path, cli_args: Sequence[str]) -> Dict[str, float]:
    """Profile ``binary`` with ``cli_args`` via NCU; return measured metrics.

    Returned values are in base SI units (seconds, bytes, raw counts).
    """
    NCU_TMP.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(NCU_TMP)
    cmd = [
        "ncu", "--csv",
        "--metrics", ",".join(NCU_METRICS),
        str(binary),
        *cli_args,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=True)
    return _parse_ncu_csv(proc.stdout)


# NCU --csv prints values in the unit listed in the "Metric Unit" column
# (e.g. "nsecond", "Kbyte"). Convert everything to seconds / bytes / counts.
_TIME_TO_SEC = {
    "second":  1.0,
    "msecond": 1e-3,
    "usecond": 1e-6,
    "nsecond": 1e-9,
    "psecond": 1e-12,
}
_BYTE_FACTORS = {
    "byte":  1.0,
    "Kbyte": 1e3, "kbyte": 1e3,
    "Mbyte": 1e6, "mbyte": 1e6,
    "Gbyte": 1e9, "gbyte": 1e9,
    "Tbyte": 1e12,
    "KiB":   1024.0,
    "MiB":   1024.0**2,
    "GiB":   1024.0**3,
}


def _normalize_unit(value: float, unit: str) -> float:
    """Best-effort conversion to base units. Unknown units pass through."""
    u = unit.strip()
    if u in _TIME_TO_SEC:   return value * _TIME_TO_SEC[u]
    if u in _BYTE_FACTORS:  return value * _BYTE_FACTORS[u]
    return value


def _parse_ncu_csv(text: str) -> Dict[str, float]:
    """Parse NCU's --csv stdout into ``{metric_name: base_unit_value}``.

    NCU prefixes its CSV with a banner; we locate the header line that
    starts with '"ID"' and stream from there.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        raise RuntimeError("could not find NCU CSV header. Output was:\n" + text)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    out: Dict[str, float] = {}
    for row in reader:
        name  = (row.get("Metric Name") or "").strip()
        unit  = (row.get("Metric Unit") or "").strip()
        value = (row.get("Metric Value") or "").replace(",", "").strip()
        if not name or not value:
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        out[name] = _normalize_unit(v, unit)
    if not out:
        raise RuntimeError("NCU returned no metrics. Output was:\n" + text)
    return out
