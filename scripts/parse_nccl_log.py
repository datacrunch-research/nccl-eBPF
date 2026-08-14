#!/usr/bin/env python3
"""Parse nccl-tests logs produced by nccl_bench.sh into one CSV.

Walks RESULTS_DIR for <tag>.log + <tag>.meta pairs, extracts the per-size
result rows (out-of-place and in-place) and joins the meta sidecar fields.

Env overrides:
  RESULTS_DIR  directory to scan (default: ./results next to this script)
  OUT_CSV      output path (default: RESULTS_DIR/combined.csv)
"""

import csv
import json
import os
import re
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path(__file__).parent / "results"))
OUT_CSV = Path(os.environ.get("OUT_CSV", RESULTS_DIR / "combined.csv"))

META_FIELDS = [
    "tag", "ts", "test", "world", "trays", "arm", "arm_kind", "policy",
    "algo", "proto", "profiler", "max_nch", "iters", "rep", "nccl",
    "nccl_lib_dir", "plugin_sha",
]
CSV_FIELDS = META_FIELDS + [
    "size_bytes", "place", "time_us", "algbw_gbs", "busbw_gbs", "avg_busbw_gbs",
]

# nccl-tests data row: size count type redop root time algbw busbw #wrong (oop)
#                      then time algbw busbw #wrong (in-place)
ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S+"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:\S+)"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:\S+)\s*$"
)
AVG_RE = re.compile(r"Avg bus bandwidth\s*:\s*([\d.]+)")


def parse_pair(log_path: Path):
    meta_path = log_path.with_suffix(".meta")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta.setdefault("tag", log_path.stem)

    text = log_path.read_text(errors="replace")
    avg = None
    m = AVG_RE.search(text)
    if m:
        avg = float(m.group(1))

    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        size, oop_t, oop_a, oop_b, ip_t, ip_a, ip_b = m.groups()
        base = {k: meta.get(k, "") for k in META_FIELDS}
        base["avg_busbw_gbs"] = avg
        rows.append({**base, "size_bytes": int(size), "place": "oop",
                     "time_us": float(oop_t), "algbw_gbs": float(oop_a),
                     "busbw_gbs": float(oop_b)})
        rows.append({**base, "size_bytes": int(size), "place": "ip",
                     "time_us": float(ip_t), "algbw_gbs": float(ip_a),
                     "busbw_gbs": float(ip_b)})
    return rows


def main():
    logs = sorted(RESULTS_DIR.glob("*.log"))
    all_rows = []
    for log in logs:
        rows = parse_pair(log)
        if not rows:
            print(f"WARN: no data rows in {log.name}")
        all_rows.extend(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"{len(all_rows)} rows from {len(logs)} logs -> {OUT_CSV}")


if __name__ == "__main__":
    main()
