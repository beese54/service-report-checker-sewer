"""
evolve_baseline.py — Baseline CSV Evolution
============================================
Appends high-confidence ACCEPTED results from a pipeline run JSON to the
baseline CSV.  When the baseline exceeds --max-rows, the lowest-confidence
rows are evicted to an archive CSV first.

Usage:
    python analytics/evolve_baseline.py \\
        --new-batch results/pipeline_run_<ts>.json \\
        --baseline monitoring/baseline.csv \\
        [--max-rows 1000] \\
        [--min-confidence 0.85] \\
        [--archive monitoring/archive_history.csv]

Golden selection criteria:
    - status == "ACCEPTED"
    - max(washing_confidence_D2, washing_confidence_U2) >= --min-confidence
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Re-use the same JSON→rows logic as monitor_performance
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_performance import json_to_rows

DEFAULT_MAX_ROWS       = 1000
DEFAULT_MIN_CONFIDENCE = 0.85
DEFAULT_ARCHIVE        = "monitoring/archive_history.csv"


# =============================================================================
# Main
# =============================================================================

def evolve(
    new_batch_path: str,
    baseline_path: str,
    max_rows: int         = DEFAULT_MAX_ROWS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    archive_path: str     = DEFAULT_ARCHIVE,
):
    # Load new batch
    with open(new_batch_path, encoding="utf-8") as f:
        run_json = json.load(f)
    new_rows = json_to_rows(run_json)
    new_df   = pd.DataFrame(new_rows)
    log.info("evolve: new_batch_n=%d source=%s", len(new_df), Path(new_batch_path).name)

    # Golden selection: ACCEPTED + max confidence >= threshold
    def _max_conf(row):
        vals = [
            v for v in [row.get("washing_confidence_D2"), row.get("washing_confidence_U2")]
            if v is not None and not (isinstance(v, float) and (v != v))  # skip NaN
        ]
        return max(vals) if vals else 0.0

    def _min_conf(row):
        vals = [
            v for v in [row.get("washing_confidence_D2"), row.get("washing_confidence_U2")]
            if v is not None and not (isinstance(v, float) and (v != v))
        ]
        return min(vals) if vals else 0.0

    new_df["_max_conf"] = new_df.apply(_max_conf, axis=1)
    golden_df = new_df[
        (new_df["status"] == "ACCEPTED") &
        (new_df["_max_conf"] >= min_confidence)
    ].copy()
    golden_df = golden_df.drop(columns=["_max_conf"])
    log.info(
        "evolve: golden_candidates=%d (status=ACCEPTED, max_conf>=%.2f)",
        len(golden_df), min_confidence,
    )

    # Load or create baseline
    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    if os.path.isfile(baseline_path):
        base_df = pd.read_csv(baseline_path)
        log.info("evolve: baseline_loaded n=%d", len(base_df))
    else:
        base_df = pd.DataFrame(columns=golden_df.columns)
        log.info("evolve: baseline not found — creating new")

    if len(golden_df) == 0:
        log.info("evolve: no golden rows to add — baseline unchanged")
        return

    # Remove duplicates: skip folders already in baseline
    existing_folders = set(base_df["folder"].astype(str)) if "folder" in base_df.columns else set()
    golden_df = golden_df[~golden_df["folder"].astype(str).isin(existing_folders)]
    new_golden = len(golden_df)
    if new_golden == 0:
        log.info("evolve: all golden folders already in baseline — no changes")
        return

    # Eviction if needed
    evicted = 0
    projected = len(base_df) + new_golden
    if projected > max_rows:
        n_evict = projected - max_rows

        # Sort baseline by min(conf_D2, conf_U2) ascending → evict lowest first
        if "washing_confidence_D2" in base_df.columns:
            base_df["_min_conf"] = base_df.apply(_min_conf, axis=1)
            evict_candidates = base_df.sort_values("_min_conf").head(n_evict)
        else:
            evict_candidates = base_df.head(n_evict)

        evict_ids = evict_candidates.index
        evicted   = len(evict_ids)

        # Archive evicted rows
        evict_rows = base_df.loc[evict_ids].copy()
        evict_rows["eviction_date"] = datetime.now().isoformat(timespec="seconds")
        if "_min_conf" in evict_rows.columns:
            evict_rows = evict_rows.drop(columns=["_min_conf"])

        os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
        if os.path.isfile(archive_path):
            evict_rows.to_csv(archive_path, mode="a", index=False, header=False)
        else:
            evict_rows.to_csv(archive_path, index=False)

        for _, row in evict_rows.iterrows():
            log.info(
                "evolve: evicted folder=%s confidence=%.4f",
                row.get("folder", "?"),
                row.get("_min_conf", float("nan")) if "_min_conf" in row else 0.0,
            )

        base_df = base_df.drop(index=evict_ids)
        if "_min_conf" in base_df.columns:
            base_df = base_df.drop(columns=["_min_conf"])

    # Append golden rows
    combined = pd.concat([base_df, golden_df], ignore_index=True)
    combined.to_csv(baseline_path, index=False)

    log.info(
        "evolve: new_golden=%d evicted=%d baseline_size=%d",
        new_golden, evicted, len(combined),
    )
    print(f"\nBaseline updated: {len(combined)} rows  (+{new_golden} added, -{evicted} evicted)")
    print(f"Saved: {baseline_path}")


def main():
    parser = argparse.ArgumentParser(description="Evolve the monitoring baseline CSV")
    parser.add_argument("--new-batch",       required=True)
    parser.add_argument("--baseline",        required=True)
    parser.add_argument("--max-rows",        type=int,   default=DEFAULT_MAX_ROWS)
    parser.add_argument("--min-confidence",  type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--archive",         default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    evolve(
        new_batch_path = args.new_batch,
        baseline_path  = args.baseline,
        max_rows       = args.max_rows,
        min_confidence = args.min_confidence,
        archive_path   = args.archive,
    )


if __name__ == "__main__":
    main()
