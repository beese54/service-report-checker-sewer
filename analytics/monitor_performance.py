"""
monitor_performance.py — Pipeline Metric Drift Monitor
=======================================================
Compares a new pipeline run JSON against a historical baseline CSV.
Uses the Kolmogorov-Smirnov test to detect distribution drift in key metrics.

Usage:
    python analytics/monitor_performance.py \\
        --baseline monitoring/baseline.csv \\
        --new-batch results/pipeline_run_<ts>.json \\
        [--output monitoring/drift_report_<ts>.json]

Output:
    - Structured JSON drift report (to --output path or stdout)
    - Structured log lines to stdout
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Numeric metrics tested for drift
DRIFT_METRICS = [
    "ransac_inliers_D",
    "ransac_inliers_U",
    "washing_confidence_D2",
    "washing_confidence_U2",
]
KS_ALPHA = 0.05  # p-value threshold for drift detection


# =============================================================================
# JSON → DataFrame
# =============================================================================

def _safe_get(d: dict, *keys, default=None):
    """Walk nested dict keys; return default if any key is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def json_to_rows(run_json: dict) -> list[dict]:
    """Extract one row per folder from a pipeline run JSON."""
    rows = []
    run_date = run_json.get("run_date", "")
    for f in run_json.get("folders", []):
        name   = f.get("name", "")
        status = f.get("status", "")
        failed = f.get("failed_stage") or ""
        detail = f.get("detail") or {}

        s2n = detail.get("stage2n") or {}
        s3n = detail.get("stage3n") or {}

        # Stage 2N
        ps = s2n.get("pair_stats") or {}
        d_pair = ps.get("D") or {}
        u_pair = ps.get("U") or {}
        ransac_D = d_pair.get("ransac_inliers") if d_pair.get("status") == "OK" else None
        ransac_U = u_pair.get("ransac_inliers") if u_pair.get("status") == "OK" else None
        stage2n_pass = bool(s2n.get("overall_pass"))

        # Stage 3N
        pr = s3n.get("pair_results") or {}
        d2 = pr.get("D2") or {}
        u2 = pr.get("U2") or {}
        conf_D2 = d2.get("washing_confidence") if d2.get("status") == "OK" else None
        conf_U2 = u2.get("washing_confidence") if u2.get("status") == "OK" else None
        tier_D2 = d2.get("washing_tier") if d2.get("status") == "OK" else None
        tier_U2 = u2.get("washing_tier") if u2.get("status") == "OK" else None
        stage3n_pass = bool(s3n.get("overall_pass"))

        # Stage 4N
        s4n = detail.get("stage4n") or {}
        pr4 = s4n.get("pair_results") or {}
        d3 = pr4.get("D3") or {}
        u3 = pr4.get("U3") or {}
        blur_D3 = _safe_get(d3, "gate", "blur_score")
        blur_U3 = _safe_get(u3, "gate", "blur_score")

        d3_status = d3.get("status", "")
        u3_status = u3.get("status", "")
        s4n_verdicts = {d3_status, u3_status} - {"", "MISSING_IMAGES"}
        stage4n_verdict = "PASS" if "PASS" in s4n_verdicts else (
            "REVIEW" if "REVIEW" in s4n_verdicts else (
                "FAIL" if "FAIL" in s4n_verdicts else ""
            )
        )

        rows.append({
            "folder":               name,
            "run_date":             run_date,
            "status":               status,
            "failed_stage":         failed,
            "ransac_inliers_D":     ransac_D,
            "ransac_inliers_U":     ransac_U,
            "washing_confidence_D2": conf_D2,
            "washing_confidence_U2": conf_U2,
            "washing_tier_D2":      tier_D2,
            "washing_tier_U2":      tier_U2,
            "stage2n_pass":         stage2n_pass,
            "stage3n_pass":         stage3n_pass,
            "blur_score_D3":        blur_D3,
            "blur_score_U3":        blur_U3,
            "stage4n_verdict":      stage4n_verdict,
        })
    return rows


# =============================================================================
# Pass-rate helpers
# =============================================================================

def _pass_rates(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"stage2n": None, "stage3n": None, "stage4n": None}
    return {
        "stage2n": round(df["stage2n_pass"].sum() / n, 4),
        "stage3n": round(df["stage3n_pass"].sum() / n, 4),
        "stage4n": round((df["stage4n_verdict"] == "PASS").sum() / n, 4),
    }


# =============================================================================
# Main
# =============================================================================

def monitor(baseline_path: str, new_batch_path: str, output_path: str | None = None):
    # Load new batch
    with open(new_batch_path, encoding="utf-8") as f:
        run_json = json.load(f)
    new_rows = json_to_rows(run_json)
    new_df   = pd.DataFrame(new_rows)
    log.info("stage=monitor new_batch_n=%d source=%s", len(new_df), Path(new_batch_path).name)

    # Load or create baseline
    if os.path.isfile(baseline_path):
        base_df = pd.read_csv(baseline_path)
        log.info("stage=monitor baseline_n=%d source=%s", len(base_df), Path(baseline_path).name)
    else:
        base_df = pd.DataFrame(columns=new_df.columns)
        log.warning("stage=monitor baseline not found — comparing against empty baseline")

    # Pass rates
    base_rates = _pass_rates(base_df)
    new_rates  = _pass_rates(new_df)
    rate_delta = {
        k: (round(new_rates[k] - base_rates[k], 4)
            if base_rates[k] is not None and new_rates[k] is not None else None)
        for k in base_rates
    }
    for stage, delta in rate_delta.items():
        if delta is not None:
            log.info("stage=monitor %s_pass_rate_delta=%.4f", stage, delta)

    # KS tests
    ks_results = {}
    for metric in DRIFT_METRICS:
        b_vals = base_df[metric].dropna().values if metric in base_df.columns else []
        n_vals = new_df[metric].dropna().values  if metric in new_df.columns  else []
        if len(b_vals) < 5 or len(n_vals) < 5:
            ks_results[metric] = {
                "statistic": None, "p_value": None, "drift_detected": None,
                "note": "insufficient data",
            }
            continue
        stat, pval = stats.ks_2samp(b_vals, n_vals)
        drift = bool(pval < KS_ALPHA)
        ks_results[metric] = {
            "statistic": round(float(stat), 4),
            "p_value":   round(float(pval), 4),
            "drift_detected": drift,
        }
        log.info(
            "stage=monitor metric=%s ks_stat=%.4f p=%.4f drift=%s",
            metric, stat, pval, drift,
        )

    # Summary sentence
    drifted = [m for m, r in ks_results.items() if r.get("drift_detected")]
    summary_parts = []
    if drifted:
        summary_parts.append(f"DRIFT DETECTED: {', '.join(drifted)}")
    for stage in ("stage2n", "stage3n", "stage4n"):
        d = rate_delta.get(stage)
        if d is not None and abs(d) >= 0.05:
            direction = "dropped" if d < 0 else "increased"
            summary_parts.append(f"{stage} pass rate {direction} {abs(d)*100:.0f}pp")
    summary = "; ".join(summary_parts) if summary_parts else "No significant drift detected"

    report = {
        "run_date":    datetime.now().isoformat(timespec="seconds"),
        "baseline_n":  len(base_df),
        "new_batch_n": len(new_df),
        "pass_rate": {
            "baseline":  base_rates,
            "new_batch": new_rates,
            "delta":     rate_delta,
        },
        "ks_tests": ks_results,
        "summary":  summary,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info("stage=monitor report_saved=%s", output_path)
    else:
        print(json.dumps(report, indent=2))

    print(f"\nSUMMARY: {summary}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Pipeline metric drift monitor")
    parser.add_argument("--baseline",   required=True,  help="Path to baseline CSV")
    parser.add_argument("--new-batch",  required=True,  help="Path to pipeline run JSON")
    parser.add_argument("--output",     default=None,   help="Output drift report JSON path")
    args = parser.parse_args()
    monitor(args.baseline, args.new_batch, args.output)


if __name__ == "__main__":
    main()
