"""
generate_report_stage3n.py — Stage 3N Washing Confidence HTML Report
=====================================================================
Reads the full pipeline run JSON and produces a per-folder HTML table
showing D2/D5 and U2/U5 washing confidence results.

Usage:
    python analytics/generate_report_stage3n.py [path/to/pipeline_run_*.json]

Output:
    pipeline_output/full_run_2/stage3n_report.html
"""

import json
import os
import sys

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker"
DEFAULT_JSON = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "pipeline_run_20260306_080308.json"
)
OUTPUT_HTML = os.path.join(
    BASE_DIR, "pipeline_output", "full_run_2",
    "stage3n_report.html"
)

PASS_TIER = "HIGH"   # only HIGH passes


# =============================================================================
# HTML helpers
# =============================================================================

def tier_chip(tier):
    if tier is None:
        return '<span class="chip chip-na">—</span>'
    colours = {
        "HIGH":   "chip-high",
        "MEDIUM": "chip-med",
        "LOW":    "chip-low",
        "FAILED": "chip-fail",
    }
    cls = colours.get(tier, "chip-na")
    return f'<span class="chip {cls}">{tier}</span>'


def conf_chip(conf):
    if conf is None:
        return '<span style="color:#aaa;">—</span>'
    pct = conf * 100
    if pct >= 65:
        cls = "chip-high"
    elif pct >= 35:
        cls = "chip-med"
    else:
        cls = "chip-low"
    return f'<span class="chip {cls}">{pct:.0f}%</span>'


def pass_badge(passed):
    if passed is True:
        return '<span class="badge badge-pass">PASS</span>'
    if passed is False:
        return '<span class="badge badge-fail">FAIL</span>'
    return '<span class="badge badge-na">—</span>'


def pipeline_badge(status):
    cls_map = {
        "ACCEPTED":             "badge-pass",
        "NEEDS_REVIEW":         "badge-rev",
        "REJECTED":             "badge-fail",
        "OBSTRUCTION_PROCESSED":"badge-obs",
    }
    cls = cls_map.get(status, "badge-na")
    return f'<span class="badge {cls}">{status}</span>'


def signal_flags(metrics):
    """Return small coloured chips for the key signals."""
    if not metrics:
        return "—"
    chips = []
    kpr = metrics.get("kp_ratio", 0)
    chips.append(
        f'<span class="sig {"sig-y" if kpr > 1.0 else "sig-n"}">kp {kpr:.1f}x</span>'
    )
    ei = metrics.get("entropy_increase", 0)
    chips.append(
        f'<span class="sig {"sig-y" if ei >= 0.3 else "sig-n"}">H+{ei:.2f}</span>'
    )
    edg = metrics.get("edge_increase_pct", 0)
    chips.append(
        f'<span class="sig {"sig-y" if edg > 0 else "sig-n"}">edge+{edg:.0f}%</span>'
    )
    lap = metrics.get("lap_increase_pct", 0)
    chips.append(
        f'<span class="sig {"sig-y" if lap > 0 else "sig-n"}">lap+{lap:.0f}%</span>'
    )
    return "".join(chips)


# =============================================================================
# Data extraction
# =============================================================================

def build_rows(folders):
    rows = []
    for folder in sorted(folders, key=lambda f: f.get("name", "")):
        detail  = folder.get("detail") or {}
        s3n     = detail.get("stage3n") or {}
        if not s3n:
            continue

        pair_results = s3n.get("pair_results") or {}
        overall_pass = s3n.get("overall_pass")

        d2 = pair_results.get("D2") or {}
        u2 = pair_results.get("U2") or {}

        rows.append({
            "name":          folder.get("name", ""),
            "pipeline":      folder.get("status", ""),
            "overall_pass":  overall_pass,
            "d2_status":     d2.get("status"),
            "d2_tier":       d2.get("washing_tier"),
            "d2_conf":       d2.get("washing_confidence"),
            "d2_metrics":    d2.get("metrics"),
            "d2_inliers":    (d2.get("sift_stats") or {}).get("ransac_inliers"),
            "u2_status":     u2.get("status"),
            "u2_tier":       u2.get("washing_tier"),
            "u2_conf":       u2.get("washing_confidence"),
            "u2_metrics":    u2.get("metrics"),
            "u2_inliers":    (u2.get("sift_stats") or {}).get("ransac_inliers"),
        })
    return rows


def summary_counts(rows):
    total    = len(rows)
    passed   = sum(1 for r in rows if r["overall_pass"] is True)
    failed   = sum(1 for r in rows if r["overall_pass"] is False)
    # pair-level tier counts (across D2 + U2)
    from collections import Counter
    tier_c = Counter()
    for r in rows:
        if r["d2_tier"]:
            tier_c[r["d2_tier"]] += 1
        if r["u2_tier"]:
            tier_c[r["u2_tier"]] += 1
    return total, passed, failed, tier_c


# =============================================================================
# HTML assembly
# =============================================================================

CSS = """
<style>
  body{font-family:Arial,sans-serif;margin:28px;color:#222;font-size:0.88rem;background:#f5f7fa;}
  h1{font-size:1.35rem;margin-bottom:4px;color:#1a237e;}
  .sub{color:#555;font-size:0.82rem;margin-bottom:20px;}
  .summary{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px;}
  .card{padding:12px 20px;border-radius:6px;min-width:110px;text-align:center;border:1px solid #ddd;background:#fff;}
  .card .num{font-size:1.8rem;font-weight:700;}
  .card .lbl{font-size:0.75rem;color:#555;margin-top:2px;}
  .legend{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;font-size:0.78rem;align-items:center;}
  .wrap{overflow-x:auto;}
  table{border-collapse:collapse;width:100%;min-width:1100px;background:#fff;}
  th,td{border:1px solid #d0d7e3;padding:5px 8px;}
  th{background:#1a237e;color:#fff;font-size:0.82rem;white-space:nowrap;}
  tr:nth-child(even){background:#f0f4ff;}
  tr:hover{background:#dce8ff;}
  .chip{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.76rem;font-weight:600;white-space:nowrap;}
  .chip-high{background:#d4edda;color:#1b5e20;border:1px solid #a5d6a7;}
  .chip-med {background:#fff3cd;color:#7c5e00;border:1px solid #ffe082;}
  .chip-low {background:#fde8e8;color:#b71c1c;border:1px solid #ef9a9a;}
  .chip-fail{background:#fce4ec;color:#880e4f;border:1px solid #f48fb1;}
  .chip-na  {background:#eeeeee;color:#757575;border:1px solid #bdbdbd;}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:700;white-space:nowrap;}
  .badge-pass{background:#1b5e20;color:#fff;}
  .badge-fail{background:#b71c1c;color:#fff;}
  .badge-rev {background:#e65100;color:#fff;}
  .badge-obs {background:#4a148c;color:#fff;}
  .badge-na  {background:#9e9e9e;color:#fff;}
  .sig{display:inline-block;padding:1px 5px;border-radius:3px;font-size:0.70rem;margin:1px;white-space:nowrap;}
  .sig-y{background:#c8e6c9;color:#1b5e20;}
  .sig-n{background:#ffcdd2;color:#b71c1c;}
  .folder-name{font-weight:600;white-space:nowrap;}
  .num-cell{text-align:right;font-variant-numeric:tabular-nums;}
</style>
"""


def build_html(folders, run_date):
    rows   = build_rows(folders)
    total, passed, failed, tier_c = summary_counts(rows)

    # Sort: failed first (to aid review), then by folder name
    rows.sort(key=lambda r: (r["overall_pass"] is True, r["name"]))

    tr_parts = []
    for r in rows:
        d2_na = r["d2_status"] not in ("OK",)
        u2_na = r["u2_status"] not in ("OK",)

        d2_tier_html = tier_chip(r["d2_tier"]) if not d2_na else '<span class="chip chip-na">N/A</span>'
        u2_tier_html = tier_chip(r["u2_tier"]) if not u2_na else '<span class="chip chip-na">N/A</span>'

        tr_parts.append(f"""
  <tr>
    <td class="folder-name">{r["name"]}</td>
    <td>{pipeline_badge(r["pipeline"])}</td>
    <td style="text-align:center;">{pass_badge(r["overall_pass"])}</td>
    <td style="text-align:center;">{d2_tier_html}</td>
    <td class="num-cell">{conf_chip(r["d2_conf"]) if not d2_na else "—"}</td>
    <td class="num-cell">{r["d2_inliers"] if r["d2_inliers"] is not None else "—"}</td>
    <td>{signal_flags(r["d2_metrics"]) if not d2_na else "—"}</td>
    <td style="text-align:center;">{u2_tier_html}</td>
    <td class="num-cell">{conf_chip(r["u2_conf"]) if not u2_na else "—"}</td>
    <td class="num-cell">{r["u2_inliers"] if r["u2_inliers"] is not None else "—"}</td>
    <td>{signal_flags(r["u2_metrics"]) if not u2_na else "—"}</td>
  </tr>""")

    tbody = "\n".join(tr_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Stage 3N — Washing Confidence Report</title>
{CSS}
</head>
<body>

<h1>WRN Service Report — Stage 3N: Washing Confidence (D2/D5 + U2/U5)</h1>
<div class="sub">Run date: {run_date} &nbsp;|&nbsp; Pass tier: <strong>{PASS_TIER}</strong>
 &nbsp;|&nbsp; Folders with Stage 3N data: {total}</div>

<div class="summary">
  <div class="card" style="border-color:#3949ab;">
    <div class="num" style="color:#3949ab;">{total}</div>
    <div class="lbl">Folders processed</div>
  </div>
  <div class="card" style="border-color:#2e7d32;">
    <div class="num" style="color:#2e7d32;">{passed}</div>
    <div class="lbl">Overall PASS</div>
  </div>
  <div class="card" style="border-color:#c62828;">
    <div class="num" style="color:#c62828;">{failed}</div>
    <div class="lbl">Overall FAIL</div>
  </div>
  <div class="card" style="border-color:#1b5e20;">
    <div class="num" style="color:#1b5e20;">{tier_c.get("HIGH", 0)}</div>
    <div class="lbl">HIGH tier pairs</div>
  </div>
  <div class="card" style="border-color:#f57f17;">
    <div class="num" style="color:#f57f17;">{tier_c.get("MEDIUM", 0)}</div>
    <div class="lbl">MEDIUM tier pairs</div>
  </div>
  <div class="card" style="border-color:#b71c1c;">
    <div class="num" style="color:#b71c1c;">{tier_c.get("LOW", 0)}</div>
    <div class="lbl">LOW tier pairs</div>
  </div>
</div>

<div class="legend">
  <strong>Tier:</strong>
  <span class="chip chip-high">HIGH (&ge;65%)</span>
  <span class="chip chip-med">MEDIUM (35–64%)</span>
  <span class="chip chip-low">LOW (&lt;35%)</span>
  &nbsp;|&nbsp;
  <strong>Signals:</strong> kp = keypoint ratio &nbsp; H = entropy increase &nbsp;
  edge/lap = texture change &nbsp; <span class="sig sig-y">green = fired</span>
  <span class="sig sig-n">red = not fired</span>
</div>

<div class="wrap">
<table>
  <thead>
    <tr>
      <th rowspan="2">Folder</th>
      <th rowspan="2">Pipeline</th>
      <th rowspan="2">S3N Pass</th>
      <th colspan="4" style="text-align:center;background:#0d47a1;">D2 / D5 (Downstream)</th>
      <th colspan="4" style="text-align:center;background:#1a237e;">U2 / U5 (Upstream)</th>
    </tr>
    <tr>
      <th>Tier</th><th>Conf</th><th>Inliers</th><th>Signals</th>
      <th>Tier</th><th>Conf</th><th>Inliers</th><th>Signals</th>
    </tr>
  </thead>
  <tbody>
{tbody}
  </tbody>
</table>
</div>

<p style="margin-top:16px;font-size:0.76rem;color:#888;">
  Sorted: FAIL folders first, then by folder name.
  Signals: kp&gt;1.0x (more keypoints after), entropy&nbsp;increase&nbsp;&ge;0.3&nbsp;nats,
  edge density increase &gt;0%, Laplacian variance increase &gt;0%.
</p>
</body>
</html>"""


# =============================================================================
# Entry point
# =============================================================================

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    print(f"Reading: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    folders  = data.get("folders", [])
    run_date = data.get("run_date", "")

    html = build_html(folders, run_date)

    out = OUTPUT_HTML
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    rows_written = sum(
        1 for r in folders
        if (r.get("detail") or {}).get("stage3n")
    )
    print(f"Rows written: {rows_written}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
