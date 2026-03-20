"""
generate_report_stage4n.py — Stage 4N Geometry-First HTML Report
=================================================================
Reads the full pipeline run JSON and produces a per-folder HTML table
showing D3/D6 and U3/U6 geometry-first results with all signal details.

Usage:
    python analytics/generate_report_stage4n.py [path/to/pipeline_run_*.json]

Output:
    pipeline_output/full_run_2/stage4n_report.html
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
    "stage4n_report.html"
)

# ── Thresholds (informational) ───────────────────────────────────────────────
THRESH_BLUR     = 35.0
THRESH_SSIM     = 0.05
THRESH_INLIER   = 0.05
THRESH_GREASE   = 2.0      # pct; above = flagged
THRESH_ENTROPY  = 0.3      # nats
THRESH_WATER    = 3.0      # combined %


# =============================================================================
# HTML helpers
# =============================================================================

def status_chip(status):
    cls_map = {
        "PASS":             "chip-pass",
        "REVIEW":           "chip-rev",
        "FAIL":             "chip-fail",
        "GATE_REJECTED":    "chip-gate",
        "ALIGNMENT_FAILED": "chip-align",
        "MISSING_IMAGES":   "chip-na",
    }
    cls = cls_map.get(status, "chip-na")
    return f'<span class="chip {cls}">{status}</span>' if status else "—"


def score_chip(score):
    if score is None:
        return "—"
    if score >= 3:
        cls = "chip-pass"
    elif score == 2:
        cls = "chip-rev"
    else:
        cls = "chip-fail"
    return f'<span class="chip {cls}">{score}/5</span>'


def verdict_badge(verdict):
    cls_map = {
        "ACCEPTED":     "badge-pass",
        "NEEDS_REVIEW": "badge-rev",
        "REJECTED":     "badge-fail",
    }
    cls = cls_map.get(verdict, "badge-na")
    return f'<span class="badge {cls}">{verdict}</span>' if verdict else "—"


def pipeline_badge(status):
    cls_map = {
        "ACCEPTED":              "badge-pass",
        "NEEDS_REVIEW":          "badge-rev",
        "REJECTED":              "badge-fail",
        "OBSTRUCTION_PROCESSED": "badge-obs",
    }
    cls = cls_map.get(status, "badge-na")
    return f'<span class="badge {cls}">{status}</span>'


def bool_sig(label, fired):
    cls = "sig-y" if fired else "sig-n"
    return f'<span class="sig {cls}">{label}</span>'


def num_sig(label, value, threshold, above_passes=True, fmt="{:.1f}"):
    if value is None:
        return f'<span class="sig sig-na">{label} —</span>'
    fired = (value >= threshold) if above_passes else (value < threshold)
    cls   = "sig-y" if fired else "sig-n"
    display = fmt.format(value)
    return f'<span class="sig {cls}">{label} {display}</span>'


def pair_signals(pair):
    """Build the signals cell for one pair result dict."""
    if not pair:
        return "—"

    gate     = pair.get("gate") or {}
    sift     = pair.get("sift_stats") or {}
    grease   = pair.get("grease") or {}
    texture  = pair.get("texture") or {}
    water    = pair.get("water") or {}

    parts = []

    # Gate signals
    parts.append(bool_sig("blur ok",   gate.get("acceptable", False)))
    parts.append(bool_sig("circle",    gate.get("circle_found", False)))

    # SIFT
    parts.append(num_sig("inliers",    sift.get("ransac_inliers"),   10,  above_passes=True,  fmt="{:.0f}"))
    parts.append(num_sig("ssim",       sift.get("ssim_score"),        THRESH_SSIM,  fmt="{:.3f}"))

    # Grease (fired = flagged = bad)
    gp = grease.get("grease_pct")
    parts.append(num_sig("grease%",    gp,  THRESH_GREASE,  above_passes=False, fmt="{:.1f}%"))

    # Texture
    parts.append(bool_sig("entropy+",  texture.get("confirmed", False)))

    # Water
    parts.append(bool_sig("water",     water.get("water_detected", False)))

    return "".join(parts)


def gate_detail(gate):
    if not gate:
        return "—"
    blur = gate.get("blur_score")
    rej  = gate.get("reject_reasons") or []
    ok   = gate.get("acceptable", False)
    blur_str = f"blur={blur:.0f}" if blur is not None else "blur=?"
    rej_str  = "; ".join(rej) if rej else "none"
    cls = "sig-y" if ok else "sig-n"
    return f'<span class="sig {cls}">{blur_str}</span> <span style="font-size:0.7rem;color:#777;">rej:{rej_str}</span>'


# =============================================================================
# Data extraction
# =============================================================================

def build_rows(folders):
    rows = []
    for folder in sorted(folders, key=lambda f: f.get("name", "")):
        detail = folder.get("detail") or {}
        s4n    = detail.get("stage4n") or {}
        if not s4n:
            continue

        pair_results = s4n.get("pair_results") or {}
        verdict      = s4n.get("folder_verdict")

        d3 = pair_results.get("D3") or {}
        u3 = pair_results.get("U3") or {}

        rows.append({
            "name":     folder.get("name", ""),
            "pipeline": folder.get("status", ""),
            "verdict":  verdict,
            "d3":       d3,
            "u3":       u3,
        })
    return rows


def summary_counts(rows):
    from collections import Counter
    verdict_c = Counter(r["verdict"] for r in rows)
    status_c  = Counter()
    for r in rows:
        for prefix in ("d3", "u3"):
            s = r[prefix].get("status")
            if s:
                status_c[s] += 1
    return verdict_c, status_c


# =============================================================================
# HTML assembly
# =============================================================================

CSS = """
<style>
  body{font-family:Arial,sans-serif;margin:28px;color:#222;font-size:0.85rem;background:#f5f7fa;}
  h1{font-size:1.3rem;margin-bottom:4px;color:#1a237e;}
  .sub{color:#555;font-size:0.80rem;margin-bottom:20px;}
  .summary{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;}
  .card{padding:10px 16px;border-radius:6px;min-width:100px;text-align:center;border:1px solid #ddd;background:#fff;}
  .card .num{font-size:1.7rem;font-weight:700;}
  .card .lbl{font-size:0.72rem;color:#555;margin-top:2px;}
  .legend{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;font-size:0.77rem;align-items:center;}
  .wrap{overflow-x:auto;}
  table{border-collapse:collapse;width:100%;min-width:1300px;background:#fff;}
  th,td{border:1px solid #cfd8dc;padding:4px 7px;}
  th{background:#1a237e;color:#fff;font-size:0.79rem;white-space:nowrap;}
  tr:nth-child(even){background:#f0f4ff;}
  tr:hover{background:#dce8ff;}
  .chip{display:inline-block;padding:2px 7px;border-radius:4px;font-size:0.74rem;font-weight:600;white-space:nowrap;}
  .chip-pass {background:#d4edda;color:#1b5e20;border:1px solid #a5d6a7;}
  .chip-rev  {background:#fff3cd;color:#7c5e00;border:1px solid #ffe082;}
  .chip-fail {background:#fde8e8;color:#b71c1c;border:1px solid #ef9a9a;}
  .chip-gate {background:#ede7f6;color:#4527a0;border:1px solid #b39ddb;}
  .chip-align{background:#e3f2fd;color:#0d47a1;border:1px solid #90caf9;}
  .chip-na   {background:#eeeeee;color:#757575;border:1px solid #bdbdbd;}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:700;white-space:nowrap;}
  .badge-pass{background:#1b5e20;color:#fff;}
  .badge-fail{background:#b71c1c;color:#fff;}
  .badge-rev {background:#e65100;color:#fff;}
  .badge-obs {background:#4a148c;color:#fff;}
  .badge-na  {background:#9e9e9e;color:#fff;}
  .sig{display:inline-block;padding:1px 5px;border-radius:3px;font-size:0.68rem;margin:1px;white-space:nowrap;}
  .sig-y  {background:#c8e6c9;color:#1b5e20;}
  .sig-n  {background:#ffcdd2;color:#b71c1c;}
  .sig-na {background:#eeeeee;color:#757575;}
  .folder-name{font-weight:600;white-space:nowrap;}
  .num-cell{text-align:right;font-variant-numeric:tabular-nums;}
</style>
"""


def build_html(folders, run_date):
    rows = build_rows(folders)
    verdict_c, status_c = summary_counts(rows)

    # Sort: REJECTED / REVIEW first to aid review
    priority = {"REJECTED": 0, "NEEDS_REVIEW": 1, "ACCEPTED": 2}
    rows.sort(key=lambda r: (priority.get(r["pipeline"], 3), r["name"]))

    tr_parts = []
    for r in rows:
        d3 = r["d3"]
        u3 = r["u3"]

        d3_status = d3.get("status")
        u3_status = u3.get("status")

        def pair_row(p):
            return (
                f'<td style="text-align:center;">{status_chip(p.get("status"))}</td>'
                f'<td style="text-align:center;">{score_chip(p.get("score"))}</td>'
                f'<td style="font-size:0.72rem;">{gate_detail(p.get("gate"))}</td>'
                f'<td style="font-size:0.72rem;">{pair_signals(p) if p.get("status") not in (None, "GATE_REJECTED", "MISSING_IMAGES") else "—"}</td>'
            )

        tr_parts.append(f"""
  <tr>
    <td class="folder-name">{r["name"]}</td>
    <td>{pipeline_badge(r["pipeline"])}</td>
    <td style="text-align:center;">{verdict_badge(r["verdict"])}</td>
    {pair_row(d3)}
    {pair_row(u3)}
  </tr>""")

    tbody = "\n".join(tr_parts)

    total     = len(rows)
    accepted  = verdict_c.get("ACCEPTED", 0)
    rejected  = verdict_c.get("REJECTED", 0)
    review    = verdict_c.get("NEEDS_REVIEW", 0)
    pair_pass = status_c.get("PASS", 0)
    pair_rev  = status_c.get("REVIEW", 0)
    pair_fail = status_c.get("FAIL", 0)
    gate_rej  = status_c.get("GATE_REJECTED", 0)
    align_fail= status_c.get("ALIGNMENT_FAILED", 0)
    missing   = status_c.get("MISSING_IMAGES", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Stage 4N — Geometry-First Report</title>
{CSS}
</head>
<body>

<h1>WRN Service Report — Stage 4N: Geometry-First (D3/D6 + U3/U6)</h1>
<div class="sub">Run date: {run_date} &nbsp;|&nbsp; Pass score: &ge;3/5
 &nbsp;|&nbsp; Folders with Stage 4N data: {total}</div>

<div class="summary">
  <div class="card" style="border-color:#3949ab;">
    <div class="num" style="color:#3949ab;">{total}</div>
    <div class="lbl">Folders</div>
  </div>
  <div class="card" style="border-color:#2e7d32;">
    <div class="num" style="color:#2e7d32;">{accepted}</div>
    <div class="lbl">Accepted</div>
  </div>
  <div class="card" style="border-color:#e65100;">
    <div class="num" style="color:#e65100;">{review}</div>
    <div class="lbl">Needs Review</div>
  </div>
  <div class="card" style="border-color:#c62828;">
    <div class="num" style="color:#c62828;">{rejected}</div>
    <div class="lbl">Rejected</div>
  </div>
  <div class="card" style="border-color:#2e7d32;background:#f1f8e9;">
    <div class="num" style="color:#2e7d32;">{pair_pass}</div>
    <div class="lbl">Pairs PASS</div>
  </div>
  <div class="card" style="border-color:#f57f17;background:#fffde7;">
    <div class="num" style="color:#f57f17;">{pair_rev}</div>
    <div class="lbl">Pairs REVIEW</div>
  </div>
  <div class="card" style="border-color:#b71c1c;background:#fce4ec;">
    <div class="num" style="color:#b71c1c;">{pair_fail}</div>
    <div class="lbl">Pairs FAIL</div>
  </div>
  <div class="card" style="border-color:#6a1b9a;">
    <div class="num" style="color:#6a1b9a;">{gate_rej}</div>
    <div class="lbl">Gate Rejected</div>
  </div>
  <div class="card" style="border-color:#0d47a1;">
    <div class="num" style="color:#0d47a1;">{align_fail}</div>
    <div class="lbl">Align Failed</div>
  </div>
  <div class="card" style="border-color:#9e9e9e;">
    <div class="num" style="color:#9e9e9e;">{missing}</div>
    <div class="lbl">Missing Images</div>
  </div>
</div>

<div class="legend">
  <strong>Score:</strong>
  <span class="chip chip-pass">PASS (&ge;3/5)</span>
  <span class="chip chip-rev">REVIEW (2/5)</span>
  <span class="chip chip-fail">FAIL (&lt;2/5)</span>
  &nbsp;|&nbsp;
  <strong>Signals (green=fired, red=not):</strong>
  blur ok &nbsp; circle detected &nbsp; inliers&ge;10 &nbsp; ssim&ge;0.05 &nbsp;
  grease&lt;2% &nbsp; entropy confirmed &nbsp; water detected
</div>

<div class="wrap">
<table>
  <thead>
    <tr>
      <th rowspan="2">Folder</th>
      <th rowspan="2">Pipeline</th>
      <th rowspan="2">Verdict</th>
      <th colspan="4" style="text-align:center;background:#0d47a1;">D3 / D6 (Downstream)</th>
      <th colspan="4" style="text-align:center;background:#1a237e;">U3 / U6 (Upstream)</th>
    </tr>
    <tr>
      <th>Status</th><th>Score</th><th>Gate</th><th>Signals</th>
      <th>Status</th><th>Score</th><th>Gate</th><th>Signals</th>
    </tr>
  </thead>
  <tbody>
{tbody}
  </tbody>
</table>
</div>

<p style="margin-top:16px;font-size:0.74rem;color:#888;">
  Sorted: Rejected/Review folders first, then by folder name.
  Signals per pair: blur gate pass | pipe circle found | RANSAC inliers &ge;10 |
  SSIM &ge;0.05 | grease &lt;2% | entropy confirmed | water detected.
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
        if (r.get("detail") or {}).get("stage4n")
    )
    print(f"Rows written: {rows_written}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
