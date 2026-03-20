"""
Generates a self-contained HTML report from results.json.

Reads : adjusted_images/obstruction/extracted_text/results.json
Writes: adjusted_images/obstruction/extracted_text/report.html
"""

import json
import os
from pathlib import Path

RESULTS_JSON = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\obstruction\extracted_text\results.json"
OUTPUT_HTML  = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images\obstruction\extracted_text\report.html"

IMAGE_COLS = ["D1", "DL", "DR", "U1", "UL", "UR"]


def conf_style(conf: float) -> str:
    if conf >= 0.80:
        return "background:#d4edda;color:#155724;border:1px solid #c3e6cb;"   # green
    if conf >= 0.50:
        return "background:#fff3cd;color:#856404;border:1px solid #ffeeba;"   # amber
    return     "background:#f8d7da;color:#721c24;border:1px solid #f5c6cb;"   # red


def badge(text: str, conf: float) -> str:
    pct = f"{conf:.0%}"
    style = conf_style(conf)
    return (
        f'<span style="display:inline-block;margin:2px 3px 2px 0;padding:3px 8px;'
        f'border-radius:4px;font-size:0.78rem;{style}">'
        f'{text} <span style="opacity:0.7;font-size:0.72rem;">({pct})</span></span>'
    )


def cell_html(image_data: dict | None) -> str:
    if not image_data or not image_data.get("raw_lines"):
        return '<td style="color:#aaa;text-align:center;">—</td>'
    badges = "".join(badge(l["text"], l["confidence"]) for l in image_data["raw_lines"])
    return f'<td style="vertical-align:top;padding:6px 8px;">{badges}</td>'


def mh_cell(folder_data: dict) -> str:
    # Collect all MH numbers found across images in this folder
    mh_set = {v["manhole_number"] for v in folder_data.values() if v.get("manhole_number")}
    if mh_set:
        val = ", ".join(sorted(mh_set))
        return (
            f'<td style="vertical-align:top;padding:6px 8px;white-space:nowrap;">'
            f'<strong style="color:#0066cc;">{val}</strong></td>'
        )
    return '<td style="color:#aaa;text-align:center;">not found</td>'


def build_html(data: dict) -> str:
    # Summary counts
    total        = len(data)
    mh_found     = sum(
        1 for fd in data.values()
        if any(v.get("manhole_number") for v in fd.values())
    )
    mh_missing   = total - mh_found

    col_headers = "".join(
        f'<th style="{TH}">{c}</th>' for c in IMAGE_COLS
    )

    rows = []
    for folder_name in sorted(data.keys()):
        folder_data = data[folder_name]
        img_cells = "".join(
            cell_html(folder_data.get(col)) for col in IMAGE_COLS
        )
        rows.append(
            f'<tr>'
            f'<td style="vertical-align:top;padding:6px 8px;white-space:nowrap;'
            f'font-weight:600;">{folder_name}</td>'
            f'{mh_cell(folder_data)}'
            f'{img_cells}'
            f'</tr>'
        )

    rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>WRN OCR Report — Obstruction Folders</title>
<style>
  body {{font-family:Arial,sans-serif;margin:30px;color:#333;font-size:0.9rem;}}
  h1   {{font-size:1.4rem;border-bottom:3px solid #0066cc;padding-bottom:8px;}}
  .summary {{display:flex;gap:16px;margin:16px 0 24px;}}
  .card {{padding:14px 20px;border-radius:6px;min-width:120px;text-align:center;}}
  .card .num {{font-size:1.8rem;font-weight:bold;}}
  .card .lbl {{font-size:0.78rem;color:#555;margin-top:2px;}}
  .legend {{display:flex;gap:12px;margin-bottom:12px;font-size:0.8rem;}}
  .lchip {{padding:2px 10px;border-radius:4px;}}
  .wrap {{overflow-x:auto;}}
  table {{border-collapse:collapse;width:100%;min-width:900px;}}
  th,td {{border:1px solid #dee2e6;}}
  th {{background:#0066cc;color:#fff;padding:8px 10px;text-align:left;
        font-size:0.85rem;white-space:nowrap;}}
  tr:nth-child(even) {{background:#f8f9fa;}}
  tr:hover {{background:#eef4ff;}}
</style>
</head>
<body>

<h1>WRN Service Report — OCR Results (Obstruction Folders)</h1>

<div class="summary">
  <div class="card" style="background:#e7f0ff;border:1px solid #b3cdf5;">
    <div class="num">{total}</div><div class="lbl">Total Folders</div>
  </div>
  <div class="card" style="background:#d4edda;border:1px solid #c3e6cb;">
    <div class="num">{mh_found}</div><div class="lbl">MH Number Found</div>
  </div>
  <div class="card" style="background:#f8d7da;border:1px solid #f5c6cb;">
    <div class="num">{mh_missing}</div><div class="lbl">MH Number Missing</div>
  </div>
</div>

<div class="legend">
  <strong>Confidence:</strong>
  <span class="lchip" style="background:#d4edda;color:#155724;border:1px solid #c3e6cb;">≥ 80% — High</span>
  <span class="lchip" style="background:#fff3cd;color:#856404;border:1px solid #ffeeba;">50–79% — Medium</span>
  <span class="lchip" style="background:#f8d7da;color:#721c24;border:1px solid #f5c6cb;">&lt; 50% — Low</span>
</div>

<div class="wrap">
<table>
  <thead>
    <tr>
      <th style="{TH}">Folder</th>
      <th style="{TH}">MH Number</th>
      {col_headers}
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
</div>

</body>
</html>"""


TH = "white-space:nowrap;"

if __name__ == "__main__":
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    html = build_html(data)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to:\n  {OUTPUT_HTML}")
    print(f"Open it in any browser to view.")
