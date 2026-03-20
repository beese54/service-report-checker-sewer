# WRN Service Report Checker — Working Principles

These principles apply to every Claude Code session in this repository.

---

## Working Principles

1. **Plan-Code Default** — Enter plan mode for any task requiring 3 or more steps. Stop and re-plan if something goes wrong mid-task.

2. **Subagent Strategy** — Use subagents for research and exploration. One task per subagent. Keep the main context clean. Delegate, don't duplicate.

3. **Self-Improvement Loop** — After any correction or unexpected outcome, capture the lesson in `tasks/lessons.md`. Review `tasks/lessons.md` at the start of each session.

4. **Verification Before Done** — Never mark a task complete without proving it works. Run tests, check logs, or verify output. If tests don't exist, add a smoke test.

5. **Demand Elegance (Balanced)** — Pause on non-trivial changes to ask: "Is there a more elegant way?" Skip this on simple one-line fixes.

6. **Autonomous Bug Fixing** — Given a bug report, fix it. No hand-holding needed. Diagnose from logs, tracebacks, and test output directly.

7. **Task Management** — Write the plan to `tasks/todo.md` with checkable items. Mark items done as work proceeds. Capture lessons in `tasks/lessons.md`.

---

## Project Overview

Automated QA pipeline for manhole CCTV service reports. A browser UI accepts a ZIP of job folders; a multi-stage CV pipeline returns per-folder verdicts.

### Pipeline Stages

| Stage | File | Images | What it checks |
|-------|------|--------|----------------|
| Level 1 — Classification | `pipeline/sort_folders.py` | All | D1+D4 or U1+U4 present? |
| Stage 2N — SIFT Alignment | `pipeline/stage2n_sift.py` | D1/D4, U1/U4 | Same pipe location |
| Stage 3N — Washing | `pipeline/stage3n_washing.py` | D2/D5, U2/U5 | Pipe was cleaned |
| Stage 4N — Geometry | `pipeline/stage4n_geometry.py` | D3/D6, U3/U6 | Grease, water, blur |
| OCR (obstruction only) | `pipeline/stage1_ocr.py` | D1, U1, DR/DL, UR/UL | Blackboard text |

### Key Files

- `pipeline.py` — main orchestrator (`run_pipeline`)
- `pipeline_config.py` — all thresholds, env-var loading
- `app/main.py` — FastAPI server (all endpoints)
- `app/checker.py` — Level 1 classification logic

### Environment Variables

Override any threshold via env var — see `pipeline_config.py` for the full list.
Key vars: `APP_PASSWORD`, `MAX_UPLOAD_MB`, `OUTPUT_DIR`, `PORT`, `DISABLE_STAGE4N`, `SKIP_EXCEL_REPORT`.

### Dependencies

PaddleOCR 2.7.3 must be installed with `--no-deps` due to an overly strict opencv pin:
```bash
pip install paddleocr==2.7.3 --no-deps
pip install -r requirements-pipeline.txt
```
Pin `numpy==1.26.4` — PaddlePaddle 2.6.2 breaks on NumPy 2.x.
