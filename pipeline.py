"""
WRN Service Report Checker — Full Pipeline Orchestrator
========================================================

Takes a directory of raw job folders, runs all five analysis stages and
returns a structured results dict.

Usage (CLI):
    python pipeline.py --input ./my_job_folders --output ./results
    python pipeline.py --input ./my_job_folders/1159097 --single --output ./results

Usage (as library):
    from pipeline import run_pipeline
    results = run_pipeline(input_dir, output_dir, progress_cb=my_callback)
"""

import argparse
import ctypes
import gc
import json
import os
import sys
from datetime import datetime


def _rss_mb() -> str:
    """Return current process RSS in MB (Linux /proc; returns '?' elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return f"{kb // 1024} MB"
    except Exception:
        pass
    return "?"


def _free_mem():
    """gc.collect() + malloc_trim to return freed pages to the OS on Linux."""
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stage runners — lazy imports so heavy deps only load when needed
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage2n(folder_path, folder_name, no_obs_output_dir, detector, progress_cb,
                 ratio_threshold_override=None):
    """Stage 2N: D1/D4 + U1/U4 SIFT alignment."""
    from pipeline.stage2n_sift import run_sift_on_folder
    return run_sift_on_folder(
        folder_path, folder_name, no_obs_output_dir,
        detector=detector, progress_cb=progress_cb,
        ratio_threshold=ratio_threshold_override,
    )


def _run_stage3n(folder_path, folder_name, no_obs_output_dir, progress_cb):
    """Stage 3N: D2/D5 + U2/U5 washing confidence."""
    from pipeline.stage3n_washing import run_d2d5_u2u5_on_folder
    return run_d2d5_u2u5_on_folder(
        folder_path, folder_name, no_obs_output_dir,
        progress_cb=progress_cb,
    )


def _run_stage4n(folder_path, folder_name, no_obs_output_dir, seg_model, progress_cb):
    """Stage 4N: D3/D6 + U3/U6 geometry-first pipeline."""
    from pipeline.stage4n_geometry import run_d3d6_u3u6_on_folder
    return run_d3d6_u3u6_on_folder(
        folder_path, folder_name, no_obs_output_dir,
        seg_model=seg_model, progress_cb=progress_cb,
    )


def _run_ocr(folder_path, folder_name, obs_output_dir, progress_cb):
    """Obstruction: OCR extraction."""
    from pipeline.stage1_ocr import run_ocr_on_single_folder
    return run_ocr_on_single_folder(
        folder_path, folder_name, obs_output_dir,
        progress_cb=progress_cb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detector / seg-model loader (optional — no error if models absent)
# ─────────────────────────────────────────────────────────────────────────────

_DETECTOR_LOADED = False
_detector        = None

def _get_detector():
    global _DETECTOR_LOADED, _detector
    if _DETECTOR_LOADED:
        return _detector
    _DETECTOR_LOADED = True
    try:
        from pipeline.stage2n_sift import load_detector
        _detector = load_detector()
    except Exception:
        _detector = None
    return _detector


_SEG_MODEL_LOADED = False
_seg_model        = None

def _get_seg_model():
    global _SEG_MODEL_LOADED, _seg_model
    if _SEG_MODEL_LOADED:
        return _seg_model
    _SEG_MODEL_LOADED = True
    try:
        from pipeline.stage4n_geometry import _load_grease_seg_model
        _seg_model = _load_grease_seg_model()
    except Exception:
        _seg_model = None
    return _seg_model


def _unload_models():
    """Release YOLO model references so memory can be reclaimed before OCR phase."""
    global _detector, _seg_model
    _detector  = None
    _seg_model = None
    _free_mem()


# ─────────────────────────────────────────────────────────────────────────────
# OCR engine init — only once
# ─────────────────────────────────────────────────────────────────────────────

_OCR_READY = False

def _ensure_ocr():
    global _OCR_READY
    if _OCR_READY:
        return
    _OCR_READY = True
    # Importing the module initialises the global PaddleOCR engine
    try:
        import pipeline.stage1_ocr  # noqa: F401
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Classify folders (no filesystem moves — pipeline owns the output layout)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_folders(input_dir):
    """
    Return lists of folder names classified as no_obstruction / obstruction.
    Handles two ZIP structures:

    Unsorted (raw job folders at root):
        input_dir/1159097/D1.jpg ...
        input_dir/146915/D1.jpg ...

    Pre-sorted (already split into subdirs):
        input_dir/no_obstruction/1159097/D1.jpg ...
        input_dir/obstruction/146915/D1.jpg ...
    """
    no_obs_dir = os.path.join(input_dir, "no_obstruction")
    obs_dir    = os.path.join(input_dir, "obstruction")

    if os.path.isdir(no_obs_dir) or os.path.isdir(obs_dir):
        # Pre-sorted structure — read subdirectory names directly
        result = {"no_obstruction": [], "obstruction": []}
        if os.path.isdir(no_obs_dir):
            result["no_obstruction"] = sorted(
                e.name for e in os.scandir(no_obs_dir) if e.is_dir()
            )
        if os.path.isdir(obs_dir):
            result["obstruction"] = sorted(
                e.name for e in os.scandir(obs_dir) if e.is_dir()
            )
        return result

    # Unsorted — classify each subfolder by image presence
    from pipeline.sort_folders import classify_only
    return classify_only(input_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder processors
# ─────────────────────────────────────────────────────────────────────────────

def _process_no_obstruction_folder(
    folder_path, folder_name, no_obs_output_dir,
    detector, seg_model, progress_cb,
):
    """
    Run stages 2N → 3N → 4N for one no-obstruction folder.

    Returns a result dict ready to be embedded in the final report.
    """
    result = {
        "name":         folder_name,
        "type":         "no_obstruction",
        "status":       None,
        "failed_stage": None,
        "detail":       {},
    }

    # ── Stage 2N ─────────────────────────────────────────────────────────────
    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 2N starting  mem={_rss_mb()}")
    try:
        s2 = _run_stage2n(folder_path, folder_name, no_obs_output_dir, detector, progress_cb)
    except Exception as exc:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "D1D4_U1U4"
        result["detail"]["stage2n_error"] = str(exc)
        if progress_cb:
            progress_cb(f"{folder_name}  REJECTED  stage=D1D4_U1U4  error={exc}")
        return result

    result["detail"]["stage2n"] = s2
    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 2N done  mem={_rss_mb()}")
    if not s2["overall_pass"]:
        # Fallback: retry with more permissive Lowe's ratio before rejecting
        from pipeline_config import RATIO_THRESHOLD_FALLBACK
        if progress_cb:
            progress_cb(f"{folder_name}  Stage 2N failed -- retrying with fallback ratio {RATIO_THRESHOLD_FALLBACK} ...")
        try:
            s2_fallback = _run_stage2n(
                folder_path, folder_name, no_obs_output_dir,
                detector, progress_cb,
                ratio_threshold_override=RATIO_THRESHOLD_FALLBACK,
            )
            if s2_fallback["overall_pass"]:
                s2_fallback["fallback_used"] = True
                s2 = s2_fallback
                result["detail"]["stage2n"] = s2
            else:
                result["status"]       = "REJECTED"
                result["failed_stage"] = "D1D4_U1U4"
                if progress_cb:
                    progress_cb(f"{folder_name}  REJECTED  stage=D1D4_U1U4  (fallback also failed)")
                return result
        except Exception:
            result["status"]       = "REJECTED"
            result["failed_stage"] = "D1D4_U1U4"
            return result

    # ── Stage 3N ─────────────────────────────────────────────────────────────
    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 3N starting  mem={_rss_mb()}")
    try:
        s3 = _run_stage3n(folder_path, folder_name, no_obs_output_dir, progress_cb)
    except Exception as exc:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "D2D5_U2U5"
        result["detail"]["stage3n_error"] = str(exc)
        if progress_cb:
            progress_cb(f"{folder_name}  REJECTED  stage=D2D5_U2U5  error={exc}")
        return result

    result["detail"]["stage3n"] = s3
    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 3N done  mem={_rss_mb()}")
    if not s3["overall_pass"]:
        # Check if any pair reached MEDIUM confidence — try Stage 4N as rescue
        medium_rescue = any(
            p.get("washing_tier") == "MEDIUM"
            for p in s3.get("pair_results", {}).values()
            if isinstance(p, dict) and p.get("status") == "OK"
        )
        if not medium_rescue:
            result["status"]       = "REJECTED"
            result["failed_stage"] = "D2D5_U2U5"
            if progress_cb:
                progress_cb(f"{folder_name}  REJECTED  stage=D2D5_U2U5  (LOW confidence)")
            return result
        # MEDIUM confidence — attempt geometry rescue via Stage 4N
        if progress_cb:
            progress_cb(f"{folder_name}  Stage 3N MEDIUM -- attempting Stage 4N geometry rescue ...")
        try:
            s4_rescue = _run_stage4n(folder_path, folder_name, no_obs_output_dir, seg_model, progress_cb)
        except Exception as exc:
            result["status"]       = "REJECTED"
            result["failed_stage"] = "D2D5_U2U5"
            result["detail"]["stage4n_rescue_error"] = str(exc)
            return result
        result["detail"]["stage4n_rescue"] = s4_rescue
        rescue_verdict = s4_rescue.get("folder_verdict", "REJECTED")
        if rescue_verdict == "ACCEPTED":
            result["status"] = "ACCEPTED"
            result["detail"]["stage3n_rescue_note"] = "Promoted by Stage 4N geometry rescue (Stage 3N MEDIUM)"
        else:
            result["status"]       = "REJECTED"
            result["failed_stage"] = "D2D5_U2U5"
        if progress_cb:
            progress_cb(f"{folder_name}  {result['status']}  stage=3N_rescue")
        return result

    # ── Stage 4N ─────────────────────────────────────────────────────────────
    # DISABLE_STAGE4N=true skips the D3/D6+U3/U6 geometry stage (useful for
    # low-resource deployments). When disabled, stage 3N HIGH confidence
    # is sufficient to ACCEPT.
    if os.environ.get("DISABLE_STAGE4N", "").lower() in ("1", "true", "yes"):
        result["status"] = "ACCEPTED"
        result["detail"]["stage4n_skipped"] = "DISABLE_STAGE4N=true"
        if progress_cb:
            progress_cb(f"{folder_name}  ACCEPTED  stage=3N  (Stage 4N disabled)")
        return result

    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 4N starting  mem={_rss_mb()}")
    try:
        s4 = _run_stage4n(folder_path, folder_name, no_obs_output_dir, seg_model, progress_cb)
    except Exception as exc:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "D3D6_U3U6"
        result["detail"]["stage4n_error"] = str(exc)
        if progress_cb:
            progress_cb(f"{folder_name}  REJECTED  stage=D3D6_U3U6  error={exc}")
        return result

    result["detail"]["stage4n"] = s4
    if progress_cb:
        progress_cb(f"STATUS:{folder_name}  Stage 4N done  mem={_rss_mb()}")
    verdict = s4.get("folder_verdict", "REJECTED")

    if verdict == "ACCEPTED":
        result["status"] = "ACCEPTED"
    elif verdict == "NEEDS_REVIEW":
        result["status"] = "NEEDS_REVIEW"
    else:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "D3D6_U3U6"

    if progress_cb:
        progress_cb(f"{folder_name}  {result['status']}  stage=4N")
    return result


def _process_obstruction_folder(
    folder_path, folder_name, obs_output_dir,
    detector, seg_model, obs_sift_output_dir,
    progress_cb,
):
    """
    Run OCR for one obstruction folder, then optionally SIFT stages if pairs exist.
    """
    _ensure_ocr()
    result = {
        "name":         folder_name,
        "type":         "obstruction",
        "status":       None,
        "failed_stage": None,
        "detail":       {},
    }

    if progress_cb:
        progress_cb(f"{folder_name}  obstruction  OCR ...")
    try:
        ocr_result = _run_ocr(folder_path, folder_name, obs_output_dir, progress_cb)
    except Exception as exc:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "OCR_FAILED"
        result["detail"]["ocr_error"] = str(exc)
        result["has_sift"] = False
        if progress_cb:
            progress_cb(f"{folder_name}  REJECTED  stage=OCR_FAILED  error={exc}")
        return result

    result["detail"]["ocr"] = {
        "images_processed": ocr_result["images_processed"],
        "images_failed":    ocr_result["images_failed"],
    }

    if ocr_result["status"] == "ACCEPTED":
        result["status"] = "OBSTRUCTION_PROCESSED"
    else:
        result["status"]       = "REJECTED"
        result["failed_stage"] = "OCR_FAILED"

    if progress_cb:
        progress_cb(
            f"{folder_name}  {result['status']}  "
            f"ocr_images={ocr_result['images_processed']}"
        )

    # Optional SIFT stages when pairs exist in an obstruction folder
    if result["status"] == "OBSTRUCTION_PROCESSED" and _has_sift_pairs(folder_path):
        if progress_cb:
            progress_cb(f"{folder_name}  obstruction+SIFT  pairs found, running stages ...")
        sift_stages = _run_sift_stages_for_obstruction(
            folder_path, folder_name, obs_sift_output_dir,
            detector, seg_model, progress_cb,
        )
        result["detail"]["sift_stages"] = sift_stages
        result["has_sift"] = True
    else:
        result["has_sift"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_dir, output_dir, progress_cb=None, single_folder=False, resume_from=None):
    """
    Run the full WRN pipeline on *input_dir*.

    Parameters
    ----------
    input_dir       : str   Directory containing raw job folders.
    output_dir      : str   Directory to write all outputs and the JSON report.
    progress_cb     : callable(msg: str), optional
                            Called after each significant event.  Receives a
                            plain-text status line.  Safe to use with SSE.
    single_folder   : bool  If True, treat *input_dir* itself as the single
                            job folder to process.

    Returns
    -------
    dict   Full structured results (also written to
           ``<output_dir>/pipeline_run_<timestamp>.json``).
    """
    os.makedirs(output_dir, exist_ok=True)

    no_obs_output = os.path.join(output_dir, "no_obstruction")
    obs_output    = os.path.join(output_dir, "obstruction")
    os.makedirs(no_obs_output, exist_ok=True)
    os.makedirs(obs_output,    exist_ok=True)

    from pipeline_config import OBS_SIFT_SUBDIR
    obs_sift_output = os.path.join(obs_output, OBS_SIFT_SUBDIR)
    os.makedirs(obs_sift_output, exist_ok=True)

    # ── Step 0: Discover folders ──────────────────────────────────────────────
    if single_folder:
        folder_name = os.path.basename(input_dir.rstrip("/\\"))
        classification = _classify_single(input_dir, folder_name)
        if classification == "no_obstruction":
            folders_map = {"no_obstruction": [folder_name], "obstruction": []}
        else:
            folders_map = {"no_obstruction": [], "obstruction": [folder_name]}
        base_dir = os.path.dirname(input_dir)
    else:
        base_dir    = input_dir
        folders_map = _classify_folders(input_dir)

    # Resume support: drop all folders (sorted) before resume_from
    if resume_from:
        folders_map["no_obstruction"] = [
            f for f in folders_map["no_obstruction"] if f >= resume_from
        ]
        folders_map["obstruction"] = [
            f for f in folders_map["obstruction"] if f >= resume_from
        ]

    all_folders = folders_map["no_obstruction"] + folders_map["obstruction"]
    total       = len(all_folders)

    if progress_cb:
        progress_cb(
            f"INIT  total={total}  "
            f"no_obstruction={len(folders_map['no_obstruction'])}  "
            f"obstruction={len(folders_map['obstruction'])}"
        )

    # ── Step 1: Load optional models ─────────────────────────────────────────
    detector  = _get_detector()
    seg_model = _get_seg_model()
    if progress_cb:
        progress_cb(f"STATUS:Models loaded  mem={_rss_mb()}")

    # ── Step 2: Process folders ───────────────────────────────────────────────
    folder_results = []
    done = 0

    # No-obstruction folders
    for folder_name in sorted(folders_map["no_obstruction"]):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            # May have been pre-sorted into a subdirectory by sort_folders
            folder_path = os.path.join(base_dir, "no_obstruction", folder_name)
        if not os.path.isdir(folder_path):
            result = {
                "name": folder_name, "type": "no_obstruction",
                "status": "REJECTED", "failed_stage": "D1D4_U1U4",
                "detail": {"error": "Folder path not found"},
            }
            folder_results.append(result)
            done += 1
            if progress_cb:
                progress_cb(f"PROGRESS  done={done}  total={total}  folder={folder_name}  status=REJECTED")
            continue

        result = _process_no_obstruction_folder(
            folder_path, folder_name, no_obs_output,
            detector, seg_model, progress_cb,
        )
        folder_results.append(result)
        done += 1
        _free_mem()
        if progress_cb:
            mem_note = f"  mem={_rss_mb()}"
            progress_cb(
                f"PROGRESS  done={done}  total={total}  "
                f"folder={folder_name}  status={result['status']}{mem_note}"
            )

    # Unload YOLO before OCR phase — PaddleOCR + YOLO together use significant RAM
    _unload_models()
    if progress_cb:
        progress_cb(f"STATUS:YOLO unloaded before OCR phase  mem={_rss_mb()}")

    # Obstruction folders
    for folder_name in sorted(folders_map["obstruction"]):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            folder_path = os.path.join(base_dir, "obstruction", folder_name)
        if not os.path.isdir(folder_path):
            result = {
                "name": folder_name, "type": "obstruction",
                "status": "REJECTED", "failed_stage": "OCR_FAILED",
                "detail": {"error": "Folder path not found"},
            }
            folder_results.append(result)
            done += 1
            if progress_cb:
                progress_cb(f"PROGRESS  done={done}  total={total}  folder={folder_name}  status=REJECTED")
            continue

        result = _process_obstruction_folder(
            folder_path, folder_name, obs_output,
            None, None, obs_sift_output,
            progress_cb,
        )
        folder_results.append(result)
        done += 1
        _free_mem()
        if progress_cb:
            mem_note = f"  mem={_rss_mb()}"
            progress_cb(
                f"PROGRESS  done={done}  total={total}  "
                f"folder={folder_name}  status={result['status']}{mem_note}"
            )

    # ── Step 3: Build summary ─────────────────────────────────────────────────
    accepted             = sum(1 for r in folder_results if r["status"] == "ACCEPTED")
    needs_review         = sum(1 for r in folder_results if r["status"] == "NEEDS_REVIEW")
    obstruction_proc     = sum(1 for r in folder_results if r["status"] == "OBSTRUCTION_PROCESSED")
    rejected             = sum(1 for r in folder_results if r["status"] == "REJECTED")
    obstruction_with_sift = sum(
        1 for r in folder_results
        if r["status"] == "OBSTRUCTION_PROCESSED" and r.get("has_sift")
    )

    rejected_breakdown = {}
    for r in folder_results:
        if r["status"] == "REJECTED" and r.get("failed_stage"):
            key = r["failed_stage"]
            rejected_breakdown[key] = rejected_breakdown.get(key, 0) + 1

    output_data = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "summary": {
            "total":                  total,
            "accepted":               accepted,
            "needs_review":           needs_review,
            "obstruction_processed":  obstruction_proc,
            "obstruction_with_sift":  obstruction_with_sift,
            "rejected":               rejected,
        },
        "rejected_breakdown": rejected_breakdown,
        "folders": folder_results,
    }

    # ── Step 4: Write JSON report ─────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"pipeline_run_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    output_data["report_path"] = json_path

    # ── Step 4.5: Write Excel report ─────────────────────────────────────────
    # Skipped when SKIP_EXCEL_REPORT=1 (useful on memory-constrained deployments)
    # to avoid loading matplotlib + openpyxl (~110 MB) in the pipeline subprocess.
    excel_path = None
    if not os.environ.get("SKIP_EXCEL_REPORT", "").strip() in ("1", "true", "yes"):
        excel_path = json_path.replace(".json", ".xlsx")
        try:
            from analytics.generate_excel_report import generate_excel_report
            generate_excel_report(output_data, excel_path)
            output_data["excel_report_path"] = excel_path
        except Exception as exc:
            excel_path = None
            if progress_cb:
                progress_cb(f"STATUS: Excel report skipped -- {exc}")

    if progress_cb:
        done_msg = (
            f"DONE  accepted={accepted}  needs_review={needs_review}  "
            f"obstruction_processed={obstruction_proc}  "
            f"obstruction_with_sift={obstruction_with_sift}  "
            f"rejected={rejected}  report={json_path}"
        )
        if excel_path:
            done_msg += f"  excel={excel_path}"
        progress_cb(done_msg)

    return output_data


def _classify_single(folder_path, folder_name):
    """Classify a single folder without moving it."""
    from pipeline.sort_folders import _has_file
    has_d = _has_file(folder_path, "D1") and _has_file(folder_path, "D4")
    has_u = _has_file(folder_path, "U1") and _has_file(folder_path, "U4")
    return "no_obstruction" if (has_d or has_u) else "obstruction"


def _has_sift_pairs(folder_path):
    """Return True if folder has a D1+D4 or U1+U4 pair (without moving it)."""
    from pipeline.sort_folders import _has_file
    has_d = _has_file(folder_path, "D1") and _has_file(folder_path, "D4")
    has_u = _has_file(folder_path, "U1") and _has_file(folder_path, "U4")
    return has_d or has_u


def _run_sift_stages_for_obstruction(
    folder_path, folder_name, obs_sift_output_dir,
    detector, seg_model, progress_cb,
):
    """Run stages 2N -> 3N -> 4N for an obstruction folder that has SIFT pairs."""
    sift_stages = {
        "ran": True,
        "stage2n": None, "stage2n_pass": False,
        "stage3n": None, "stage3n_pass": False,
        "stage4n": None,
        "sift_verdict": "REJECTED",
        "error": None,
    }

    # Stage 2N
    try:
        s2 = _run_stage2n(folder_path, folder_name, obs_sift_output_dir, detector, progress_cb)
    except Exception as exc:
        sift_stages.update({"sift_verdict": "FAILED", "error": f"Stage 2N: {exc}"})
        return sift_stages
    sift_stages["stage2n"] = s2
    sift_stages["stage2n_pass"] = s2["overall_pass"]
    if not s2["overall_pass"]:
        return sift_stages  # sift_verdict stays "REJECTED"

    # Stage 3N
    try:
        s3 = _run_stage3n(folder_path, folder_name, obs_sift_output_dir, progress_cb)
    except Exception as exc:
        sift_stages.update({"sift_verdict": "FAILED", "error": f"Stage 3N: {exc}"})
        return sift_stages
    sift_stages["stage3n"] = s3
    sift_stages["stage3n_pass"] = s3["overall_pass"]
    if not s3["overall_pass"]:
        return sift_stages

    # Stage 4N
    try:
        s4 = _run_stage4n(folder_path, folder_name, obs_sift_output_dir, seg_model, progress_cb)
    except Exception as exc:
        sift_stages.update({"sift_verdict": "FAILED", "error": f"Stage 4N: {exc}"})
        return sift_stages
    sift_stages["stage4n"] = s4
    sift_stages["sift_verdict"] = s4.get("folder_verdict", "REJECTED")
    return sift_stages


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WRN Service Report Checker — Full Pipeline"
    )
    parser.add_argument("--input",  required=True, help="Input directory of job folders")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--single", action="store_true",
                        help="Treat --input as a single job folder (not a batch)")
    parser.add_argument("--resume-from", default=None, metavar="FOLDER",
                        help="Skip all folders sorted before FOLDER (resume after a crash)")
    args = parser.parse_args()

    def print_cb(msg):
        print(msg)

    result = run_pipeline(args.input, args.output, progress_cb=print_cb,
                          single_folder=args.single, resume_from=args.resume_from)

    s = result["summary"]
    print()
    print("-" * 55)
    print("PIPELINE COMPLETE")
    print("-" * 55)
    print(f"  Total folders          : {s['total']}")
    print(f"  Accepted               : {s['accepted']}")
    print(f"  Needs review           : {s['needs_review']}")
    print(f"  Obstruction processed  : {s['obstruction_processed']}")
    print(f"  Obstruction with SIFT  : {s['obstruction_with_sift']}")
    print(f"  Rejected               : {s['rejected']}")
    if result["rejected_breakdown"]:
        print()
        print("  Rejected by stage:")
        for stage, count in result["rejected_breakdown"].items():
            print(f"    {stage:<20} {count}")
    print("-" * 55)
    print(f"  Report: {result['report_path']}")
    if result.get("excel_report_path"):
        print(f"  Excel:  {result['excel_report_path']}")


if __name__ == "__main__":
    main()
