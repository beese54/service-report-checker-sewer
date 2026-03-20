"""
Core logic for the Level 1 obstruction check.

A folder is classified as 'no_obstruction' if it contains:
  - D1 AND D4  (downstream pair), OR
  - U1 AND U4  (upstream pair)

Any other combination is classified as 'obstruction'.
File extensions are ignored so the check works for both .jpg and .png.
"""

import os


def _has_file(folder: str, base_name: str) -> bool:
    """Return True if a file with the given base name exists in folder (any extension)."""
    for f in os.listdir(folder):
        base, _ = os.path.splitext(f)
        if base == base_name:
            return True
    return False


def classify_folder(folder_path: str) -> str:
    """Return 'no_obstruction' or 'obstruction' for a single folder."""
    has_d_pair = _has_file(folder_path, "D1") and _has_file(folder_path, "D4")
    has_u_pair = _has_file(folder_path, "U1") and _has_file(folder_path, "U4")
    return "no_obstruction" if (has_d_pair or has_u_pair) else "obstruction"


def check_all_folders(base_dir: str) -> list[dict]:
    """
    Scan all immediate subfolders of base_dir and classify each one.

    Returns a list of dicts sorted by folder name:
        [{"name": "1159097", "result": "no_obstruction"}, ...]
    """
    results = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            result = classify_folder(entry.path)
            results.append({"name": entry.name, "result": result})
    results.sort(key=lambda x: x["name"])
    return results
