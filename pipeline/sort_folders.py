import os
import shutil

_VALID_EXTS = {".jpg", ".jpeg", ".png"}


def _has_file(folder, name):
    """Check if a file with the given base name exists with a supported image extension."""
    for f in os.listdir(folder):
        base, ext = os.path.splitext(f)
        if base == name and ext.lower() in _VALID_EXTS:
            return True
    return False


def sort_folders(base_dir, no_obstruction_dir=None, obstruction_dir=None):
    """
    Classify and move every immediate subfolder of *base_dir* into
    no_obstruction/ or obstruction/ subdirectories.

    Parameters
    ----------
    base_dir : str
        Directory containing raw job folders (e.g. "1159097", "146915", …).
    no_obstruction_dir : str, optional
        Destination for no-obstruction folders. Defaults to
        ``<base_dir>/no_obstruction``.
    obstruction_dir : str, optional
        Destination for obstruction folders. Defaults to
        ``<base_dir>/obstruction``.

    Returns
    -------
    dict
        ``{"no_obstruction": ["1159097", ...], "obstruction": ["146915", ...]}``
    """
    if no_obstruction_dir is None:
        no_obstruction_dir = os.path.join(base_dir, "no_obstruction")
    if obstruction_dir is None:
        obstruction_dir = os.path.join(base_dir, "obstruction")

    os.makedirs(no_obstruction_dir, exist_ok=True)
    os.makedirs(obstruction_dir, exist_ok=True)

    result = {"no_obstruction": [], "obstruction": []}

    skip_names = {
        os.path.basename(no_obstruction_dir),
        os.path.basename(obstruction_dir),
    }

    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue
        if entry.name in skip_names:
            continue

        folder = entry.path
        has_d_pair = _has_file(folder, "D1") and _has_file(folder, "D4")
        has_u_pair = _has_file(folder, "U1") and _has_file(folder, "U4")

        if has_d_pair or has_u_pair:
            dest = os.path.join(no_obstruction_dir, entry.name)
            shutil.move(folder, dest)
            result["no_obstruction"].append(entry.name)
        else:
            dest = os.path.join(obstruction_dir, entry.name)
            shutil.move(folder, dest)
            result["obstruction"].append(entry.name)

    return result


def classify_only(base_dir):
    """
    Like sort_folders but does NOT move anything — only returns the
    classification of each immediate subfolder.

    Returns
    -------
    dict
        ``{"no_obstruction": ["1159097", ...], "obstruction": ["146915", ...]}``
    """
    result = {"no_obstruction": [], "obstruction": []}
    skip = {"no_obstruction", "obstruction"}

    for entry in os.scandir(base_dir):
        if not entry.is_dir() or entry.name in skip:
            continue
        has_d_pair = _has_file(entry.path, "D1") and _has_file(entry.path, "D4")
        has_u_pair = _has_file(entry.path, "U1") and _has_file(entry.path, "U4")
        key = "no_obstruction" if (has_d_pair or has_u_pair) else "obstruction"
        result[key].append(entry.name)

    return result


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\allti\OneDrive\Documents\wrn_service_report_checker\adjusted_images"
    result = sort_folders(BASE_DIR)
    moved_no  = len(result["no_obstruction"])
    moved_obs = len(result["obstruction"])
    for name in result["no_obstruction"]:
        print(f"[no_obstruction] {name}")
    for name in result["obstruction"]:
        print(f"[obstruction]    {name}")
    print(f"\nDone. {moved_no} folders -> no_obstruction, {moved_obs} folders -> obstruction.")
