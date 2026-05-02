from pathlib import Path
import sys

import numpy as np


EXPECTED_SESSIONS = range(1, 12)
OUTPUT_DIR = Path("outputs")
LABEL_DIR = OUTPUT_DIR / "label_tables"
WINDOW_DIR = OUTPUT_DIR / "window_data"
WIDEBAND_WINDOW_DIR = OUTPUT_DIR / "window_data_wideband"


def fail(message):
    print(f"ERROR: {message}")
    return False


def count_labels(y):
    return int(y.size), int(np.sum(y == 0)), int(np.sum(y == 1))


def load_y(path):
    with np.load(path) as data:
        if "y" not in data:
            raise KeyError(f"{path} does not contain a 'y' array")
        return np.asarray(data["y"])


def check_file_count(folder, pattern, expected_count):
    files = sorted(folder.glob(pattern)) if folder.is_dir() else []
    ok = len(files) == expected_count
    if not ok:
        fail(f"Expected {expected_count} files matching {folder / pattern}, found {len(files)}")
    return ok


def print_table(rows):
    headers = [
        "session",
        "normal_n",
        "normal_l0",
        "normal_l1",
        "wide_n",
        "wide_l0",
        "wide_l1",
        "match",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }

    header_line = "  ".join(header.rjust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)

    for row in rows:
        print("  ".join(str(row[header]).rjust(widths[header]) for header in headers))


def main():
    print("Current working directory:", Path.cwd())

    ok = True
    ok = check_file_count(LABEL_DIR, "session_*_labels.csv", 11) and ok
    ok = check_file_count(WINDOW_DIR, "session_*_windows.npz", 11) and ok
    ok = check_file_count(WIDEBAND_WINDOW_DIR, "session_*_wideband_windows.npz", 11) and ok

    buggy_folders = sorted(
        path for path in Path.cwd().glob("outputs_buggy_restored_*") if path.is_dir()
    )
    for folder in buggy_folders:
        print(f"WARNING: Found restored buggy output folder: {folder}")

    rows = []
    for session in EXPECTED_SESSIONS:
        label_path = LABEL_DIR / f"session_{session}_labels.csv"
        normal_path = WINDOW_DIR / f"session_{session}_windows.npz"
        wide_path = WIDEBAND_WINDOW_DIR / f"session_{session}_wideband_windows.npz"

        for path in [label_path, normal_path, wide_path]:
            if not path.is_file():
                ok = fail(f"Missing required output file: {path}") and ok

        if not normal_path.is_file() or not wide_path.is_file():
            continue

        try:
            normal_y = load_y(normal_path)
            wide_y = load_y(wide_path)
        except Exception as exc:
            ok = fail(str(exc)) and ok
            continue

        normal_n, normal_l0, normal_l1 = count_labels(normal_y)
        wide_n, wide_l0, wide_l1 = count_labels(wide_y)
        match = (
            normal_n == wide_n
            and normal_l0 == wide_l0
            and normal_l1 == wide_l1
        )
        ok = match and ok

        rows.append(
            {
                "session": session,
                "normal_n": normal_n,
                "normal_l0": normal_l0,
                "normal_l1": normal_l1,
                "wide_n": wide_n,
                "wide_l0": wide_l0,
                "wide_l1": wide_l1,
                "match": match,
            }
        )

    if rows:
        print()
        print_table(rows)

    print()
    print("ALL MATCH:", ok)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
