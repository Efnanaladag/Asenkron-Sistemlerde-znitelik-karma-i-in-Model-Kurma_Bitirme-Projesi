import csv
import os
import sys

import numpy as np

import config
from run_cross_session_csp_5band import load_all_sessions
from train_csp_5band_baseline import run_loso_csp_5band_baseline


RESULTS_FOLDER = os.path.join(config.OUTPUT_DIR, "ablation_results")
COMPONENT_VALUES = [2, 4, 6, 8]
WIDEBAND_WINDOW_FOLDER = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
EXPECTED_WIDEBAND_WINDOW_COUNT = 11


def preflight_wideband_inputs():
    print("Current working directory:", os.getcwd())

    if not os.path.isdir(WIDEBAND_WINDOW_FOLDER):
        found_count = 0
    else:
        found_count = len(
            [
                filename for filename in os.listdir(WIDEBAND_WINDOW_FOLDER)
                if filename.startswith("session_")
                and filename.endswith("_wideband_windows.npz")
            ]
        )

    print("Wideband window file count:", found_count)

    if found_count != EXPECTED_WIDEBAND_WINDOW_COUNT:
        raise RuntimeError(
            f"Expected 11 wideband window files before ablation, found {found_count}. "
            "Run windowing_wideband.py for sessions 1..11 first."
        )


def get_session_range_from_cli():
    """
    Opsiyonel session araligini alir.

    Kullanim:
    python run_ablation_csp_components.py
    python run_ablation_csp_components.py 1 3
    """
    if len(sys.argv) == 1:
        return None, None

    if len(sys.argv) == 3:
        start_session = int(sys.argv[1])
        end_session = int(sys.argv[2])

        if start_session > end_session:
            raise ValueError("Baslangic session, bitis session'dan buyuk olamaz.")

        return start_session, end_session

    raise ValueError(
        "Kullanim: python run_ablation_csp_components.py [start_session end_session]"
    )


def save_csv(rows, save_path):
    if len(rows) == 0:
        raise ValueError(f"Kaydedilecek satir yok: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Kaydedildi: {save_path}")


def build_summary_row(csp_components, fold_results):
    auc_values = [float(row["roc_auc"]) for row in fold_results]
    bal_values = [float(row["balanced_accuracy"]) for row in fold_results]

    return {
        "csp_components": int(csp_components),
        "mean_auc": round(float(np.mean(auc_values)), 6),
        "std_auc": round(float(np.std(auc_values)), 6),
        "mean_balanced_accuracy": round(float(np.mean(bal_values)), 6),
        "std_balanced_accuracy": round(float(np.std(bal_values)), 6),
        "fold_count": int(len(fold_results)),
    }


def main():
    preflight_wideband_inputs()
    start_session, end_session = get_session_range_from_cli()

    print("Session wideband window dosyalari yukleniyor...")
    session_data = load_all_sessions(start_session, end_session)

    summary_rows = []

    for csp_components in COMPONENT_VALUES:
        print(f"\n===== CSP COMPONENT ABLATION: {csp_components} =====")

        fold_results, _ = run_loso_csp_5band_baseline(
            session_data=session_data,
            verbose=True,
            csp_components=csp_components,
        )

        per_value_path = os.path.join(
            RESULTS_FOLDER,
            f"csp_components_{csp_components}_loso_results.csv",
        )
        save_csv(fold_results, per_value_path)

        summary_rows.append(build_summary_row(csp_components, fold_results))

    summary_path = os.path.join(RESULTS_FOLDER, "csp_components_ablation_summary.csv")
    save_csv(summary_rows, summary_path)

    print("\nCSP components ablation tamamlandi.")


if __name__ == "__main__":
    main()
