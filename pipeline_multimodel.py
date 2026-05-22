"""Phase 5: Batch Processing + Multi-Model Architecture (CWRU).

This module implements an end-to-end pipeline:
1) Batch scan a folder of CWRU `.mat` files.
2) Extract time-domain statistical features from fixed-size windows.
3) Build a single feature table tagged with (Label, Load_HP).
4) Train one IsolationForest model per load condition (Load_HP).

Design goals:
- Production-friendly: logging (no ad-hoc prints), clear structure, type hints.
- Multi-model: each load condition has its own anomaly detector.

Expected folder layout:
- archive/97.mat, 98.mat, 99.mat, 100.mat (Normal, loads 0..3 HP)
- archive/105.mat, 106.mat, 107.mat, 108.mat (Fault, loads 0..3 HP)

Run:
    python pipeline_multimodel.py
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1) Mapping: file id -> (Label, Load_HP)
# -----------------------------------------------------------------------------

CWRU_MAPPING: dict[str, tuple[int, int]] = {
    # Normal
    "97": (0, 0),
    "98": (0, 1),
    "99": (0, 2),
    "100": (0, 3),
    # Fault (inner race example)
    "105": (1, 0),
    "106": (1, 1),
    "107": (1, 2),
    "108": (1, 3),
}


# -----------------------------------------------------------------------------
# 2) Helpers
# -----------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with timestamp + level.

    Args:
        level: Logging level (default INFO).
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _extract_file_id(mat_path: str) -> str:
    """Extract the numeric file id from a .mat filename.

    Example: '/path/to/97.mat' -> '97'

    Args:
        mat_path: Path to a .mat file.

    Returns:
        The numeric part of the filename (without extension).
    """

    base = os.path.basename(mat_path)
    file_id, _ext = os.path.splitext(base)
    return file_id


def _find_de_time_key(mat_dict: dict[str, Any]) -> str:
    """Find the key containing Drive End time-domain signal.

    CWRU MAT files typically store the vibration array under a key that contains
    '_DE_time'. This function searches for that key.

    Args:
        mat_dict: Dictionary returned by scipy.io.loadmat.

    Returns:
        The matched key name.

    Raises:
        ValueError: If no suitable key is found.
    """

    candidates = [k for k in mat_dict.keys() if "_DE_time" in k]
    if not candidates:
        raise ValueError("Không tìm thấy key chứa '_DE_time' trong file .mat")

    # Deterministic choice if multiple keys exist.
    candidates.sort()
    return candidates[0]


def _create_windows(signal_array: np.ndarray, window_size: int = 1200) -> np.ndarray:
    """Split a 1D signal into fixed-size, non-overlapping windows.

    Notes:
        - Uses non-overlapping windows for batch processing stability.
        - Any remainder samples at the end that cannot form a complete window
          are discarded.

    Args:
        signal_array: 1D vibration array.
        window_size: Window size in samples (default 1200).

    Returns:
        2D array shaped (num_windows, window_size).

    Raises:
        ValueError: If input is not 1D or window_size is invalid.
    """

    if window_size <= 0:
        raise ValueError("window_size phải > 0")

    arr = np.asarray(signal_array).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"signal_array phải là mảng 1D; nhận được shape={arr.shape}")

    num_windows = arr.shape[0] // window_size
    usable_len = num_windows * window_size
    if usable_len == 0:
        return np.empty((0, window_size), dtype=np.float64)

    windows_2d = arr[:usable_len].reshape(num_windows, window_size)
    return windows_2d.astype(np.float64, copy=False)


def _features_from_windows(
    windows_2d: np.ndarray,
    label: int,
    load_hp: int,
) -> pd.DataFrame:
    """Compute statistical features per window and attach metadata.

    Features per window:
        - RMS: $\sqrt{mean(x^2)}$
        - Variance: $var(x)$
        - Kurtosis: Pearson kurtosis (fisher=False)
        - Peak_to_Peak: $max(x) - min(x)$

    Args:
        windows_2d: 2D windows array (num_windows, window_size).
        label: 0=Normal, 1=Fault.
        load_hp: Motor load condition in HP.

    Returns:
        DataFrame with columns:
        RMS, Variance, Kurtosis, Peak_to_Peak, Load_HP, Label

    Raises:
        ValueError: If windows_2d is not 2D.
    """

    windows = np.asarray(windows_2d)
    if windows.ndim != 2:
        raise ValueError(f"windows_2d phải là mảng 2D; nhận được shape={windows.shape}")

    rms_list: list[float] = []
    var_list: list[float] = []
    kurt_list: list[float] = []
    ptp_list: list[float] = []

    for window in windows:
        w = np.asarray(window, dtype=np.float64)
        rms_list.append(float(np.sqrt(np.mean(w**2))))
        var_list.append(float(np.var(w)))
        kurt_list.append(float(kurtosis(w, fisher=False, bias=False)))
        ptp_list.append(float(np.max(w) - np.min(w)))

    return pd.DataFrame(
        {
            "RMS": rms_list,
            "Variance": var_list,
            "Kurtosis": kurt_list,
            "Peak_to_Peak": ptp_list,
            "Load_HP": int(load_hp),
            "Label": int(label),
        }
    )


# -----------------------------------------------------------------------------
# 3) Batch feature extraction
# -----------------------------------------------------------------------------

def extract_features_from_folder(folder_path: str) -> pd.DataFrame:
    """Batch scan `.mat` files and build a master feature table.

    Algorithm:
        1) Scan all `*.mat` in folder.
        2) For each file, extract file id and look up (Label, Load_HP) from
           `CWRU_MAPPING`. Files not in the mapping are skipped.
        3) Load MAT with `scipy.io.loadmat`, locate the key containing `_DE_time`.
        4) Slice the 1D vibration signal into windows of 1200 samples.
        5) Compute RMS/Variance/Kurtosis/Peak_to_Peak per window.
        6) Attach `Load_HP` and `Label`, then concatenate into a master DataFrame.
        7) Save to `features_multiload.csv`.

    Args:
        folder_path: Path to the folder containing CWRU `.mat` files.

    Returns:
        master_df: Combined DataFrame containing features + metadata.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If no features could be extracted.
    """

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder_path}")

    pattern = os.path.join(folder_path, "*.mat")
    mat_files = sorted(glob.glob(pattern))
    LOGGER.info("Tìm thấy %d file .mat trong %s", len(mat_files), folder_path)

    frames: list[pd.DataFrame] = []

    for mat_path in mat_files:
        file_id = _extract_file_id(mat_path)
        if file_id not in CWRU_MAPPING:
            LOGGER.warning("Bỏ qua file không có trong mapping: %s", mat_path)
            continue

        label, load_hp = CWRU_MAPPING[file_id]

        try:
            mat_dict = loadmat(mat_path)
            key = _find_de_time_key(mat_dict)
            raw_signal = mat_dict[key]
        except Exception as exc:  # noqa: BLE001 - batch job should continue
            LOGGER.exception("Lỗi đọc file %s: %s", mat_path, exc)
            continue

        signal_array = np.asarray(raw_signal).squeeze().astype(np.float64, copy=False)

        try:
            windows_2d = _create_windows(signal_array, window_size=1200)
        except ValueError as exc:
            LOGGER.exception("Lỗi windowing file %s: %s", mat_path, exc)
            continue

        if windows_2d.size == 0:
            LOGGER.warning("File %s không đủ dữ liệu để tạo window", mat_path)
            continue

        df_feat = _features_from_windows(windows_2d, label=label, load_hp=load_hp)
        frames.append(df_feat)

        LOGGER.info(
            "Đã xử lý %s | id=%s | Label=%d | Load_HP=%d | windows=%d",
            os.path.basename(mat_path),
            file_id,
            label,
            load_hp,
            len(df_feat),
        )

    if not frames:
        raise ValueError(
            "Không trích xuất được feature nào. Hãy kiểm tra mapping và dữ liệu .mat."
        )

    master_df = pd.concat(frames, ignore_index=True)

    output_csv = "features_multiload.csv"
    master_df.to_csv(output_csv, index=False)
    LOGGER.info("Đã lưu dataset feature: %s | rows=%d", output_csv, len(master_df))

    return master_df


# -----------------------------------------------------------------------------
# 4) Train one model per load
# -----------------------------------------------------------------------------

def train_multiload_models(csv_path: str) -> None:
    """Train one IsolationForest model per load condition.

    Splitting logic (anomaly detection):
        For each Load_HP group:
        - Train set: 80% of Normal-only windows (Label=0)
        - Test set: remaining 20% of Normal + 100% of Fault (Label=1)

    For each trained model:
        - Evaluate using classification_report on the test set.
        - Save the model to `iso_forest_load_{load_hp}HP.pkl`.

    Args:
        csv_path: Path to `features_multiload.csv`.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"RMS", "Variance", "Kurtosis", "Peak_to_Peak", "Load_HP", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {sorted(missing)}")

    for load_hp, df_load in df.groupby("Load_HP", sort=True):
        df_normal = df_load[df_load["Label"] == 0].copy()
        df_fault = df_load[df_load["Label"] == 1].copy()

        if df_normal.empty:
            LOGGER.warning("Load %s HP: không có Normal (Label=0), bỏ qua.", load_hp)
            continue

        df_train_normal, df_test_normal = train_test_split(
            df_normal,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        feature_cols = ["RMS", "Variance", "Kurtosis", "Peak_to_Peak"]

        X_train = df_train_normal[feature_cols]

        X_test = pd.concat(
            [
                df_test_normal[feature_cols],
                df_fault[feature_cols],
            ],
            ignore_index=True,
        )

        y_test = pd.concat(
            [
                df_test_normal["Label"],
                df_fault["Label"],
            ],
            ignore_index=True,
        )

        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_train)

        # IsolationForest: +1=inlier, -1=outlier
        raw_pred = model.predict(X_test)
        y_pred = (raw_pred == -1).astype(int)

        title = f"Đánh giá Model Load {int(load_hp)} HP"
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Fault"],
            zero_division=0,
        )

        LOGGER.info("%s\n%s", title, report)

        model_path = f"iso_forest_load_{int(load_hp)}HP.pkl"
        joblib.dump(model, model_path)
        LOGGER.info("Đã lưu model: %s", model_path)


# -----------------------------------------------------------------------------
# 5) Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Run Phase 5 end-to-end pipeline."""

    configure_logging()

    folder_path = os.path.join("archive")
    extract_features_from_folder(folder_path)
    train_multiload_models("features_multiload.csv")

    LOGGER.info("Hoàn thành Phase 5: Batch + Multi-Model theo Load_HP")


if __name__ == "__main__":
    main()
