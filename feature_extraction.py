"""Feature engineering for CWRU Bearing Dataset (Predictive Maintenance).

Stage 2: split vibration time series into fixed-size windows, extract
statistical features per window, assign labels, and export a CSV dataset.

Expected input files:
- archive/97.mat   (Normal)
- archive/105.mat  (Fault)

Run:
    python feature_extraction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from data_preparation import load_cwru_data


def create_windows(signal_array: np.ndarray, window_size: int = 1200) -> np.ndarray:
    """Split a 1D signal into fixed-size windows.

    The function slices the 1D array into consecutive, non-overlapping windows
    of size `window_size`. Any remainder at the end that cannot form a complete
    window is discarded.

    Args:
        signal_array: 1D numpy array of vibration values.
        window_size: Window size in samples. Default 1200 samples (0.1s @ 12kHz).

    Returns:
        A 2D numpy array with shape (num_windows, window_size).

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


def extract_features(window_2d_array: np.ndarray, label: int) -> pd.DataFrame:
    """Extract statistical features for each window and attach a label.

    Features per window (row):
    - RMS: Root Mean Square, computed as sqrt(mean(x^2))
    - Variance: np.var(x)
    - Kurtosis: scipy.stats.kurtosis(x)
    - Peak_to_Peak: max(x) - min(x)

    Args:
        window_2d_array: 2D numpy array from `create_windows`.
        label: Integer label to assign to all windows (e.g., 0=Normal, 1=Fault).

    Returns:
        A DataFrame with columns: RMS, Variance, Kurtosis, Peak_to_Peak, Label.

    Raises:
        ValueError: If input array is not 2D.
    """

    windows = np.asarray(window_2d_array)
    if windows.ndim != 2:
        raise ValueError(
            f"window_2d_array phải là mảng 2D; nhận được shape={windows.shape}"
        )

    rms_list: list[float] = []
    var_list: list[float] = []
    kurt_list: list[float] = []
    ptp_list: list[float] = []

    for window in windows:
        window = np.asarray(window, dtype=np.float64)

        rms_value = float(np.sqrt(np.mean(window**2)))
        var_value = float(np.var(window))
        kurt_value = float(kurtosis(window, fisher=False, bias=False))
        ptp_value = float(np.max(window) - np.min(window))

        rms_list.append(rms_value)
        var_list.append(var_value)
        kurt_list.append(kurt_value)
        ptp_list.append(ptp_value)

    df = pd.DataFrame(
        {
            "RMS": rms_list,
            "Variance": var_list,
            "Kurtosis": kurt_list,
            "Peak_to_Peak": ptp_list,
            "Label": int(label),
        }
    )
    return df


def main() -> None:
    """Load CWRU signals, window them, extract features, then export CSV."""

    normal_path = Path("archive/97.mat")
    fault_path = Path("archive/105.mat")

    try:
        df_normal = load_cwru_data(normal_path)
        df_fault = load_cwru_data(fault_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Gợi ý: Hãy đặt file vào đúng đường dẫn 'archive/97.mat' và 'archive/105.mat'.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Lỗi khi đọc dữ liệu: {exc}", file=sys.stderr)
        sys.exit(1)

    normal_signal = df_normal["Vibration"].to_numpy(dtype=np.float64)
    fault_signal = df_fault["Vibration"].to_numpy(dtype=np.float64)

    normal_windows = create_windows(normal_signal, window_size=1200)
    fault_windows = create_windows(fault_signal, window_size=1200)

    df_normal_features = extract_features(normal_windows, label=0)
    df_fault_features = extract_features(fault_windows, label=1)

    df_features = pd.concat([df_normal_features, df_fault_features], ignore_index=True)

    output_path = Path("features_dataset.csv")
    df_features.to_csv(output_path, index=False)

    print("=== Features: head() ===")
    print(df_features.head())
    print("\n=== Features: tail() ===")
    print(df_features.tail())
    print(f"\nTotal samples (rows): {len(df_features)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
