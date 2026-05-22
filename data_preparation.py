"""Data acquisition and exploratory visualization for the CWRU Bearing Dataset.

This script loads vibration data from CWRU .mat files (Drive End accelerometer
signal), converts it into a tidy pandas DataFrame, prints quick diagnostics,
then plots a side-by-side (subplot) comparison between Normal and Fault states.

Assumed folder structure:
- data/normal.mat
- data/fault.mat

Run:
    python data_preparation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_cwru_data(file_path: str | Path, sampling_rate: int = 12_000) -> pd.DataFrame:
    """Load CWRU vibration data from a .mat file into a DataFrame.

    The CWRU .mat file loaded by `scipy.io.loadmat` returns a dictionary.
    The actual Drive End accelerometer time-series is stored under a key that
    contains the substring '_DE_time'. This function automatically finds that
    key, extracts the numpy array, then constructs a DataFrame with:

    - `Vibration`: vibration amplitude signal (1D)
    - `Time`: synthetic time axis in seconds using the provided sampling rate

    Args:
        file_path: Path to the .mat file.
        sampling_rate: Sampling rate in Hz (default: 12000 for 12 kHz).

    Returns:
        A pandas DataFrame containing `Time` and `Vibration` columns.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If no key containing '_DE_time' is found or data is invalid.
    """

    mat_path = Path(file_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {mat_path}")

    try:
        mat_dict: dict[str, Any] = loadmat(mat_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Không thể đọc file .mat: {mat_path}") from exc

    de_time_keys = [key for key in mat_dict.keys() if "_DE_time" in key]
    if not de_time_keys:
        raise ValueError(
            "Không tìm thấy key chứa '_DE_time' trong file .mat. "
            "Vui lòng kiểm tra cấu trúc dữ liệu của file."
        )

    # If multiple candidates exist, pick the first deterministic (sorted) key.
    de_time_key = sorted(de_time_keys)[0]
    signal = mat_dict.get(de_time_key)

    if signal is None:
        raise ValueError(f"Key '{de_time_key}' tồn tại nhưng không có dữ liệu.")

    signal_array = np.asarray(signal).squeeze()
    if signal_array.ndim != 1:
        raise ValueError(
            f"Dữ liệu '{de_time_key}' không phải 1D sau khi squeeze; "
            f"shape={signal_array.shape}."
        )

    if sampling_rate <= 0:
        raise ValueError("sampling_rate phải > 0")

    time_axis = np.arange(signal_array.shape[0], dtype=np.float64) / float(sampling_rate)

    df = pd.DataFrame({"Time": time_axis, "Vibration": signal_array.astype(np.float64)})
    return df


def plot_vibration_comparison(
    df_normal: pd.DataFrame,
    df_fault: pd.DataFrame,
    duration_sec: float = 0.1,
) -> None:
    """Plot Normal vs Fault vibration signals for an initial time window.

    Creates 2 subplots for easy visual comparison.

    Args:
        df_normal: DataFrame for Normal state, must contain `Time` and `Vibration`.
        df_fault: DataFrame for Fault state, must contain `Time` and `Vibration`.
        duration_sec: Duration (seconds) to display from the start (default: 0.1s).

    Raises:
        ValueError: If required columns are missing or duration is invalid.
    """

    required_cols = {"Time", "Vibration"}
    if not required_cols.issubset(df_normal.columns):
        missing = required_cols - set(df_normal.columns)
        raise ValueError(f"df_normal thiếu cột: {sorted(missing)}")

    if not required_cols.issubset(df_fault.columns):
        missing = required_cols - set(df_fault.columns)
        raise ValueError(f"df_fault thiếu cột: {sorted(missing)}")

    if duration_sec <= 0:
        raise ValueError("duration_sec phải > 0")

    import matplotlib.pyplot as plt

    normal_view = df_normal[df_normal["Time"] <= duration_sec]
    fault_view = df_fault[df_fault["Time"] <= duration_sec]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)

    axes[0].plot(
        normal_view["Time"],
        normal_view["Vibration"],
        label="Normal",
        linewidth=1.0,
    )
    axes[0].set_title("CWRU - Normal (Drive End) | First 0.1s")
    axes[0].set_ylabel("Vibration Amplitude")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        fault_view["Time"],
        fault_view["Vibration"],
        label="Fault",
        linewidth=1.0,
    )
    axes[1].set_title("CWRU - Fault (Drive End) | First 0.1s")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Vibration Amplitude")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Entry point: load data, print quick checks, and plot comparison."""

    # Sửa đường dẫn trỏ thẳng vào file trong thư mục archive
    normal_path = Path("archive/97.mat")
    fault_path = Path("archive/105.mat")
    try:
        df_normal = load_cwru_data(normal_path)
        df_fault = load_cwru_data(fault_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Gợi ý: Hãy tạo thư mục 'data/' và đặt 2 file 'normal.mat' và 'fault.mat' vào đó.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Lỗi khi đọc dữ liệu: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=== Normal: head() ===")
    print(df_normal.head())
    print("\n=== Normal: info() ===")
    df_normal.info()

    plot_vibration_comparison(df_normal=df_normal, df_fault=df_fault)


if __name__ == "__main__":
    main()
