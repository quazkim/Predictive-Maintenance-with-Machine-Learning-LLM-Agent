"""Train an Isolation Forest anomaly detector for Predictive Maintenance.

Stage 3: Model training for anomaly detection.

Dataset:
- features_dataset.csv with columns: RMS, Variance, Kurtosis, Peak_to_Peak, Label
- Label convention: 0 = Normal, 1 = Fault

Key idea:
- Train IsolationForest ONLY on Normal data (Label=0)
- Evaluate on remaining Normal + ALL Fault data

Run:
    python model_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def prepare_data(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load CSV and split into train/test for anomaly detection.

    Splitting logic (critical for anomaly detection):
    - Training set: ONLY 80% of Normal samples (Label=0)
    - Testing set: remaining 20% of Normal + ALL Fault samples (Label=1)

    Args:
        csv_path: Path to the feature dataset CSV.

    Returns:
        X_train: Feature matrix for training (Normal-only).
        X_test: Feature matrix for testing (Normal remainder + all Fault).
        y_test: Ground-truth labels for X_test (0 Normal, 1 Fault).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing or dataset is empty.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    df = pd.read_csv(path)
    required_cols = {"RMS", "Variance", "Kurtosis", "Peak_to_Peak", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {sorted(missing)}")

    df_normal = df[df["Label"] == 0].copy()
    df_fault = df[df["Label"] == 1].copy()

    if df_normal.empty:
        raise ValueError("Không có dữ liệu Normal (Label=0) để huấn luyện.")

    # Train/test split only on normal samples.
    df_train_normal, df_test_normal = train_test_split(
        df_normal,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    X_train = df_train_normal.drop(columns=["Label"])

    X_test = pd.concat(
        [
            df_test_normal.drop(columns=["Label"]),
            df_fault.drop(columns=["Label"]),
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

    if X_test.empty:
        raise ValueError("Tập test rỗng. Hãy kiểm tra lại dữ liệu đầu vào.")

    return X_train, X_test, y_test


def train_anomaly_model(X_train: pd.DataFrame) -> IsolationForest:
    """Train an IsolationForest anomaly detector on normal-only data.

    Args:
        X_train: Training feature matrix (Normal-only).

    Returns:
        Trained IsolationForest model.
    """

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )
    model.fit(X_train)
    return model


def evaluate_and_save_model(
    model: IsolationForest,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str | Path = "isolation_forest_model.pkl",
) -> None:
    """Evaluate model, print metrics, and save it using joblib.

    IsolationForest prediction convention:
    - +1 => inlier (Normal)
    - -1 => outlier (Anomaly)

    We convert it to match our dataset labels:
    - 0 => Normal
    - 1 => Fault/Anomaly

    Args:
        model: Trained IsolationForest.
        X_test: Test feature matrix.
        y_test: Ground-truth labels (0 Normal, 1 Fault).
        model_path: Output path to save the trained model.
    """

    raw_pred = model.predict(X_test)
    y_pred = (raw_pred == -1).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("=== Confusion Matrix (labels: 0=Normal, 1=Fault/Anomaly) ===")
    print(cm)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Fault"],
        zero_division=0,
    )
    print("\n=== Classification Report ===")
    print(report)

    output_path = Path(model_path)
    joblib.dump(model, output_path)
    print(f"Saved model to: {output_path.resolve()}")


def main() -> None:
    """End-to-end training/evaluation pipeline."""

    csv_path = Path("features_dataset.csv")

    try:
        X_train, X_test, y_test = prepare_data(csv_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Gợi ý: Hãy đảm bảo file 'features_dataset.csv' tồn tại trong thư mục dự án.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Lỗi dữ liệu: {exc}", file=sys.stderr)
        sys.exit(1)

    model = train_anomaly_model(X_train)
    evaluate_and_save_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
