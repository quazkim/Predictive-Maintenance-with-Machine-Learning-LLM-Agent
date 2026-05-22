"""LLM integration wrapper for predictive maintenance reporting.

Stage 4: simulate a real-time anomaly event, score it with a trained
IsolationForest model, and (if anomalous) generate a Vietnamese maintenance
report using Google Gemini.

Prerequisites:
- features_dataset.csv
- isolation_forest_model.pkl
- Environment variable: GEMINI_API_KEY

Run:
    python llm_integration.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai


FEATURE_COLUMNS: list[str] = ["RMS", "Variance", "Kurtosis", "Peak_to_Peak"]


def simulate_realtime_data(csv_path: str) -> dict[str, float]:
    """Simulate real-time incoming data by sampling one Fault row.

    Reads the features dataset and randomly picks one row with Label == 1.

    Args:
        csv_path: Path to features_dataset.csv.

    Returns:
        A dict containing: RMS, Variance, Kurtosis, Peak_to_Peak.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing or no Fault samples exist.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = set(FEATURE_COLUMNS + ["Label"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {sorted(missing)}")

    df_fault = df[df["Label"] == 1]
    if df_fault.empty:
        raise ValueError("Không có dòng Fault (Label=1) để mô phỏng realtime.")

    row = df_fault.sample(n=1).iloc[0]

    metrics: dict[str, float] = {
        "RMS": float(row["RMS"]),
        "Variance": float(row["Variance"]),
        "Kurtosis": float(row["Kurtosis"]),
        "Peak_to_Peak": float(row["Peak_to_Peak"]),
    }
    return metrics


def generate_maintenance_report(metrics_dict: dict[str, float]) -> str:
    """Generate a Vietnamese maintenance report using Google Gemini.

    The function expects GEMINI_API_KEY to be set in the environment.

    Args:
        metrics_dict: Dictionary containing vibration metrics.

    Returns:
        Generated report text.

    Raises:
        ValueError: If GEMINI_API_KEY is not configured.
    """

    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "Thiếu GEMINI_API_KEY. Hãy cấu hình biến môi trường GEMINI_API_KEY (hoặc trong file .env) trước khi chạy."
        )

    client = genai.Client()

    system_instruction = (
        "Bạn là một Chuyên gia Kỹ thuật Bảo trì hệ thống cơ điện tại một nhà máy thông minh. "
        "Nhiệm vụ của bạn là phân tích các thông số rung động bất thường và đưa ra cảnh báo "
        "ngắn gọn, chuyên nghiệp."
    )

    rms = float(metrics_dict["RMS"])
    kurt = float(metrics_dict["Kurtosis"])
    ptp = float(metrics_dict["Peak_to_Peak"])

    user_prompt = f"""Dữ liệu đo rung động bất thường (từ hệ thống giám sát):
- RMS: {rms:.6f}
- Kurtosis: {kurt:.6f}
- Peak-to-Peak: {ptp:.6f}

Hãy trả về báo cáo tiếng Việt gồm đúng 3 phần, có tiêu đề rõ ràng:
(1) Đánh giá tình trạng
(2) Nhận định rủi ro vật lý (đặc biệt khi Kurtosis cao)
(3) Đề xuất hành động khẩn cấp

Yêu cầu văn phong: ngắn gọn, chuyên nghiệp, hướng hành động, phù hợp bối cảnh bảo trì công nghiệp.
"""

    prompt = f"""{system_instruction}

{user_prompt}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def main() -> None:
    """Load model, simulate an event, score it, and optionally call Gemini."""

    load_dotenv()

    csv_path = "features_dataset.csv"
    model_path = "isolation_forest_model.pkl"

    if not os.path.exists(model_path):
        print(f"Không tìm thấy model file: {model_path}")
        print("Gợi ý: Hãy chạy model_training.py để tạo 'isolation_forest_model.pkl'.")
        return

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Không thể load model: {exc}")
        return

    try:
        metrics = simulate_realtime_data(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        print(f"Lỗi khi đọc dữ liệu realtime giả lập: {exc}")
        return

    sample_df = pd.DataFrame([metrics], columns=FEATURE_COLUMNS)

    try:
        pred = model.predict(sample_df)
    except Exception as exc:  # noqa: BLE001
        print(f"Lỗi khi dự đoán với model: {exc}")
        return

    if int(pred[0]) == -1:
        print("🚨 PHÁT HIỆN BẤT THƯỜNG! Đang kích hoạt AI Agent phân tích...")
        try:
            report = generate_maintenance_report(metrics)
        except ValueError as exc:
            print(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Lỗi khi gọi Gemini API: {exc}")
            return

        print("\n=== BÁO CÁO CHẨN ĐOÁN (Gemini) ===\n")
        print(report)


if __name__ == "__main__":
    main()
