# Predictive Maintenance with Machine Learning and Generative AI

## Overview
An end-to-end artificial intelligence pipeline designed for industrial predictive maintenance. This project analyzes time-series vibration data from motor bearings, detects mechanical anomalies using unsupervised machine learning, and leverages a Generative AI Agent (Google Gemini) to automatically generate actionable diagnostic reports.

## Key Features
* **Signal Processing & Feature Engineering:** Processes raw vibration `.mat` files (CWRU Dataset) using sliding windows to extract time-domain physical features including Root Mean Square (RMS), Variance, Kurtosis, and Peak-to-Peak amplitude.
* **Anomaly Detection:** Implements an `Isolation Forest` model trained exclusively on healthy operational data to detect out-of-distribution mechanical faults without requiring historical failure data.
* **LLM Diagnostic Agent:** Integrates the `google-genai` SDK to parse anomaly metrics in real-time and generate natural language maintenance reports, effectively acting as an automated reliability engineer.

## Project Structure
```text
.
├── archive/                      # CWRU raw data files (local; NOT tracked by git)
├── data_preparation.py           # Loads .mat files into pandas DataFrames & plots signals
├── feature_extraction.py         # Slices data into windows & extracts physical features
├── features_dataset.csv          # Engineered dataset (generated; NOT tracked by git)
├── model_training.py             # Trains and evaluates the Isolation Forest model
├── isolation_forest_model.pkl    # Saved ML model artifact (generated; NOT tracked by git)
├── llm_integration.py            # Simulates real-time streaming and triggers the LLM Agent
├── .env                          # Environment variables (API Keys - Not tracked by git)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignore rules (keeps secrets & data out of repo)
└── README.md                     # Project documentation

```

## Installation & Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib google-genai python-dotenv
```

3. Configure the LLM API key (do not commit secrets):

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
GEMINI_API_KEY="your_api_key_here"
```

## System Execution Pipeline

Execute the modules in the following order:

### Phase 1: Data Preparation & Visualization

```bash
python data_preparation.py
```

### Phase 2: Feature Engineering

```bash
python feature_extraction.py
```

### Phase 3: Model Training

```bash
python model_training.py
```

### Phase 4: LLM Agent Integration

```bash
python llm_integration.py
```

## Technologies Used

* Core Data Stack: Python, Pandas, NumPy, SciPy
* Machine Learning: Scikit-Learn (Isolation Forest)
* Generative AI: Google GenAI SDK (Gemini 2.5 Flash)
* Domain: Mechanical Vibration Analysis, Reliability Engineering

## Author

Kim Anh Quân  
Computer Science | Hanoi University of Science and Technology (HUST)

## License

This project is for educational and portfolio demonstration purposes. Data provided by the Case Western Reserve University (CWRU) Bearing Data Center.