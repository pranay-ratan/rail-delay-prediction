# 🚂 Canadian Rail Incident & Delay Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-lightgreen)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34-red?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.21-blue)](https://plotly.com/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas)](https://pandas.pydata.org/)

**Live Demo:** [railriskcanadaxyz]([https://rail-incident-prediction.streamlit.app](https://www.railwayriskcanada.xyz/))  
**Author:** Pranay Ratan — BSc Data Science, Simon Fraser University  
**Contact:** [LinkedIn](https://linkedin.com/in/pranayratan) · [GitHub](https://github.com/pranayratan)

---

## 🎯 Project Overview

This end-to-end machine learning project builds an **incident severity prediction system** for Canada's rail network, using 25 years of Transport Canada Railway Occurrence Statistics enriched with Open-Meteo weather data and Statistics Canada route density proxies.

Built with **CPKC (Canadian Pacific Kansas City)** in mind — North America's only transnational railroad — this system demonstrates how historical occurrence patterns, geographic risk factors, and seasonal signals can be combined to produce **proactive risk intelligence** for operations teams.

### Why This Matters for CPKC

Canada's rail network moves ~$400 billion in goods annually. Every unplanned incident carries:
- Direct infrastructure repair costs ($500K–$10M+ per derailment)
- Regulatory fines under the Transportation of Dangerous Goods Act
- Crew safety consequences and service disruption cascades

Predicting high-severity incidents **before they occur** — rather than responding after — allows corridor managers to deploy maintenance crews, adjust train speeds, and prioritize inspections with data-driven precision.

---

## 📊 Dataset

| Dataset | Source | Access |
|---------|--------|--------|
| Railway Occurrence Statistics | [Transport Canada / Open Canada](https://open.canada.ca/data/en/dataset/1dc5304e-6d54-4b15-a907-cd5e23f69c25) | CKAN API — automatic |
| Monthly Weather by Province | [Open-Meteo ERA5 Archive](https://open-meteo.com/) | Free REST API — no key |
| Route Density Proxy | Statistics Canada Track Km + 2021 Census | Embedded constants |

All data is loaded **programmatically** — zero manual downloads required.

---

## 🏗️ Project Structure

```text
rail-delay-prediction/
│
├── data/
│   ├── raw/                        ← Transport Canada CSV + weather parquet cache
│   └── processed/                  ← Feature-engineered parquet (15+ features)
│
├── notebooks/
│   └── rail_incident_prediction.ipynb   ← End-to-end walkthrough (Steps 1–7)
│
├── src/
│   ├── data_loader.py              ← CKAN API + Open-Meteo + quality report
│   ├── preprocessing.py            ← Cleaning, feature engineering (15+ features)
│   ├── models.py                   ← 4 classifiers, 10-fold CV, SHAP, tuning
│   └── visualizations.py           ← Matplotlib + Plotly charts (10+ figures)
│
├── models/
│   └── final_model.joblib          ← Serialized best estimator + feature names
│
├── outputs/
│   ├── figures/                    ← All publication-quality saved plots
│   └── results/
│       └── model_metrics.csv       ← Full CV metrics for all models
│
├── website/                        ← Streamlit multi-page dashboard
│   ├── app.py                      ← Entry point + sidebar
│   ├── pages/
│   │   ├── 01_overview.py          ← Hero, stats, key findings
│   │   ├── 02_eda.py               ← Interactive EDA with filters
│   │   ├── 03_model_results.py     ← Metrics, ROC, SHAP, confusion matrix
│   │   └── 04_live_predictor.py    ← Real-time risk scoring form
│   └── assets/
│       └── cpkc_theme.css          ← CPKC red (#C8102E) theme
│
├── README.md
├── requirements.txt                ← Version-pinned dependencies
├── Procfile                        ← Streamlit Cloud deployment
├── runtime.txt                     ← Python 3.11.9
└── .gitignore
```

---

## ⚙️ Installation & Usage

### 1. Clone & Install

```bash
git clone https://github.com/pranayratan/rail-delay-prediction.git
cd rail-delay-prediction
python -m venv venv && source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook (Full Pipeline)

```bash
cd notebooks
jupyter notebook rail_incident_prediction.ipynb
```

Run all cells top-to-bottom. The notebook will:
- Fetch Transport Canada data via API (cached after first run)
- Fetch Open-Meteo weather data
- Engineer 15+ features
- Train 4 ML models with 10-fold CV
- Run hyperparameter tuning
- Compute SHAP values
- Save `models/final_model.joblib` and all figures

### 3. Launch Streamlit Dashboard

```bash
streamlit run website/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🚀 Deploy to Streamlit Cloud (One-Click)

1. Push repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set **Main file path** to `website/app.py`
4. Click **Deploy** — Streamlit Cloud reads `requirements.txt` and `runtime.txt` automatically

The `Procfile` is also included for Heroku/Render compatibility.

---

## 🔬 Key Findings

- **~23% decline** in total rail occurrences from peak (2005) to 2024, validating the impact of TC safety regulations
- **Winter months carry 1.4× higher severity probability** than summer — driven by track contraction, ice, and reduced visibility
- **Ontario + Quebec account for ~40% of incidents** (network density), but Saskatchewan/Alberta lead in derailment rate
- **XGBoost achieved ROC-AUC > 0.95** on 10-fold stratified CV — substantially outperforming the Logistic Regression baseline (AUC ~0.80)
- **Rolling 12-month incident count, province risk score, and cargo type** are the top 3 SHAP predictors — targeting the top-10% highest-scoring corridor-season combinations would cover an estimated **62% of all high-severity incidents**

---

## 🤖 Models & Performance

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 0.741 | 0.740 | 0.802 |
| Random Forest | 0.879 | 0.877 | 0.931 |
| XGBoost | 0.893 | 0.892 | 0.949 |
| LightGBM | 0.888 | 0.886 | 0.943 |
| **XGBoost (Tuned)** | **0.901** | **0.900** | **0.956** |

All metrics from 10-fold stratified cross-validation. Best model saved to `models/final_model.joblib`.

---

## 🛠️ Feature Engineering (15+ Features)

| Feature | Description |
|---------|-------------|
| `season` | Winter/Spring/Summer/Fall from month |
| `decade` | 2000s/2010s/2020s grouping |
| `is_weekend` | Binary weekend flag |
| `years_since_2000` | Continuous time trend feature |
| `province_risk_score` | Province encoded by historical incident frequency |
| `route_density_score` | Track-km per 100k population (Statistics Canada) |
| `incident_type_encoded` | Label encoded incident type |
| `cargo_risk` | Ordinal risk by cargo (1=low, 3=high) |
| `multi_fatality` | Binary: 2+ fatalities |
| `incident_severity` | **Target** — binary HIGH/LOW |
| `rolling_12m_incidents` | 12-month rolling count per province |
| `cumulative_incidents_province` | Province-level cumulative count |
| `season_x_province_risk` | Interaction: season × province risk |
| `cargo_x_type_risk` | Interaction: cargo risk × incident type |
| `temp_x_cargo_risk` | Interaction: temperature × cargo risk |
| `density_x_rolling` | Interaction: route density × rolling incidents |

---

## 📁 Data Quality Report

The `data_loader.print_data_quality_report()` function produces a full report on:
- Shape, dtypes, memory usage
- Null counts and percentages per column
- Duplicate row count
- Unique value counts per column

Missing value fill strategies are fully documented in `src/preprocessing.py`.

---

## 📄 License

This project uses Transport Canada open data under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).

---

*Built by Pranay Ratan · SFU Data Science · April 2026*
