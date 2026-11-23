# Canadian Rail Incident Severity: Pipeline Demo

An end-to-end machine-learning pipeline and dashboard for predicting whether a Canadian rail occurrence will be high-severity, built as a portfolio project with rail operators (CPKC) in mind. **It currently runs on generated data, so it shows the workflow, not findings about Canadian rail.**

**Live demo:** not currently online · **Code:** this repo

![Dashboard screenshot](docs/screenshot-dashboard.png)
<!-- TODO: add docs/screenshot-dashboard.png (Streamlit overview page) and docs/screenshot-predictor.png (live risk predictor) -->

**Stack:** Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit · Plotly · FastAPI + React (alternate front end)

## Status: read this first
- **The data is generated, not real.** `src/data_loader.py` asks the Open Canada CKAN API for a Transport Canada dataset ID that no longer exists, so it always falls back to `_generate_synthetic_dataset()`. All 8,500 rows in `data/processed/incidents_featured.parquet` have `_source_file = "synthetic_data"`.
- **The model scores are not meaningful yet.** The target (`incident_severity`) is defined from incident type, fatalities and injuries, and those same columns are also used as features. That is target leakage: `outputs/results/model_metrics.csv` shows Random Forest, XGBoost and LightGBM all at 1.000 accuracy and ROC-AUC (logistic regression: 0.749 ROC-AUC).
- **No key results are claimed.** Earlier versions of this README listed metrics and findings that the code did not produce; they have been removed.

Next steps: load the real Transportation Safety Board rail occurrence data, drop leaked columns from the feature set, and re-run.

## What it does
1. **Load** occurrence records (currently generated), monthly weather per province (Open-Meteo, with a generated fallback) and route-density constants (`src/data_loader.py`).
2. **Clean and engineer features**: season, rolling 12-month incidents per province, province risk score, cargo risk, weather and interaction terms (`src/preprocessing.py`).
3. **Train and compare** logistic regression, random forest, XGBoost and LightGBM with stratified 10-fold CV, tune the best one, and explain it with SHAP (`src/models.py`).
4. **Explore** results in a Streamlit dashboard: overview, EDA, model results and a live risk predictor (`website/`).

## Run it locally

```bash
git clone https://github.com/pranay-ratan/rail-delay-prediction.git
cd rail-delay-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/rail_incident_prediction.ipynb   # full pipeline
streamlit run website/app.py                                 # dashboard at http://localhost:8501
```

An alternate React front end (`frontend/`) talks to a FastAPI backend (`backend/server.py`).

## Project structure

```text
data/processed/        feature-engineered parquet (generated data)
notebooks/             end-to-end walkthrough
src/                   data_loader, preprocessing, models, visualizations
outputs/results/       model_metrics.csv
website/               Streamlit multi-page dashboard
frontend/, backend/    React + FastAPI version of the dashboard
```

## Licence
Code by Pranay Ratan. If real Transport Canada / TSB data is added, it is used under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).

[LinkedIn](https://www.linkedin.com/in/pranay-ratan/) · [GitHub](https://github.com/pranay-ratan)
