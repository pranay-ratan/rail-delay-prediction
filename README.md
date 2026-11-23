# Canadian Rail Accident Severity Prediction

End-to-end ML on **real** Transportation Safety Board of Canada data: given that a rail accident happens,
how likely is it to **kill or seriously injure** someone, using only information known *before* the accident?

- **React dashboard + FastAPI** (Vercel): https://www.railwayriskcanada.xyz (domain must be re-pointed to the Vercel project; see Deploy)
- **Streamlit dashboard**: _add the `*.streamlit.app` URL after deploying (steps below)_
- **Author:** Pranay Ratan, BSc Data Science, Simon Fraser University · [GitHub](https://github.com/pranay-ratan)

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · React · Streamlit · Plotly

---

## Results (all numbers produced by `python -m src.train`)

**Data:** 28,233 TSB-reported rail accidents, 2001-2025 (25 years). **Target:** at least one fatality or serious
injury, **11.1%** of accidents, so the classes are imbalanced. **37 pre-incident features.**

### 10-fold stratified cross-validation

| Model | ROC-AUC | PR-AUC | F1 (severe) | Precision | Recall | Balanced acc. | Accuracy |
|---|---|---|---|---|---|---|---|
| Majority-class baseline | 0.500 | 0.111 | 0.000 | 0.000 | 0.000 | 0.500 | 0.889 |
| Logistic Regression | 0.866 | 0.436 | 0.449 | 0.301 | 0.887 | 0.814 | 0.758 |
| Random Forest | 0.888 | 0.495 | 0.504 | 0.370 | 0.786 | 0.809 | 0.827 |
| XGBoost | 0.888 | 0.498 | 0.505 | 0.375 | 0.776 | 0.807 | 0.831 |
| LightGBM | 0.890 | 0.503 | 0.498 | 0.361 | 0.804 | 0.813 | 0.819 |
| **XGBoost (tuned, final)** | **0.892** (±0.008) | 0.502 | 0.479 | 0.333 | 0.859 | 0.821 | 0.792 |

### Time-based holdout: train 2001-2020, test on unseen 2021-2025 (4,550 accidents)

| Model | ROC-AUC | PR-AUC (base 0.131) | F1 | Recall |
|---|---|---|---|---|
| **XGBoost (tuned, final)** | **0.875** | 0.472 | 0.495 | 0.850 |
| Random Forest | 0.875 | 0.486 | 0.510 | 0.792 |
| Logistic Regression | 0.869 | 0.476 | 0.464 | 0.914 |
| XGBoost (default) | 0.869 | 0.455 | 0.497 | 0.654 |
| LightGBM | 0.863 | 0.433 | 0.497 | 0.694 |

**Read accuracy with care:** always predicting "not severe" scores 88.9% accuracy with zero recall. ROC-AUC and
PR-AUC are the meaningful metrics here.

**Top SHAP drivers:** main-line track, train speed, train proceeding (vs switching), switching, tonnage,
temperature. Fatal and serious-injury accidents cluster on main-line running at speed (crossings, trespassers);
the far more common yard and switching accidents are rarely severe.

---

## What was fixed (and why the old numbers were wrong)

An audit of the previous version of this repo found:

1. **The data was synthetic.** The Transport Canada CKAN package ID in the loader did not exist, so the loader
   silently fell back to a random generator (`_generate_synthetic_dataset`, seed 42). All 8,500 "incidents"
   matched that generator exactly. The Open-Meteo request was also invalid (`monthly=` is not supported), so the
   weather was synthetic too. The "StatCan route density" was a hard-coded table.
2. **Target leakage.** The target was a rule over `incident_type`, `fatalities` and `injuries`, and all three were
   model inputs (plus `multi_fatality`, `cargo_x_type_risk`). A single decision tree on those three columns scored
   100%, which is why RF/XGBoost/LightGBM showed 1.0 on every metric.
3. **The published 90.1% / 0.956 never came from a run**: they were typed into the README; the notebook had no
   outputs and the only results file showed 1.0. On the synthetic data with leakage removed, every model scored
   ROC-AUC ≈ 0.50 (chance), as expected.

The rebuilt pipeline:

- **Real data:** TSB Rail Occurrence Database (RODS), accidents only, joined to each accident's lead train;
  daily Open-Meteo ERA5 weather for the accident date. No synthetic fallback: missing files raise an error.
- **Pre-incident features only** (`src/features.py`, an explicit allow-list): year, month, hour, weekend, province,
  mile post, track type (main/yard), train activity (proceeding/switching/standing), train type, speed, cars,
  locomotives, tonnage, daily weather, and provincial accident history computed **only from strictly earlier
  dates**. Accident type, injuries, damage, evacuation, etc. are outcomes and are excluded.
- **Missingness leak handled:** TSB fills in more fields for serious accidents. Blank categories are merged with
  "other", numeric gaps are median-imputed inside each CV fold, and the leak is measured: a model that sees only
  *which fields are blank* reaches ROC-AUC 0.658, while the tuned model scores 0.886 on fully complete records only
  (vs 0.892 overall), so it is not relying on blanks. Dropping train speed costs only ~0.016 AUC.
- **Evaluation:** 10-fold stratified CV for all models, XGBoost tuned with `RandomizedSearchCV` (30 candidates ×
  10 folds) on the 2001-2020 years only, plus the untouched 2021-2025 holdout. Real ROC curves come from
  out-of-fold predictions; the API's probabilities are calibrated against observed out-of-fold severe rates.
- **Tests** (`tests/`, 41 tests): feature/target rules, past-only history, API endpoints, training/serving parity
  (the API's feature code is a byte-identical copy of training's), and every Streamlit page.

---

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Get the data (once).** The TSB server rejects scripted downloads, so download these three CSVs in a browser from
[Rail occurrence data from January 1983 to present](https://open.canada.ca/data/en/dataset/4f8d9b0d-c4af-4be2-b751-3f5a403993ff)
and save them to `data/raw/tsb/`:
`RODSdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv`, `RODSdb_MDOTW_VW_TRAIN_PUBLIC.csv`, `RODSdb-dd.csv` (dictionary).

**2. Train, evaluate, copy artifacts into the backend, test:**

```bash
python -m src.train              # ~4 min; writes outputs/, models/, data/processed/
python scripts/sync_backend.py   # copies model + data + results into backend/
pytest
```

**3. Run the apps:**

```bash
cd backend && uvicorn server:app --port 8000          # API: http://localhost:8000/api/health
cd frontend && npm install && REACT_APP_BACKEND_URL=http://localhost:8000 npm start
streamlit run website/app.py                           # from the repo root
```

---

## Deploy

### Vercel (React + FastAPI, one project)

`vercel.json` defines two services: `frontend/` (Create React App) and `backend/` (FastAPI, `server:app`), with
`/api/*` routed to the API (the path is passed through unchanged) and everything else to React.

1. Import the GitHub repo in Vercel. Root directory: repo root (`./`). Vercel reads `vercel.json`.
2. Environment variables: none required. Leave `REACT_APP_BACKEND_URL` unset (same origin).
   `CI=false` is not needed: the frontend builds with zero ESLint warnings under `CI=true`.
3. Python 3.12 is pinned in `backend/.python-version`; `backend/requirements.txt` uses `xgboost-cpu`
   (the default Linux `xgboost` wheel pulls in ~200 MB of NVIDIA libraries) and ships the data as gzip CSV
   instead of parquet, keeping the function bundle ≈ 280 MB (limit 500 MB).
4. After deploying, open `/api/health`: it should show `"model_loaded": true, "data_loaded": true`.
5. Add the custom domain `railwayriskcanada.xyz` in Vercel and update the DNS records at the registrar
   to the values Vercel shows.

### Streamlit Community Cloud

1. https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub*.
2. Repository `pranay-ratan/rail-delay-prediction`, branch `main` (after merging), main file `website/app.py`.
3. **Advanced settings** → Python **3.12**. No secrets needed. Dependencies come from `website/requirements.txt`.
4. Deploy, then paste the `*.streamlit.app` URL at the top of this README.

---

## Project structure

```text
src/
  data_loader.py     TSB CSV loader + Open-Meteo daily weather (cached, rate-limit aware)
  features.py        the pre-incident feature definition (shared with the API)
  preprocessing.py   target, past-only history, weather join
  models.py          baseline + 4 classifiers, 10-fold CV, time holdout, XGBoost tuning, SHAP
  train.py           end-to-end run that writes every artifact
scripts/sync_backend.py   copies model + data + results into backend/
backend/             FastAPI app (server.py, predictor.py, resources.py, insights.py, data/, models/)
frontend/            React dashboard (4 pages)
website/             Streamlit dashboard (reuses backend/ modules)
outputs/             metrics CSVs, ROC/calibration JSON, SHAP plots
tests/               pytest suite
```

## Limitations

- The model predicts severity **conditional on an accident having happened**; it does not predict whether an
  accident will happen (TSB data contains only occurrences).
- "Speed just before the accident" is recorded after the fact by TSB; it is a pre-impact condition, but in real use
  it would be a planned or authorized speed.
- Weather is province-level (one point per province, at the median accident location), not per accident site.
- Not intended for operational safety decisions.
