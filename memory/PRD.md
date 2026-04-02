# Rail Risk Intelligence Dashboard — PRD

## Original Problem Statement
Build a production-grade interactive web dashboard that visualizes and predicts rail incident severity using machine learning. Replace the existing Streamlit app with an enterprise-grade React+FastAPI dashboard with dark theme, interactive Plotly charts, and ML-powered predictions.

## Architecture
- **Backend**: FastAPI (Python) on port 8001 — serves incident data, model metrics, and ML predictions
- **Frontend**: React on port 3000 — dark-themed enterprise dashboard with Plotly.js (CDN)
- **ML Model**: Random Forest classifier trained on 8,500 incidents with 25 engineered features
- **Data**: Transport Canada Railway Occurrence Statistics (parquet format)

## Tech Stack
- React 18, Plotly.js (CDN), react-countup, lucide-react, framer-motion
- FastAPI, pandas, scikit-learn, XGBoost, LightGBM, joblib
- Custom dark theme CSS (no Tailwind — pure CSS for performance)

## User Personas
1. **Rail Operations Manager** — Needs executive dashboard view, quick risk assessment
2. **Safety Engineer** — Needs deep dive into data, model transparency, SHAP interpretation
3. **Data Scientist** — Needs model comparison, feature importance, correlation analysis

## Core Requirements (Static)
- Dark theme (#0D1117 background) with CPKC brand colors
- Left sidebar navigation with model status indicator
- 4 pages: Executive Dashboard, Deep Dive Explorer, Model Command Center, Live Risk Assessor
- Interactive Plotly charts with dark theme styling
- ML-powered risk predictions with feature contribution analysis
- No emoji usage — professional enterprise language throughout

## What's Been Implemented (Jan 2026)
1. FastAPI backend with 12 API endpoints (health, stats, provinces, annual, seasonal, type, heatmap, correlation, severity, models/metrics, feature-importance, predict)
2. React frontend with 4 fully interactive pages
3. Custom Plot component using Plotly.js CDN (avoids memory issues with npm plotly.js)
4. Executive Dashboard: animated stat counters, annual trend, seasonal distribution, choropleth map, donut chart, province risk table
5. Deep Dive Explorer: 4 tabs (Geographic heatmap, Temporal trends with rolling 12m, Correlation matrix, Incident typology)
6. Model Command Center: Leaderboard table, ROC curve comparison with model selection, Feature importance (top 15), Radar chart
7. Live Risk Assessor: Input form with chips/sliders, Gauge chart, Feature contributions, Recommendations, Similar incidents, Prediction history
8. Streamlit config updated for dark theme (GitHub/Streamlit Cloud deployment)
9. Trained Random Forest model saved to /app/models/final_model.joblib

## Prioritized Backlog
### P0 (Done)
- All 4 dashboard pages fully functional
- All backend APIs operational
- Dark theme consistent across all pages
- ML predictions working

### P1 (Next)
- SHAP beeswarm plot on Model Command Center (requires shap computation on backend)
- What-If Analysis with real-time sliders affecting prediction probability
- Export prediction as PDF/CSV
- Confusion Matrix visualization (interactive)

### P2 (Future)
- Real-time data pipeline from Transport Canada API
- User authentication and role-based access
- Prediction audit trail stored in MongoDB
- Mobile-optimized responsive layout improvements
- Progressive Web App (PWA) support

## Next Tasks
1. Add SHAP interpretation section with beeswarm and dependence plots
2. Implement What-If Analysis with dynamic sliders
3. Add prediction export (PDF/CSV)
4. Improve prediction calibration for more realistic probability distribution
5. Push updated code to GitHub for Streamlit Cloud deployment
