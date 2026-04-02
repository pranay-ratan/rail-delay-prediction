# Project Brief: CPKC Rail Risk Intelligence Dashboard

You are building a production-grade interactive web dashboard for CPKC (Canadian Pacific Kansas City), North America's only transnational railroad. The dashboard visualizes and predicts rail incident severity using machine learning.

---

## What Exists Now

A Streamlit-based app with 4 pages:
- **Overview** — Project summary, motivation, key stats
- **EDA** — Static charts and basic interactivity
- **Model Results** — Metrics tables and ROC curves
- **Live Predictor** — Basic form with dropdowns

Tech stack: Python 3.13, pandas, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly

---

## Core Problems to Solve

1. **The current UI is generic** — It looks like a default Streamlit app. Needs branded, polished enterprise feel.
2. **Predictor is slow** — Running model inference on every submission is clunky.
3. **No real visualization depth** — Charts are basic, no drill-down, no interactivity layers.
4. **Missing narrative** — Pages don't tell a story for decision-makers (rail operations managers, safety engineers).
5. **No actionable insights** — Results are shown but there's no "so what?" for the business user.

---

## Vision

Build a beautiful, fast, interactive **Rail Risk Intelligence Platform** that a CPKC operations manager would bookmark. It should feel like a tool people use daily, not a homework submission.

Think: Datadog / Palantir / Grafana level of polish. Clean dark theme, responsive, with real data visualizations that invite exploration.

---

## Design Requirements

### Visual Identity
- **Dark mode by default** — Think deep navy (#0D1117), not pure black
- **Brand colors**: CPKC red (#C8102E), steel blue (#1A3A5C), green (#2E8B57) for positive
- **Typography**: Inter or SF Pro — clean, modern, professional
- **Cards-based layout** — Information organized in cards with subtle shadows, rounded corners
- **Animated transitions** — Smooth page transitions, loading spinners, chart animations

### Layout Structure
Replace Streamlit's default sidebar with a professional **left-nav panel** with:
- Logo + "Rail Risk Intelligence" at top
- Clean nav items with icons
- Status indicator at bottom ("Model: Random Forest • v1.0")
- User section with author info

### Page 1: Executive Dashboard (Home)
**What it should be:** A single-screen overview of the entire system
- Hero card with animated counter stats:
  - "8,500 incidents analyzed"
  - "25 predictive features"
  - "98.2% ROC-AUC accuracy"
  - "13 provinces/territories covered"
- Risk heatmap of Canada showing incident density by province (interactive map)
- Quick "Check Route Risk" widget — enter a province, get instant risk score
- Recent predictions table with severity badges
- Mini trend chart showing incident rate over time

### Page 2: Deep Dive Explorer (Replaces EDA)
**What it should be:** An interactive data exploration tool
- **Tabbed interface:**
  - **Tab 1: Geographic View** — Interactive Canada map with chloropleth. Click a province → show incident breakdown, top incident types, rolling trend
  - **Tab 2: Temporal View** — Time series with filters (year range, month, season). Seasonal decomposition chart with observed/trend/residual
  - **Tab 3: Correlation Explorer** — Interactive feature correlation matrix. Click a cell → show scatter plot of those two features
  - **Tab 4: Incident Typology** — Sankey diagram or sunburst showing incident type → severity → outcome relationships

### Page 3: Model Command Center (Replaces Model Results)
**What it should be:** Full model transparency and comparison
- **Model leaderboard** — Ranked table with sortable columns (AUC, F1, Precision, Recall, Accuracy)
- **Model detail cards** — Click a model to expand:
  - Parameter summary
  - Confusion matrix (interactive — hover to see counts/percentages)
  - Precision-Recall curve + ROC curve side by side
  - Feature importance bar chart (top 15 features)
  - Training time and model size
- **Model comparison mode** — Checkbox to overlay ROC curves for multiple models
- **What-If analysis section** — Sliders for feature values → see how predicted probability changes in real-time (uses pre-loaded model)
- **SHAP section**:
  - Beeswarm summary plot (interactive)
  - Individual prediction explanation — "Why did the model flag this as high risk?"
  - Feature dependence plots for top 3 features

### Page 4: Live Risk Assessor (Replaces Live Predictor)
**What it should be:** The money page — where users get instant answers
- **Input form that feels like a tool, not a form:**
  - Province selector with visual map highlight
  - Incident type dropdown with descriptions/tooltips
  - Cargo type selector with color-coded risk levels (red = Dangerous Goods, green = Grain)
  - Temperature slider with seasonal reference
  - Date picker with season auto-detect
  - Track class radio buttons styled as chips
  - "Advanced" section (collapsed) for mileage, subdivision, weather features
- **Results panel that appears instantly:**
  - Large gauge chart showing risk probability (0-100%)
  - Severity badge: LOW / MEDIUM / HIGH with color coding
  - "Top 3 factors driving this prediction" — ranked list with impact direction
  - "Similar historical incidents" — 5 examples from training data
  - Recommendation card: "Based on similar cases, consider: [actionable bullet points]"
  - "Compare with baseline" — toggle to see average risk for this province/incident type
- **Prediction history** — Last 10 assessments shown below (stored in session state)
- **Export button** — Download prediction as PDF/CSV

---

## Technical Requirements

### Performance
- Pre-load model at app startup; predictions should take <200ms
- Cache all charts aggressively (Streamlit @st.cache_data)
- Lazy load heavy charts (only render when user scrolls to them)
- Use Plotly for interactive charts, not matplotlib (faster rendering in browser)

### State Management
- Use `st.session_state` for prediction history
- Remember user's last inputs for quick re-runs
- Persist model in `/models/final_model.joblib`

### Responsiveness
- Mobile-friendly: forms should stack vertically on small screens
- Charts should use Plotly's responsive mode
- Use Streamlit's `st.columns` for grid layouts, with breakpoints consideration

### Error Handling
- Graceful messages if model fails to load
- Input validation on the predictor page
- "No matching historical incidents" → helpful empty state

---

## What Not To Do
- No default Streamlit theme — use custom dark theme
- No basic `st.bar_chart()` — use Plotly throughout
- No "loading… please wait" without a spinner
- No markdown-only pages — every page should have visual elements
- No model metrics without context — always explain what they mean
- No wall-of-text explanations — use cards, badges, and bullet points

---

## Files to Preserve

The existing backend code is solid. Do NOT change:
- `src/data_loader.py` — Data loading logic
- `src/preprocessing.py` — Feature engineering
- `src/models.py` — Model training and SHAP
- `src/visualizations.py` — Chart generation
- `models/final_model.joblib` — Trained model
- `data/processed/incidents_featured.parquet` — Processed data

**Only modify:** `website/app.py`, `website/pages/*.py`, `website/assets/`

The model expects 25 feature columns:
```
['year', 'month', 'day', 'province_risk_score', 'route_density_score',
 'incident_type_encoded', 'cargo_risk', 'multi_fatality',
 'rolling_12m_incidents', 'cumulative_incidents_province',
 'avg_temp_c', 'total_precip_mm', 'avg_wind_kmh',
 'season_num', 'season_x_province_risk', 'cargo_x_type_risk',
 'temp_x_cargo_risk', 'density_x_rolling', 'years_since_2000',
 'is_weekend', 'fatalities', 'injuries', 'evacuations',
 'mile_post', 'temperature_c']
```

---

## Success Criteria

A successful result is a dashboard that:
1. Looks like a product, not a prototype
2. Loads in <3 seconds
3. Makes predictions feel instant
4. Tells a story: problem → data → insight → action
5. Would impress someone in a job interview