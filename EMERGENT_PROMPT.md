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

Repository: https://github.com/pranay-ratan/rail-delay-prediction

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

### Tone & Professionalism
- **NO random emojis anywhere** — This is a professional enterprise tool, not a casual internal app. Absolutely avoid decorative emojis like 🚂📊🤖🎯🔥💡🔍📋✅❌ as bullet points, section headers, or decoration. Use only purposeful icons if they serve a clear functional navigation purpose.
- **Professional language** — All labels, tooltips, and descriptions should read like they belong in a corporate risk management platform. Use precise terminology: "Incident Severity Score" not "How risky is this?"
- **No playful copy** — Avoid "Let's go!", "Awesome!", "Here you go!" Language should be crisp, neutral, and action-oriented.

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
  - **Tab 1: Geographic View** — Interactive Canada map with choropleth. Click a province → show incident breakdown, top incident types, rolling trend
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

## Graphics & Visualization Standards

### Chart Quality
- **Custom styling only**: No default Plotly or matplotlib styles. Apply the dark theme consistently across all charts.
- **Clear visual hierarchy**: Titles > axis labels > data > annotations. Use font weight and size to guide the eye.
- **Insight-first design**: Every chart should answer a question. Put the insight (e.g., "Derailments are 40% more likely in winter") as a text callout ABOVE the chart, not below it.
- **Avoid chart junk**: No 3D effects, no decorative elements, no redundant legends. Every pixel should serve a purpose.

### Chart Implementation Notes
- **Risk map**: Use Plotly `choropleth` with a continuous color scale from dark green (low) to dark red (high). Add dropdown for incident type filtering.
- **Trend lines**: Plotly `line` with shaded confidence band. Annotate significant events or seasonal peaks.
- **Feature importance**: Horizontal bars sorted descending, with each bar colored by whether increasing that feature increases or decreases risk.
- **Model comparison**: Overlaid ROC curves with AUC in legend. No area fill — keep it clean.
- **Live predictor gauge**: Build with Plotly `indicator` chart but style it to match the dark theme. Segment into LOW/MEDIUM/HIGH zones with distinct colors.
- **Waterfall chart**: For the risk assessor, show baseline probability → feature contributions → final prediction. Use green bars for risk-decreasing factors, red for risk-increasing.

### Insight Presentation Standards
Every page must have an **insight summary block** that distills what the user is looking at:
- Use compact stat callouts: `22% of all incidents | Ontario | 1,870 cases`
- Use color-coded severity badges: `LOW` (green), `MEDIUM` (amber), `HIGH` (red)
- Use horizontal rule separators between sections, not card borders for every section
- Include brief interpretive text: "An AUC of 0.98 means the model correctly ranks a random HIGH-risk incident above a random LOW-risk incident 98% of the time."
- Every chart must be paired with a one-sentence takeaway that answers "so what?"

### What-If Visualizations
On the Live Risk Assessor, the prediction result should include:
- **Risk waterfall**: Show how the prediction moves from baseline (e.g., 35%) to final (e.g., 72%) driven by each input feature
- **Feature contribution bars**: "Province: Ontario (+8%)", "Cargo: Dangerous Goods (+15%)", "Season: Winter (+4%)"
- **Confidence indicator**: "Model confidence: HIGH (this input is well within the training data distribution)" vs "Model confidence: LOW (some input values are unusual — interpret with caution)"

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
- **NO random emojis** — Do not use emojis as bullet points, section headers, or decoration. This is an enterprise risk intelligence tool.
- **No generic Streamlit widgets** — No default `st.metric`, `st.expander`, `st.selectbox` without custom theming. Every component should be styled to match the dark theme.
- **No "toy project" aesthetics** — This should not look like a Kaggle notebook or class project. It should look like software a Fortune 500 company licenses.

---

## Files to Preserve

The existing backend code is solid. Do NOT change:
- `src/data_loader.py` — Data loading logic
- `src/preprocessing.py` — Feature engineering
- `src/models.py` — Model training and SHAP
- `src/visualizations.py` — Chart generation
- `models/final_model.joblib` — Trained model
- `data/processed/incidents_featured.parquet` — Processed data

**Only modify:** `website/app.py`, `website/pages/*.py`, `website/assets/`, `.streamlit/config.toml`

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

## Deployment Instructions

When you are done building, all changes must be committed and pushed to the repository:

1. **Commit all changes** to https://github.com/pranay-ratan/rail-delay-prediction
2. The repo already has a `Procfile` and `requirements.txt` — ensure `requirements.txt` includes any new packages you add
3. **Ensure the app deploys to Streamlit Cloud** by pushing to the `main` branch — the entire project should be deployable with a single push
4. Do NOT leave any files uncommitted. All code, assets, and configs must be in the repo
5. Follow the existing file convention: `website/pages/01_overview.py`, `website/pages/02_eda.py`, etc.
6. Create or update `.streamlit/config.toml` to enable the dark theme and custom styling by default, so deployment works immediately without manual configuration
7. After committing, verify that `streamlit run website/app.py` works from a clean checkout
8. The goal: push to `main` → Streamlit Cloud auto-deploys → URL works in 30 seconds with zero manual configuration

---

## Success Criteria

A successful result is a dashboard that:
1. Looks like a product, not a prototype
2. Loads in <3 seconds
3. Makes predictions feel instant
4. Tells a story: problem → data → insight → action
5. Would impress someone in a job interview
6. Has zero emoji abuse — reads as professional enterprise software
7. Every chart has a clear insight, not just data dumped on screen
8. Custom dark theme applied consistently — no default widget styling leaks through
9. All changes committed to GitHub — the repo is the single source of truth