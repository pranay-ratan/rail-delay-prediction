// Shared Plotly theme and small helpers for all dashboard pages.

export const DARK_LAYOUT = {
  paper_bgcolor: '#161B22',
  plot_bgcolor: '#0D1117',
  font: { family: 'Inter, sans-serif', color: '#8B949E', size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  yaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  hoverlabel: { bgcolor: '#161B22', bordercolor: '#30363D', font: { color: '#E6EDF3', size: 12 } },
};

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };

export const SERIES_COLORS = ['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA'];

/**
 * Fetch JSON and fail loudly on HTTP errors (instead of rendering an error body).
 * @param {string} url
 * @returns {Promise<any>}
 */
export async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}

/**
 * Severity label relative to the dataset's base rate.
 * @param {number} ratePct severe rate in percent
 * @param {number} baseRatePct dataset severe rate in percent
 */
export function relativeLevel(ratePct, baseRatePct) {
  if (ratePct >= 2 * baseRatePct) return 'HIGH';
  if (ratePct >= baseRatePct) return 'MEDIUM';
  return 'LOW';
}

export const LEVEL_BADGE = { HIGH: 'badge-high', MEDIUM: 'badge-medium', LOW: 'badge-low' };

/** Human-readable feature names. */
export const FEATURE_LABELS = {
  track_main: 'Main-line track',
  track_yard: 'Yard track',
  approx_speed_mph: 'Train speed',
  activity_proceeding: 'Train proceeding',
  activity_switching: 'Switching',
  activity_standing: 'Standing',
  train_freight: 'Freight train',
  train_passenger: 'Passenger train',
  train_cars_only: 'Cut of cars / single car',
  train_track_unit: 'Track unit',
  train_tonnage: 'Train tonnage',
  num_cars: 'Number of cars',
  num_locomotives: 'Locomotives',
  wx_temp_c: 'Temperature (day)',
  wx_precip_mm: 'Precipitation (day)',
  wx_wind_kmh: 'Wind (day)',
  hour: 'Hour of day',
  year: 'Year',
  month: 'Month',
  mile_post: 'Mile post',
  is_weekend: 'Weekend',
  rolling_12m_incidents: 'Province accidents, past 12 mo',
  prior_incidents_province: 'Province accidents, all prior',
  prior_severe_rate_province: 'Province past severe rate',
  incident_severity: 'SEVERE (target)',
};

export const label = (f) => FEATURE_LABELS[f] || f.replace(/^province_/, 'Province: ').replace(/_/g, ' ');
