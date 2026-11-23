import React, { useState, useCallback } from 'react';
import Plot from '../components/Plot';
import { DARK_LAYOUT, PLOT_CONFIG, LEVEL_BADGE } from '../components/chartTheme';

const PROVINCES = [
  'Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador',
  'Northwest Territories', 'Nova Scotia', 'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan',
];
const TRACK = [['main', 'Main line'], ['yard', 'Yard'], ['other', 'Other / siding']];
const ACTIVITY = [['proceeding', 'Proceeding'], ['switching', 'Switching'], ['standing', 'Standing'], ['other', 'Other']];
const TRAIN = [['freight', 'Freight'], ['passenger', 'Passenger'], ['cars_only', 'Cut of cars'], ['track_unit', 'Track unit'], ['other', 'Other']];
const OPTIONAL_NUMBERS = [
  ['approx_speed_mph', 'Train speed (mph)', 0, 120],
  ['num_cars', 'Number of cars', 0, 400],
  ['num_locomotives', 'Locomotives', 0, 20],
  ['train_tonnage', 'Tonnage', 0, 40000],
  ['wx_temp_c', 'Temperature (°C)', -60, 50],
  ['wx_precip_mm', 'Precipitation (mm)', 0, 300],
];
const RISK_COLORS = { HIGH: '#FF4444', MEDIUM: '#D29922', LOW: '#2E8B57' };
const HISTORY_LIMIT = 10;

const INITIAL_FORM = {
  province: 'Ontario', year: 2025, month: 1, hour: 12, is_weekend: false,
  track_type: 'main', activity: 'proceeding', train_type: 'freight',
  approx_speed_mph: '40', num_cars: '', num_locomotives: '', train_tonnage: '', wx_temp_c: '', wx_precip_mm: '',
};

/** Blank optional fields are sent as null ("unknown") rather than as 0. */
function toPayload(form) {
  const numbers = Object.fromEntries(OPTIONAL_NUMBERS.map(([k]) => [k, form[k] === '' ? null : Number(form[k])]));
  return { ...form, ...numbers };
}

function Chips({ options, value, onChange, testid }) {
  return (
    <div className="chip-group">
      {options.map(([v, text]) => (
        <button key={v} className={`chip ${value === v ? 'active' : ''}`} onClick={() => onChange(v)} data-testid={`${testid}-${v}`}>
          {text}
        </button>
      ))}
    </div>
  );
}

export default function LiveRiskAssessor({ api }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const set = (field, value) => setForm(prev => ({ ...prev, [field]: value }));

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${api}/api/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(toPayload(form)),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid input');
      setResult(data);
      setHistory(prev => [
        { ...form, id: data.timestamp, probability: data.probability, risk_level: data.risk_level },
        ...prev.slice(0, HISTORY_LIMIT - 1),
      ]);
    } catch (e) {
      setError(`Prediction failed: ${e.message}`);
    }
    setLoading(false);
  }, [api, form]);

  const maxContribution = result ? Math.max(...result.contributions.map(c => Math.abs(c.value)), 1e-6) : 1;

  return (
    <div data-testid="live-risk-assessor">
      <div className="page-header">
        <h2>Live Risk Assessor</h2>
        <p>If an accident happens under these conditions, how likely is it to kill or seriously injure someone?</p>
      </div>

      <div className="grid-1-2">
        <div>
          <div className="card" style={{ position: 'sticky', top: 20 }}>
            <div className="card-header"><div className="card-title">Scenario (pre-incident information only)</div></div>
            <div className="card-body">
              <div className="form-group">
                <label className="form-label">Province</label>
                <select className="form-select" value={form.province} onChange={e => set('province', e.target.value)} data-testid="select-province">
                  {PROVINCES.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>

              <div className="form-group"><label className="form-label">Track</label>
                <Chips options={TRACK} value={form.track_type} onChange={v => set('track_type', v)} testid="track" /></div>
              <div className="form-group"><label className="form-label">Train activity</label>
                <Chips options={ACTIVITY} value={form.activity} onChange={v => set('activity', v)} testid="activity" /></div>
              <div className="form-group"><label className="form-label">Train type</label>
                <Chips options={TRAIN} value={form.train_type} onChange={v => set('train_type', v)} testid="train" /></div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">Year: {form.year}</label>
                  <input type="range" className="form-slider" min={2001} max={2030} value={form.year} onChange={e => set('year', parseInt(e.target.value, 10))} data-testid="slider-year" />
                </div>
                <div className="form-group">
                  <label className="form-label">Month: {form.month}</label>
                  <input type="range" className="form-slider" min={1} max={12} value={form.month} onChange={e => set('month', parseInt(e.target.value, 10))} data-testid="slider-month" />
                </div>
                <div className="form-group">
                  <label className="form-label">Hour: {form.hour}:00</label>
                  <input type="range" className="form-slider" min={0} max={23} value={form.hour} onChange={e => set('hour', parseInt(e.target.value, 10))} data-testid="slider-hour" />
                </div>
                <div className="form-group">
                  <label className="form-label">Day</label>
                  <Chips options={[['weekday', 'Weekday'], ['weekend', 'Weekend']]} value={form.is_weekend ? 'weekend' : 'weekday'}
                    onChange={v => set('is_weekend', v === 'weekend')} testid="day" />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {OPTIONAL_NUMBERS.map(([key, text, min, max]) => (
                  <div className="form-group" key={key}>
                    <label className="form-label">{text}</label>
                    <input type="number" className="form-select" min={min} max={max} placeholder="unknown"
                      value={form[key]} onChange={e => set(key, e.target.value)} data-testid={`input-${key}`} />
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 8 }}>
                Blank = unknown. Missing weather uses the Open-Meteo median for that province and month.
              </div>

              <button className="btn btn-primary" style={{ width: '100%', marginTop: 8 }} onClick={handleSubmit} disabled={loading} data-testid="predict-button">
                {loading ? 'Scoring...' : 'Assess severity risk'}
              </button>
              {error && <div style={{ color: '#FF6B6B', fontSize: 12, marginTop: 8 }} data-testid="predict-error">{error}</div>}
            </div>
          </div>
        </div>

        <div>
          {result ? (
            <>
              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">Result: {result.model}</div>
                  <span className={`badge ${LEVEL_BADGE[result.risk_level]}`}>{result.risk_level} RISK</span>
                </div>
                <div className="card-body" data-testid="risk-result">
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                    <div data-testid="risk-gauge-chart">
                      <Plot
                        data={[{
                          type: 'indicator', mode: 'gauge+number', value: result.probability * 100,
                          number: { suffix: '%', valueformat: '.1f', font: { size: 36, color: RISK_COLORS[result.risk_level], family: 'JetBrains Mono' } },
                          gauge: {
                            axis: { range: [0, 100], tickcolor: '#6E7681', tickfont: { color: '#6E7681' } },
                            bar: { color: RISK_COLORS[result.risk_level], thickness: 0.3 },
                            bgcolor: '#0D1117', borderwidth: 1, bordercolor: '#30363D',
                            steps: [
                              { range: [0, result.base_rate * 100], color: 'rgba(46,139,87,0.12)' },
                              { range: [result.base_rate * 100, result.base_rate * 200], color: 'rgba(210,153,34,0.12)' },
                              { range: [result.base_rate * 200, 100], color: 'rgba(200,16,46,0.12)' },
                            ],
                          },
                        }]}
                        layout={{ ...DARK_LAYOUT, height: 250, margin: { l: 30, r: 30, t: 30, b: 10 } }}
                        config={PLOT_CONFIG}
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div style={{ fontSize: 13, color: '#8B949E', lineHeight: 1.6 }}>
                      <p style={{ marginTop: 0 }}>
                        Among past accidents the model scored like this one, <b style={{ color: '#E6EDF3' }}>{(result.probability * 100).toFixed(1)}%</b>{' '}
                        killed or seriously injured someone: <b style={{ color: '#E6EDF3' }}>{result.relative_risk}×</b> the overall rate
                        of {(result.base_rate * 100).toFixed(1)}%.
                      </p>
                      <p>
                        HIGH ≥ 2× base rate, MEDIUM ≥ 1×. Province history as of {result.inputs.history_as_of}:{' '}
                        {result.inputs.rolling_12m_incidents} accidents in the prior 12 months.
                      </p>
                      {(result.inputs.weather_filled_from_climatology.length > 0 || result.inputs.imputed_by_model.length > 0) && (
                        <p style={{ fontSize: 11, color: '#6E7681' }}>
                          {result.inputs.weather_filled_from_climatology.length > 0 && `Weather from climatology: ${result.inputs.weather_filled_from_climatology.join(', ')}. `}
                          {result.inputs.imputed_by_model.length > 0 && `Unknown, filled with training medians: ${result.inputs.imputed_by_model.join(', ')}.`}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">Why: factor contributions</div>
                  <div className="card-subtitle">This prediction's SHAP values, in {result.contribution_units}</div>
                </div>
                <div className="card-body" data-testid="contributions">
                  {result.contributions.map(c => (
                    <div className="contribution-bar" key={c.feature}>
                      <div className="contribution-name">{c.feature}</div>
                      <div className="contribution-track">
                        <div className={`contribution-fill ${c.direction === 'increase' ? 'positive' : 'negative'}`}
                          style={{ width: `${(Math.abs(c.value) / maxContribution) * 100}%` }} />
                      </div>
                      <div className="contribution-value" style={{ color: c.direction === 'increase' ? '#FF6B6B' : '#5CB87A' }}>
                        {c.value > 0 ? '+' : ''}{c.value.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {result.similar_incidents.length > 0 && (
                <div className="card" style={{ marginBottom: 20 }}>
                  <div className="card-header">
                    <div className="card-title">Similar real accidents</div>
                    <div className="card-subtitle">Most recent TSB accidents with the same province, track and train type</div>
                  </div>
                  <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="similar-incidents">
                    <table className="data-table">
                      <thead><tr><th>Date</th><th>Type</th><th>Train</th><th>Deaths</th><th>Serious inj.</th><th>Outcome</th></tr></thead>
                      <tbody>
                        {result.similar_incidents.map(inc => (
                          <tr key={`${inc.date}-${inc.type}-${inc.train}-${inc.fatalities}-${inc.serious_injuries}`}>
                            <td className="mono">{inc.date}</td>
                            <td>{inc.type}</td>
                            <td>{inc.train}</td>
                            <td className="mono">{inc.fatalities}</td>
                            <td className="mono">{inc.serious_injuries}</td>
                            <td><span className={`badge ${inc.severity === 'SEVERE' ? 'badge-high' : 'badge-low'}`}>{inc.severity}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card">
              <div className="card-body" style={{ textAlign: 'center', padding: '80px 40px' }}>
                <div style={{ fontSize: 16, fontWeight: 600, color: '#E6EDF3', marginBottom: 8 }}>Describe a scenario and submit</div>
                <div style={{ fontSize: 13, color: '#6E7681', maxWidth: 380, margin: '0 auto' }}>
                  The model only uses information known before an accident: place, time, track, what the train was
                  doing, the train itself, weather and the province's accident history.
                </div>
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-header">
                <div className="card-title">This session</div>
                <div className="card-subtitle">Last {history.length} assessments</div>
              </div>
              <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="prediction-history">
                <table className="data-table">
                  <thead><tr><th>Province</th><th>Track</th><th>Activity</th><th>Train</th><th>Speed</th><th>Probability</th><th>Risk</th></tr></thead>
                  <tbody>
                    {history.map(h => (
                      <tr key={h.id}>
                        <td>{h.province}</td><td>{h.track_type}</td><td>{h.activity}</td><td>{h.train_type}</td>
                        <td className="mono">{h.approx_speed_mph || '—'}</td>
                        <td className="mono" style={{ fontWeight: 600 }}>{(h.probability * 100).toFixed(1)}%</td>
                        <td><span className={`badge ${LEVEL_BADGE[h.risk_level]}`}>{h.risk_level}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
