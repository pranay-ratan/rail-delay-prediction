import React, { useState, useCallback } from 'react';
import Plot from '../components/Plot';

const DARK_LAYOUT = {
  paper_bgcolor: '#161B22',
  plot_bgcolor: '#0D1117',
  font: { family: 'Inter, sans-serif', color: '#8B949E', size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  yaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  hoverlabel: { bgcolor: '#161B22', bordercolor: '#30363D', font: { color: '#E6EDF3', size: 12 } },
};

const PROVINCES = [
  "Alberta", "British Columbia", "Manitoba", "New Brunswick",
  "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
  "Nunavut", "Ontario", "Prince Edward Island", "Quebec",
  "Saskatchewan", "Yukon",
];

const INCIDENT_TYPES = [
  "Derailment", "Main Track Train Collision", "Crossing Collision",
  "Employee Fatality", "Non-main Track Collision", "Employee Injury",
  "Dangerous Goods Release", "Fire or Explosion", "Other Occurrence",
];

const CARGO_TYPES = [
  { name: "Dangerous Goods", risk: 3 },
  { name: "Crude Oil", risk: 3 },
  { name: "Coal", risk: 2 },
  { name: "Potash", risk: 2 },
  { name: "Grain", risk: 1 },
  { name: "General Freight", risk: 1 },
  { name: "Intermodal", risk: 1 },
  { name: "Passenger", risk: 2 },
];

const SEASONS = ["Winter", "Spring", "Summer", "Fall"];

export default function LiveRiskAssessor({ api }) {
  const [form, setForm] = useState({
    province: 'Ontario',
    incident_type: 'Derailment',
    cargo_type: 'Dangerous Goods',
    season: 'Winter',
    year: 2024,
    month: 1,
    rolling_12m: 45,
    fatalities: 0,
    injuries: 0,
    is_weekend: false,
    mile_post: 150,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${api}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
        setHistory(prev => [
          { ...form, probability: data.probability, risk_level: data.risk_level, timestamp: data.timestamp },
          ...prev.slice(0, 9),
        ]);
      }
    } catch (e) { console.error('Prediction error:', e); }
    setLoading(false);
  }, [api, form]);

  const getRiskColor = (level) => {
    if (level === 'HIGH') return '#FF4444';
    if (level === 'MEDIUM') return '#D29922';
    return '#2E8B57';
  };

  return (
    <div data-testid="live-risk-assessor">
      <div className="page-header">
        <h2>Live Risk Assessor</h2>
        <p>Configure a rail corridor scenario and get instant severity prediction with driving factors</p>
      </div>

      <div className="grid-1-2">
        <div>
          <div className="card" style={{ position: 'sticky', top: 20 }}>
            <div className="card-header">
              <div className="card-title">Corridor Configuration</div>
            </div>
            <div className="card-body">
              <div className="form-group">
                <label className="form-label">Province / Region</label>
                <select className="form-select" value={form.province} onChange={e => handleChange('province', e.target.value)} data-testid="select-province">
                  {PROVINCES.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Incident Type</label>
                <select className="form-select" value={form.incident_type} onChange={e => handleChange('incident_type', e.target.value)} data-testid="select-incident-type">
                  {INCIDENT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Cargo Type</label>
                <div className="chip-group">
                  {CARGO_TYPES.map(c => (
                    <button
                      key={c.name}
                      className={`chip ${form.cargo_type === c.name ? 'active' : ''}`}
                      onClick={() => handleChange('cargo_type', c.name)}
                      data-testid={`cargo-${c.name.replace(/\s+/g, '-').toLowerCase()}`}
                      style={c.risk === 3 ? { borderColor: form.cargo_type === c.name ? 'rgba(200,16,46,0.5)' : undefined } : {}}
                    >
                      {c.name}
                      {c.risk === 3 && <span style={{ color: '#FF6B6B', marginLeft: 4, fontSize: 10 }}>HIGH</span>}
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Season</label>
                <div className="chip-group">
                  {SEASONS.map(s => (
                    <button
                      key={s}
                      className={`chip ${form.season === s ? 'active' : ''}`}
                      onClick={() => handleChange('season', s)}
                      data-testid={`season-${s.toLowerCase()}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">Year: {form.year}</label>
                  <input type="range" className="form-slider" min={2000} max={2025} value={form.year} onChange={e => handleChange('year', parseInt(e.target.value))} data-testid="slider-year" />
                </div>
                <div className="form-group">
                  <label className="form-label">Month: {form.month}</label>
                  <input type="range" className="form-slider" min={1} max={12} value={form.month} onChange={e => handleChange('month', parseInt(e.target.value))} data-testid="slider-month" />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Rolling 12-Month Incidents: {form.rolling_12m}</label>
                <input type="range" className="form-slider" min={0} max={200} value={form.rolling_12m} onChange={e => handleChange('rolling_12m', parseInt(e.target.value))} data-testid="slider-rolling" />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">Fatalities: {form.fatalities}</label>
                  <input type="range" className="form-slider" min={0} max={5} value={form.fatalities} onChange={e => handleChange('fatalities', parseInt(e.target.value))} data-testid="slider-fatalities" />
                </div>
                <div className="form-group">
                  <label className="form-label">Injuries: {form.injuries}</label>
                  <input type="range" className="form-slider" min={0} max={15} value={form.injuries} onChange={e => handleChange('injuries', parseInt(e.target.value))} data-testid="slider-injuries" />
                </div>
              </div>

              <button
                className="btn btn-primary"
                style={{ width: '100%', marginTop: 8 }}
                onClick={handleSubmit}
                disabled={loading}
                data-testid="predict-button"
              >
                {loading ? 'Analyzing...' : 'Assess Corridor Risk'}
              </button>
            </div>
          </div>
        </div>

        <div>
          {result ? (
            <>
              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">Risk Assessment Result</div>
                  <span className={`badge ${result.risk_level === 'HIGH' ? 'badge-high' : result.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}`}>
                    {result.risk_level} RISK
                  </span>
                </div>
                <div className="card-body" data-testid="risk-result">
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                    <div data-testid="risk-gauge-chart">
                      <Plot
                        data={[{
                          type: 'indicator',
                          mode: 'gauge+number',
                          value: result.probability * 100,
                          number: { suffix: '%', font: { size: 36, color: getRiskColor(result.risk_level), family: 'JetBrains Mono' } },
                          gauge: {
                            axis: { range: [0, 100], tickcolor: '#6E7681', tickfont: { color: '#6E7681' } },
                            bar: { color: getRiskColor(result.risk_level), thickness: 0.3 },
                            bgcolor: '#0D1117',
                            borderwidth: 1, bordercolor: '#30363D',
                            steps: [
                              { range: [0, 40], color: 'rgba(46,139,87,0.12)' },
                              { range: [40, 70], color: 'rgba(210,153,34,0.12)' },
                              { range: [70, 100], color: 'rgba(200,16,46,0.12)' },
                            ],
                          },
                        }]}
                        layout={{
                          ...DARK_LAYOUT,
                          height: 250,
                          margin: { l: 30, r: 30, t: 30, b: 10 },
                        }}
                        config={{ displayModeBar: false, responsive: true }}
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#6E7681', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>
                        Scenario Summary
                      </div>
                      <div style={{ display: 'grid', gap: 8 }}>
                        {[
                          ['Province', form.province],
                          ['Incident Type', form.incident_type],
                          ['Cargo', form.cargo_type],
                          ['Season', form.season],
                          ['Rolling 12m', `${form.rolling_12m} incidents`],
                          ['Year', form.year],
                        ].map(([k, v]) => (
                          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #30363D' }}>
                            <span style={{ color: '#6E7681', fontSize: 12 }}>{k}</span>
                            <span className="mono" style={{ color: '#E6EDF3', fontSize: 12 }}>{v}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">Feature Contributions</div>
                  <div className="card-subtitle">Top factors driving this prediction from baseline ({(result.baseline * 100).toFixed(0)}%)</div>
                </div>
                <div className="card-body" data-testid="contributions">
                  {result.contributions.map((c, i) => (
                    <div className="contribution-bar" key={i}>
                      <div className="contribution-name">{c.feature}</div>
                      <div className="contribution-track">
                        <div
                          className={`contribution-fill ${c.direction === 'increase' ? 'positive' : 'negative'}`}
                          style={{ width: `${Math.min(Math.abs(c.value) * 500, 100)}%` }}
                        />
                      </div>
                      <div className="contribution-value" style={{ color: c.direction === 'increase' ? '#FF6B6B' : '#5CB87A' }}>
                        {c.direction === 'increase' ? '+' : '-'}{Math.abs(c.value * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">Recommended Actions</div>
                </div>
                <div className="card-body" data-testid="recommendations">
                  {result.recommendations.map((rec, i) => (
                    <div key={i} className={`recommendation ${result.risk_level === 'HIGH' ? 'high-rec' : result.risk_level === 'MEDIUM' ? 'medium-rec' : 'low-rec'}`}>
                      <span style={{ color: '#6E7681', fontWeight: 700, fontSize: 11 }}>{i + 1}.</span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              </div>

              {result.similar_incidents.length > 0 && (
                <div className="card" style={{ marginBottom: 20 }}>
                  <div className="card-header">
                    <div className="card-title">Similar Historical Incidents</div>
                    <div className="card-subtitle">Matching province and incident type from training data</div>
                  </div>
                  <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="similar-incidents">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Year</th>
                          <th>Month</th>
                          <th>Province</th>
                          <th>Type</th>
                          <th>Cargo</th>
                          <th>Severity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.similar_incidents.map((inc, i) => (
                          <tr key={i}>
                            <td className="mono">{inc.year}</td>
                            <td className="mono">{inc.month}</td>
                            <td>{inc.province}</td>
                            <td>{inc.type}</td>
                            <td>{inc.cargo}</td>
                            <td>
                              <span className={`badge ${inc.severity === 'HIGH' ? 'badge-high' : 'badge-low'}`}>
                                {inc.severity}
                              </span>
                            </td>
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
                <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: '#E6EDF3', marginBottom: 8 }}>Configure and Submit</div>
                <div style={{ fontSize: 13, color: '#6E7681', maxWidth: 360, margin: '0 auto' }}>
                  Set corridor parameters in the form and click "Assess Corridor Risk" to generate a real-time severity prediction with full factor analysis.
                </div>
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-header">
                <div className="card-title">Prediction History</div>
                <div className="card-subtitle">Last {history.length} assessments this session</div>
              </div>
              <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="prediction-history">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Province</th>
                      <th>Type</th>
                      <th>Cargo</th>
                      <th>Season</th>
                      <th>Probability</th>
                      <th>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => (
                      <tr key={i}>
                        <td>{h.province}</td>
                        <td>{h.incident_type}</td>
                        <td>{h.cargo_type}</td>
                        <td>{h.season}</td>
                        <td className="mono" style={{ fontWeight: 600 }}>{(h.probability * 100).toFixed(1)}%</td>
                        <td>
                          <span className={`badge ${h.risk_level === 'HIGH' ? 'badge-high' : h.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}`}>
                            {h.risk_level}
                          </span>
                        </td>
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
