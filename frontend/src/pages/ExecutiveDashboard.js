import React, { useState, useEffect, useCallback } from 'react';
import CountUp from 'react-countup';
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

function StatCard({ label, value, suffix, decimals, color, change }) {
  return (
    <div className="stat-card" data-testid={`stat-${label.replace(/\s+/g, '-').toLowerCase()}`}>
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${color || ''}`}>
        <CountUp end={value} duration={1.5} separator="," suffix={suffix || ''} decimals={decimals || 0} />
      </div>
      {change && <div className="stat-change">{change}</div>}
    </div>
  );
}

export default function ExecutiveDashboard({ stats, api }) {
  const [annualData, setAnnualData] = useState(null);
  const [provinceData, setProvinceData] = useState(null);
  const [seasonData, setSeasonData] = useState(null);
  const [typeData, setTypeData] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [annual, provinces, seasonal, types] = await Promise.all([
        fetch(`${api}/api/incidents/annual`).then(r => r.json()),
        fetch(`${api}/api/provinces`).then(r => r.json()),
        fetch(`${api}/api/incidents/by-season`).then(r => r.json()),
        fetch(`${api}/api/incidents/by-type`).then(r => r.json()),
      ]);
      setAnnualData(annual);
      setProvinceData(provinces);
      setSeasonData(seasonal);
      setTypeData(types);
    } catch (e) { console.error('Dashboard data fetch error:', e); }
  }, [api]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return (
    <div data-testid="executive-dashboard">
      <div className="page-header">
        <h2>Executive Dashboard</h2>
        <p>System-wide overview of rail incident intelligence across the Canadian network</p>
      </div>

      <div className="stats-grid" data-testid="stats-grid">
        <StatCard label="Incidents Analyzed" value={stats?.total_incidents || 0} color="red" change="Transport Canada data" />
        <StatCard label="Predictive Features" value={stats?.feature_count || 25} color="blue" change="Engineered features" />
        <StatCard label="ROC-AUC Accuracy" value={stats?.best_auc || 0} suffix="%" decimals={1} color="green" change={`Best: ${stats?.best_model || 'Random Forest'}`} />
        <StatCard label="Provinces Covered" value={stats?.provinces_covered || 0} change="All territories included" />
      </div>

      <div className="grid-2-1">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Annual Incident Trend</div>
              <div className="card-subtitle">Rail incidents peaked mid-2000s, declining with improved safety regulation</div>
            </div>
          </div>
          <div className="card-body" data-testid="annual-trend-chart">
            {annualData ? (
              <Plot
                data={[
                  {
                    x: annualData.years, y: annualData.high_risk,
                    type: 'bar', name: 'High Risk',
                    marker: { color: 'rgba(200,16,46,0.7)' },
                    hovertemplate: '<b>%{x}</b><br>High Risk: %{y}<extra></extra>',
                  },
                  {
                    x: annualData.years, y: annualData.low_risk,
                    type: 'bar', name: 'Low Risk',
                    marker: { color: 'rgba(26,58,92,0.7)' },
                    hovertemplate: '<b>%{x}</b><br>Low Risk: %{y}<extra></extra>',
                  },
                  {
                    x: annualData.years, y: annualData.total,
                    type: 'scatter', mode: 'lines+markers', name: 'Total',
                    line: { color: '#E6EDF3', width: 2 },
                    marker: { size: 4, color: '#E6EDF3' },
                    hovertemplate: '<b>%{x}</b><br>Total: %{y}<extra></extra>',
                  }
                ]}
                layout={{
                  ...DARK_LAYOUT,
                  barmode: 'stack',
                  height: 340,
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.15, font: { size: 11, color: '#8B949E' } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : <div className="loading-overlay"><div className="spinner" /> Loading chart...</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Seasonal Distribution</div>
              <div className="card-subtitle">Winter carries highest severity risk</div>
            </div>
          </div>
          <div className="card-body" data-testid="season-chart">
            {seasonData ? (
              <Plot
                data={[{
                  x: seasonData.total,
                  y: seasonData.seasons,
                  type: 'bar', orientation: 'h',
                  marker: {
                    color: ['#1A3A5C', '#2E8B57', '#D29922', '#C8102E'],
                  },
                  text: seasonData.total.map(String),
                  textposition: 'outside',
                  textfont: { color: '#8B949E', size: 11 },
                  hovertemplate: '<b>%{y}</b>: %{x} incidents<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT,
                  height: 340,
                  showlegend: false,
                  xaxis: { ...DARK_LAYOUT.xaxis, title: '' },
                  yaxis: { ...DARK_LAYOUT.yaxis, autorange: 'reversed' },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : <div className="loading-overlay"><div className="spinner" /> Loading...</div>}
          </div>
        </div>
      </div>

      <div className="insight-block">
        <div className="insight-label">Key Insight</div>
        <p>Winter months (Dec-Feb) carry 1.4x higher incident severity probability than summer, driven by track contraction, ice formation, and reduced visibility. Ontario and Quebec account for approximately 40% of all reported incidents.</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Risk by Province</div>
              <div className="card-subtitle">Geographic distribution of incident density across Canada</div>
            </div>
          </div>
          <div className="card-body" data-testid="province-risk-chart">
            {provinceData ? (
              <Plot
                data={[{
                  type: 'choropleth',
                  locationmode: 'country names',
                  locations: provinceData.provinces.map(p => p.province),
                  z: provinceData.provinces.map(p => p.incidents),
                  text: provinceData.provinces.map(p => `${p.province}: ${p.incidents} incidents (${p.risk_pct}% high risk)`),
                  colorscale: [[0, '#0D1117'], [0.3, '#1A3A5C'], [0.6, '#D29922'], [1, '#C8102E']],
                  showscale: true,
                  colorbar: {
                    title: { text: 'Incidents', font: { color: '#8B949E', size: 11 } },
                    tickfont: { color: '#8B949E' },
                    bgcolor: 'transparent',
                  },
                  hovertemplate: '%{text}<extra></extra>',
                  geo: 'geo',
                }]}
                layout={{
                  ...DARK_LAYOUT,
                  height: 380,
                  geo: {
                    scope: 'north america',
                    showlakes: false,
                    showland: true,
                    landcolor: '#161B22',
                    bgcolor: '#0D1117',
                    showframe: false,
                    coastlinecolor: '#30363D',
                    countrycolor: '#30363D',
                    subunitcolor: '#30363D',
                    projection: { type: 'natural earth' },
                    center: { lat: 56, lon: -96 },
                    lonaxis: { range: [-140, -50] },
                    lataxis: { range: [42, 72] },
                  },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : <div className="loading-overlay"><div className="spinner" /> Loading map...</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Incident Type Distribution</div>
              <div className="card-subtitle">Derailments dominate at 30% of all occurrences</div>
            </div>
          </div>
          <div className="card-body" data-testid="type-chart">
            {typeData ? (
              <Plot
                data={[{
                  labels: typeData.types,
                  values: typeData.total,
                  type: 'pie',
                  hole: 0.55,
                  marker: {
                    colors: ['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA', '#FF6B35', '#6E7681', '#444'],
                    line: { color: '#161B22', width: 2 },
                  },
                  textinfo: 'percent',
                  textfont: { color: '#E6EDF3', size: 11 },
                  hovertemplate: '<b>%{label}</b><br>%{value} incidents (%{percent})<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT,
                  height: 380,
                  showlegend: true,
                  legend: { font: { size: 10, color: '#8B949E' }, x: 1.05, y: 0.5 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : <div className="loading-overlay"><div className="spinner" /> Loading...</div>}
          </div>
        </div>
      </div>

      {provinceData && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Province Risk Table</div>
              <div className="card-subtitle">Ranked by total incident count with severity indicators</div>
            </div>
          </div>
          <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="province-table">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Province</th>
                  <th>Total Incidents</th>
                  <th>High Risk</th>
                  <th>Risk %</th>
                  <th>Top Type</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {provinceData.provinces
                  .sort((a, b) => b.incidents - a.incidents)
                  .map(p => (
                    <tr key={p.province}>
                      <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{p.province}</td>
                      <td className="mono">{p.incidents.toLocaleString()}</td>
                      <td className="mono">{p.high_risk.toLocaleString()}</td>
                      <td className="mono">{p.risk_pct}%</td>
                      <td>{p.top_incident_type}</td>
                      <td>
                        <span className={`badge ${p.risk_pct >= 70 ? 'badge-high' : p.risk_pct >= 50 ? 'badge-medium' : 'badge-low'}`}>
                          {p.risk_pct >= 70 ? 'HIGH' : p.risk_pct >= 50 ? 'MEDIUM' : 'LOW'}
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
  );
}
