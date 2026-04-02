import React, { useState, useEffect, useCallback } from 'react';
import Plot from '../components/Plot';

const DARK_LAYOUT = {
  paper_bgcolor: '#161B22',
  plot_bgcolor: '#0D1117',
  font: { family: 'Inter, sans-serif', color: '#8B949E', size: 12 },
  margin: { l: 140, r: 20, t: 40, b: 50 },
  xaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  yaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  hoverlabel: { bgcolor: '#161B22', bordercolor: '#30363D', font: { color: '#E6EDF3', size: 12 } },
};

export default function DeepDiveExplorer({ api }) {
  const [activeTab, setActiveTab] = useState('geographic');
  const [heatmapData, setHeatmapData] = useState(null);
  const [annualData, setAnnualData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [severityData, setSeverityData] = useState(null);
  const [rollingData, setRollingData] = useState(null);

  const fetchAll = useCallback(async () => {
    try {
      const [hm, annual, corr, sev, rolling] = await Promise.all([
        fetch(`${api}/api/incidents/heatmap`).then(r => r.json()),
        fetch(`${api}/api/incidents/annual`).then(r => r.json()),
        fetch(`${api}/api/incidents/correlation`).then(r => r.json()),
        fetch(`${api}/api/incidents/severity-by-type`).then(r => r.json()),
        fetch(`${api}/api/incidents/rolling`).then(r => r.json()),
      ]);
      setHeatmapData(hm);
      setAnnualData(annual);
      setCorrelationData(corr);
      setSeverityData(sev);
      setRollingData(rolling);
    } catch (e) { console.error('Explorer data error:', e); }
  }, [api]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const tabs = [
    { id: 'geographic', label: 'Geographic View' },
    { id: 'temporal', label: 'Temporal View' },
    { id: 'correlation', label: 'Correlation Explorer' },
    { id: 'typology', label: 'Incident Typology' },
  ];

  return (
    <div data-testid="deep-dive-explorer">
      <div className="page-header">
        <h2>Deep Dive Explorer</h2>
        <p>Interactive data exploration across 25 years of Canadian rail occurrence data</p>
      </div>

      <div className="tabs" data-testid="explorer-tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
            data-testid={`tab-${t.id}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === 'geographic' && (
        <div data-testid="geographic-view">
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="card-header">
              <div>
                <div className="card-title">Province vs Incident Type Heatmap</div>
                <div className="card-subtitle">Ontario and Quebec dominate raw counts; Saskatchewan and Alberta show disproportionate derailment rates</div>
              </div>
            </div>
            <div className="card-body">
              {heatmapData ? (
                <Plot
                  data={[{
                    z: heatmapData.values,
                    x: heatmapData.incident_types,
                    y: heatmapData.provinces,
                    type: 'heatmap',
                    colorscale: [[0, '#0D1117'], [0.25, '#1A3A5C'], [0.5, '#D29922'], [1, '#C8102E']],
                    hovertemplate: '<b>%{y}</b> / %{x}<br>Count: %{z}<extra></extra>',
                    colorbar: {
                      title: { text: 'Count', font: { color: '#8B949E' } },
                      tickfont: { color: '#8B949E' },
                    },
                  }]}
                  layout={{
                    ...DARK_LAYOUT,
                    height: 520,
                    margin: { l: 180, r: 60, t: 20, b: 120 },
                    xaxis: { ...DARK_LAYOUT.xaxis, tickangle: -35, tickfont: { size: 10, color: '#8B949E' } },
                    yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 11, color: '#8B949E' } },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : <div className="loading-overlay"><div className="spinner" /> Loading heatmap...</div>}
            </div>
          </div>
          <div className="insight-block">
            <div className="insight-label">Insight</div>
            <p>Saskatchewan and Alberta rank highest for derailment rates relative to population, reflecting the high volume of bulk commodity unit trains (grain, potash, crude oil) on mixed terrain. These are priority corridors for proactive inspection scheduling.</p>
          </div>
        </div>
      )}

      {activeTab === 'temporal' && (
        <div data-testid="temporal-view">
          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Incident Trend Over Time</div>
                  <div className="card-subtitle">Total incidents declined ~23% from peak to present</div>
                </div>
              </div>
              <div className="card-body">
                {annualData ? (
                  <Plot
                    data={[
                      {
                        x: annualData.years, y: annualData.total,
                        type: 'scatter', mode: 'lines+markers', name: 'Total',
                        line: { color: '#C8102E', width: 2.5 },
                        marker: { size: 5, color: '#C8102E' },
                        fill: 'tozeroy', fillcolor: 'rgba(200,16,46,0.08)',
                        hovertemplate: '<b>%{x}</b>: %{y} incidents<extra></extra>',
                      },
                      {
                        x: annualData.years, y: annualData.high_risk,
                        type: 'scatter', mode: 'lines', name: 'High Risk',
                        line: { color: '#D29922', width: 1.5, dash: 'dot' },
                        hovertemplate: '<b>%{x}</b>: %{y} high risk<extra></extra>',
                      },
                    ]}
                    layout={{
                      ...DARK_LAYOUT, height: 350,
                      legend: { font: { color: '#8B949E', size: 11 }, x: 0, y: 1.1, orientation: 'h' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                ) : <div className="loading-overlay"><div className="spinner" /></div>}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Rolling 12-Month Trend by Province</div>
                  <div className="card-subtitle">Top 6 provinces by incident volume</div>
                </div>
              </div>
              <div className="card-body">
                {rollingData ? (
                  <Plot
                    data={Object.entries(rollingData.provinces).map(([prov, data], i) => ({
                      x: data.dates, y: data.values,
                      type: 'scatter', mode: 'lines', name: prov,
                      line: { color: ['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA'][i], width: 2 },
                      hovertemplate: `<b>${prov}</b><br>%{x}: %{y:.1f}<extra></extra>`,
                    }))}
                    layout={{
                      ...DARK_LAYOUT, height: 350,
                      legend: { font: { color: '#8B949E', size: 10 }, orientation: 'h', y: -0.2 },
                      xaxis: { ...DARK_LAYOUT.xaxis, tickformat: '%Y' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                ) : <div className="loading-overlay"><div className="spinner" /></div>}
              </div>
            </div>
          </div>
          <div className="insight-block">
            <div className="insight-label">Insight</div>
            <p>The rolling 12-month view reveals structural shifts: Ontario surged mid-decade with increased intermodal traffic, while Saskatchewan tracks closely with grain export cycles, peaking post-harvest September through November. This temporal pattern is the top SHAP predictor in the model.</p>
          </div>
        </div>
      )}

      {activeTab === 'correlation' && (
        <div data-testid="correlation-view">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Feature Correlation Matrix</div>
                <div className="card-subtitle">Pearson correlations between engineered features reveal multicollinearity and interaction effects</div>
              </div>
            </div>
            <div className="card-body">
              {correlationData ? (
                <Plot
                  data={[{
                    z: correlationData.values,
                    x: correlationData.features.map(f => f.replace(/_/g, ' ')),
                    y: correlationData.features.map(f => f.replace(/_/g, ' ')),
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    zmid: 0, zmin: -1, zmax: 1,
                    hovertemplate: '%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>',
                    colorbar: {
                      title: { text: 'Pearson r', font: { color: '#8B949E' } },
                      tickfont: { color: '#8B949E' },
                    },
                  }]}
                  layout={{
                    ...DARK_LAYOUT,
                    height: 650,
                    margin: { l: 160, r: 60, t: 20, b: 140 },
                    xaxis: { ...DARK_LAYOUT.xaxis, tickangle: -45, tickfont: { size: 9, color: '#8B949E' } },
                    yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 9, color: '#8B949E' }, autorange: 'reversed' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : <div className="loading-overlay"><div className="spinner" /> Loading correlations...</div>}
            </div>
          </div>
          <div className="insight-block">
            <div className="insight-label">Insight</div>
            <p>Strong positive correlation between route_density_score and rolling_12m_incidents confirms that high-traffic corridors accumulate incidents. The interaction feature density_x_rolling captures this compound risk effectively, explaining its high SHAP importance.</p>
          </div>
        </div>
      )}

      {activeTab === 'typology' && (
        <div data-testid="typology-view">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Incident Type Severity Breakdown</div>
                <div className="card-subtitle">High vs Low severity distribution across incident categories</div>
              </div>
            </div>
            <div className="card-body">
              {severityData ? (
                <Plot
                  data={[
                    {
                      y: severityData.data.map(d => d.type),
                      x: severityData.data.map(d => d.high),
                      type: 'bar', orientation: 'h', name: 'High Severity',
                      marker: { color: 'rgba(200,16,46,0.75)' },
                      hovertemplate: '<b>%{y}</b><br>High: %{x}<extra></extra>',
                    },
                    {
                      y: severityData.data.map(d => d.type),
                      x: severityData.data.map(d => d.low),
                      type: 'bar', orientation: 'h', name: 'Low Severity',
                      marker: { color: 'rgba(26,58,92,0.75)' },
                      hovertemplate: '<b>%{y}</b><br>Low: %{x}<extra></extra>',
                    },
                  ]}
                  layout={{
                    ...DARK_LAYOUT,
                    height: 450,
                    barmode: 'stack',
                    yaxis: { ...DARK_LAYOUT.yaxis, autorange: 'reversed', tickfont: { size: 11 } },
                    legend: { font: { color: '#8B949E', size: 11 }, orientation: 'h', y: -0.15 },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : <div className="loading-overlay"><div className="spinner" /></div>}
            </div>
          </div>

          {severityData && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-header">
                <div>
                  <div className="card-title">Severity Rate by Incident Type</div>
                  <div className="card-subtitle">Percentage of incidents classified as high severity</div>
                </div>
              </div>
              <div className="card-body" style={{ padding: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Incident Type</th>
                      <th>Total</th>
                      <th>High Risk</th>
                      <th>Low Risk</th>
                      <th>Severity Rate</th>
                      <th>Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {severityData.data.map(d => {
                      const rate = ((d.high / d.total) * 100).toFixed(1);
                      return (
                        <tr key={d.type}>
                          <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{d.type}</td>
                          <td className="mono">{d.total.toLocaleString()}</td>
                          <td className="mono" style={{ color: '#FF6B6B' }}>{d.high.toLocaleString()}</td>
                          <td className="mono">{d.low.toLocaleString()}</td>
                          <td className="mono">{rate}%</td>
                          <td>
                            <span className={`badge ${rate >= 75 ? 'badge-high' : rate >= 50 ? 'badge-medium' : 'badge-low'}`}>
                              {rate >= 75 ? 'HIGH' : rate >= 50 ? 'MEDIUM' : 'LOW'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="insight-block" style={{ marginTop: 20 }}>
            <div className="insight-label">Insight</div>
            <p>Derailments, Main Track Collisions, and Dangerous Goods Releases carry the highest severity rates. Employee Fatality incidents, while less frequent, have near-100% high severity classification. The model leverages incident_type_encoded as a top-5 predictive feature.</p>
          </div>
        </div>
      )}
    </div>
  );
}
