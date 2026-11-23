import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Plot from '../components/Plot';
import { DARK_LAYOUT, PLOT_CONFIG, SERIES_COLORS, LEVEL_BADGE, getJson, label, relativeLevel } from '../components/chartTheme';

const Loading = ({ text }) => <div className="loading-overlay"><div className="spinner" /> {text}</div>;
const TARGET = 'incident_severity';
const TOP_CORRELATIONS = 3;

/** Features most correlated (by |r|) with the severity target. */
function strongestWithTarget(corr) {
  const t = corr.features.indexOf(TARGET);
  if (t < 0) return [];
  return corr.features
    .map((f, i) => ({ feature: f, r: corr.values[t][i] }))
    .filter(d => d.feature !== TARGET && d.r != null)
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))
    .slice(0, TOP_CORRELATIONS);
}

export default function DeepDiveExplorer({ api }) {
  const [activeTab, setActiveTab] = useState('geographic');
  const [heatmapData, setHeatmapData] = useState(null);
  const [annualData, setAnnualData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [severityData, setSeverityData] = useState(null);
  const [rollingData, setRollingData] = useState(null);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);

  const fetchAll = useCallback(async () => {
    try {
      const [hm, annual, corr, sev, rolling, ins] = await Promise.all([
        getJson(`${api}/api/incidents/heatmap`),
        getJson(`${api}/api/incidents/annual`),
        getJson(`${api}/api/incidents/correlation`),
        getJson(`${api}/api/incidents/severity-by-type`),
        getJson(`${api}/api/incidents/rolling`),
        getJson(`${api}/api/insights`),
      ]);
      setHeatmapData(hm);
      setAnnualData(annual);
      setCorrelationData(corr);
      setSeverityData(sev);
      setRollingData(rolling);
      setInsights(ins);
    } catch (e) {
      setError('Could not load explorer data from the API.');
    }
  }, [api]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const strongest = useMemo(() => (correlationData ? strongestWithTarget(correlationData) : []), [correlationData]);
  const baseRate = insights?.severe_rate_pct;

  const tabs = [
    { id: 'geographic', label: 'Geographic View' },
    { id: 'temporal', label: 'Temporal View' },
    { id: 'correlation', label: 'Correlation Explorer' },
    { id: 'typology', label: 'Accident Typology' },
  ];

  return (
    <div data-testid="deep-dive-explorer">
      <div className="page-header">
        <h2>Deep Dive Explorer</h2>
        <p>Interactive exploration of 25 years of TSB rail accident data</p>
      </div>

      {error && <div className="insight-block"><p>{error}</p></div>}

      <div className="tabs" data-testid="explorer-tabs">
        {tabs.map(t => (
          <button key={t.id} className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)} data-testid={`tab-${t.id}`}>
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === 'geographic' && (
        <div data-testid="geographic-view">
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="card-header">
              <div>
                <div className="card-title">Province vs Accident Type</div>
                <div className="card-subtitle">Accident counts for the 12 most common accident types</div>
              </div>
            </div>
            <div className="card-body">
              {heatmapData ? (
                <Plot
                  data={[{
                    z: heatmapData.values, x: heatmapData.incident_types, y: heatmapData.provinces, type: 'heatmap',
                    colorscale: [[0, '#0D1117'], [0.25, '#1A3A5C'], [0.5, '#D29922'], [1, '#C8102E']],
                    hovertemplate: '<b>%{y}</b> / %{x}<br>Count: %{z}<extra></extra>',
                    colorbar: { title: { text: 'Count', font: { color: '#8B949E' } }, tickfont: { color: '#8B949E' } },
                  }]}
                  layout={{
                    ...DARK_LAYOUT, height: 520,
                    margin: { l: 180, r: 60, t: 20, b: 160 },
                    xaxis: { ...DARK_LAYOUT.xaxis, tickangle: -35, tickfont: { size: 10, color: '#8B949E' } },
                    yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 11, color: '#8B949E' } },
                  }}
                  config={PLOT_CONFIG}
                  style={{ width: '100%' }}
                />
              ) : <Loading text="Loading heatmap..." />}
            </div>
          </div>
          {insights && (
            <div className="insight-block">
              <div className="insight-label">Insight (computed)</div>
              <p>
                {insights.top_provinces.map(p => `${p.province} reports ${p.share_pct}%`).join(' and ')} of all
                accidents. Most accidents are low-consequence yard and non-main-track events; the most common
                type is {insights.most_common_type.type} ({insights.most_common_type.share_pct}%).
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'temporal' && (
        <div data-testid="temporal-view">
          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Accident Trend Over Time</div>
                  <div className="card-subtitle">
                    {insights ? `${insights.change_peak_to_last_pct}% from the ${insights.peak_year} peak to ${insights.last_year}` : '…'}
                  </div>
                </div>
              </div>
              <div className="card-body">
                {annualData ? (
                  <Plot
                    data={[
                      {
                        x: annualData.years, y: annualData.total, type: 'scatter', mode: 'lines+markers', name: 'All accidents',
                        line: { color: '#C8102E', width: 2.5 }, marker: { size: 5, color: '#C8102E' },
                        fill: 'tozeroy', fillcolor: 'rgba(200,16,46,0.08)',
                        hovertemplate: '<b>%{x}</b>: %{y} accidents<extra></extra>',
                      },
                      {
                        x: annualData.years, y: annualData.high_risk, type: 'scatter', mode: 'lines', name: 'Severe',
                        line: { color: '#D29922', width: 1.5, dash: 'dot' },
                        hovertemplate: '<b>%{x}</b>: %{y} severe<extra></extra>',
                      },
                    ]}
                    layout={{ ...DARK_LAYOUT, height: 350, legend: { font: { color: '#8B949E', size: 11 }, x: 0, y: 1.1, orientation: 'h' } }}
                    config={PLOT_CONFIG}
                    style={{ width: '100%' }}
                  />
                ) : <Loading text="" />}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Rolling 12-Month Accidents by Province</div>
                  <div className="card-subtitle">{rollingData?.description || ''} · top 6 provinces</div>
                </div>
              </div>
              <div className="card-body">
                {rollingData ? (
                  <Plot
                    data={Object.entries(rollingData.provinces).map(([prov, d], i) => ({
                      x: d.dates, y: d.values, type: 'scatter', mode: 'lines', name: prov,
                      line: { color: SERIES_COLORS[i % SERIES_COLORS.length], width: 2 },
                      hovertemplate: `<b>${prov}</b><br>%{x}: %{y}<extra></extra>`,
                    }))}
                    layout={{
                      ...DARK_LAYOUT, height: 350,
                      legend: { font: { color: '#8B949E', size: 10 }, orientation: 'h', y: -0.2 },
                      xaxis: { ...DARK_LAYOUT.xaxis, tickformat: '%Y' },
                    }}
                    config={PLOT_CONFIG}
                    style={{ width: '100%' }}
                  />
                ) : <Loading text="" />}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'correlation' && (
        <div data-testid="correlation-view">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Feature Correlation Matrix</div>
                <div className="card-subtitle">Pearson r between model inputs and the severity target (pairwise, ignoring blanks)</div>
              </div>
            </div>
            <div className="card-body">
              {correlationData ? (
                <Plot
                  data={[{
                    z: correlationData.values,
                    x: correlationData.features.map(label), y: correlationData.features.map(label),
                    type: 'heatmap', colorscale: 'RdBu', zmid: 0, zmin: -1, zmax: 1,
                    hovertemplate: '%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>',
                    colorbar: { title: { text: 'Pearson r', font: { color: '#8B949E' } }, tickfont: { color: '#8B949E' } },
                  }]}
                  layout={{
                    ...DARK_LAYOUT, height: 650,
                    margin: { l: 190, r: 60, t: 20, b: 170 },
                    xaxis: { ...DARK_LAYOUT.xaxis, tickangle: -45, tickfont: { size: 9, color: '#8B949E' } },
                    yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 9, color: '#8B949E' }, autorange: 'reversed' },
                  }}
                  config={PLOT_CONFIG}
                  style={{ width: '100%' }}
                />
              ) : <Loading text="Loading correlations..." />}
            </div>
          </div>
          {strongest.length > 0 && (
            <div className="insight-block">
              <div className="insight-label">Insight (computed)</div>
              <p>
                Strongest linear relationships with severity:{' '}
                {strongest.map(d => `${label(d.feature)} (r = ${d.r.toFixed(2)})`).join(', ')}. Correlations are
                weak overall: severity depends on combinations of factors, which is why tree models beat logistic
                regression here.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'typology' && (
        <div data-testid="typology-view">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Severity by Accident Type</div>
                <div className="card-subtitle">Severe = at least one fatality or serious injury</div>
              </div>
            </div>
            <div className="card-body">
              {severityData ? (
                <Plot
                  data={[
                    {
                      y: severityData.data.map(d => d.type), x: severityData.data.map(d => d.high),
                      type: 'bar', orientation: 'h', name: 'Severe', marker: { color: 'rgba(200,16,46,0.75)' },
                      hovertemplate: '<b>%{y}</b><br>Severe: %{x}<extra></extra>',
                    },
                    {
                      y: severityData.data.map(d => d.type), x: severityData.data.map(d => d.low),
                      type: 'bar', orientation: 'h', name: 'Not severe', marker: { color: 'rgba(26,58,92,0.75)' },
                      hovertemplate: '<b>%{y}</b><br>Not severe: %{x}<extra></extra>',
                    },
                  ]}
                  layout={{
                    ...DARK_LAYOUT, height: 450, barmode: 'stack',
                    margin: { l: 240, r: 20, t: 20, b: 40 },
                    yaxis: { ...DARK_LAYOUT.yaxis, autorange: 'reversed', tickfont: { size: 11 } },
                    legend: { font: { color: '#8B949E', size: 11 }, orientation: 'h', y: -0.15 },
                  }}
                  config={PLOT_CONFIG}
                  style={{ width: '100%' }}
                />
              ) : <Loading text="" />}
            </div>
          </div>

          {severityData && baseRate != null && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-header">
                <div>
                  <div className="card-title">Severe Rate by Accident Type</div>
                  <div className="card-subtitle">Level vs the overall severe rate ({baseRate}%)</div>
                </div>
              </div>
              <div className="card-body" style={{ padding: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr><th>Accident Type</th><th>Total</th><th>Severe</th><th>Not Severe</th><th>Severe Rate</th><th>Level</th></tr>
                  </thead>
                  <tbody>
                    {severityData.data.map(d => {
                      const rate = (d.high / d.total) * 100;
                      const level = relativeLevel(rate, baseRate);
                      return (
                        <tr key={d.type}>
                          <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{d.type}</td>
                          <td className="mono">{d.total.toLocaleString()}</td>
                          <td className="mono" style={{ color: '#FF6B6B' }}>{d.high.toLocaleString()}</td>
                          <td className="mono">{d.low.toLocaleString()}</td>
                          <td className="mono">{rate.toFixed(1)}%</td>
                          <td><span className={`badge ${LEVEL_BADGE[level]}`}>{level}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {insights && (
            <div className="insight-block" style={{ marginTop: 20 }}>
              <div className="insight-label">Insight (computed)</div>
              <p>
                Severity is concentrated in a few accident types:{' '}
                {insights.deadliest_types.map(t => `${t.type} ${t.severe_rate_pct}%`).join(', ')}. Accident type is
                only known after the fact, so it is deliberately <b>not</b> a model input; the model has to infer
                risk from conditions known beforehand (track, speed, train type, activity, weather, history).
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
