import React, { useState, useEffect, useCallback } from 'react';
import CountUp from 'react-countup';
import Plot from '../components/Plot';
import { DARK_LAYOUT, PLOT_CONFIG, LEVEL_BADGE, getJson, relativeLevel } from '../components/chartTheme';

function StatCard({ label, value, suffix, decimals, color, change }) {
  return (
    <div className="stat-card" data-testid={`stat-${label.replace(/\s+/g, '-').toLowerCase()}`}>
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${color || ''}`}>
        {value == null ? '—' : (
          <CountUp end={value} duration={1.5} separator="," suffix={suffix || ''} decimals={decimals || 0} />
        )}
      </div>
      {change && <div className="stat-change">{change}</div>}
    </div>
  );
}

const Loading = ({ text }) => <div className="loading-overlay"><div className="spinner" /> {text}</div>;

export default function ExecutiveDashboard({ stats, api }) {
  const [annualData, setAnnualData] = useState(null);
  const [provinceData, setProvinceData] = useState(null);
  const [seasonData, setSeasonData] = useState(null);
  const [typeData, setTypeData] = useState(null);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [annual, provinces, seasonal, types, ins] = await Promise.all([
        getJson(`${api}/api/incidents/annual`),
        getJson(`${api}/api/provinces`),
        getJson(`${api}/api/incidents/by-season`),
        getJson(`${api}/api/incidents/by-type`),
        getJson(`${api}/api/insights`),
      ]);
      setAnnualData(annual);
      setProvinceData(provinces);
      setSeasonData(seasonal);
      setTypeData(types);
      setInsights(ins);
    } catch (e) {
      setError('Could not load dashboard data from the API.');
    }
  }, [api]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const baseRate = insights?.severe_rate_pct;
  const seasonRates = insights?.severe_rate_by_season_pct;

  return (
    <div data-testid="executive-dashboard">
      <div className="page-header">
        <h2>Executive Dashboard</h2>
        <p>Canadian rail accidents reported to the Transportation Safety Board, {stats?.year_range || '…'}</p>
      </div>

      {error && <div className="insight-block"><p>{error}</p></div>}

      <div className="stats-grid" data-testid="stats-grid">
        <StatCard label="Accidents Analyzed" value={stats?.total_incidents} color="red" change={`TSB RODS, ${stats?.year_range || ''}`} />
        <StatCard label="Predictive Features" value={stats?.feature_count} color="blue" change="Known before an accident" />
        <StatCard label="ROC-AUC" value={stats?.best_auc} suffix="%" decimals={1} color="green"
          change={stats ? `${stats.best_model}, 10-fold CV · ${stats.holdout_auc}% on 2021-25` : ''} />
        <StatCard label="Severe Accidents" value={stats?.high_risk_pct} suffix="%" decimals={1}
          change="Fatality or serious injury" />
      </div>

      <div className="grid-2-1">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Annual Accident Trend</div>
              <div className="card-subtitle">
                {insights
                  ? `Peak ${insights.peak_year} (${insights.peak_count.toLocaleString()}); ${insights.last_year}: ${insights.last_count.toLocaleString()} (${insights.change_peak_to_last_pct}% vs peak)`
                  : '…'}
              </div>
            </div>
          </div>
          <div className="card-body" data-testid="annual-trend-chart">
            {annualData ? (
              <Plot
                data={[
                  {
                    x: annualData.years, y: annualData.high_risk, type: 'bar', name: 'Severe',
                    marker: { color: 'rgba(200,16,46,0.7)' },
                    hovertemplate: '<b>%{x}</b><br>Severe: %{y}<extra></extra>',
                  },
                  {
                    x: annualData.years, y: annualData.low_risk, type: 'bar', name: 'Not severe',
                    marker: { color: 'rgba(26,58,92,0.7)' },
                    hovertemplate: '<b>%{x}</b><br>Not severe: %{y}<extra></extra>',
                  },
                ]}
                layout={{
                  ...DARK_LAYOUT, barmode: 'stack', height: 340, showlegend: true,
                  legend: { orientation: 'h', y: -0.15, font: { size: 11, color: '#8B949E' } },
                }}
                config={PLOT_CONFIG}
                style={{ width: '100%' }}
              />
            ) : <Loading text="Loading chart..." />}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Severe Rate by Season</div>
              <div className="card-subtitle">Share of accidents with a fatality or serious injury</div>
            </div>
          </div>
          <div className="card-body" data-testid="season-chart">
            {seasonData ? (
              <Plot
                data={[{
                  x: seasonData.severe_rate_pct, y: seasonData.seasons, type: 'bar', orientation: 'h',
                  marker: { color: ['#1A3A5C', '#2E8B57', '#D29922', '#C8102E'] },
                  text: seasonData.severe_rate_pct.map((v, i) => `${v}% of ${seasonData.total[i].toLocaleString()}`),
                  textposition: 'outside', textfont: { color: '#8B949E', size: 11 },
                  hovertemplate: '<b>%{y}</b>: %{x}% severe<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 340, showlegend: false,
                  margin: { l: 70, r: 90, t: 20, b: 40 },
                  xaxis: { ...DARK_LAYOUT.xaxis, ticksuffix: '%' },
                  yaxis: { ...DARK_LAYOUT.yaxis, autorange: 'reversed' },
                }}
                config={PLOT_CONFIG}
                style={{ width: '100%' }}
              />
            ) : <Loading text="Loading..." />}
          </div>
        </div>
      </div>

      {insights && seasonRates && (
        <div className="insight-block">
          <div className="insight-label">Key Insight (computed from the data)</div>
          <p>
            {insights.severe_rate_pct}% of accidents killed or seriously injured someone. Summer has the highest
            severe rate ({seasonRates.Summer}%) and winter the lowest ({seasonRates.Winter}%).{' '}
            {insights.top_provinces.map(p => `${p.province} (${p.share_pct}%)`).join(' and ')} report the most
            accidents. Accidents fell {Math.abs(insights.change_peak_to_last_pct)}% from the {insights.peak_year} peak
            to {insights.last_year}.
          </p>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Accidents by Province</div>
              <div className="card-subtitle">Bar colour = severe rate</div>
            </div>
          </div>
          <div className="card-body" data-testid="province-risk-chart">
            {provinceData ? (
              <Plot
                data={[{
                  type: 'bar', orientation: 'h',
                  y: provinceData.provinces.map(p => p.province),
                  x: provinceData.provinces.map(p => p.incidents),
                  marker: {
                    color: provinceData.provinces.map(p => p.risk_pct),
                    colorscale: [[0, '#1A3A5C'], [0.5, '#D29922'], [1, '#C8102E']],
                    showscale: true,
                    colorbar: { title: { text: 'Severe %', font: { color: '#8B949E', size: 11 } }, tickfont: { color: '#8B949E' } },
                  },
                  customdata: provinceData.provinces.map(p => p.risk_pct),
                  hovertemplate: '<b>%{y}</b><br>%{x} accidents · %{customdata}% severe<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 380,
                  margin: { l: 170, r: 20, t: 20, b: 40 },
                  yaxis: { ...DARK_LAYOUT.yaxis, autorange: 'reversed' },
                }}
                config={PLOT_CONFIG}
                style={{ width: '100%' }}
              />
            ) : <Loading text="Loading..." />}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Accident Type Distribution</div>
              <div className="card-subtitle">
                {insights ? `${insights.most_common_type.type}: ${insights.most_common_type.share_pct}% of accidents` : '…'}
              </div>
            </div>
          </div>
          <div className="card-body" data-testid="type-chart">
            {typeData ? (
              <Plot
                data={[{
                  labels: typeData.types, values: typeData.total, type: 'pie', hole: 0.55,
                  marker: {
                    colors: ['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA', '#FF6B35', '#6E7681', '#444'],
                    line: { color: '#161B22', width: 2 },
                  },
                  textinfo: 'percent', textfont: { color: '#E6EDF3', size: 11 },
                  hovertemplate: '<b>%{label}</b><br>%{value} accidents (%{percent})<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 380, showlegend: true,
                  legend: { font: { size: 10, color: '#8B949E' }, x: 1.05, y: 0.5 },
                }}
                config={PLOT_CONFIG}
                style={{ width: '100%' }}
              />
            ) : <Loading text="Loading..." />}
          </div>
        </div>
      </div>

      {provinceData && baseRate != null && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Province Table</div>
              <div className="card-subtitle">Level vs the national severe rate ({baseRate}%): HIGH ≥ 2×, MEDIUM ≥ 1×</div>
            </div>
          </div>
          <div className="card-body" style={{ padding: 0, overflow: 'auto' }} data-testid="province-table">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Province</th><th>Accidents</th><th>Severe</th><th>Severe %</th><th>Most Common Type</th><th>Level</th>
                </tr>
              </thead>
              <tbody>
                {provinceData.provinces.map(p => {
                  const level = relativeLevel(p.risk_pct, baseRate);
                  return (
                    <tr key={p.province}>
                      <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{p.province}</td>
                      <td className="mono">{p.incidents.toLocaleString()}</td>
                      <td className="mono">{p.high_risk.toLocaleString()}</td>
                      <td className="mono">{p.risk_pct}%</td>
                      <td>{p.top_incident_type}</td>
                      <td><span className={`badge ${LEVEL_BADGE[level]}`}>{level}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
