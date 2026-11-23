import React, { useState, useEffect, useCallback } from 'react';
import Plot from '../components/Plot';
import { DARK_LAYOUT, PLOT_CONFIG, SERIES_COLORS, getJson, label } from '../components/chartTheme';

const fmt = (v) => (v == null ? '—' : v.toFixed(3));
const Loading = ({ text }) => <div className="loading-overlay"><div className="spinner" /> {text}</div>;
const BASELINE = 'Majority-class baseline';

function MetricsTable({ rows, testid, extra }) {
  return (
    <table className="data-table" data-testid={testid}>
      <thead>
        <tr>
          <th>#</th><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1 (severe)</th>
          <th>Precision</th><th>Recall</th><th>Balanced acc.</th><th>Accuracy</th>{extra && <th>{extra.title}</th>}
        </tr>
      </thead>
      <tbody>
        {rows.map((m, i) => (
          <tr key={m.name} className={i === 0 ? 'highlight' : ''}>
            <td className="mono">{i + 1}</td>
            <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{m.name}</td>
            <td className="mono" style={{ color: '#2E8B57', fontWeight: 600 }}>{fmt(m.roc_auc)}</td>
            <td className="mono">{fmt(m.pr_auc)}</td>
            <td className="mono">{fmt(m.f1)}</td>
            <td className="mono">{fmt(m.precision)}</td>
            <td className="mono">{fmt(m.recall)}</td>
            <td className="mono">{fmt(m.balanced_accuracy)}</td>
            <td className="mono">{fmt(m.accuracy)}</td>
            {extra && <td>{extra.render(m)}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelCommandCenter({ api }) {
  const [metrics, setMetrics] = useState(null);
  const [roc, setRoc] = useState(null);
  const [importance, setImportance] = useState(null);
  const [selected, setSelected] = useState([]);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [met, rocData, imp] = await Promise.all([
        getJson(`${api}/api/models/metrics`),
        getJson(`${api}/api/models/roc`),
        getJson(`${api}/api/models/feature-importance`),
      ]);
      setMetrics(met);
      setRoc(rocData);
      setImportance(imp);
      setSelected(met.models.map(m => m.name));
    } catch (e) {
      setError('Could not load model results from the API.');
    }
  }, [api]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const toggle = (name) => setSelected(prev => (prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]));

  if (error) return <div className="insight-block"><p>{error}</p></div>;
  if (!metrics || metrics.models.length === 0) return <Loading text="Loading model metrics..." />;

  const final = metrics.models.find(m => m.name === metrics.final_model);
  const finalHoldout = metrics.holdout.find(m => m.name === metrics.final_model);
  const baseRatePct = (metrics.positive_rate * 100).toFixed(1);
  const curves = roc ? Object.entries(roc.curves).filter(([name]) => selected.includes(name)) : [];

  return (
    <div data-testid="model-command-center">
      <div className="page-header">
        <h2>Model Command Center</h2>
        <p>Four classifiers vs a majority-class baseline, predicting whether an accident kills or seriously injures someone</p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Leaderboard: 10-fold stratified cross-validation</div>
            <div className="card-subtitle">{metrics.cv}</div>
          </div>
          <span className="badge badge-best">Final model: {metrics.final_model}</span>
        </div>
        <div className="card-body" style={{ padding: 0, overflow: 'auto' }}>
          <MetricsTable rows={metrics.models} testid="model-leaderboard" extra={{
            title: 'ROC',
            render: (m) => (
              <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)}
                style={{ accentColor: '#C8102E', cursor: 'pointer' }}
                data-testid={`compare-${m.name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`} />
            ),
          }} />
        </div>
      </div>

      {final && (
        <div className="insight-block" style={{ marginBottom: 24 }}>
          <div className="insight-label">How to read this</div>
          <p>
            Only {baseRatePct}% of accidents are severe, so always predicting "not severe" already scores{' '}
            {(metrics.models.find(m => m.name === BASELINE)?.accuracy * 100).toFixed(1)}% accuracy with zero recall.
            That is why accuracy is not the headline metric. ROC-AUC {final.roc_auc.toFixed(3)} means the model ranks a
            random severe accident above a random non-severe one {(final.roc_auc * 100).toFixed(0)}% of the time;
            PR-AUC {final.pr_auc.toFixed(3)} compares with {(metrics.positive_rate).toFixed(3)} for random guessing.
            {finalHoldout && ` Trained on 2001-2020 and tested on unseen 2021-2025 accidents, it scores ROC-AUC ${finalHoldout.roc_auc.toFixed(3)}.`}
          </p>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">ROC Curves</div>
              <div className="card-subtitle">Real curves from {roc?.source || 'out-of-fold predictions'}; tick models above to compare</div>
            </div>
          </div>
          <div className="card-body" data-testid="roc-curves-chart">
            <Plot
              data={[
                ...curves.map(([name, c], i) => ({
                  x: c.fpr, y: c.tpr, type: 'scatter', mode: 'lines', name: `${name} (AUC ${c.auc.toFixed(3)})`,
                  line: { color: SERIES_COLORS[i % SERIES_COLORS.length], width: 2.5 },
                  hovertemplate: `${name}<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>`,
                })),
                { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'Chance', line: { color: '#6E7681', width: 1, dash: 'dash' } },
              ]}
              layout={{
                ...DARK_LAYOUT, height: 480, margin: { l: 50, r: 20, t: 20, b: 140 },
                xaxis: { ...DARK_LAYOUT.xaxis, title: { text: 'False positive rate', font: { size: 12, color: '#8B949E' } }, range: [0, 1] },
                yaxis: { ...DARK_LAYOUT.yaxis, title: { text: 'True positive rate', font: { size: 12, color: '#8B949E' } }, range: [0, 1.02] },
                legend: { font: { color: '#8B949E', size: 10 }, orientation: 'h', x: 0, y: -0.2, yanchor: 'top' },
              }}
              config={PLOT_CONFIG}
              style={{ width: '100%' }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">What drives the prediction (SHAP)</div>
              <div className="card-subtitle">{importance?.method || ''}</div>
            </div>
          </div>
          <div className="card-body" data-testid="feature-importance-chart">
            {importance ? (
              <Plot
                data={[{
                  y: [...importance.features].reverse().map(label), x: [...importance.importances].reverse(),
                  type: 'bar', orientation: 'h',
                  marker: { color: [...importance.importances].reverse().map((v, i, a) => (i >= a.length - 3 ? '#C8102E' : i >= a.length - 7 ? '#D29922' : '#1A3A5C')) },
                  hovertemplate: '<b>%{y}</b><br>mean |SHAP| %{x:.3f}<extra></extra>',
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 420, margin: { l: 210, r: 20, t: 20, b: 40 },
                  xaxis: { ...DARK_LAYOUT.xaxis, title: { text: 'Mean |SHAP| (log-odds)', font: { size: 12, color: '#8B949E' } } },
                  yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 10 } },
                }}
                config={PLOT_CONFIG}
                style={{ width: '100%' }}
              />
            ) : <Loading text="" />}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Time-based holdout</div>
            <div className="card-subtitle">{metrics.time_holdout}: no shuffling, so no future data leaks into training</div>
          </div>
        </div>
        <div className="card-body" style={{ padding: 0, overflow: 'auto' }}>
          <MetricsTable rows={metrics.holdout} testid="holdout-table" />
        </div>
      </div>

      {metrics.leakage_checks.length > 0 && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Leakage checks</div>
              <div className="card-subtitle">TSB fills in more fields for serious accidents; these checks measure how much that matters</div>
            </div>
          </div>
          <div className="card-body" style={{ padding: 0, overflow: 'auto' }}>
            <table className="data-table" data-testid="leakage-table">
              <thead><tr><th>Check</th><th>ROC-AUC</th><th>PR-AUC</th><th>Rows</th><th>Meaning</th></tr></thead>
              <tbody>
                {metrics.leakage_checks.map(c => (
                  <tr key={c.model}>
                    <td style={{ fontWeight: 600, color: '#E6EDF3' }}>{c.model}</td>
                    <td className="mono">{fmt(c.roc_auc)}</td>
                    <td className="mono">{fmt(c.pr_auc)}</td>
                    <td className="mono">{c.n_rows.toLocaleString()}</td>
                    <td>{c.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {importance && (
        <div className="insight-block" style={{ marginTop: 24 }}>
          <div className="insight-label">Model interpretation (from SHAP)</div>
          <p>
            The strongest drivers are {importance.features.slice(0, 3).map(label).join(', ')}. In plain terms: fatal and
            serious-injury accidents cluster on main-line running at speed (crossings and trespassers), while the far more
            common yard and switching accidents are rarely severe.
          </p>
        </div>
      )}
    </div>
  );
}
