import React, { useState, useEffect, useCallback } from 'react';
import './index.css';
import ExecutiveDashboard from './pages/ExecutiveDashboard';
import DeepDiveExplorer from './pages/DeepDiveExplorer';
import ModelCommandCenter from './pages/ModelCommandCenter';
import LiveRiskAssessor from './pages/LiveRiskAssessor';
import { BarChart3, Search, Cpu, Shield, Menu, X } from 'lucide-react';

const API = process.env.REACT_APP_BACKEND_URL || '';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Executive Dashboard', icon: BarChart3 },
  { id: 'explorer', label: 'Deep Dive Explorer', icon: Search },
  { id: 'models', label: 'Model Command Center', icon: Cpu },
  { id: 'predictor', label: 'Live Risk Assessor', icon: Shield },
];

function Sidebar({ active, onNavigate, sidebarOpen, onClose }) {
  return (
    <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} data-testid="sidebar">
      <div className="sidebar-logo">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg, #C8102E, #8B0000)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, fontWeight: 800, color: '#fff'
          }}>RR</div>
          <div>
            <h1>Rail Risk Intelligence</h1>
            <span>Rail Incident Analytics Platform</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => (
          <div
            key={item.id}
            className={`nav-item ${active === item.id ? 'active' : ''}`}
            onClick={() => { onNavigate(item.id); onClose(); }}
            data-testid={`nav-${item.id}`}
          >
            <item.icon className="nav-icon" />
            <span>{item.label}</span>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status" data-testid="model-status">
          <div className="status-dot" />
          <span>Model: Random Forest v1.0</span>
        </div>
        <div className="sidebar-author">
          Pranay Ratan<br />
          BSc Data Science, SFU
        </div>
      </div>
    </aside>
  );
}

function App() {
  const [page, setPage] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [stats, setStats] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/stats`);
      if (res.ok) setStats(await res.json());
    } catch (e) { console.error('Stats fetch error:', e); }
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  const renderPage = () => {
    switch (page) {
      case 'dashboard': return <ExecutiveDashboard stats={stats} api={API} />;
      case 'explorer': return <DeepDiveExplorer api={API} />;
      case 'models': return <ModelCommandCenter api={API} />;
      case 'predictor': return <LiveRiskAssessor api={API} />;
      default: return <ExecutiveDashboard stats={stats} api={API} />;
    }
  };

  return (
    <div className="app-layout">
      <button
        className="mobile-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        data-testid="mobile-menu-toggle"
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <Sidebar
        active={page}
        onNavigate={setPage}
        sidebarOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="main-content" data-testid="main-content">
        {renderPage()}
        <div className="disclaimer">
          Built with Transport Canada public data. For demonstration purposes only.
          All predictions are illustrative and not intended for operational safety decisions.
        </div>
      </main>
    </div>
  );
}

export default App;
