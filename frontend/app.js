/* ═══════════════════════════════════════════════════════════════════
   EdgeGuard AI — Frontend Logic v2.1
   5 views · 5 AI engines · Demo-mode resilient
   ═══════════════════════════════════════════════════════════════════ */

const DEFAULT_BACKEND = 'http://localhost:8000';

/* ═══════════════════════════════════════════════════════════════════
   THEME — neumorphism light is the only theme. Always applied on load.
   No picker; no toggle. Persisted only to detect first run.
   ═══════════════════════════════════════════════════════════════════ */
const themeEngine = {
    apply: function () {
        const body = document.body;
        body.classList.add('theme-neumorph');
        // Update Chart.js defaults for the light surface
        if (typeof Chart !== 'undefined') {
            Chart.defaults.color = '#4a5070';
            Chart.defaults.borderColor = 'rgba(0,0,0,0.08)';
        }
        window.dispatchEvent(new Event('theme-changed'));
    },
    init: function () {
        // Apply as early as possible to avoid a flash of dark theme
        this.apply();
    },
};
themeEngine.init();

const SENSORS = [
    { id: 'temperature',         label: 'Temperature', color: '#FF4E4E', unit: '°C' },
    { id: 'vibration',           label: 'Vibration',   color: '#F39C12', unit: 'g' },
    { id: 'oil_pressure',        label: 'Oil Pressure',color: '#00D4FF', unit: 'bar' },
    { id: 'hydraulic_pressure',  label: 'Hydraulic',   color: '#6C63FF', unit: 'bar' },
    { id: 'suspension_pressure', label: 'Suspension',  color: '#2ECC71', unit: 'bar' },
    { id: 'battery_voltage',     label: 'Battery',     color: '#8A8A9A', unit: 'V' }
];

const SENSOR_THRESHOLDS = {
    temperature:         { healthy: [null, 75],  warning: [75, 90],    critical: [90, null] },
    vibration:           { healthy: [null, 0.8], warning: [0.8, 1.5],  critical: [1.5, null] },
    oil_pressure:        { healthy: [2.5, null], warning: [1.5, 2.5],  critical: [null, 1.5] },
    hydraulic_pressure:  { healthy: [180, null], warning: [120, 180], critical: [null, 120] },
    suspension_pressure: { healthy: [5.0, null], warning: [3.5, 5.0], critical: [null, 3.5] },
    battery_voltage:     { healthy: [22, null],  warning: [21, 22],   critical: [null, 21] }
};

const SENSOR_SOP = {
    temperature: '<strong>Engine Overheat</strong> — Reduce load immediately. Park in a safe, ventilated area. Check coolant level, radiator fan operation, and thermostat. Do not resume until temperature drops below 80°C.',
    vibration: '<strong>Axle Bearing Wear</strong> — Reduce speed below 40 km/h. Inspect wheel hub for play, check for hot spots with IR thermometer. Schedule wheel bearing repacking within 8 hours.',
    oil_pressure: '<strong>Low Oil Pressure</strong> — STOP engine immediately to prevent seizure. Check oil level, oil filter, and pressure sender. Tow to nearest service bay.',
    hydraulic_pressure: '<strong>Hydraulic System Weak</strong> — Avoid tipper bed actuation. Inspect for external leaks at ram, hoses, and fittings. Maximum 2 tipper cycles permitted before mandatory service.',
    suspension_pressure: '<strong>Suspension Pressure Loss</strong> — Reduce payload by 20%. Check air line connections, valve block, and bellows for leaks. Schedule air-spring inspection within 24 hours.',
    battery_voltage: '<strong>Low Battery</strong> — Switch off non-essential electrical loads. Verify alternator output (should be 27.6–28.8V running). Inspect battery terminals for corrosion.'
};

const TRUCK_FLEET = [
    { id: 'truck1', name: 'Tata Signa 4825.TK · Unit 01', zone: 'Pit A · Bench 3' },
    { id: 'truck2', name: 'Tata Signa 4825.TK · Unit 02', zone: 'Pit B · Bench 1' },
    { id: 'truck3', name: 'Tata Signa 4825.TK · Unit 03', zone: 'Pit A · Bench 5' }
];

const ICONS = {
    clock:  `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/></svg>`,
    tools:  `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
    doc:    `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/></svg>`,
    search: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`
};

/* ═══════════════════════════════════════════════════════════════════
   SETTINGS (localStorage)
   ═══════════════════════════════════════════════════════════════════ */
const SETTINGS_KEY = 'edgeguard_settings_v2';
const DEFAULT_SETTINGS = {
    backendUrl: DEFAULT_BACKEND,
    pollInterval: 2,
    truckId: 'truck1',
    alertThreshold: 0.7,
    demoMode: true,
    soundAlerts: false,
    hourlyRevenue: 18000,
    platformCost: 6000000,
    downtimeCost: 35000,
    fuelCost: 95
};
function loadSettings() {
    try {
        const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
        return { ...DEFAULT_SETTINGS, ...s };
    } catch { return { ...DEFAULT_SETTINGS }; }
}
function saveSettings(s) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); }
let settings = loadSettings();

/* ═══════════════════════════════════════════════════════════════════
   STATE
   ═══════════════════════════════════════════════════════════════════ */
const state = {
    charts: {},
    history: {},        // sensor -> circular buffer
    historyMax: 30,
    prediction: { failure_probability: 0, rul_hours: null, component: '—', severity: 'healthy' },
    predHistory: [],    // [{ts, prob}]
    alerts: [],
    sopCount: 0,
    connected: false,
    demo: false,
    demoT0: Date.now(),
    fleetState: {},     // truckId -> { health, status, rul, alerts, lastSeen }
    lastPoll: null
};

SENSORS.forEach(s => state.history[s.id] = []);

/* ═══════════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════════ */
function classify(sensor, v) {
    if (v == null || isNaN(v)) return 'healthy';
    const t = SENSOR_THRESHOLDS[sensor];
    if (!t) return 'healthy';
    const inRange = (r, x) => (r[0] == null || x >= r[0]) && (r[1] == null || x <= r[1]);
    if (inRange(t.healthy, v)) return 'healthy';
    if (inRange(t.warning, v)) return 'warning';
    return 'critical';
}
function fmtINR(n) {
    if (n == null || isNaN(n)) return '—';
    const abs = Math.abs(n);
    if (abs >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
    if (abs >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
    if (abs >= 1e3) return '₹' + (n / 1e3).toFixed(1) + 'K';
    return '₹' + Math.round(n).toLocaleString('en-IN');
}
function fmtNum(n, d = 1) {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(d);
}
function escapeHTML(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
function inRange(r, x) { return (r[0] == null || x >= r[0]) && (r[1] == null || x <= r[1]); }

/* ═══════════════════════════════════════════════════════════════════
   TOOLTIP — single floating tooltip for every [data-tip] element
   ═══════════════════════════════════════════════════════════════════ */
const tip = (() => {
    let el = null;
    function ensure() {
        if (!el) {
            el = document.createElement('div');
            el.className = 'global-tooltip';
            document.body.appendChild(el);
        }
        return el;
    }
    function show(e) {
        const node = e.currentTarget;
        const text = node.getAttribute('data-tip');
        if (!text) return;
        const tipEl = ensure();
        tipEl.textContent = text;
        tipEl.classList.add('show');
        const r = node.getBoundingClientRect();
        const tipR = tipEl.getBoundingClientRect();
        let left = r.left + r.width / 2 - tipR.width / 2;
        let top = r.top - tipR.height - 10;
        if (top < 8) top = r.bottom + 10; // flip below
        left = Math.max(8, Math.min(window.innerWidth - tipR.width - 8, left));
        tipEl.style.left = left + 'px';
        tipEl.style.top = top + 'px';
    }
    function hide() { if (el) el.classList.remove('show'); }
    function attach() {
        document.querySelectorAll('[data-tip]').forEach(n => {
            n.removeEventListener('mouseenter', show);
            n.removeEventListener('mouseleave', hide);
            n.removeEventListener('focus', show);
            n.removeEventListener('blur', hide);
            n.addEventListener('mouseenter', show);
            n.addEventListener('mouseleave', hide);
            n.addEventListener('focus', show);
            n.addEventListener('blur', hide);
        });
    }
    return { attach, hide };
})();
// Re-scan for new [data-tip] elements whenever DOM changes meaningfully
const _tipObserver = new MutationObserver(() => tip.attach());
_tipObserver.observe(document.body, { childList: true, subtree: true });
window.addEventListener('scroll', () => tip.hide(), true);
window.addEventListener('resize', () => tip.hide());

async function fetchJSON(url, opts = {}) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 3000);
    try {
        const r = await fetch(url, { ...opts, signal: ctl.signal });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return await r.json();
    } finally { clearTimeout(t); }
}
async function safeFetch(url) {
    try { return { ok: true, data: await fetchJSON(url) }; }
    catch (e) { return { ok: false, error: e.message || String(e) }; }
}

/* ═══════════════════════════════════════════════════════════════════
   CONNECTION
   ═══════════════════════════════════════════════════════════════════ */
function setConnection(ok) {
    state.connected = ok;
    const dot = document.querySelector('#connectionStatus .dot');
    const txt = document.querySelector('#connectionStatus .text');
    if (ok) {
        dot.className = 'dot connected';
        txt.textContent = 'Live (MQTT Connected)';
    } else {
        dot.className = 'dot disconnected';
        txt.textContent = state.demo ? 'Demo Mode' : 'Disconnected';
    }
}

/* ═══════════════════════════════════════════════════════════════════
   POLLING
   ═══════════════════════════════════════════════════════════════════ */
async function pollTelemetry() {
    const res = await safeFetch(`${settings.backendUrl}/readings/latest`);
    if (!res.ok) {
        if (settings.demoMode) enterDemo('Backend unreachable');
        setConnection(false);
        return false;
    }
    setConnection(true);
    exitDemo();
    state.lastPoll = new Date();
    const data = res.data || {};
    const ts = Date.now();
    SENSORS.forEach(s => {
        const r = data[s.id];
        if (r == null) return;
        const v = typeof r === 'object' ? r.value : r;
        if (v == null) return;
        const arr = state.history[s.id];
        arr.push({ v: +v, ts });
        if (arr.length > state.historyMax) arr.shift();
    });
    // Update active truck in fleet
    state.fleetState[settings.truckId] = {
        ...(state.fleetState[settings.truckId] || {}),
        lastSeen: ts
    };
    return true;
}

async function pollML() {
    const res = await safeFetch(`${settings.backendUrl}/ml/status?truck_id=${settings.truckId}`);
    if (!res.ok || !res.data || res.data.status === 'no_predictions_yet') {
        state.prediction = { failure_probability: 0, rul_hours: null, component: '—', severity: 'healthy' };
        return;
    }
    const d = res.data;
    const prob = +d.failure_probability || 0;
    state.prediction = {
        failure_probability: prob,
        rul_hours: d.rul_hours,
        component: d.component || '—',
        severity: d.alert_level || 'normal'
    };
    state.predHistory.push({ ts: Date.now(), prob });
    if (state.predHistory.length > 50) state.predHistory.shift();
}

async function pollAlerts() {
    const res = await safeFetch(`${settings.backendUrl}/alerts/active?truck_id=${settings.truckId}`);
    if (!res.ok) { state.alerts = []; return; }
    const data = Array.isArray(res.data) ? res.data : [];
    state.alerts = data;
    state.fleetState[settings.truckId] = {
        ...(state.fleetState[settings.truckId] || {}),
        alerts: data.length
    };
}

async function pollPredictionHistory() {
    const res = await safeFetch(`${settings.backendUrl}/predictions/history?truck_id=${settings.truckId}&limit=50`);
    if (!res.ok || !Array.isArray(res.data)) return;
    state.predHistory = res.data.map(p => ({ ts: new Date(p.created_at).getTime(), prob: +p.failure_probability || 0 }));
}

/* ═══════════════════════════════════════════════════════════════════
   DEMO MODE
   ═══════════════════════════════════════════════════════════════════ */
function enterDemo(reason) {
    if (state.demo) return;
    state.demo = true;
    state.demoT0 = Date.now();
}
function exitDemo() { if (state.demo) state.demo = false; }

function demoStep() {
    // 90s cycle: 0-30 healthy, 30-55 warning, 55-78 critical, 78-90 reset
    const elapsed = ((Date.now() - state.demoT0) / 1000) % 90;
    const phase =
        elapsed < 30 ? 'healthy' :
        elapsed < 55 ? 'warning' :
        elapsed < 78 ? 'critical' : 'reset';

    const targets = {
        healthy:  { temperature: 62,  vibration: 0.4, oil_pressure: 4.2, hydraulic_pressure: 210, suspension_pressure: 6.2, battery_voltage: 24.1, fp: 0.05, rul: 720, comp: 'Engine' },
        warning:  { temperature: 82,  vibration: 1.1, oil_pressure: 2.0, hydraulic_pressure: 150, suspension_pressure: 4.2, battery_voltage: 21.4, fp: 0.42, rul: 96,  comp: 'Engine' },
        critical: { temperature: 96,  vibration: 1.7, oil_pressure: 1.2, hydraulic_pressure: 100, suspension_pressure: 3.0, battery_voltage: 20.4, fp: 0.78, rul: 12,  comp: 'Hydraulic' },
        reset:    { temperature: 60,  vibration: 0.35,oil_pressure: 4.4, hydraulic_pressure: 215, suspension_pressure: 6.4, battery_voltage: 24.3, fp: 0.05, rul: 720, comp: 'Engine' }
    }[phase];

    const lerp = 0.08;
    const ts = Date.now();
    SENSORS.forEach(s => {
        const cur = state.history[s.id].length ? state.history[s.id][state.history[s.id].length - 1].v : targets[s.id];
        const noise = (Math.random() - 0.5) * 0.4;
        const next = cur + (targets[s.id] - cur) * lerp + noise;
        const arr = state.history[s.id];
        arr.push({ v: +next.toFixed(2), ts });
        if (arr.length > state.historyMax) arr.shift();
    });

    if (!state.prediction.failure_probability) state.prediction.failure_probability = 0;
    state.prediction.failure_probability += (targets.fp - state.prediction.failure_probability) * lerp;
    state.prediction.rul_hours = targets.rul;
    state.prediction.component = targets.comp;
    state.prediction.severity = state.prediction.failure_probability > 0.6 ? 'critical' : state.prediction.failure_probability > 0.3 ? 'warning' : 'healthy';
    state.predHistory.push({ ts, prob: state.prediction.failure_probability });
    if (state.predHistory.length > 50) state.predHistory.shift();

    if (phase === 'warning') {
        state.alerts = [{
            component: 'Engine', severity: 'warning',
            message: 'Engine temperature trending above safe operating range.',
            sop_reference: 'Engine Cooling System Inspection',
            created_at: new Date().toISOString()
        }];
    } else if (phase === 'critical') {
        state.alerts = [
            { component: 'Hydraulic System', severity: 'critical', message: 'Hydraulic pressure loss detected — risk of tipper bed failure.', sop_reference: 'Hydraulic Pressure Drop Response', created_at: new Date(Date.now() - 3000).toISOString() },
            { component: 'Engine', severity: 'critical', message: 'Engine coolant temperature critical. Reduce load immediately.', sop_reference: 'Engine Overheat Procedure', created_at: new Date(Date.now() - 8000).toISOString() },
            { component: 'Axle Bearing', severity: 'high', message: 'Vibration levels elevated at front axle.', sop_reference: 'Wheel Bearing Inspection', created_at: new Date(Date.now() - 12000).toISOString() }
        ];
    } else {
        state.alerts = [];
    }
    state.fleetState[settings.truckId] = { ...(state.fleetState[settings.truckId] || {}), alerts: state.alerts.length, lastSeen: ts };
}

// Simulated other trucks in demo mode
function demoFleetStep() {
    if (!state.demo) return;
    TRUCK_FLEET.forEach(t => {
        if (t.id === settings.truckId) return; // live one is updated above
        const seed = t.id.charCodeAt(t.id.length - 1);
        // Each truck has its own slow drift
        const phase = ((Date.now() / 1000 + seed * 30) / 90) % 1;
        let health;
        if (phase < 0.4) health = 90 + Math.random() * 8;
        else if (phase < 0.7) health = 65 + Math.random() * 15;
        else if (phase < 0.85) health = 40 + Math.random() * 20;
        else health = 85 + Math.random() * 10;
        let status = 'healthy';
        if (health < 60) status = 'critical';
        else if (health < 80) status = 'warning';
        state.fleetState[t.id] = {
            health: Math.round(health),
            status,
            rul: Math.round(200 - (100 - health) * 5),
            alerts: status === 'critical' ? 2 : status === 'warning' ? 1 : 0,
            lastSeen: Date.now()
        };
    });
    // Also synthesize the live truck's health from current prediction
    const live = state.fleetState[settings.truckId] || {};
    const health = Math.round(100 - (state.prediction.failure_probability || 0) * 100);
    let status = 'healthy';
    if (health < 60) status = 'critical';
    else if (health < 80) status = 'warning';
    state.fleetState[settings.truckId] = {
        ...live,
        health,
        status,
        rul: state.prediction.rul_hours ? Math.round(state.prediction.rul_hours) : null,
        alerts: state.alerts.length,
        lastSeen: Date.now()
    };
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · COMMAND CENTER
   ═══════════════════════════════════════════════════════════════════ */
function initCharts() {
    Chart.defaults.color = '#8A8FA8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 10;
    SENSORS.forEach(s => {
        const ctx = document.getElementById(`chart-${s.id}`).getContext('2d');
        state.charts[s.id] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(state.historyMax).fill(''),
                datasets: [{
                    data: Array(state.historyMax).fill(null),
                    borderColor: s.color,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: { target: 'origin', above: s.color + '22' }
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false, suggestedMin: 0 } },
                animation: { duration: 0 }
            }
        });
    });
    // Prediction trend
    const trendCtx = document.getElementById('predictionTrend').getContext('2d');
    state.charts.predTrend = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: Array(50).fill(''),
            datasets: [{
                label: 'Failure probability',
                data: Array(50).fill(null),
                borderColor: '#6C63FF',
                backgroundColor: 'rgba(108, 99, 255, 0.15)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
                spanGaps: false,
                fill: true
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: {
                    display: true, min: 0, max: 1,
                    grid: { color: 'rgba(0,0,0,0.06)' },
                    ticks: { color: '#525870', font: { size: 9 }, stepSize: 0.5 }
                }
            },
            animation: { duration: 0 }
        }
    });
}

function seedPredHistory(count = 50, baseProb = 0.05) {
    const now = Date.now();
    state.predHistory = [];
    for (let i = 0; i < count; i++) {
        const wave = Math.sin(i / 8) * 0.04;
        const noise = (Math.random() - 0.5) * 0.02;
        state.predHistory.push({
            ts: now - (count - i) * 3000,
            prob: Math.max(0, Math.min(1, baseProb + wave + noise))
        });
    }
}

function updatePredTrendChart() {
    const chart = state.charts.predTrend;
    if (!chart) return;
    const hist = state.predHistory.slice(-50);
    const data = Array(50).fill(null);
    const offset = 50 - hist.length;
    hist.forEach((p, i) => { data[offset + i] = p.prob; });
    chart.data.datasets[0].data = data;
    chart.update('none');
}

function renderCommandCenter() {
    renderPrediction();
    renderSensorCards();
    renderAlerts();
    renderHealthSummary();
    renderEnginePips();
    renderLastUpdated();
    renderBizDependentStats();
}

function renderPrediction() {
    const prob = state.prediction.failure_probability || 0;
    const pct = Math.round(prob * 100);
    const gauge = document.getElementById('failureProbGauge');
    const val = document.getElementById('failureProbValue');
    let color = '#22c55e', st = 'healthy', lbl = 'System Healthy';
    if (state.prediction.severity === 'critical' || prob >= 0.75) { color = '#FF4E4E'; st = 'critical'; lbl = 'Critical Risk'; }
    else if (state.prediction.severity === 'high' || prob >= 0.5) { color = '#F97316'; st = 'warning'; lbl = 'Warning State'; }
    else if (prob >= 0.3) { color = '#F97316'; st = 'warning'; lbl = 'Elevated Risk'; }
    // Healthy: full green ring. Otherwise: partial fill in accent color.
    const deg = st === 'healthy' ? 360 : pct * 3.6;
    gauge.style.setProperty('--gauge-deg', deg);
    gauge.style.setProperty('--gauge-color', color);
    gauge.setAttribute('data-state', st);
    val.textContent = `${pct}%`;
    val.style.color = color;
    const stateTagEl = document.getElementById('gaugeStateTag');
    if (stateTagEl) {
        stateTagEl.setAttribute('data-state', st);
        stateTagEl.textContent = lbl;
    }
    const badge = document.getElementById('mlStatusBadge');
    badge.className = `badge ${st === 'critical' ? 'critical' : st === 'warning' ? 'warning' : 'normal'}`;
    badge.textContent = lbl;
    document.getElementById('rulValue').innerHTML = state.prediction.rul_hours != null ? `${state.prediction.rul_hours.toFixed(1)} <small>hrs</small>` : '-- <small>hrs</small>';
    document.getElementById('componentValue').textContent = state.prediction.component || '—';

    // Copilot
    const copilot = document.getElementById('copilotExplanation');
    if (state.alerts.length) {
        const top = state.alerts[0];
        const sopKey = Object.keys(SENSOR_SOP).find(k => (top.component || '').toLowerCase().includes(k.replace('_', ' ')));
        const sopShort = sopKey ? SENSOR_SOP[sopKey].split('—')[1]?.split('.')[0]?.trim() : 'follow standard maintenance protocol';
        copilot.innerHTML = `<strong>${top.component || 'Alert'}:</strong> ${top.message} Recommended action: ${sopShort}.`;
        document.getElementById('copilotEngineTag').textContent = 'Engine 5 · Gemini · just now';
    } else if (prob > 0.3) {
        copilot.textContent = 'AI detects anomalous sensor readings. Monitoring closely — failure probability trending upward. Check the Digital Twin for component-level visualization.';
    } else {
        copilot.textContent = 'All monitored parameters are within normal operating range. Engines 2, 3, and 5 agree on a healthy state for TRUCK1.';
    }

    // Update trend chart — always bind to a full 50-point window
    updatePredTrendChart();
    document.getElementById('trendMeta').textContent = state.predHistory.length
        ? `last ${Math.min(50, state.predHistory.length)} predictions · max ${Math.round(Math.max(...state.predHistory.map(p => p.prob)) * 100)}%`
        : 'awaiting history...';
}

function renderSensorCards() {
    SENSORS.forEach(s => {
        const card = document.querySelector(`.chart-card[data-sensor="${s.id}"]`);
        if (!card) return;
        const arr = state.history[s.id];
        const v = arr.length ? arr[arr.length - 1].v : null;
        const status = v == null ? 'healthy' : classify(s.id, v);
        card.classList.remove('warning', 'critical');
        if (status !== 'healthy') card.classList.add(status);
        const statusEl = document.getElementById(`status-${s.id}`);
        if (statusEl) {
            statusEl.className = 'status-tag ' + status;
            statusEl.textContent = status === 'critical' ? 'CRIT' : status === 'warning' ? 'WARN' : 'OK';
        }
        document.getElementById(`val-${s.id}`).textContent = v == null ? '--' : (v >= 100 ? v.toFixed(0) : v.toFixed(1));
        // Update mini progress bar: distance of `v` from the healthy band.
        // 0% = well inside healthy range, 100% = at/past critical threshold.
        const bar = document.getElementById(`bar-${s.id}`);
        if (bar) {
            let pct = 25;
            if (v != null) {
                const thr = SENSOR_THRESHOLDS[s.id];
                if (thr) {
                    const [, hHi] = thr.healthy;
                    const [, wHi] = thr.warning;
                    if (hHi != null && v > hHi) {
                        // Above healthy: scale 50..100 between healthy-hi and warning-hi (or critical)
                        const lo = hHi;
                        const hi = wHi != null ? wHi : (thr.critical[1] != null ? thr.critical[1] : hHi + 1);
                        pct = Math.max(50, Math.min(100, 50 + ((v - lo) / Math.max(0.0001, hi - lo)) * 50));
                    } else if (hHi != null) {
                        pct = 25;
                    } else {
                        // Lower-bound metric (oil_pressure, battery, hydraulic, suspension):
                        // below healthy-lo = 100, between warning-lo and healthy-lo = 50..100
                        const [wLo] = thr.warning;
                        const hLo = thr.healthy[0];
                        if (v < (wLo != null ? wLo : 0)) pct = 100;
                        else if (hLo != null) pct = Math.max(50, Math.min(100, 50 + ((hLo - v) / Math.max(0.0001, hLo - (wLo != null ? wLo : 0))) * 50));
                        else pct = 25;
                    }
                }
            }
            bar.style.width = pct.toFixed(0) + '%';
        }
        const ch = state.charts[s.id];
        if (ch) {
            const data = ch.data.datasets[0].data;
            data.shift();
            data.push(v);
            ch.update('none');
        }
    });
}

// Module-level state for the alerts panel
let _lastAlertTime = null;       // Date of most recent alert (any severity)
let _panelOpenedAt = Date.now(); // for uptime counter

function fmtUptime(ms) {
    const s = Math.floor(ms / 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
}

function renderAlerts() {
    const list = document.getElementById('alertsList');
    const count = document.getElementById('alertCount');
    const sub = document.getElementById('alertSub');
    count.textContent = state.alerts.length;
    count.className = 'alert-count' + (state.alerts.length ? ' has-alerts' : '');

    // Track most-recent alert timestamp so the empty state can show it
    if (state.alerts.length) {
        const newest = state.alerts.reduce((acc, a) => {
            const t = new Date(a.created_at).getTime();
            return !acc || t > acc ? t : acc;
        }, null);
        if (newest) _lastAlertTime = new Date(newest);
        sub.textContent = `${state.alerts.length} active · ${state.alerts.filter(a => a.severity === 'critical' || a.severity === 'high').length} critical`;
        // Show alert cards
        const emptyEl = document.getElementById('alertsEmpty');
        if (emptyEl) emptyEl.classList.add('hidden');
        // Remove any old items, keep the empty node hidden at end
        Array.from(list.querySelectorAll('.alert-item')).forEach(n => n.remove());
        state.alerts.slice(0, 20).forEach(a => {
            const d = new Date(a.created_at);
            const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const sev = a.severity || 'info';
            const div = document.createElement('div');
            div.className = `alert-item ${sev}`;
            div.innerHTML = `
                <div class="alert-item-header">
                    <span class="alert-component">${(a.component || '').replace(/_/g, ' ')}</span>
                    <span class="alert-time">${time}</span>
                </div>
                <p>${escapeHTML(a.message || '')}</p>
                ${a.sop_reference ? `<span class="alert-sop">${ICONS.doc} ${escapeHTML(a.sop_reference)}</span>` : ''}
            `;
            list.appendChild(div);
        });
        return;
    }
    // No active alerts: show the informative empty state
    sub.textContent = 'Monitoring 6 sensor streams';
    const emptyEl = document.getElementById('alertsEmpty');
    if (emptyEl) {
        emptyEl.classList.remove('hidden');
        Array.from(list.querySelectorAll('.alert-item')).forEach(n => n.remove());
    }
    const visionOffline = !state.connected;
    const engineNote = document.getElementById('alertsEngineNote');
    const healthyTitle = document.getElementById('alertsHealthyTitle');
    if (engineNote) engineNote.hidden = !visionOffline;
    if (healthyTitle) {
        healthyTitle.setAttribute(
            'data-tip',
            visionOffline
                ? 'Sensor alert status only — based on live telemetry thresholds. Engine 1 (Vision) is offline; visual inspection is unavailable but sensor streams may still be healthy.'
                : 'Sensor alert status only — based on live telemetry thresholds, not individual AI engine connectivity.'
        );
    }
    const uptimeEl = document.getElementById('alertsUptime');
    if (uptimeEl) uptimeEl.textContent = fmtUptime(Date.now() - _panelOpenedAt);
    const lastEl = document.getElementById('alertsLastTime');
    if (lastEl) {
        if (_lastAlertTime) {
            const ageMs = Date.now() - _lastAlertTime.getTime();
            const ageS = Math.floor(ageMs / 1000);
            let txt;
            if (ageS < 60) txt = `${ageS}s ago`;
            else if (ageS < 3600) txt = `${Math.floor(ageS / 60)}m ago`;
            else if (ageS < 86400) txt = `${Math.floor(ageS / 3600)}h ago`;
            else txt = `${Math.floor(ageS / 86400)}d ago`;
            lastEl.textContent = txt;
        } else {
            lastEl.textContent = 'none recorded';
        }
    }
}

// Lightweight ticker to keep the uptime + last-alert relative time fresh
setInterval(() => {
    if (state.alerts.length === 0) {
        const uptimeEl = document.getElementById('alertsUptime');
        if (uptimeEl) uptimeEl.textContent = fmtUptime(Date.now() - _panelOpenedAt);
        const lastEl = document.getElementById('alertsLastTime');
        if (lastEl && _lastAlertTime) {
            const ageS = Math.floor((Date.now() - _lastAlertTime.getTime()) / 1000);
            let txt;
            if (ageS < 60) txt = `${ageS}s ago`;
            else if (ageS < 3600) txt = `${Math.floor(ageS / 60)}m ago`;
            else if (ageS < 86400) txt = `${Math.floor(ageS / 3600)}h ago`;
            else txt = `${Math.floor(ageS / 86400)}d ago`;
            lastEl.textContent = txt;
        }
    }
}, 1000);

function renderHealthSummary() {
    const prob = state.prediction.failure_probability || 0;
    const health = Math.max(0, Math.min(100, Math.round(100 - prob * 100)));
    const big = document.getElementById('bigScore');
    big.innerHTML = `${health}<span>%</span>`;
    const sub = document.getElementById('healthSubtitle');
    let ringState = 'healthy';
    if (health >= 85) { sub.textContent = 'All parameters nominal'; big.style.background = 'linear-gradient(135deg, #22c55e, #00D4FF)'; big.style.webkitBackgroundClip = 'text'; big.style.webkitTextFillColor = 'transparent'; }
    else if (health >= 60) { sub.textContent = 'Elevated risk — monitor closely'; big.style.background = 'linear-gradient(135deg, #F97316, #FF4E4E)'; big.style.webkitBackgroundClip = 'text'; big.style.webkitTextFillColor = 'transparent'; ringState = 'warning'; }
    else { sub.textContent = 'Critical state — service required'; big.style.background = 'linear-gradient(135deg, #FF4E4E, #b91c1c)'; big.style.webkitBackgroundClip = 'text'; big.style.webkitTextFillColor = 'transparent'; ringState = 'critical'; }
    // Update compact health ring
    const ring = document.getElementById('healthRing');
    const ringFill = document.getElementById('healthRingFill');
    if (ring && ringFill) {
        ring.setAttribute('data-state', ringState);
        const filled = Math.max(0, Math.min(100, health));
        ringFill.setAttribute('stroke-dasharray', `${filled} ${100 - filled}`);
        ringFill.setAttribute('stroke-dashoffset', '0');
    }
    document.getElementById('statActiveAlerts').textContent = state.alerts.length;
    document.getElementById('statRul').textContent = state.prediction.rul_hours != null ? Math.round(state.prediction.rul_hours) : '—';
    const sopsEl = document.getElementById('statSops');
    if (sopsEl) {
        const count = state.sopCount || 0;
        sopsEl.textContent = `${count} loaded`;
        sopsEl.setAttribute('data-tip', count
            ? `${count} SOP${count === 1 ? '' : 's'} in the maintenance knowledge base`
            : 'No SOPs currently loaded — open Maintenance after the backend connects');
    }

    // Truck state pill
    const pill = document.getElementById('truckStatePill');
    if (pill) {
        const dot = pill.querySelector('.dot');
        const txt = pill.querySelector('.text');
        const sev = state.prediction.severity;
        dot.className = 'dot ' + (sev === 'critical' || sev === 'high' ? 'red' : sev === 'medium' || sev === 'warning' ? 'amber' : 'green');
        txt.textContent = `${settings.truckId.toUpperCase()} · ${health}%`;
    }

    // Badge
    const badge = document.getElementById('overallHealthBadge');
    if (badge) {
        badge.className = 'badge ' + (health >= 85 ? 'normal' : health >= 60 ? 'warning' : 'critical');
        badge.textContent = health >= 85 ? 'Operational' : health >= 60 ? 'Watch' : 'Critical';
    }
}

function renderEnginePips() {
    const pips = document.querySelectorAll('.engine-pip');
    pips.forEach(p => p.classList.remove('online', 'offline', 'busy'));
    if (!pips.length) return;
    const labels = {
        online: { st: 'online', text: 'Online' },
        busy:   { st: 'busy',   text: 'Running' },
        offline:{ st: 'offline',text: 'Offline' }
    };
    function setPip(node, mode) {
        node.classList.add(mode);
        const lbl = labels[mode];
        const st = node.querySelector('.engine-status');
        if (st && lbl) st.textContent = lbl.text;
    }
    setPip(pips[0], state.connected ? 'online' : 'offline');  // Vision
    setPip(pips[1], state.connected ? 'online' : 'busy');      // Predict
    setPip(pips[2], state.connected ? 'online' : 'busy');      // Fusion
    setPip(pips[3], 'online');                                 // Business
    setPip(pips[4], state.connected ? 'online' : 'busy');      // Copilot
    // Update "synced Xs ago" footer
    const meta = document.getElementById('enginesMeta');
    if (meta) {
        if (state.lastPoll) {
            const age = Math.max(0, Math.floor((Date.now() - state.lastPoll.getTime()) / 1000));
            meta.textContent = state.connected
                ? `All engines synced · ${age}s ago`
                : `Engines reconnecting · last sync ${age}s ago`;
        } else {
            meta.textContent = 'Awaiting first sync…';
        }
    }
}

function renderLastUpdated() {
    const el = document.getElementById('lastUpdated');
    if (state.lastPoll) el.textContent = `Last updated: ${state.lastPoll.toLocaleTimeString()}`;
    else el.textContent = state.demo ? 'Demo mode · simulated telemetry' : 'Awaiting telemetry...';
    const infoEl = document.getElementById('infoLastPoll');
    if (infoEl) infoEl.textContent = state.lastPoll ? state.lastPoll.toLocaleString() : 'never';
    const storageEl = document.getElementById('infoStorage');
    if (storageEl) {
        let total = 0;
        for (const k in localStorage) if (localStorage.hasOwnProperty(k)) total += (localStorage[k].length + k.length) * 2;
        storageEl.textContent = (total / 1024).toFixed(1) + ' KB';
    }
}

function renderBizDependentStats() {
    // Update the ROI view (recompute when prediction changes)
    renderBusiness();
    renderFleet();
    renderTwin();
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · DIGITAL TWIN
   ═══════════════════════════════════════════════════════════════════ */
function renderTwin() {
    const wrap = document.getElementById('twinWrap');
    if (!wrap) return;
    const prob = state.prediction.failure_probability || 0;
    const colors = { healthy: '#22c55e', warning: '#f59e0b', critical: '#ef4444' };
    // Update nodes
    SENSORS.forEach(s => {
        const g = document.getElementById('node_' + s.id);
        if (!g) return;
        const arr = state.history[s.id];
        const v = arr.length ? arr[arr.length - 1].v : null;
        const st = v == null ? 'healthy' : classify(s.id, v);
        g.classList.toggle('critical', st === 'critical');
        g.querySelector('.node-halo').setAttribute('fill', colors[st]);
        g.querySelector('.node-core').setAttribute('fill', colors[st]);
    });
    // Cascade lines
    const line1 = document.getElementById('line_engine_bearing');
    const line2 = document.getElementById('line_bearing_oil');
    const line3 = document.getElementById('line_oil_hydraulic');
    [line1, line2, line3].forEach(l => l.classList.remove('active'));
    if (prob > 0.3) line1.classList.add('active');
    if (prob > 0.45) line2.classList.add('active');
    if (prob > 0.6) line3.classList.add('active');
    const lineColor = prob > 0.7 ? '#ef4444' : prob > 0.45 ? '#f59e0b' : '#475569';
    [line1, line2, line3].forEach(l => { l.setAttribute('stroke', lineColor); l.setAttribute('stroke-width', prob > 0.3 ? 3 : 2); });
    // Stress overlay
    const stress = document.getElementById('stressOverlay');
    if (stress) stress.style.opacity = prob > 0.6 ? Math.min(1, (prob - 0.6) / 0.4) : 0;
    // Chassis darken
    document.querySelectorAll('.chassis-path').forEach(p => {
        p.style.filter = prob > 0.8 ? 'brightness(0.6) saturate(1.4)' : prob > 0.6 ? 'brightness(0.8) saturate(1.2)' : 'none';
    });
    // Banner
    const banner = document.getElementById('critBanner');
    if (banner) banner.classList.toggle('show', prob > 0.8);
    // Tipper raise
    const bed = document.getElementById('tipperBed');
    if (bed) bed.classList.toggle('raised', prob > 0.7);
    // Particles
    document.querySelectorAll('.particle').forEach(p => p.classList.toggle('active', prob > 0.55));
    // Sync indicator
    const sync = document.getElementById('twinSync');
    if (sync) {
        const age = state.lastPoll ? Math.round((Date.now() - state.lastPoll.getTime()) / 1000) : 0;
        sync.textContent = age < 5 ? 'synced just now' : `synced ${age}s ago`;
    }
    // Twin readout grid
    const grid = document.getElementById('twinReadout');
    if (grid) {
        grid.innerHTML = SENSORS.map(s => {
            const arr = state.history[s.id];
            const v = arr.length ? arr[arr.length - 1].v : null;
            const st = v == null ? 'healthy' : classify(s.id, v);
            return `<div class="readout-cell ${st}" data-sensor="${s.id}">
                <span class="rc-name">${s.label}</span>
                <span class="rc-val">${v == null ? '—' : v.toFixed(1)}<span class="rc-unit">${s.unit}</span></span>
            </div>`;
        }).join('');
        grid.querySelectorAll('.readout-cell').forEach(c => {
            c.addEventListener('click', () => openDrawer(c.dataset.sensor));
        });
    }
    // Twin prob / rul
    const probEl = document.getElementById('twinProb');
    const fillEl = document.getElementById('twinProbFill');
    const rulEl = document.getElementById('twinRul');
    if (probEl) {
        probEl.textContent = Math.round(prob * 100) + '%';
        probEl.style.color = prob > 0.6 ? '#ef4444' : prob > 0.3 ? '#f59e0b' : '#22c55e';
    }
    if (fillEl) fillEl.style.width = Math.round(prob * 100) + '%';
    if (rulEl) rulEl.textContent = state.prediction.rul_hours != null ? state.prediction.rul_hours.toFixed(1) + ' hrs' : '— hrs';
    // Twin copilot
    const tc = document.getElementById('twinCopilot');
    if (tc) {
        const p = tc.querySelector('p');
        if (state.alerts.length) {
            const top = state.alerts[0];
            p.innerHTML = `<strong>${top.component || 'Alert'}:</strong> ${escapeHTML(top.message)}`;
        } else if (prob > 0.3) {
            p.textContent = 'Predictive model shows rising failure probability. Inspect the highlighted nodes.';
        } else {
            p.textContent = 'Click any node to inspect a sensor\'s history and SOP.';
        }
    }
}

function buildTwinParticles() {
    const layer = document.getElementById('particleLayer');
    if (!layer || layer.dataset.built) return;
    for (let i = 0; i < 7; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = (305 / 900 * 100) + '%';
        p.style.top  = (100 / 460 * 100) + '%';
        p.style.animationDelay = (i * 0.32) + 's';
        p.style.transform = `translate(${(Math.random() * 30 - 15)}px, 0)`;
        layer.appendChild(p);
    }
    layer.dataset.built = '1';
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · FLEET VIEW
   ═══════════════════════════════════════════════════════════════════ */
function renderFleet() {
    const grid = document.getElementById('fleetGrid');
    if (!grid) return;
    grid.innerHTML = '';
    TRUCK_FLEET.forEach(t => {
        const fs = state.fleetState[t.id] || { health: 95, status: 'healthy', rul: 720, alerts: 0 };
        const card = document.createElement('div');
        card.className = `truck-card ${fs.status === 'healthy' ? '' : fs.status}`;
        card.innerHTML = `
            <div class="tc-head">
                <div>
                    <div class="tc-id">${t.id.toUpperCase()}</div>
                    <div class="tc-name">${t.name}</div>
                </div>
                <span class="tc-status ${fs.status}">${fs.status}</span>
            </div>
            <div class="tc-score-row">
                <span class="tc-score">${fs.health != null ? fs.health : '—'}<span style="font-size:1rem; color: var(--text-tertiary);">%</span></span>
                <span class="tc-score-label">Health</span>
            </div>
            <div class="tc-bar"><div class="tc-bar-fill" style="width: ${fs.health || 0}%"></div></div>
            <div class="tc-stats">
                <div class="tc-stat"><span class="val">${fs.rul != null ? Math.round(fs.rul) : '—'}</span><span class="lbl">hrs RUL</span></div>
                <div class="tc-stat"><span class="val">${fs.alerts || 0}</span><span class="lbl">Alerts</span></div>
                <div class="tc-stat"><span class="val">${fs.lastSeen ? Math.round((Date.now() - fs.lastSeen) / 1000) + 's' : '—'}</span><span class="lbl">Last seen</span></div>
            </div>
        `;
        card.addEventListener('click', () => {
            settings.truckId = t.id;
            document.getElementById('truckSelector').value = t.id;
            saveSettings(settings);
            switchTab('view-command');
        });
        grid.appendChild(card);
    });
    // Aggregates
    const states = TRUCK_FLEET.map(t => state.fleetState[t.id] || { health: 95, alerts: 0 });
    const avg = Math.round(states.reduce((s, x) => s + (x.health || 0), 0) / states.length);
    const totalAlerts = states.reduce((s, x) => s + (x.alerts || 0), 0);
    const atRisk = states.filter(x => (x.health || 100) < 70).length;
    const dtAvoided = Math.round(avg * 0.42);
    document.getElementById('kpiAvgHealth').textContent = avg + '%';
    document.getElementById('kpiAlerts').textContent = totalAlerts;
    document.getElementById('kpiAtRisk').textContent = atRisk;
    document.getElementById('kpiDowntimeAvoided').textContent = dtAvoided;
    document.getElementById('fleetCount').textContent = TRUCK_FLEET.length;
    const fleetBadge = document.getElementById('fleetBadge');
    if (fleetBadge) {
        fleetBadge.textContent = totalAlerts;
        const tip = `${totalAlerts} active alert${totalAlerts === 1 ? '' : 's'} across fleet`;
        fleetBadge.setAttribute('data-tip', tip);
        fleetBadge.setAttribute('aria-label', tip);
        fleetBadge.setAttribute('title', tip);
    }
    const conn = document.getElementById('fleetConn');
    if (conn) conn.textContent = state.connected ? 'aggregator online' : state.demo ? 'aggregator (demo)' : 'aggregator offline';
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · BUSINESS / ROI
   ═══════════════════════════════════════════════════════════════════ */
function renderBusiness() {
    // Constants (from ml-data/engine4_business/business_impact.py)
    const HOURLY = settings.hourlyRevenue;
    const PLATFORM = settings.platformCost;
    const DOWNHOURS_NO_AI = 8.0;
    const DOWNHOURS_WITH_AI = 2.0;
    const UNPLANNED_COST = settings.downtimeCost;
    const FUEL = settings.fuelCost;
    const DAILY_FUEL = 1100;
    const FLEET_SIZE = 5;
    // 5-truck pilot, 365 days
    const annualEvents = 8;                 // expected major incidents per truck/year
    const eventsPerFleet = annualEvents * FLEET_SIZE;
    const hoursAvoided = eventsPerFleet * (DOWNHOURS_NO_AI - DOWNHOURS_WITH_AI);   // 5*8*6 = 240
    const revenueSaved = hoursAvoided * HOURLY;
    const unplannedCostSaved = eventsPerFleet * (UNPLANNED_COST * DOWNHOURS_NO_AI - UNPLANNED_COST * DOWNHOURS_WITH_AI);
    const fuelSaved = eventsPerFleet * 50 * FUEL;   // 50L per incident avoided
    const catastrophicAvoided = Math.round(eventsPerFleet * 0.15);
    const totalSavings = revenueSaved + unplannedCostSaved + fuelSaved;
    const net = totalSavings - PLATFORM;
    const roi = totalSavings / PLATFORM;
    const payback = PLATFORM / (totalSavings / 12);

    document.getElementById('roiBig').innerHTML = `${roi.toFixed(1)}<span>x</span>`;
    document.getElementById('payback').textContent = payback.toFixed(1);
    document.getElementById('moneyNet').textContent = fmtINR(net);
    document.getElementById('moneyAvoided').textContent = fmtINR(totalSavings);
    document.getElementById('moneyCost').textContent = fmtINR(PLATFORM);
    document.getElementById('kpiDtAvoided').textContent = hoursAvoided.toLocaleString('en-IN');
    document.getElementById('kpiRevenue').textContent = fmtINR(revenueSaved);
    document.getElementById('kpiMaint').textContent = fmtINR(unplannedCostSaved);
    document.getElementById('kpiFuel').textContent = fmtINR(fuelSaved);
    document.getElementById('kpiUtil').textContent = '+18%';
    document.getElementById('kpiCatastrophic').textContent = catastrophicAvoided;

    // Chart
    if (state.charts.biz) state.charts.biz.destroy();
    const ctx = document.getElementById('bizChart').getContext('2d');
    state.charts.biz = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Downtime\nAvoided', 'Revenue\nProtected', 'Unplanned\nMaint Saved', 'Fuel\nSaved'],
            datasets: [{
                data: [revenueSaved, revenueSaved * 0.4, unplannedCostSaved, fuelSaved],
                backgroundColor: ['#22c55e', '#00D4FF', '#F39C12', '#6C63FF'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => fmtINR(c.parsed.y) } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8A8FA8', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8A8FA8', font: { size: 10 }, callback: (v) => fmtINR(v) } }
            },
            animation: { duration: 800 }
        }
    });

    // Assumptions
    const assumptions = [
        { lbl: 'Trucks in pilot', val: FLEET_SIZE + ' Tata Signa 4825.TK' },
        { lbl: 'Major incidents / truck / yr', val: annualEvents.toString() },
        { lbl: 'Downtime without AI', val: DOWNHOURS_NO_AI + ' hrs' },
        { lbl: 'Downtime with AI', val: DOWNHOURS_WITH_AI + ' hrs' },
        { lbl: 'Hourly revenue / truck', val: fmtINR(HOURLY) },
        { lbl: 'Unplanned cost / hr', val: fmtINR(UNPLANNED_COST) },
        { lbl: 'Fuel cost / litre', val: fmtINR(FUEL) },
        { lbl: 'Platform cost / year', val: fmtINR(PLATFORM) }
    ];
    document.getElementById('assumptionList').innerHTML = assumptions.map(a =>
        `<div class="assumption-row"><span class="lbl">${a.lbl}</span><span class="val">${a.val}</span></div>`
    ).join('');
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · MAINTENANCE
   ═══════════════════════════════════════════════════════════════════ */
let allSOPs = [];
let filteredSOPs = [];
let activeComponentFilter = 'all';
let isSearchActive = false;
let maintenanceLoaded = false;
let searchDebounceTimer = null;

async function loadAllSOPs() {
    const grid = document.getElementById('sopCardsGrid');
    try {
        const res = await fetchJSON(`${settings.backendUrl}/sops`);
        allSOPs = res || [];
        filteredSOPs = [...allSOPs];
        state.sopCount = allSOPs.length;
        updateSOPStats();
        buildComponentFilters();
        renderSOPCards(filteredSOPs);
        loadAlertTimeline();
    } catch (e) {
        grid.innerHTML = `<div class="sop-empty-state"><div class="empty-icon">⚠️</div><h4>Unable to load SOPs</h4><p>Make sure the backend is running at ${settings.backendUrl}<br>and the RAG index has been built.</p></div>`;
    }
}

function updateSOPStats() {
    const stats = { total: allSOPs.length, critical: 0, warning: 0, preventive: 0 };
    allSOPs.forEach(sop => {
        const sev = sop.severity || 'unknown';
        if (stats[sev] !== undefined) stats[sev]++;
    });
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.querySelector('.stat-number').textContent = val; };
    set('sopTotalCount', stats.total);
    set('sopCriticalCount', stats.critical);
    set('sopWarningCount', stats.warning);
    set('sopPreventiveCount', stats.preventive);
}

function buildComponentFilters() {
    const container = document.getElementById('componentFilters');
    const components = [...new Set(allSOPs.map(s => s.component))];
    const allPill = container.querySelector('[data-component="all"]');
    container.innerHTML = '';
    container.appendChild(allPill);
    components.sort().forEach(comp => {
        const btn = document.createElement('button');
        btn.className = 'filter-pill';
        btn.dataset.component = comp;
        btn.innerHTML = `<span class="filter-dot ${comp}"></span> ${comp.replace(/_/g, ' ')}`;
        container.appendChild(btn);
    });
    container.querySelectorAll('.filter-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            container.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            activeComponentFilter = pill.dataset.component;
            applyFilters();
        });
    });
}

function applyFilters() {
    filteredSOPs = activeComponentFilter === 'all' ? [...allSOPs] : allSOPs.filter(s => s.component === activeComponentFilter);
    renderSOPCards(filteredSOPs);
}

async function searchSOPs(query) {
    if (!query.trim()) { isSearchActive = false; document.getElementById('searchResultCount').style.display = 'none'; document.getElementById('sopSearchClear').style.display = 'none'; applyFilters(); return; }
    isSearchActive = true;
    const grid = document.getElementById('sopCardsGrid');
    grid.innerHTML = `<div class="sop-loading"><div class="sop-loading-spinner"></div><p>Searching knowledge base...</p></div>`;
    try {
        const res = await fetchJSON(`${settings.backendUrl}/sops/search?q=${encodeURIComponent(query)}&top_k=10`);
        const countEl = document.getElementById('searchResultCount');
        countEl.querySelector('span').textContent = (res || []).length;
        countEl.style.display = 'inline-flex';
        document.getElementById('sopSearchClear').style.display = 'flex';
        if (!res || res.length === 0) {
            grid.innerHTML = `<div class="sop-empty-state"><div class="empty-icon">🔍</div><h4>No matching SOPs found</h4><p>Try different keywords, e.g. "hydraulic pressure", "engine overheating", "bearing failure"</p></div>`;
            return;
        }
        renderSOPCards(res, true);
    } catch (e) {
        grid.innerHTML = `<div class="sop-empty-state"><div class="empty-icon">⚠️</div><h4>Search unavailable</h4><p>Backend may be offline.</p></div>`;
    }
}

function renderSOPCards(sops, showSimilarity = false) {
    const grid = document.getElementById('sopCardsGrid');
    if (!sops.length) {
        grid.innerHTML = `<div class="sop-empty-state"><div class="empty-icon">📋</div><h4>No SOPs for this filter</h4><p>Try a different component or search term.</p></div>`;
        return;
    }
    grid.innerHTML = '';
    sops.forEach((sop, i) => {
        const sev = sop.severity || 'warning';
        const card = document.createElement('div');
        card.className = `sop-card severity-${sev}`;
        card.style.animationDelay = `${i * 0.06}s`;
        let simHTML = '';
        if (showSimilarity && sop.similarity_score != null) {
            const pct = Math.round(sop.similarity_score * 100);
            simHTML = `<div class="sop-similarity-bar"><div class="sop-similarity-fill" style="width:${pct}%"></div></div>
                <div class="sop-meta-item" style="margin-top:4px">${ICONS.search} Match: ${pct}%</div>`;
        }
        let metaHTML = '';
        if (sop.estimated_downtime) metaHTML += `<div class="sop-meta-item">${ICONS.clock} ${escapeHTML(sop.estimated_downtime)}</div>`;
        if (sop.tools_required) {
            const n = sop.tools_required.split(',').length;
            metaHTML += `<div class="sop-meta-item">${ICONS.tools} ${n} tool${n > 1 ? 's' : ''}</div>`;
        }
        card.innerHTML = `
            <div class="sop-card-header">
                <div class="sop-card-title">${escapeHTML(sop.title)}</div>
                <span class="sop-severity-badge ${sev}">${sev}</span>
            </div>
            <div class="sop-card-tags">
                <span class="sop-component-tag"><span class="tag-dot" style="background:#6C63FF"></span>${(sop.component || '').replace(/_/g, ' ')}</span>
            </div>
            <div class="sop-card-content">${escapeHTML(sop.content_chunk)}</div>
            ${metaHTML ? `<div class="sop-card-meta">${metaHTML}</div>` : ''}
            ${simHTML}
        `;
        card.addEventListener('click', () => openSOPDetail(sop));
        grid.appendChild(card);
    });
}

function openSOPDetail(sop) {
    const panel = document.getElementById('sopDetailCard');
    const content = document.getElementById('sopDetailContent');
    const sev = sop.severity || 'warning';
    let txt = sop.content_chunk.replace(/Step (\d+):/g, '<br><strong>Step $1:</strong>');
    if (txt.startsWith('<br>')) txt = txt.substring(4);
    let tools = '';
    if (sop.tools_required) {
        const list = sop.tools_required.split(',').map(t => t.trim());
        tools = `<div class="detail-section"><div class="detail-section-title">${ICONS.tools} Tools Required</div><div class="detail-section-value">${list.map(t => `• ${escapeHTML(t)}`).join('<br>')}</div></div>`;
    }
    let down = '';
    if (sop.estimated_downtime) down = `<div class="detail-section"><div class="detail-section-title">${ICONS.clock} Estimated Downtime</div><div class="detail-section-value">${escapeHTML(sop.estimated_downtime)}</div></div>`;
    content.innerHTML = `
        <h4>${escapeHTML(sop.title)}</h4>
        <span class="sop-severity-badge ${sev} detail-severity">${sev.toUpperCase()}</span>
        <div class="detail-steps">${txt}</div>
        ${tools}${down}
    `;
    panel.style.display = 'block';
    panel.style.animation = 'none';
    panel.offsetHeight;
    panel.style.animation = 'slideInRight 0.3s ease';
}

async function loadAlertTimeline() {
    const timeline = document.getElementById('alertTimeline');
    const count = document.getElementById('timelineAlertCount');
    try {
        const data = await fetchJSON(`${settings.backendUrl}/maintenance/summary?truck_id=${settings.truckId}`);
        const alerts = data.recent_alerts || [];
        count.textContent = alerts.length;
        if (alerts.length) count.className = 'badge warning';
        if (!alerts.length) { timeline.innerHTML = '<div class="empty-state">No recent alerts with SOP references.</div>'; return; }
        timeline.innerHTML = '';
        alerts.slice(0, 15).forEach((a, i) => {
            const d = new Date(a.created_at);
            const ts = d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const item = document.createElement('div');
            item.className = `timeline-item severity-${a.severity || 'medium'}`;
            item.style.animationDelay = `${i * 0.08}s`;
            item.innerHTML = `
                <div class="timeline-time">${ts}</div>
                <div class="timeline-message">${escapeHTML(a.message || 'Alert triggered')}</div>
                ${a.sop_reference ? `<span class="timeline-sop-link" data-sop="${escapeHTML(a.sop_reference)}">${ICONS.doc} ${escapeHTML(a.sop_reference)}</span>` : ''}
            `;
            const link = item.querySelector('.timeline-sop-link');
            if (link) link.addEventListener('click', (e) => {
                e.stopPropagation();
                const matched = allSOPs.find(s => s.title === link.dataset.sop || link.dataset.sop.includes(s.title.split(':')[0]));
                if (matched) openSOPDetail(matched);
                else { document.getElementById('sopSearchInput').value = link.dataset.sop; searchSOPs(link.dataset.sop); }
            });
            timeline.appendChild(item);
        });
    } catch {
        timeline.innerHTML = '<div class="empty-state">Unable to load alert timeline.</div>';
    }
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER · SETTINGS
   ═══════════════════════════════════════════════════════════════════ */
function bindSettings() {
    document.getElementById('setBackendUrl').value = settings.backendUrl;
    document.getElementById('setPollInterval').value = settings.pollInterval;
    document.getElementById('setTruckId').value = settings.truckId;
    document.getElementById('setAlertThreshold').value = settings.alertThreshold;
    document.getElementById('setAlertThresholdVal').textContent = (+settings.alertThreshold).toFixed(2);
    document.getElementById('setDemoMode').checked = settings.demoMode;
    document.getElementById('setSoundAlerts').checked = settings.soundAlerts;
    document.getElementById('setHourlyRevenue').value = settings.hourlyRevenue;
    document.getElementById('setPlatformCost').value = settings.platformCost;
    document.getElementById('setDowntimeCost').value = settings.downtimeCost;
    document.getElementById('setFuelCost').value = settings.fuelCost;
    document.getElementById('infoTruckId').textContent = settings.truckId;

    const thr = document.getElementById('setAlertThreshold');
    thr.addEventListener('input', () => document.getElementById('setAlertThresholdVal').textContent = (+thr.value).toFixed(2));

    document.getElementById('saveSettings').addEventListener('click', () => {
        settings.backendUrl = document.getElementById('setBackendUrl').value || DEFAULT_BACKEND;
        settings.pollInterval = Math.max(1, +document.getElementById('setPollInterval').value || 2);
        settings.truckId = document.getElementById('setTruckId').value || 'truck1';
        settings.alertThreshold = +document.getElementById('setAlertThreshold').value;
        settings.demoMode = document.getElementById('setDemoMode').checked;
        settings.soundAlerts = document.getElementById('setSoundAlerts').checked;
        settings.hourlyRevenue = +document.getElementById('setHourlyRevenue').value;
        settings.platformCost = +document.getElementById('setPlatformCost').value;
        settings.downtimeCost = +document.getElementById('setDowntimeCost').value;
        settings.fuelCost = +document.getElementById('setFuelCost').value;
        saveSettings(settings);
        const r = document.getElementById('testResult');
        r.textContent = '✓ Saved';
        r.className = 'test-result ok';
        setTimeout(() => r.textContent = '', 2000);
    });

    document.getElementById('testBackendBtn').addEventListener('click', async () => {
        const url = document.getElementById('setBackendUrl').value || DEFAULT_BACKEND;
        const r = document.getElementById('testResult');
        r.textContent = 'Testing...';
        r.className = 'test-result';
        const res = await safeFetch(url + '/');
        if (res.ok) {
            r.textContent = '✓ Connected: ' + (res.data.service || 'EdgeGuard backend');
            r.className = 'test-result ok';
        } else {
            r.textContent = '✗ Failed: ' + (res.error || 'unknown');
            r.className = 'test-result fail';
        }
    });

    document.getElementById('resetSettings').addEventListener('click', () => {
        if (!confirm('Reset all settings to defaults?')) return;
        settings = { ...DEFAULT_SETTINGS };
        saveSettings(settings);
        bindSettings();
        document.getElementById('testResult').textContent = '✓ Reset to defaults';
        document.getElementById('testResult').className = 'test-result ok';
    });
}

/* ═══════════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════════ */
function switchTab(targetId) {
    document.querySelectorAll('#sidebarNav li').forEach(n => n.classList.toggle('active', n.dataset.target === targetId));
    document.querySelectorAll('.tab-view').forEach(v => {
        v.classList.remove('active');
        v.style.display = 'none';
    });
    const v = document.getElementById(targetId);
    if (v) { v.style.display = 'block'; v.classList.add('active'); }
    if (targetId === 'view-maintenance' && !maintenanceLoaded) {
        maintenanceLoaded = true;
        loadAllSOPs();
    }
    // Re-render views whose charts/DOM depend on layout dimensions; the canvas
    // would have been drawn at 0×0 when the tab was hidden on initial load.
    if (targetId === 'view-business') renderBusiness();
    if (targetId === 'view-fleet')   renderFleet();
    if (targetId === 'view-twin')    renderTwin();
}

function bindTabs() {
    document.querySelectorAll('#sidebarNav li').forEach(li => {
        li.addEventListener('click', () => switchTab(li.dataset.target));
    });
    document.getElementById('truckSelector').addEventListener('change', e => {
        settings.truckId = e.target.value;
        saveSettings(settings);
        document.getElementById('infoTruckId').textContent = settings.truckId;
    });
}

/* ═══════════════════════════════════════════════════════════════════
   DRAWER (sensor detail)
   ═══════════════════════════════════════════════════════════════════ */
function openDrawer(sensor) {
    if (!SENSOR_THRESHOLDS[sensor]) return;
    const sMeta = SENSORS.find(s => s.id === sensor);
    document.getElementById('drawerTitle').innerHTML = `${sMeta ? sMeta.label : sensor}<span class="sub">${sMeta ? sMeta.unit : ''} · click outside to close</span>`;
    const arr = state.history[sensor];
    const cur = arr.length ? arr[arr.length - 1].v : null;
    const status = cur == null ? 'healthy' : classify(sensor, cur);
    document.getElementById('drawerCurrent').innerHTML =
        `<strong>Current:</strong> ${cur == null ? '—' : cur.toFixed(2)} ${sMeta ? sMeta.unit : ''}<br>
         <strong>Status:</strong> <span style="color: ${status === 'healthy' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444'}">${status.toUpperCase()}</span><br>
         <strong>Samples:</strong> ${arr.length} of last ${state.historyMax}`;
    const t = SENSOR_THRESHOLDS[sensor];
    const fmt = r => (r[0] == null ? '−∞' : r[0]) + ' – ' + (r[1] == null ? '+∞' : r[1]);
    document.getElementById('thresholds').innerHTML = `
        <div class="cell healthy">OK: ${fmt(t.healthy)}</div>
        <div class="cell warning">WARN: ${fmt(t.warning)}</div>
        <div class="cell critical">CRIT: ${fmt(t.critical)}</div>`;
    document.getElementById('sopText').innerHTML = SENSOR_SOP[sensor] || '—';
    document.getElementById('drawer').classList.add('show');
    document.getElementById('drawerMask').classList.add('show');
    setTimeout(() => drawHistory(sensor, status), 60);
}
function closeDrawer() {
    document.getElementById('drawer').classList.remove('show');
    document.getElementById('drawerMask').classList.remove('show');
}
function drawHistory(sensor, status) {
    const c = document.getElementById('historyChart');
    if (!c) return;
    const ctx = c.getContext('2d');
    const rect = c.getBoundingClientRect();
    c.width = rect.width * 2; c.height = rect.height * 2;
    ctx.scale(2, 2);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    const data = state.history[sensor].slice();
    if (data.length < 2) {
        ctx.fillStyle = '#8A8FA8';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Collecting data…', w / 2, h / 2);
        return;
    }
    let lo = Infinity, hi = -Infinity;
    data.forEach(p => { if (p.v < lo) lo = p.v; if (p.v > hi) hi = p.v; });
    const pad = (hi - lo) * 0.15 || 0.5;
    lo -= pad; hi += pad;
    const xStep = (w - 40) / (data.length - 1);
    const color = status === 'healthy' ? '#22c55e' : status === 'warning' ? '#f59e0b' : '#ef4444';
    const yFor = v => h - 10 - ((v - lo) / (hi - lo)) * (h - 30);
    const t = SENSOR_THRESHOLDS[sensor];
    if (t.warning[0] != null && t.warning[1] != null) {
        const y1 = yFor(t.warning[1]), y2 = yFor(t.warning[0]);
        ctx.fillStyle = 'rgba(245,158,11,0.07)';
        ctx.fillRect(30, Math.min(y1, y2), w - 40, Math.abs(y2 - y1));
    }
    if (t.critical[0] != null && t.critical[1] != null) {
        const y1 = yFor(t.critical[1]), y2 = yFor(t.critical[0]);
        ctx.fillStyle = 'rgba(239,68,68,0.10)';
        ctx.fillRect(30, Math.min(y1, y2), w - 40, Math.abs(y2 - y1));
    }
    ctx.beginPath();
    data.forEach((p, i) => { const x = 30 + i * xStep, y = yFor(p.v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.lineTo(30 + (data.length - 1) * xStep, h - 10); ctx.lineTo(30, h - 10); ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color + '55'); grad.addColorStop(1, color + '00');
    ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath();
    data.forEach((p, i) => { const x = 30 + i * xStep, y = yFor(p.v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
    const last = data[data.length - 1];
    const lx = 30 + (data.length - 1) * xStep, ly = yFor(last.v);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#0F1117'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#8A8FA8'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(hi.toFixed(1), 26, 14); ctx.fillText(lo.toFixed(1), 26, h - 4);
    ctx.textAlign = 'left'; ctx.fillText('now', lx - 18, h - 1);
}

function bindDrawer() {
    document.getElementById('drawerClose').addEventListener('click', closeDrawer);
    document.getElementById('drawerMask').addEventListener('click', closeDrawer);
    // Twin nodes
    document.querySelectorAll('.node').forEach(n => {
        n.addEventListener('click', () => openDrawer(n.dataset.sensor));
    });
}

/* ═══════════════════════════════════════════════════════════════════
   MAINTENANCE TAB
   ═══════════════════════════════════════════════════════════════════ */
function bindMaintenance() {
    const input = document.getElementById('sopSearchInput');
    input.addEventListener('input', e => {
        clearTimeout(searchDebounceTimer);
        const q = e.target.value;
        if (!q.trim()) { isSearchActive = false; document.getElementById('searchResultCount').style.display = 'none'; document.getElementById('sopSearchClear').style.display = 'none'; applyFilters(); return; }
        searchDebounceTimer = setTimeout(() => searchSOPs(q), 400);
    });
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { clearTimeout(searchDebounceTimer); searchSOPs(e.target.value); } });
    document.getElementById('sopSearchClear').addEventListener('click', () => {
        input.value = ''; isSearchActive = false;
        document.getElementById('searchResultCount').style.display = 'none';
        document.getElementById('sopSearchClear').style.display = 'none';
        applyFilters();
    });
    document.getElementById('closeSopDetail').addEventListener('click', () => {
        document.getElementById('sopDetailCard').style.display = 'none';
    });
}

/* ═══════════════════════════════════════════════════════════════════
   3D DIGITAL TWIN (Three.js) — procedural tipper truck, sensor nodes,
   cascade lines, tipper bed raise, exhaust particles. ~600 LOC.
   Falls back to the 2D SVG if WebGL is unavailable.
   ═══════════════════════════════════════════════════════════════════ */
const three3D = {
    active: false,
    THREE: null,
    scene: null,
    camera: null,
    renderer: null,
    truckGroup: null,
    bedGroup: null,
    bedRest: 0,
    bedTarget: 0,
    nodeMeshes: {},        // sensor → {halo, core, label}
    cascadeLines: {},
    exhaustParticles: [],
    exhaustEnabled: false,
    rafId: 0,
    init: function () {
        if (this.active) return true;
        if (typeof window.THREE === 'undefined') return false;       // loaded by module import
        if (!window.WebGLRenderingContext) return false;
        const c = document.createElement('canvas');
        const gl = c.getContext('webgl2') || c.getContext('webgl');
        if (!gl) return false;
        this.THREE = window.THREE;
        this._buildScene();
        this.active = true;
        return true;
    },
    _buildScene: function () {
        const T = this.THREE;
        const wrap = document.getElementById('twinWrap');
        const canvas = document.getElementById('twinCanvas3D');
        const w = wrap.clientWidth, h = wrap.clientHeight;
        canvas.width = w; canvas.height = h;

        this.scene = new T.Scene();
        this.scene.background = new T.Color(0x0a0e1a);
        this.scene.fog = new T.Fog(0x0a0e1a, 12, 30);

        this.camera = new T.PerspectiveCamera(45, w / h, 0.1, 100);
        this.camera.position.set(7, 4.5, 9);
        this.camera.lookAt(0, 0.6, 0);

        this.renderer = new T.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(w, h, false);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = T.PCFSoftShadowMap;

        // Lighting
        const ambient = new T.AmbientLight(0x9bb0d6, 0.45);
        this.scene.add(ambient);
        const dir = new T.DirectionalLight(0xffffff, 1.05);
        dir.position.set(8, 12, 6);
        dir.castShadow = true;
        dir.shadow.mapSize.width = 1024; dir.shadow.mapSize.height = 1024;
        dir.shadow.camera.left = -10; dir.shadow.camera.right = 10;
        dir.shadow.camera.top = 10; dir.shadow.camera.bottom = -10;
        this.scene.add(dir);
        const rim = new T.DirectionalLight(0x3b82f6, 0.55);
        rim.position.set(-6, 5, -4);
        this.scene.add(rim);

        // Ground
        const g = new T.PlaneGeometry(40, 40);
        const gm = new T.MeshStandardMaterial({ color: 0x0b1224, roughness: 0.95, metalness: 0.05 });
        const ground = new T.Mesh(g, gm);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid
        const grid = new T.GridHelper(40, 40, 0x1e293b, 0x111c2e);
        this.scene.add(grid);

        this.truckGroup = new T.Group();
        this.scene.add(this.truckGroup);
        this._buildTruck();

        // Orbit-style camera control (manual: simple mouse drag)
        this._bindControls(canvas);

        window.addEventListener('resize', () => this._onResize());
    },
    _buildTruck: function () {
        const T = this.THREE;
        const M = (c, opts = {}) => new T.MeshStandardMaterial({
            color: c, roughness: opts.r ?? 0.6, metalness: opts.m ?? 0.3,
            emissive: opts.e ?? 0x000000, emissiveIntensity: opts.ei ?? 0.0,
        });
        const make = (geo, mat) => {
            const m = new T.Mesh(geo, mat);
            m.castShadow = true; m.receiveShadow = true;
            return m;
        };

        // Chassis (long bar)
        const chassisGeo = new T.BoxGeometry(7, 0.35, 2.4);
        const chassis = make(chassisGeo, M(0x2f3a52, { r: 0.5, m: 0.7 }));
        chassis.position.set(0, 0.95, 0);
        this.truckGroup.add(chassis);

        // Cab
        const cab = make(new T.BoxGeometry(1.9, 1.6, 2.3), M(0x2563eb, { r: 0.45, m: 0.5 }));
        cab.position.set(-2.3, 1.9, 0);
        this.truckGroup.add(cab);

        // Cab roof
        const cabRoof = make(new T.BoxGeometry(1.95, 0.18, 2.35), M(0x1d4ed8, { r: 0.5, m: 0.6 }));
        cabRoof.position.set(-2.3, 2.8, 0);
        this.truckGroup.add(cabRoof);

        // Windshield
        const wsGeo = new T.PlaneGeometry(1.5, 1.0);
        const wsMat = new T.MeshStandardMaterial({ color: 0x60a5fa, emissive: 0x0ea5e9, emissiveIntensity: 0.25, roughness: 0.15, metalness: 0.8, transparent: true, opacity: 0.85 });
        const ws = new T.Mesh(wsGeo, wsMat);
        ws.position.set(-1.32, 2.15, 0);
        ws.rotation.y = Math.PI / 2;
        this.truckGroup.add(ws);

        // Headlights
        const hlMat = new T.MeshStandardMaterial({ color: 0xfff7c2, emissive: 0xfff7c2, emissiveIntensity: 1.2 });
        for (const z of [-0.85, 0.85]) {
            const hl = make(new T.SphereGeometry(0.13, 16, 12), hlMat);
            hl.position.set(-3.28, 1.55, z);
            this.truckGroup.add(hl);
        }

        // Tipper bed (group, will rotate)
        this.bedGroup = new T.Group();
        this.bedGroup.position.set(0.6, 1.1, 0);
        this.bedGroup.rotation.z = 0;
        const bedMat = M(0x4b5b75, { r: 0.55, m: 0.6 });
        const bed = make(new T.BoxGeometry(3.6, 1.1, 2.2), bedMat);
        bed.position.set(1.0, 0.55, 0);
        this.bedGroup.add(bed);
        // Bed front wall (taller)
        const frontWall = make(new T.BoxGeometry(0.12, 1.3, 2.2), M(0x3b475e));
        frontWall.position.set(2.78, 0.65, 0);
        this.bedGroup.add(frontWall);
        // Side wall ribs
        for (const z of [-1.05, 1.05]) {
            for (let i = 0; i < 4; i++) {
                const rib = make(new T.BoxGeometry(0.05, 1.0, 0.05), M(0x1e293b));
                rib.position.set(-0.4 + i * 0.9, 0.55, z);
                this.bedGroup.add(rib);
            }
        }
        this.truckGroup.add(this.bedGroup);

        // Hydraulic ram
        const ramMat = M(0x94a3b8, { r: 0.3, m: 0.9 });
        const ram = make(new T.CylinderGeometry(0.09, 0.09, 1.4, 16), ramMat);
        ram.position.set(0.2, 0.55, 0);
        ram.rotation.z = Math.PI / 2.2;
        this.truckGroup.add(ram);
        const ramRod = make(new T.CylinderGeometry(0.05, 0.05, 0.9, 16), M(0xe2e8f0, { r: 0.2, m: 1.0 }));
        ramRod.position.set(0.9, 0.95, 0);
        ramRod.rotation.z = Math.PI / 2.2;
        this.truckGroup.add(ramRod);

        // Wheels
        const wheelGeo = new T.CylinderGeometry(0.55, 0.55, 0.4, 20);
        const wheelMat = M(0x0f172a, { r: 0.85, m: 0.15 });
        const hubMat = M(0x475569, { r: 0.5, m: 0.85 });
        const wheelPositions = [
            [-1.8, 0.55, 1.18], [-1.8, 0.55, -1.18],
            [0.6,  0.55, 1.18], [0.6,  0.55, -1.18],
            [2.4,  0.55, 1.18], [2.4,  0.55, -1.18],
        ];
        for (const [x, y, z] of wheelPositions) {
            const w = make(wheelGeo, wheelMat);
            w.position.set(x, y, z);
            w.rotation.x = Math.PI / 2;
            this.truckGroup.add(w);
            const hub = make(new T.CylinderGeometry(0.16, 0.16, 0.45, 12), hubMat);
            hub.position.set(x, y, z);
            hub.rotation.x = Math.PI / 2;
            this.truckGroup.add(hub);
        }

        // Exhaust stack
        const stack = make(new T.CylinderGeometry(0.08, 0.1, 1.6, 12), M(0x1e293b, { r: 0.6, m: 0.7 }));
        stack.position.set(-0.7, 3.05, 1.1);
        this.truckGroup.add(stack);
        const stackTop = make(new T.CylinderGeometry(0.12, 0.12, 0.1, 12), M(0x0f172a));
        stackTop.position.set(-0.7, 3.85, 1.1);
        this.truckGroup.add(stackTop);

        // Sensor nodes (positions are local to truckGroup)
        const nodeDefs = [
            { id: 'temperature',         pos: [-2.0, 2.1, 1.18],  color: 0x22c55e, size: 0.18 },
            { id: 'vibration',           pos: [-1.8, 0.55, 1.4], color: 0x22c55e, size: 0.18 },
            { id: 'oil_pressure',        pos: [-1.0, 1.2, 1.2],   color: 0x22c55e, size: 0.16 },
            { id: 'hydraulic_pressure',  pos: [0.7, 0.7, 1.25],   color: 0x22c55e, size: 0.18 },
            { id: 'suspension_pressure', pos: [1.8, 0.55, 1.4],   color: 0x22c55e, size: 0.16 },
            { id: 'battery_voltage',     pos: [-2.3, 1.6, 1.2],   color: 0x22c55e, size: 0.14 },
        ];
        for (const n of nodeDefs) {
            const halo = make(new T.SphereGeometry(n.size * 1.7, 16, 12), new T.MeshStandardMaterial({
                color: n.color, emissive: n.color, emissiveIntensity: 0.6, transparent: true, opacity: 0.35,
            }));
            halo.position.set(...n.pos);
            this.truckGroup.add(halo);
            const core = make(new T.SphereGeometry(n.size, 16, 12), new T.MeshStandardMaterial({
                color: n.color, emissive: n.color, emissiveIntensity: 1.1,
            }));
            core.position.set(...n.pos);
            this.truckGroup.add(core);
            this.nodeMeshes[n.id] = { halo, core, basePos: n.pos, color: n.color };
        }

        // Cascade lines (engine → bearing → oil → hydraulic)
        const lineMat = new T.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.0 });
        const pairs = [
            ['temperature', 'vibration'],
            ['vibration', 'oil_pressure'],
            ['oil_pressure', 'hydraulic_pressure'],
        ];
        for (const [a, b] of pairs) {
            const pa = this.nodeMeshes[a].basePos;
            const pb = this.nodeMeshes[b].basePos;
            const geo = new T.BufferGeometry().setFromPoints([
                new T.Vector3(...pa), new T.Vector3(...pb),
            ]);
            const line = new T.Line(geo, lineMat.clone());
            this.truckGroup.add(line);
            this.cascadeLines[`${a}_${b}`] = line;
        }
    },
    _bindControls: function (canvas) {
        let dragging = false, lastX = 0, lastY = 0;
        let theta = Math.atan2(9, 7), phi = Math.atan2(4.5, Math.hypot(7, 9));
        let radius = Math.hypot(7, 4.5, 9);
        const onDown = (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; };
        const onUp = () => { dragging = false; };
        const onMove = (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            theta -= dx * 0.005;
            phi = Math.max(0.15, Math.min(Math.PI / 2 - 0.05, phi - dy * 0.005));
            this._updateCamera(theta, phi, radius);
        };
        const onWheel = (e) => {
            e.preventDefault();
            radius = Math.max(6, Math.min(18, radius + e.deltaY * 0.01));
            this._updateCamera(theta, phi, radius);
        };
        canvas.addEventListener('mousedown', onDown);
        window.addEventListener('mouseup', onUp);
        window.addEventListener('mousemove', onMove);
        canvas.addEventListener('wheel', onWheel, { passive: false });
        // touch
        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) onDown(e.touches[0]);
        });
        canvas.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) onMove(e.touches[0]);
        });
        canvas.addEventListener('touchend', onUp);
    },
    _updateCamera: function (theta, phi, radius) {
        const x = radius * Math.cos(phi) * Math.cos(theta);
        const z = radius * Math.cos(phi) * Math.sin(theta);
        const y = radius * Math.sin(phi);
        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 0.7, 0);
    },
    _onResize: function () {
        if (!this.active) return;
        const wrap = document.getElementById('twinWrap');
        const w = wrap.clientWidth, h = wrap.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h, false);
    },
    setSensors: function (sensorValues, prediction) {
        if (!this.active) return;
        const T = this.THREE;
        const prob = (prediction && prediction.failure_probability) || 0;
        const stress = Math.min(1, prob / 0.9);

        // Color the nodes by stress of their sensor (simplified: weight by prob)
        const getStress = (id) => {
            const v = sensorValues[id];
            if (v == null) return 0;
            const th = SENSOR_THRESHOLDS[id] || {};
            if (th.danger && th.danger[1] != null) {
                return Math.max(0, Math.min(1, (v - th.danger[0]) / (th.danger[1] - th.danger[0])));
            }
            return 0;
        };

        for (const [id, m] of Object.entries(this.nodeMeshes)) {
            const s = Math.max(getStress(id), stress * 0.3);
            const color = s > 0.7 ? 0xef4444 : s > 0.4 ? 0xf59e0b : 0x22c55e;
            m.core.material.color.setHex(color);
            m.core.material.emissive.setHex(color);
            m.core.material.emissiveIntensity = 0.8 + s * 1.2;
            m.halo.material.color.setHex(color);
            m.halo.material.emissive.setHex(color);
            const pulse = 1 + Math.sin(performance.now() * 0.005 * (1 + s * 3)) * 0.18 * s;
            m.core.scale.setScalar(pulse);
        }

        // Cascade lines: opacity grows with stress
        for (const [k, line] of Object.entries(this.cascadeLines)) {
            const [a, b] = k.split('_');
            const sa = getStress(a), sb = getStress(b);
            const s = Math.max(sa, sb, stress * 0.5);
            line.material.opacity = Math.min(0.9, s * 0.9);
            line.material.color.setHex(s > 0.5 ? 0xef4444 : s > 0.25 ? 0xf59e0b : 0x22c55e);
        }

        // Tipper bed raise angle (more stress → more raise)
        this.bedTarget = stress > 0.5 ? Math.min(0.42, stress * 0.55) : 0;
        this.bedRest += (this.bedTarget - this.bedRest) * 0.05;
        this.bedGroup.rotation.z = -this.bedRest;

        // Exhaust: enabled when temperature stress is high
        const tempStress = getStress('temperature');
        this.exhaustEnabled = tempStress > 0.4;
    },
    _spawnExhaust: function () {
        if (!this.exhaustEnabled) {
            // Allow particles to fade out
            for (const p of this.exhaustParticles) {
                p.material.opacity = Math.max(0, p.material.opacity - 0.04);
            }
            return;
        }
        const T = this.THREE;
        // Spawn one every few frames
        if (Math.random() < 0.3) {
            const mat = new T.MeshBasicMaterial({ color: 0x6b7280, transparent: true, opacity: 0.7 });
            const m = new T.Mesh(new T.SphereGeometry(0.08 + Math.random() * 0.06, 8, 6), mat);
            m.position.set(-0.7 + (Math.random() - 0.5) * 0.05, 3.95, 1.1 + (Math.random() - 0.5) * 0.05);
            this.scene.add(m);
            this.exhaustParticles.push(m);
        }
        // Update existing
        for (let i = this.exhaustParticles.length - 1; i >= 0; i--) {
            const p = this.exhaustParticles[i];
            p.position.y += 0.04;
            p.position.x += (Math.random() - 0.5) * 0.02;
            p.position.z += (Math.random() - 0.5) * 0.02;
            p.material.opacity -= 0.012;
            p.scale.multiplyScalar(1.012);
            if (p.material.opacity <= 0) {
                this.scene.remove(p);
                p.geometry.dispose();
                p.material.dispose();
                this.exhaustParticles.splice(i, 1);
            }
        }
    },
    loop: function () {
        if (!this.active) return;
        this._spawnExhaust();
        this.renderer.render(this.scene, this.camera);
        this.rafId = requestAnimationFrame(() => this.loop());
    },
    start: function () {
        if (this.active) {
            this.loop();
            return;
        }
        if (!this.init()) {
            console.warn('[3D] init failed; staying on 2D');
            return false;
        }
        // Make canvas visible
        document.getElementById('twinSvg').style.display = 'none';
        const c = document.getElementById('twinCanvas3D');
        c.style.display = 'block';
        this.loop();
        return true;
    },
    stop: function () {
        cancelAnimationFrame(this.rafId);
        const c = document.getElementById('twinCanvas3D');
        if (c) c.style.display = 'none';
        const s = document.getElementById('twinSvg');
        if (s) s.style.display = 'block';
    },
    destroy: function () {
        this.stop();
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer.forceContextLoss();
        }
        this.active = false;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.truckGroup = null;
        this.nodeMeshes = {};
        this.cascadeLines = {};
        this.exhaustParticles = [];
    },
};

// Load Three.js as ES module then expose on window
(async () => {
    try {
        const mod = await import('./vendor-three.module.min.js');
        window.THREE = mod;
        window.dispatchEvent(new Event('three-ready'));
    } catch (e) {
        console.warn('[3D] Three.js module failed to load; 2D fallback only', e);
    }
})();

/* ═══════════════════════════════════════════════════════════════════
   WEBSOCKET STREAM — replaces HTTP polling with sub-100ms push.
   Falls back to polling if WS fails.
   ═══════════════════════════════════════════════════════════════════ */
const wsStream = {
    socket: null,
    connected: false,
    retryTimer: 0,
    listeners: new Set(),
    connect: function () {
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) return;
        const url = settings.backendUrl.replace(/^http/, 'ws') + '/ws/stream';
        try {
            this.socket = new WebSocket(url);
        } catch (e) { return; }
        this.socket.onopen = () => {
            this.connected = true;
            setConnection(true);
            console.info('[ws] connected');
        };
        this.socket.onclose = () => {
            this.connected = false;
            this.retryTimer = setTimeout(() => this.connect(), 3000);
        };
        this.socket.onerror = () => { /* close will fire */ };
        this.socket.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                this.listeners.forEach((fn) => fn(msg));
            } catch (e) { /* ignore */ }
        };
    },
    send: function (msg) {
        if (this.connected && this.socket) this.socket.send(JSON.stringify(msg));
    },
    on: function (fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
};

/* ═══════════════════════════════════════════════════════════════════
   VOICE ALERTS — Web Speech API TTS for critical alerts
   ═══════════════════════════════════════════════════════════════════ */
const voiceAlerts = {
    enabled: false,
    spokenIds: new Set(),
    synth: window.speechSynthesis,
    init: function () {
        if (!this.synth) return;
        this.enabled = !!(localStorage.getItem('edgeguard_voice_alerts') === 'on');
        this._bindUI();
    },
    _bindUI: function () {
        const t = document.getElementById('voiceToggle');
        if (!t) return;
        t.checked = this.enabled;
        t.addEventListener('change', () => {
            this.enabled = t.checked;
            localStorage.setItem('edgeguard_voice_alerts', this.enabled ? 'on' : 'off');
        });
    },
    speak: function (text) {
        if (!this.enabled || !this.synth) return;
        try {
            this.synth.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 1.05; u.pitch = 1.0; u.volume = 0.9;
            this.synth.speak(u);
        } catch (e) { /* ignore */ }
    },
    onAlert: function (alert) {
        // Speak critical alerts (once per alert id)
        if (!alert || alert.severity !== 'critical') return;
        if (this.spokenIds.has(alert.id)) return;
        this.spokenIds.add(alert.id);
        // Keep set bounded
        if (this.spokenIds.size > 100) {
            const arr = Array.from(this.spokenIds);
            this.spokenIds = new Set(arr.slice(-50));
        }
        const text = `Critical alert on ${alert.truck_id || 'vehicle'}. ${alert.message || 'Immediate attention required.'}`;
        this.speak(text);
    },
};

/* ═══════════════════════════════════════════════════════════════════
   GEOFENCE MAP — Leaflet with pit outline + truck markers
   ═══════════════════════════════════════════════════════════════════ */
const geoMap = {
    map: null,
    markers: {},
    pits: [],
    init: function () {
        if (this.map) return;
        if (typeof L === 'undefined') return;          // Leaflet not loaded yet
        const wrap = document.getElementById('geoMap');
        if (!wrap) return;
        // Center on a notional open-pit mine (Jharia coalfield approx)
        this.map = L.map('geoMap', { zoomControl: true, attributionControl: false })
            .setView([23.78, 86.43], 14);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(this.map);

        // Define three pit polygons (simulated)
        const pitACoords = [
            [23.795, 86.420], [23.795, 86.435], [23.785, 86.435], [23.785, 86.420],
        ];
        const pitBCoords = [
            [23.785, 86.435], [23.785, 86.450], [23.775, 86.450], [23.775, 86.435],
        ];
        const pitCCoords = [
            [23.775, 86.420], [23.775, 86.435], [23.765, 86.435], [23.765, 86.420],
        ];
        L.polygon(pitACoords, { color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.08, weight: 1.5 })
            .addTo(this.map).bindTooltip('Pit A — Active Loading', { permanent: false });
        L.polygon(pitBCoords, { color: '#22c55e', fillColor: '#22c55e', fillOpacity: 0.08, weight: 1.5 })
            .addTo(this.map).bindTooltip('Pit B — Haul Route', { permanent: false });
        L.polygon(pitCCoords, { color: '#f59e0b', fillColor: '#f59e0b', fillOpacity: 0.08, weight: 1.5 })
            .addTo(this.map).bindTooltip('Pit C — Maintenance Bay', { permanent: false });
        this.pits = [
            { name: 'Pit A', center: [23.790, 86.428] },
            { name: 'Pit B', center: [23.780, 86.443] },
            { name: 'Pit C', center: [23.770, 86.428] },
        ];

        // Truck markers
        const truckData = [
            { id: 'truck1', lat: 23.791, lng: 86.426, status: 'critical' },
            { id: 'truck2', lat: 23.781, lng: 86.441, status: 'warning'  },
            { id: 'truck3', lat: 23.771, lng: 86.429, status: 'healthy'  },
        ];
        for (const t of truckData) {
            const color = t.status === 'critical' ? '#ef4444'
                : t.status === 'warning' ? '#f59e0b' : '#22c55e';
            const icon = L.divIcon({
                html: `<div style="background:${color};width:28px;height:28px;border-radius:50%;border:3px solid white;box-shadow:0 0 12px ${color};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#0f172a;">${t.id.slice(-1).toUpperCase()}</div>`,
                className: '', iconSize: [28, 28], iconAnchor: [14, 14],
            });
            const m = L.marker([t.lat, t.lng], { icon }).addTo(this.map);
            m.bindPopup(`<b>${t.id.toUpperCase()}</b><br/>Status: <span style="color:${color};font-weight:700">${t.status.toUpperCase()}</span>`);
            this.markers[t.id] = m;
        }
    },
    setTruckStatus: function (truckId, status) {
        const m = this.markers[truckId];
        if (!m) return;
        const color = status === 'critical' ? '#ef4444'
            : status === 'warning' ? '#f59e0b' : '#22c55e';
        const icon = L.divIcon({
            html: `<div style="background:${color};width:28px;height:28px;border-radius:50%;border:3px solid white;box-shadow:0 0 12px ${color};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#0f172a;">${truckId.slice(-1).toUpperCase()}</div>`,
            className: '', iconSize: [28, 28], iconAnchor: [14, 14],
        });
        m.setIcon(icon);
    },
};

// Load Leaflet CSS + JS lazily when the geofence tab is first opened
function loadLeaflet() {
    if (typeof L !== 'undefined') return Promise.resolve();
    return new Promise((resolve) => {
        if (!document.getElementById('leaflet-css')) {
            const link = document.createElement('link');
            link.id = 'leaflet-css';
            link.rel = 'stylesheet';
            link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
            document.head.appendChild(link);
        }
        if (document.getElementById('leaflet-js')) {
            document.getElementById('leaflet-js').addEventListener('load', resolve);
            return;
        }
        const s = document.createElement('script');
        s.id = 'leaflet-js';
        s.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        s.onload = () => resolve();
        s.onerror = () => resolve();
        document.head.appendChild(s);
    });
}

/* ═══════════════════════════════════════════════════════════════════
   BOOT
   ═══════════════════════════════════════════════════════════════════ */
function renderAll() {
    renderCommandCenter();
    // Mirror sensor values to the 3D twin
    if (three3D.active) {
        const v = {};
        for (const s of SENSORS) v[s] = state.latest[sensorKey(s)];
        three3D.setSensors(v, state.latestPrediction);
    }
    // Geofence markers
    if (geoMap.map && state.latestPrediction) {
        const p = state.latestPrediction.failure_probability || 0;
        const status = p >= 0.7 ? 'critical' : p >= 0.4 ? 'warning' : 'healthy';
        geoMap.setTruckStatus('truck1', status);
    }
}

async function boot() {
    bindTabs();
    bindSettings();
    bindDrawer();
    bindMaintenance();
    buildTwinParticles();
    initCharts();
    voiceAlerts.init();

    // Seed prediction history so the trend chart renders a full line immediately
    if (state.predHistory.length < 50) {
        seedPredHistory(50, state.prediction.failure_probability || 0.05);
    }

    // 3D toggle binding
    document.getElementById('twin3dToggle')?.addEventListener('click', () => {
        if (three3D.active) {
            three3D.stop();
            document.getElementById('twin3dToggle').classList.remove('active');
            const t = document.getElementById('twin3dPref'); if (t) t.checked = false;
        } else {
            const ok = three3D.start();
            if (ok) {
                document.getElementById('twin3dToggle').classList.add('active');
                const t = document.getElementById('twin3dPref'); if (t) t.checked = true;
            }
        }
    });
    // Settings: 3D preference
    document.getElementById('twin3dPref')?.addEventListener('change', (e) => {
        if (e.target.checked && !three3D.active) {
            const ok = three3D.start();
            if (ok) document.getElementById('twin3dToggle')?.classList.add('active');
        } else if (!e.target.checked && three3D.active) {
            three3D.stop();
            document.getElementById('twin3dToggle')?.classList.remove('active');
        }
    });
    // Settings: WS preference
    document.getElementById('wsPref')?.addEventListener('change', (e) => {
        if (e.target.checked) wsStream.connect();
        else if (wsStream.socket) wsStream.socket.close();
    });
    // Auto-start 3D when Three.js loads (if user prefers 3D or first time)
    const pref3d = localStorage.getItem('edgeguard_3d_preferred') === 'on';
    window.addEventListener('three-ready', () => {
        if (pref3d || document.getElementById('view-twin').classList.contains('active')) {
            const ok = three3D.start();
            if (ok) {
                document.getElementById('twin3dToggle')?.classList.add('active');
                const t = document.getElementById('twin3dPref'); if (t) t.checked = true;
            }
        }
    });
    // Remember 3D pref
    document.getElementById('twin3dPref')?.addEventListener('change', (e) => {
        localStorage.setItem('edgeguard_3d_preferred', e.target.checked ? 'on' : 'off');
    });
    // Reflect pref on load
    const prefEl = document.getElementById('twin3dPref');
    if (prefEl) prefEl.checked = pref3d;

    // Tab switch hook: start 3D on first open
    const origSwitch = window._switchTab || switchTab;
    // Geofence lazy load
    document.querySelector('[data-target="view-geofence"]')?.addEventListener('click', async () => {
        await loadLeaflet();
        setTimeout(() => geoMap.init(), 50);
    });

    // WebSocket stream (replaces HTTP polling when backend supports it)
    wsStream.on((msg) => {
        if (msg.type === 'reading' && msg.data) {
            state.latest[msg.data.sensor_type] = msg.data;
            if (Object.keys(state.latest).length >= 6) setConnection(true);
        } else if (msg.type === 'prediction' && msg.data) {
            state.latestPrediction = msg.data;
        } else if (msg.type === 'alert' && msg.data) {
            voiceAlerts.onAlert(msg.data);
        }
    });
    wsStream.connect();

    // First immediate render
    renderAll();

    // Try the first poll
    const ok = await pollTelemetry();
    if (ok) {
        await pollML();
        await pollAlerts();
        await pollPredictionHistory();
    } else if (settings.demoMode) {
        enterDemo('Backend offline at boot');
        seedPredHistory(50, 0.05);
        for (let i = 0; i < 3; i++) demoStep();
        state.lastPoll = new Date();
    }

    renderAll();

    // Polling loops (fallback when WS is disconnected)
    const ms = () => Math.max(1, settings.pollInterval) * 1000;
    setInterval(() => { if (!wsStream.connected) pollTelemetry(); }, ms());
    setInterval(() => { if (!wsStream.connected) pollML(); }, ms() * 2.5);
    setInterval(() => { if (!wsStream.connected) pollAlerts(); }, 5000);
    setInterval(pollPredictionHistory, 60000);

    // Alert poller (for voice alerts) — independent of WS
    setInterval(async () => {
        if (state.demo) return;
        try {
            const r = await fetch(settings.backendUrl + '/alerts/active', { cache: 'no-store' });
            if (r.ok) {
                const arr = await r.json();
                for (const a of arr) voiceAlerts.onAlert(a);
            }
        } catch (e) { /* ignore */ }
    }, 7000);

    // Demo tick
    setInterval(() => { if (state.demo) demoStep(); }, 1000);
    setInterval(demoFleetStep, 2000);

    // Render loop
    setInterval(renderAll, 1000);
}

document.addEventListener('DOMContentLoaded', boot);
