/* ═══════════════════════════════════════════════════════════════
   FlashGuard — Real-Time Flash Crash Detection Frontend
   ═══════════════════════════════════════════════════════════════ */

const API = "";

// ── State ──────────────────────────────────────────────────────
let priceChart = null;
let riskHistoryChart = null;
let timelineChart = null;
let autoRefreshActive = false;
let autoRefreshTimer = null;
let countdown = 0;
let countdownTimer = null;
const riskHistory = [];           // {time, risk}
const REFRESH_INTERVAL = 60;      // seconds

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initControls();
    loadModels();
});


function initTabs() {
    document.querySelectorAll(".nav-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-section").forEach(s => s.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(`section-${tab.dataset.tab}`)?.classList.add("active");
        });
    });
}


function initControls() {
    document.getElementById("preset-select").addEventListener("change", e => {
        if (e.target.value) document.getElementById("ticker-input").value = e.target.value;
    });

    document.getElementById("btn-predict").addEventListener("click", () => runPrediction());
    document.getElementById("btn-timeline").addEventListener("click", runTimeline);
    document.getElementById("btn-auto").addEventListener("click", toggleAutoRefresh);

    document.getElementById("ticker-input").addEventListener("keydown", e => {
        if (e.key === "Enter") runPrediction();
    });
    document.getElementById("timeline-ticker").addEventListener("keydown", e => {
        if (e.key === "Enter") runTimeline();
    });
}


async function loadModels() {
    try {
        const res = await fetch(`${API}/api/models`);
        const models = await res.json();
        const sel = document.getElementById("model-select");
        sel.innerHTML = "";

        // Prioritize minute model
        const sorted = [...models].sort((a, b) => {
            if (a.name.includes("minute")) return -1;
            if (b.name.includes("minute")) return 1;
            return 0;
        });

        sorted.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.name;
            const short = m.name.replace(".keras","").replace(".h5","");
            opt.textContent = `${short} (${m.timesteps}×${m.features})`;
            sel.appendChild(opt);
        });

        setConnection(true);
    } catch {
        showToast("Cannot connect to API server");
        setConnection(false);
    }
}


function setConnection(online) {
    const el = document.getElementById("connection-indicator");
    el.className = `conn-badge ${online ? "online" : "offline"}`;
    el.querySelector(".conn-text").textContent = online ? "Connected" : "Offline";
}


// ═══════════════════════════════════════════════════════════════
// LIVE PREDICTION
// ═══════════════════════════════════════════════════════════════

async function runPrediction(silent = false) {
    const ticker = document.getElementById("ticker-input").value.trim();
    const model = document.getElementById("model-select").value;
    const interval = document.getElementById("interval-select").value;

    if (!ticker) return showToast("Enter a ticker symbol");
    if (!model) return showToast("No model available");

    // Choose period based on interval
    const periodMap = { "1m": "5d", "5m": "5d", "15m": "1mo", "30m": "1mo", "1h": "3mo", "1d": "6mo" };
    const period = periodMap[interval] || "6mo";

    if (!silent) showLoading("live");

    try {
        const res = await fetch(`${API}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker, model, period, interval, threshold: 0.20 }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        renderLive(data, interval);
        setConnection(true);
    } catch (err) {
        showToast("Prediction failed: " + err.message);
        if (!silent) hideLoading("live");
        setConnection(false);
    }
}


function renderLive(data, interval) {
    hideLoading("live");
    document.getElementById("live-empty")?.classList.add("hidden");
    document.getElementById("live-results").classList.remove("hidden");

    const prob = data.probability;
    const pct = data.risk_pct;
    const band = data.band;

    // Risk ring
    drawRiskRing(prob);

    // Risk value
    const pctEl = document.getElementById("risk-pct");
    animateNumber(pctEl, pct);
    pctEl.style.color = band === "STABLE" ? "var(--green)" : band === "ELEVATED" ? "var(--amber)" : "var(--red)";

    // Badge
    const badge = document.getElementById("risk-badge");
    badge.textContent = band;
    badge.className = `risk-badge ${band === "STABLE" ? "stable" : band === "ELEVATED" ? "elevated" : "high"}`;

    // Description
    const desc = document.getElementById("risk-description");
    if (band === "STABLE") desc.textContent = "No significant crash risk detected";
    else if (band === "ELEVATED") desc.textContent = "Elevated volatility — monitor closely";
    else desc.textContent = "⚠️ High crash probability detected!";

    // Hero border color
    const hero = document.getElementById("risk-hero");
    hero.className = `risk-hero ${band === "STABLE" ? "stable" : band === "ELEVATED" ? "elevated" : "high"}`;

    // Stats
    document.getElementById("stat-ticker").textContent = data.ticker.replace(".NS", "");
    document.getElementById("stat-price").textContent = "₹" + data.latest_close.toLocaleString("en-IN");
    document.getElementById("stat-date").textContent = data.latest_date;
    document.getElementById("stat-model").textContent = data.model.replace(".keras","").replace(".h5","");
    document.getElementById("stat-interval").textContent = interval.toUpperCase();
    document.getElementById("stat-shape").textContent = `${data.timesteps} × ${data.features}`;

    // Chart subtitle
    document.getElementById("chart-subtitle").textContent = `${data.ohlc.length} bars · ${interval}`;

    // Price chart
    drawPriceChart(data.ohlc, data.ticker);

    // Risk history
    const now = new Date();
    riskHistory.push({
        time: now.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        risk: pct,
    });
    if (riskHistory.length > 30) riskHistory.shift();
    drawRiskHistory();
}


function animateNumber(el, target) {
    const start = parseFloat(el.textContent) || 0;
    const diff = target - start;
    const duration = 600;
    const startTime = performance.now();

    function update(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = (start + diff * ease).toFixed(1);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}


function drawRiskRing(prob) {
    const canvas = document.getElementById("risk-ring");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = 200;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";

    const cx = size / 2, cy = size / 2, r = 82;
    const lineWidth = 10;
    const startAngle = -0.5 * Math.PI;
    const fullAngle = 2 * Math.PI;

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, fullAngle);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.stroke();

    // Track marks
    for (let i = 0; i < 60; i++) {
        const angle = startAngle + (i / 60) * fullAngle;
        const len = i % 5 === 0 ? 8 : 4;
        const x1 = cx + (r + 6) * Math.cos(angle);
        const y1 = cy + (r + 6) * Math.sin(angle);
        const x2 = cx + (r + 6 + len) * Math.cos(angle);
        const y2 = cy + (r + 6 + len) * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineWidth = i % 5 === 0 ? 1.5 : 0.5;
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.stroke();
    }

    // Value arc
    const endAngle = startAngle + prob * fullAngle;
    const gradient = ctx.createConicGradient(startAngle, cx, cy);
    gradient.addColorStop(0, "#34d399");
    gradient.addColorStop(0.3, "#fbbf24");
    gradient.addColorStop(0.6, "#f97316");
    gradient.addColorStop(1, "#f87171");

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = gradient;
    ctx.lineCap = "round";
    ctx.stroke();

    // Glow at end
    if (prob > 0.01) {
        const ex = cx + r * Math.cos(endAngle);
        const ey = cy + r * Math.sin(endAngle);
        const glow = ctx.createRadialGradient(ex, ey, 0, ex, ey, 12);
        const color = prob < 0.13 ? "52,211,153" : prob < 0.5 ? "251,191,36" : "248,113,113";
        glow.addColorStop(0, `rgba(${color},0.5)`);
        glow.addColorStop(1, `rgba(${color},0)`);
        ctx.beginPath();
        ctx.arc(ex, ey, 12, 0, fullAngle);
        ctx.fillStyle = glow;
        ctx.fill();
    }
}


function drawPriceChart(ohlc, ticker) {
    const ctx = document.getElementById("price-chart").getContext("2d");
    if (priceChart) priceChart.destroy();

    const labels = ohlc.map(d => d.date);
    const closes = ohlc.map(d => d.close);
    const volumes = ohlc.map(d => d.volume);
    const colors = ohlc.map(d => d.close >= d.open ? "rgba(52,211,153,0.6)" : "rgba(248,113,113,0.6)");

    const gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, "rgba(56,189,248,0.15)");
    gradient.addColorStop(1, "rgba(56,189,248,0)");

    priceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    type: "line", label: "Close", data: closes,
                    borderColor: "#38bdf8", backgroundColor: gradient,
                    borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3,
                    yAxisID: "y", order: 0,
                },
                {
                    type: "bar", label: "Volume", data: volumes,
                    backgroundColor: colors, yAxisID: "y1", order: 1,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(6,10,19,0.95)", titleColor: "#e2e8f0",
                    bodyColor: "#8b9dc3", borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1, cornerRadius: 8,
                    callbacks: {
                        label: ctx => {
                            if (ctx.dataset.label === "Close") return `₹${ctx.raw.toLocaleString("en-IN")}`;
                            return `Vol: ${ctx.raw.toLocaleString()}`;
                        },
                    },
                },
            },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.02)" }, ticks: { color: "#4a5d82", maxRotation: 45, font: { size: 9 } } },
                y: { position: "left", grid: { color: "rgba(255,255,255,0.02)" }, ticks: { color: "#4a5d82", font: { size: 10 } } },
                y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { display: false } },
            },
        },
    });
}


function drawRiskHistory() {
    const ctx = document.getElementById("risk-history-chart").getContext("2d");
    if (riskHistoryChart) riskHistoryChart.destroy();

    const n = riskHistory.length;
    document.getElementById("risk-history-count").textContent = `${n} prediction${n !== 1 ? "s" : ""}`;

    const gradient = ctx.createLinearGradient(0, 0, 0, 160);
    gradient.addColorStop(0, "rgba(129,140,248,0.2)");
    gradient.addColorStop(1, "rgba(129,140,248,0)");

    riskHistoryChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: riskHistory.map(r => r.time),
            datasets: [{
                data: riskHistory.map(r => r.risk),
                borderColor: "#818cf8", backgroundColor: gradient,
                borderWidth: 2, pointRadius: 3, pointBackgroundColor: "#818cf8",
                pointHoverRadius: 5, fill: true, tension: 0.3,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(6,10,19,0.95)", titleColor: "#e2e8f0",
                    bodyColor: "#8b9dc3", cornerRadius: 8,
                    callbacks: { label: ctx => `Risk: ${ctx.raw.toFixed(2)}%` },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: "#4a5d82", font: { size: 9 } } },
                y: { min: 0, grid: { color: "rgba(255,255,255,0.02)" }, ticks: { color: "#4a5d82", font: { size: 10 } } },
            },
        },
    });
}


// ═══════════════════════════════════════════════════════════════
// AUTO-REFRESH
// ═══════════════════════════════════════════════════════════════

function toggleAutoRefresh() {
    autoRefreshActive = !autoRefreshActive;
    const btn = document.getElementById("btn-auto");
    const timerEl = document.getElementById("refresh-timer");

    if (autoRefreshActive) {
        btn.textContent = "ON";
        btn.classList.add("on");
        countdown = REFRESH_INTERVAL;
        startAutoRefresh();
        // Also run immediately
        runPrediction();
    } else {
        btn.textContent = "OFF";
        btn.classList.remove("on");
        clearInterval(autoRefreshTimer);
        clearInterval(countdownTimer);
        timerEl.textContent = "";
    }
}


function startAutoRefresh() {
    const timerEl = document.getElementById("refresh-timer");
    countdown = REFRESH_INTERVAL;

    clearInterval(countdownTimer);
    countdownTimer = setInterval(() => {
        countdown--;
        timerEl.textContent = countdown > 0 ? `${countdown}s` : "…";
        if (countdown <= 0) {
            runPrediction(true);
            countdown = REFRESH_INTERVAL;
        }
    }, 1000);
}


// ═══════════════════════════════════════════════════════════════
// TIMELINE
// ═══════════════════════════════════════════════════════════════

async function runTimeline() {
    const ticker = document.getElementById("timeline-ticker").value.trim();
    const model = document.getElementById("model-select").value;
    const period = document.getElementById("timeline-period").value;

    if (!ticker) return showToast("Enter a ticker symbol");

    showLoading("timeline");

    try {
        const res = await fetch(`${API}/api/timeline`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker, model, period }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        renderTimeline(data);
    } catch (err) {
        showToast("Timeline failed: " + err.message);
        hideLoading("timeline");
    }
}


function renderTimeline(data) {
    hideLoading("timeline");
    document.getElementById("timeline-empty")?.classList.add("hidden");
    document.getElementById("timeline-results").classList.remove("hidden");
    document.getElementById("timeline-subtitle").textContent =
        `${data.ticker} · ${data.points.length} points`;

    const ctx = document.getElementById("timeline-chart").getContext("2d");
    if (timelineChart) timelineChart.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 0, 380);
    gradient.addColorStop(0, "rgba(248,113,113,0.2)");
    gradient.addColorStop(0.5, "rgba(251,191,36,0.05)");
    gradient.addColorStop(1, "rgba(52,211,153,0)");

    timelineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.points.map(p => p.date),
            datasets: [{
                label: "Risk %", data: data.points.map(p => p.risk),
                borderColor: "#f59e0b", backgroundColor: gradient,
                borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
                fill: true, tension: 0.3,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(6,10,19,0.95)", titleColor: "#e2e8f0",
                    bodyColor: "#8b9dc3", cornerRadius: 8,
                    callbacks: { label: ctx => `Risk: ${ctx.raw.toFixed(2)}%` },
                },
            },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.02)" }, ticks: { color: "#4a5d82", maxRotation: 45, font: { size: 9 } } },
                y: { min: 0, grid: { color: "rgba(255,255,255,0.02)" }, ticks: { color: "#4a5d82" } },
            },
        },
    });
}


// ═══════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════

function showLoading(section) {
    document.getElementById(`${section}-loading`).classList.remove("hidden");
    document.getElementById(`${section}-results`)?.classList.add("hidden");
    document.getElementById(`${section}-empty`)?.classList.add("hidden");
}
function hideLoading(section) {
    document.getElementById(`${section}-loading`).classList.add("hidden");
}

function showToast(msg) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity 0.3s"; setTimeout(() => t.remove(), 300); }, 4000);
}
