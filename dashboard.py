"""
GRU Flash Crash Risk Prediction — Enhanced Dashboard
=====================================================
Modes:
  1. Live Market    — single-stock prediction with optional auto-refresh
  2. Risk Timeline  — rolling GRU risk curve overlaid on a candlestick chart
  3. Portfolio Scan — risk ranking across a NIFTY watchlist
  4. Compare Models — run all available model artifacts on the same data
  5. Upload CSV     — batch file-based inference
"""
from pathlib import Path
import logging
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ── register custom layers for improved model ────────────────────────────────
import sys, importlib
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
try:
    from custom_layers import Attention as _Attention
    _CUSTOM_OBJECTS = {"Attention": _Attention}
except ImportError:
    _CUSTOM_OBJECTS = {}

# ── silence noisy yfinance logs ──────────────────────────────────────────────
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── constants ─────────────────────────────────────────────────────────────────
NIFTY_PRESETS: dict[str, str] = {
    "— pick a preset —": "",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "ITC Ltd": "ITC.NS",
    "Axis Bank": "AXISBANK.NS",
}

DEFAULT_WATCHLIST: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]

MODEL_CANDIDATES: list[str] = [
    "improved_flash_crash_model.keras",
    "improved_minute_model.keras",
    "best_gru_model_improved.h5",
    "gru_model_final_improved.h5",
    "flash_crash_model.keras",
    "flash_crash_lstm.keras",
    "flash_crash_gru_model.keras",
    "enhanced_flash_crash_model.keras",
    "flash_crash_lstm_model.keras",
    "best_lstm_model.h5",
    "lstm_model_final.h5",
]

FEATURES_30x5: list[str] = [
    "return",
    "volume_change",
    "volatility_5",
    "volatility_10",
    "momentum_5",
]

FEATURES_20x14: list[str] = [
    "Open", "High", "Low", "Close", "Volume", "VWAP", "return",
    "volatility", "momentum", "volume_change", "vwap_diff",
    "high_low_spread", "open_close_return", "turnover_change",
]

FEATURES_30x10: list[str] = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return",
    "price_acceleration",
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRU Flash Crash Risk",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0b1120;
  --surface: rgba(15, 23, 42, 0.7);
  --surface-solid: #0f172a;
  --ink: #e2e8f0;
  --muted: #94a3b8;
  --accent: #06d6a0;
  --accent-glow: rgba(6, 214, 160, 0.15);
  --danger: #ef4444;
  --danger-glow: rgba(239, 68, 68, 0.15);
  --warning: #f59e0b;
  --border: rgba(148, 163, 184, 0.12);
  --radius: 16px;
}

.stApp {
  background: var(--bg) !important;
  font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label { font-size: .82rem !important; }

/* hero banner */
.hero-bar {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.5rem;
  margin-bottom: .8rem;
  display: flex; align-items: center; justify-content: space-between;
}
.hero-bar h1 {
  margin: 0; font-size: 1.4rem; font-weight: 700;
  color: #f1f5f9; letter-spacing: -.3px;
}
.hero-bar .hero-sub {
  font-size: .78rem; color: var(--muted); margin-top: .15rem;
}
.hero-bar .hero-badge {
  background: var(--accent-glow); color: var(--accent);
  border: 1px solid rgba(6,214,160,.25);
  padding: .3rem .7rem; border-radius: 999px;
  font-size: .72rem; font-weight: 600; letter-spacing: .5px;
}

/* risk result card */
.risk-result {
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.5rem;
  text-align: center;
  transition: transform .2s, box-shadow .2s;
}
.risk-result:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(0,0,0,.3);
}
.risk-pct {
  font-size: 3rem; font-weight: 800; line-height: 1;
  margin: .3rem 0; letter-spacing: -1px;
}
.risk-pct.low  { color: var(--accent); }
.risk-pct.mid  { color: var(--warning); }
.risk-pct.high { color: var(--danger); }
.risk-label {
  display: inline-block;
  padding: .25rem .75rem; border-radius: 999px;
  font-size: .75rem; font-weight: 700; letter-spacing: 1px;
  color: #fff;
}
.risk-label.low  { background: var(--accent); }
.risk-label.mid  { background: var(--warning); }
.risk-label.high { background: var(--danger); }
.risk-meta {
  font-size: .72rem; color: var(--muted); margin-top: .5rem;
}

/* glass cards */
.glass-card {
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
}
.glass-card h3 {
  margin: 0 0 .3rem; font-size: .85rem; font-weight: 600;
  color: var(--muted); text-transform: uppercase; letter-spacing: .8px;
}
.glass-card .val {
  font-size: 1.3rem; font-weight: 700; color: var(--ink);
}

/* force dark text in main area */
.stApp, .stApp p, .stApp span, .stApp label, .stApp .stMarkdown {
  color: var(--ink) !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color: #f1f5f9 !important;
}

/* plotly container */
.stPlotlyChart { border-radius: var(--radius); overflow: hidden; }
</style>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def build_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = df.copy()

    # ── base returns (MUST match training scripts exactly) ────────────────
    out["return"]             = out["Close"].pct_change()           # pct_change for daily & minute models
    out["log_return"]         = np.log(out["Close"] / out["Close"].shift(1))  # separate log-return

    # ── volatility at multiple scales ─────────────────────────────────────
    out["volatility_5"]       = out["return"].rolling(5).std()
    out["volatility_10"]      = out["return"].rolling(10).std()
    out["volatility_20"]      = out["return"].rolling(20).std()
    out["volatility"]         = out["return"].rolling(10).std()     # alias for daily model

    # ── momentum ──────────────────────────────────────────────────────────
    out["momentum_5"]         = out["Close"].pct_change(5)
    out["momentum_10"]        = out["Close"].pct_change(10)
    out["momentum"]           = out["Close"] - out["Close"].shift(5)  # absolute for daily model

    # ── volume / turnover ─────────────────────────────────────────────────
    out["volume_change"]      = out["Volume"].pct_change()
    out["VWAP"]               = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["vwap_diff"]          = (out["Close"] - out["VWAP"]) / out["VWAP"].replace(0, np.nan)
    out["turnover_change"]    = (out["Close"] * out["Volume"]).pct_change()

    # ── price structure ───────────────────────────────────────────────────
    out["high_low_spread"]    = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["open_close_return"]  = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)

    # ── minute-model extras ───────────────────────────────────────────────
    out["price_acceleration"] = out["return"].diff()

    return out


def build_sequence(
    df: pd.DataFrame, timesteps: int, feature_count: int
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    prepared = build_common_columns(flatten_columns(df))
    if feature_count == 5:
        features = FEATURES_30x5
        ff = prepared[features].dropna().copy()
        sc = StandardScaler()
        ff.loc[:, features] = sc.fit_transform(ff[features])
    elif feature_count == 10:
        features = FEATURES_30x10
        ff = prepared[features].dropna().copy()
        sc = StandardScaler()
        ff.loc[:, features] = sc.fit_transform(ff[features])
    elif feature_count == 14:
        features = FEATURES_20x14
        ff = prepared[features].dropna().copy()
    else:
        raise ValueError(
            f"Model expects {feature_count} features — app supports 5, 10, or 14."
        )
    if len(ff) < timesteps:
        raise ValueError(
            f"Need {timesteps} rows, only {len(ff)} available after feature engineering."
        )
    seq = np.expand_dims(ff.tail(timesteps).to_numpy(dtype=np.float32), axis=0)
    return seq, ff, features


def compute_risk_timeline(
    df: pd.DataFrame,
    model,
    timesteps: int,
    feature_count: int,
    step: int = 1,
) -> pd.DataFrame:
    """Run every sliding window of ``step`` rows and return a risk curve."""
    prepared = build_common_columns(flatten_columns(df))
    if feature_count == 5:
        features = FEATURES_30x5
        ff = prepared[features].dropna().copy()
        sc = StandardScaler()
        ff.loc[:, features] = sc.fit_transform(ff[features])
    elif feature_count == 10:
        features = FEATURES_30x10
        ff = prepared[features].dropna().copy()
        sc = StandardScaler()
        ff.loc[:, features] = sc.fit_transform(ff[features])
    else:
        features = FEATURES_20x14
        ff = prepared[features].dropna().copy()
    arr = ff.to_numpy(dtype=np.float32)
    idx = ff.index
    probs, dates = [], []
    for i in range(timesteps, len(arr), step):
        seq = np.expand_dims(arr[i - timesteps:i], axis=0)
        p = float(np.clip(model.predict(seq, verbose=0).reshape(-1)[0], 0.0, 1.0))
        probs.append(p)
        dates.append(idx[i - 1])
    if not probs:
        raise ValueError("Not enough rows to build a risk timeline.")
    return pd.DataFrame({"crash_risk": probs}, index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def discover_models() -> list[Path]:
    return [Path(c) for c in MODEL_CANDIDATES if Path(c).exists()]


@st.cache_resource
def load_trained_model(path_str: str):
    return load_model(path_str, compile=False, custom_objects=_CUSTOM_OBJECTS)


def get_model_signature(model) -> tuple[int, int]:
    shape = getattr(model, "input_shape", None)
    if not shape or len(shape) != 3:
        raise ValueError("Expected 3-D model input (batch, timesteps, features).")
    return int(shape[1]), int(shape[2])


def predict_probability(model, sequence: np.ndarray) -> float:
    return float(model.predict(sequence, verbose=0).reshape(-1)[0])


def normalize_prob(p: float) -> float:
    return 0.0 if np.isnan(p) else float(np.clip(p, 0.0, 1.0))


def risk_band(prob: float, threshold: float) -> tuple[str, str]:
    if prob >= threshold:
        return "HIGH RISK", "#b91c1c"
    if prob >= threshold * 0.65:
        return "ELEVATED", "#d97706"
    return "STABLE", "#15803d"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def _demo_freq(interval: str) -> str:
    return {
        "1m": "min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "h",
        "1d": "B",
    }.get(interval, "B")


@st.cache_data(ttl=600, show_spinner=False)
def _fetch(ticker: str, period: str, interval: str, cache_bust: int | None = None) -> pd.DataFrame:
    # NOTE: `cache_bust` is intentionally unused; it exists to vary Streamlit's cache key
    # so auto-refresh can reliably force new data instead of reusing a stale cached response.
    _ = cache_bust
    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)


def load_market_data(
    ticker: str,
    period: str,
    interval: str,
    max_retries: int = 3,
    *,
    cache_bust: int | None = None,
) -> pd.DataFrame:
    last_err = ""
    for attempt in range(max_retries):
        try:
            df = _fetch(ticker, period, interval, cache_bust)
            if not df.empty:
                return df
            last_err = "Empty response."
        except Exception as exc:
            last_err = str(exc)
        time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))
    raise ValueError(f"Could not fetch {ticker} after {max_retries} retries. {last_err}")


def _period_rows(period: str, interval: str) -> int:
    # rough approximations for demo generation only
    trading_days = {
        "1d": 1,
        "5d": 5,
        "7d": 7,
        "1mo": 22,
        "3mo": 66,
        "6mo": 132,
        "1y": 252,
        "2y": 504,
    }.get(period, 132)

    if interval in {"1m", "5m", "15m", "30m"}:
        # ~390 minutes per trading day
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}[interval]
        return max(60, (trading_days * 390) // minutes)
    if interval in {"60m", "1h"}:
        return max(30, trading_days * 6)
    return max(30, trading_days)


def generate_demo_data(period: str, interval: str, seed: int = 42) -> pd.DataFrame:
    rows = _period_rows(period, interval)
    freq = _demo_freq(interval)
    rng  = np.random.default_rng(seed)
    ts   = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    ret  = rng.normal(0, 0.004 if interval == "1h" else 0.01, rows)
    close  = 2000.0 * np.cumprod(1 + ret)
    open_  = np.r_[close[0], close[:-1]]
    high   = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low    = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.integers(500_000, 5_000_000, rows).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=ts,
    )


def safe_load(
    ticker: str,
    period: str,
    interval: str,
    use_fallback: bool,
    *,
    cache_bust: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    try:
        return load_market_data(ticker, period, interval, cache_bust=cache_bust), False
    except Exception:
        if use_fallback:
            return generate_demo_data(period, interval), True
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def gauge_chart(prob: float, threshold: float) -> go.Figure:
    _, color = risk_band(prob, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": "Crash Probability", "font": {"size": 13, "color": "#486581"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [0, threshold * 65],   "color": "#dcfce7"},
                {"range": [threshold * 65, threshold * 100], "color": "#fef9c3"},
                {"range": [threshold * 100, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#7f1d1d", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def candlestick_with_risk(
    ohlcv: pd.DataFrame, risk_df: pd.DataFrame, threshold: float, ticker: str
) -> go.Figure:
    ohlcv = flatten_columns(ohlcv)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=[f"{ticker} — Price", "GRU Crash-Risk Probability"],
    )
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"], high=ohlcv["High"],
        low=ohlcv["Low"],   close=ohlcv["Close"],
        name="Price",
        increasing_line_color="#15803d",
        decreasing_line_color="#b91c1c",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=risk_df.index, y=risk_df["crash_risk"],
        mode="lines", name="Crash Risk",
        line=dict(color="#0f766e", width=2),
        fill="tozeroy", fillcolor="rgba(15,118,110,0.12)",
    ), row=2, col=1)

    high_risk = risk_df[risk_df["crash_risk"] >= threshold]
    if not high_risk.empty:
        fig.add_trace(go.Scatter(
            x=high_risk.index, y=high_risk["crash_risk"],
            mode="markers", name="High-risk",
            marker=dict(color="#b91c1c", size=5),
        ), row=2, col=1)

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#b91c1c",
        annotation_text=f"Threshold {threshold:.0%}",
        annotation_position="top right", row=2, col=1,
    )
    fig.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig


def feature_bar_chart(ff: pd.DataFrame, features: list[str]) -> go.Figure:
    last = ff[features].tail(1).T.reset_index()
    last.columns = ["Feature", "Value"]
    fig = px.bar(
        last, x="Value", y="Feature", orientation="h",
        color="Value",
        color_continuous_scale=["#15803d", "#fef9c3", "#b91c1c"],
        title="Feature snapshot — latest window",
    )
    fig.update_layout(
        height=max(260, len(features) * 28),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis_title="",
    )
    return fig


def portfolio_bar_chart(port_df: pd.DataFrame, threshold: float) -> go.Figure:
    df = port_df.dropna(subset=["Crash Risk"]).sort_values("Crash Risk", ascending=True)
    colors = [
        "#b91c1c" if r >= threshold else "#d97706" if r >= threshold * 0.65 else "#15803d"
        for r in df["Crash Risk"]
    ]
    fig = go.Figure(go.Bar(
        x=df["Crash Risk"], y=df["Ticker"],
        orientation="h", marker_color=colors,
        text=[f"{v:.1%}" for v in df["Crash Risk"]],
        textposition="outside",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#7f1d1d",
        annotation_text="Threshold", annotation_position="top",
    )
    fig.update_layout(
        height=max(300, len(df) * 52),
        xaxis=dict(tickformat=".0%", range=[0, 1.1]),
        margin=dict(l=20, r=60, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def model_comparison_chart(comp_df: pd.DataFrame, threshold: float) -> go.Figure:
    df = comp_df.dropna(subset=["Crash Risk"]).sort_values("Crash Risk")
    colors = [
        "#b91c1c" if r >= threshold else "#d97706" if r >= threshold * 0.65 else "#0f766e"
        for r in df["Crash Risk"]
    ]
    fig = go.Figure(go.Bar(
        x=df["Model"], y=df["Crash Risk"],
        marker_color=colors,
        text=[f"{v:.1%}" for v in df["Crash Risk"]],
        textposition="outside",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#7f1d1d",
        annotation_text=f"Threshold {threshold:.0%}",
        annotation_position="top right",
    )
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1.1]),
        height=370,
        margin=dict(l=10, r=10, t=30, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-35,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if "pred_log" not in st.session_state:
    st.session_state["pred_log"] = []


def log_prediction(ticker: str, prob: float, band: str, model_name: str) -> None:
    st.session_state["pred_log"].insert(0, {
        "Time":    time.strftime("%H:%M:%S"),
        "Ticker":  ticker,
        "Risk %":  f"{prob:.1%}",
        "Band":    band,
        "Model":   model_name,
    })
    st.session_state["pred_log"] = st.session_state["pred_log"][:30]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

available_models = discover_models()
if not available_models:
    st.error("No model files found in the deployment directory.")
    st.stop()

st.sidebar.markdown("## ⚙️ Controls")

selected_name = st.sidebar.selectbox("Model", [p.name for p in available_models])
selected_path = next(p for p in available_models if p.name == selected_name)

try:
    model        = load_trained_model(str(selected_path))
    timesteps, feature_count = get_model_signature(model)
except Exception as exc:
    st.sidebar.error(f"Model load error: {exc}")
    st.stop()

st.sidebar.success(f"✓ {selected_path.name}")
st.sidebar.caption(f"Input: {timesteps} steps × {feature_count} features")

threshold = st.sidebar.slider("Risk threshold", 0.05, 0.95, 0.20, 0.05, format="%.2f")
mode = st.sidebar.radio(
    "Mode",
    ["Live Market", "Risk Timeline", "Portfolio Scan", "Compare Models", "Upload CSV"],
)
st.sidebar.markdown("---")

# auto-refresh controls only shown in Live Market mode
auto_refresh  = False
refresh_secs  = 60
if mode == "Live Market":
    auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
    if auto_refresh:
        refresh_secs = st.sidebar.select_slider(
            "Interval (s)", options=[15, 30, 60, 120, 300], value=60
        )
        # non-blocking JS-based page rerun
        st_autorefresh(interval=refresh_secs * 1000, key="live_autorefresh")

use_demo_fallback = st.sidebar.checkbox("Demo fallback if rate-limited", value=True)

# ── hero bar ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-bar'>
  <div>
    <h1>📉 Flash Crash Risk Prediction</h1>
    <div class='hero-sub'>{selected_path.stem} · {timesteps}×{feature_count} · threshold {threshold:.0%}</div>
  </div>
  <span class='hero-badge'>● LIVE</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — LIVE MARKET
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "Live Market":
    # ── controls row ──────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 1.5, 1.5, 1.5])
    with ctrl1:
        preset_name = st.selectbox("NIFTY preset", list(NIFTY_PRESETS.keys()), label_visibility="collapsed")
        preset_val  = NIFTY_PRESETS[preset_name]
        if "lm_ticker" not in st.session_state:
            st.session_state["lm_ticker"] = preset_val or "RELIANCE.NS"
        elif preset_val:
            st.session_state["lm_ticker"] = preset_val
    with ctrl2:
        ticker = st.text_input("Ticker", key="lm_ticker", label_visibility="collapsed", placeholder="Ticker e.g. RELIANCE.NS").strip().upper()
    with ctrl3:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m", "1h", "1d"], index=1, label_visibility="collapsed")
    with ctrl4:
        if interval == "1m":
            period_opts = ["1d", "5d", "7d"]; period_default = "5d"
        elif interval in {"5m", "15m", "30m"}:
            period_opts = ["5d", "7d", "1mo"]; period_default = "7d"
        elif interval in {"60m", "1h"}:
            period_opts = ["1mo", "3mo", "6mo", "1y", "2y"]; period_default = "6mo"
        else:
            period_opts = ["1mo", "3mo", "6mo", "1y", "2y"]; period_default = "6mo"
        period = st.selectbox(
            "Period", period_opts,
            index=max(0, period_opts.index(period_default) if period_default in period_opts else 0),
            label_visibility="collapsed",
        )
    with ctrl5:
        run_btn = st.button("⚡ Predict", type="primary", use_container_width=True)

    # ── intraday notice (compact) ─────────────────────────────────────────
    _is_intraday = interval in {"1m", "5m", "15m", "30m", "60m", "1h"}
    _is_minute_model = "minute" in selected_path.name.lower()
    if _is_intraday and not _is_minute_model:
        st.caption("⚠️ Using daily model on intraday data — select **improved_minute_model.keras** for better accuracy")

    # ── run prediction ────────────────────────────────────────────────────
    if run_btn or auto_refresh:
        if not ticker:
            st.warning("Enter a valid ticker symbol.")
            st.stop()

        with st.spinner("Fetching market data…"):
            cache_bust = int(time.time() // max(15, int(refresh_secs))) if auto_refresh else None
            market, is_demo = safe_load(
                ticker, period, interval, use_demo_fallback, cache_bust=cache_bust
            )
            market = flatten_columns(market)

        if is_demo:
            st.caption("📡 Demo data — Yahoo Finance rate-limited")

        try:
            seq, ff, feature_list = build_sequence(market, timesteps, feature_count)
            prob = normalize_prob(predict_probability(model, seq))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        band, color = risk_band(prob, threshold)
        log_prediction(ticker, prob, band, selected_path.stem)

        # determine risk CSS class
        _cls = "low" if prob < threshold else ("mid" if prob < 0.5 else "high")

        # ── result layout: risk card | price chart ────────────────────────
        risk_col, chart_col = st.columns([1, 2.5])

        with risk_col:
            st.markdown(f"""
            <div class='risk-result'>
              <div style='font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600'>
                Crash Probability
              </div>
              <div class='risk-pct {_cls}'>{prob:.1%}</div>
              <span class='risk-label {_cls}'>{band}</span>
              <div class='risk-meta'>
                {ticker} · {interval} · {time.strftime('%H:%M:%S')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # key stats cards
            last_close = market["Close"].iloc[-1] if len(market) > 0 else 0
            day_ret = market["Close"].pct_change().iloc[-1] * 100 if len(market) > 1 else 0
            st.markdown(f"""
            <div class='glass-card' style='margin-top:.8rem'>
              <h3>Last Price</h3>
              <div class='val'>₹{last_close:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card' style='margin-top:.5rem'>
              <h3>Period Return</h3>
              <div class='val' style='color:{"var(--accent)" if day_ret >= 0 else "var(--danger)"}'>{day_ret:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with chart_col:
            candle = go.Figure(go.Candlestick(
                x=market.index,
                open=market["Open"], high=market["High"],
                low=market["Low"],   close=market["Close"],
                name=ticker,
                increasing_line_color="#06d6a0",
                decreasing_line_color="#ef4444",
            ))
            candle.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=35, b=10),
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.5)",
                title=dict(text=f"{ticker} — {period} {interval}", font=dict(color="#e2e8f0", size=14)),
                xaxis=dict(gridcolor="rgba(148,163,184,0.08)", color="#94a3b8"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.08)", color="#94a3b8"),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(candle, use_container_width=True)

        # ── expandable details ────────────────────────────────────────────
        with st.expander("📊 Feature snapshot & engineering details"):
            st.plotly_chart(feature_bar_chart(ff, feature_list), use_container_width=True)
            st.dataframe(ff.tail(5), use_container_width=True)

        if st.session_state["pred_log"]:
            with st.expander("🕘 Prediction history", expanded=False):
                st.dataframe(
                    pd.DataFrame(st.session_state["pred_log"]),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── auto-refresh info ─────────────────────────────────────────────
        if auto_refresh:
            st.caption(f"🔄 Auto-refreshing every {refresh_secs}s")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — RISK TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "Risk Timeline":
    st.subheader("📈 Rolling GRU Risk Curve over Full History")
    st.caption(
        "Runs the GRU on every sliding window to produce a continuous risk-probability "
        "curve aligned with the price chart — great for spotting structural stress periods."
    )

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        tl_preset  = st.selectbox("NIFTY preset", list(NIFTY_PRESETS.keys()), key="tl_pre")
        tl_pval    = NIFTY_PRESETS[tl_preset]
    with rc2:
        tl_ticker  = st.text_input("Ticker", value=tl_pval or "RELIANCE.NS", key="tl_t").strip().upper()
    with rc3:
        tl_period  = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="tl_p")
    with rc4:
        tl_interval = st.selectbox("Interval", ["1d"], key="tl_i")

    tl_step = st.slider(
        "Stride (rows between predictions)", 1, 10, 1,
        help="1 = densest resolution; higher = faster compute, coarser curve",
    )

    if st.button("▶ Compute risk timeline", type="primary"):
        with st.spinner(f"Fetching {tl_ticker} and running rolling GRU predictions…"):
            try:
                tl_df, tl_demo = safe_load(tl_ticker, tl_period, tl_interval, use_demo_fallback)
                if tl_demo:
                    st.warning("Using demo stream (provider rate-limited).")
                tl_risk = compute_risk_timeline(tl_df, model, timesteps, feature_count, step=tl_step)
            except Exception as exc:
                st.error(f"Timeline failed: {exc}")
                st.stop()

        high_windows = int((tl_risk["crash_risk"] >= threshold).sum())
        tm1, tm2, tm3 = st.columns(3)
        tm1.metric("Windows evaluated", len(tl_risk))
        tm2.metric("High-risk windows",  high_windows)
        tm3.metric("% time at high risk", f"{high_windows / len(tl_risk):.1%}")

        st.plotly_chart(
            candlestick_with_risk(tl_df, tl_risk, threshold, tl_ticker),
            use_container_width=True,
        )

        with st.expander("Risk data table"):
            styled = (
                tl_risk
                .rename(columns={"crash_risk": "Crash Risk"})
                .style.format("{:.2%}")
            )
            st.dataframe(styled, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — PORTFOLIO SCAN
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "Portfolio Scan":
    st.subheader("🔍 NIFTY Portfolio Risk Scanner")
    st.caption("Scan multiple stocks and rank them by current GRU crash-risk score.")

    watchlist_str = st.text_area(
        "Tickers (comma-separated)",
        value=", ".join(DEFAULT_WATCHLIST),
        height=80,
    )
    ps1, ps2 = st.columns(2)
    ps_period   = ps1.selectbox("Period",   ["3mo", "6mo", "1y"], index=1, key="ps_p")
    ps_interval = ps2.selectbox("Interval", ["1d"],               index=0, key="ps_i")

    if st.button("🔍 Scan portfolio", type="primary"):
        tickers = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
            st.stop()

        rows = []
        prog = st.progress(0)
        info = st.empty()
        for i, tk in enumerate(tickers):
            info.caption(f"Scanning {tk}… ({i+1}/{len(tickers)})")
            try:
                df_tk, demo_tk = safe_load(tk, ps_period, ps_interval, use_demo_fallback)
                sq_tk, _, _    = build_sequence(df_tk, timesteps, feature_count)
                pr_tk          = normalize_prob(predict_probability(model, sq_tk))
                bd_tk          = risk_band(pr_tk, threshold)[0]
                rows.append({"Ticker": tk, "Crash Risk": pr_tk, "Band": bd_tk,
                             "Source": "demo" if demo_tk else "live"})
            except Exception as exc:
                rows.append({"Ticker": tk, "Crash Risk": None, "Band": str(exc), "Source": "—"})
            prog.progress((i + 1) / len(tickers))

        info.empty(); prog.empty()

        port_df = pd.DataFrame(rows)
        valid   = port_df.dropna(subset=["Crash Risk"])

        if not valid.empty:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Scanned",       len(rows))
            p2.metric("High-risk",     int((valid["Crash Risk"] >= threshold).sum()))
            p3.metric("Avg risk",      f"{valid['Crash Risk'].mean():.1%}")
            p4.metric("Peak risk",     f"{valid['Crash Risk'].max():.1%}")

            st.plotly_chart(portfolio_bar_chart(valid, threshold), use_container_width=True)

        st.dataframe(
            port_df.style.format({"Crash Risk": "{:.1%}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 4 — COMPARE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "Compare Models":
    st.subheader("⚖️ Model Comparison")
    st.caption(
        "Run every available model artifact on the same price history and compare "
        "their crash-risk scores side by side."
    )

    cm1, cm2 = st.columns(2)
    with cm1:
        cm_preset  = st.selectbox("NIFTY preset", list(NIFTY_PRESETS.keys()), key="cm_pre")
        cm_pval    = NIFTY_PRESETS[cm_preset]
        cm_ticker  = st.text_input("Ticker", value=cm_pval or "RELIANCE.NS", key="cm_t").strip().upper()
    with cm2:
        cm_period   = st.selectbox("Period",   ["3mo", "6mo", "1y"], index=1, key="cm_p")
        cm_interval = st.selectbox("Interval", ["1d"],               index=0, key="cm_i")

    if st.button("⚖️ Compare all models", type="primary"):
        with st.spinner("Loading data…"):
            try:
                cm_df, cm_demo = safe_load(cm_ticker, cm_period, cm_interval, use_demo_fallback)
                if cm_demo:
                    st.warning("Using demo market stream.")
            except Exception as exc:
                st.error(f"Data fetch failed: {exc}")
                st.stop()

        comp_rows = []
        for mpath in available_models:
            try:
                m_      = load_trained_model(str(mpath))
                t_, f_  = get_model_signature(m_)
                sq_, *_ = build_sequence(cm_df, t_, f_)
                pr_     = normalize_prob(predict_probability(m_, sq_))
                comp_rows.append({
                    "Model": mpath.name,
                    "Crash Risk": pr_,
                    "Band": risk_band(pr_, threshold)[0],
                    "Shape": f"{t_}×{f_}",
                })
            except Exception as exc:
                comp_rows.append({"Model": mpath.name, "Crash Risk": None,
                                  "Band": f"Error: {exc}", "Shape": "—"})

        comp_df = pd.DataFrame(comp_rows)
        valid_c = comp_df.dropna(subset=["Crash Risk"])

        if not valid_c.empty:
            v1, v2, v3 = st.columns(3)
            v1.metric("Models run",  len(comp_rows))
            v2.metric("Avg risk",    f"{valid_c['Crash Risk'].mean():.1%}")
            agreement = (valid_c["Band"] == valid_c["Band"].mode()[0]).mean()
            v3.metric("Agreement",   f"{agreement:.0%}")
            st.plotly_chart(model_comparison_chart(valid_c, threshold), use_container_width=True)

        st.dataframe(
            comp_df.style.format({"Crash Risk": "{:.1%}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 5 — UPLOAD CSV
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("📂 Upload CSV for Inference")
    st.caption(
        f"CSV must contain: Open, High, Low, Close, Volume. "
        f"Minimum {timesteps + 15} rows required for robust feature engineering."
    )

    uploaded = st.file_uploader("Drop CSV here", type=["csv"])
    if uploaded and st.button("▶ Run prediction", type="primary"):
        try:
            up_df = pd.read_csv(uploaded)
            seq, ff, feature_list = build_sequence(up_df, timesteps, feature_count)
            prob = normalize_prob(predict_probability(model, seq))
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        band, color = risk_band(prob, threshold)
        log_prediction("CSV upload", prob, band, selected_path.stem)

        u1, u2 = st.columns([1, 2])
        with u1:
            st.plotly_chart(gauge_chart(prob, threshold), use_container_width=True)
            st.markdown(
                f"<span class='risk-chip' style='background:{color}'>{band}</span>",
                unsafe_allow_html=True,
            )
            if prob >= threshold:
                st.error(f"High crash-risk signal ≥ {threshold:.0%}")
            else:
                st.success("No high-risk signal under current threshold")
        with u2:
            st.plotly_chart(feature_bar_chart(ff, feature_list), use_container_width=True)
            with st.expander("Engineered rows (last 5)"):
                st.dataframe(ff.tail(5), use_container_width=True)
