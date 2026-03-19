"""
Flash Crash Prediction — REST API Server
==========================================
Flask API that wraps the trained GRU models for the HTML/CSS/JS frontend.
Endpoints:
    GET  /api/models              — list available models
    POST /api/predict             — single-ticker prediction
    POST /api/portfolio           — multi-ticker portfolio scan
    POST /api/timeline            — rolling risk timeline
"""

import json, sys, traceback, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ── custom layer registration ────────────────────────────────────────────────
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from custom_layers import Attention
    CUSTOM_OBJECTS = {"Attention": Attention}
except ImportError:
    CUSTOM_OBJECTS = {}

# ── app ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ── model registry ───────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    "improved_flash_crash_model.keras",
    "improved_minute_model.keras",
    "best_gru_model_improved.h5",
    "gru_model_final_improved.h5",
    "flash_crash_model.keras",
]

_loaded_models: dict[str, object] = {}


def _discover():
    return [c for c in MODEL_CANDIDATES if (MODEL_DIR / c).exists()]


def _get_model(name: str):
    if name not in _loaded_models:
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {name}")
        _loaded_models[name] = load_model(str(path), compile=False,
                                          custom_objects=CUSTOM_OBJECTS)
    return _loaded_models[name]


def _model_sig(model) -> tuple[int, int]:
    s = model.input_shape
    return int(s[1]), int(s[2])


# ── feature engineering ──────────────────────────────────────────────────────

FEATURES_14 = [
    "Open", "High", "Low", "Close", "Volume", "VWAP", "return",
    "volatility", "momentum", "volume_change", "vwap_diff",
    "high_low_spread", "open_close_return", "turnover_change",
]

FEATURES_10 = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return", "price_acceleration",
]

FEATURES_5 = [
    "return", "volume_change", "volatility_5", "volatility_10", "momentum_5",
]


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to title-cased OHLCV."""
    rename_map = {
        c: {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
             "date": "Date"}.get(str(c).strip().lower(), str(c).strip())
        for c in df.columns
    }
    return df.rename(columns=rename_map)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    o = _normalize_columns(_flatten(df.copy()))
    o["return"] = o["Close"].pct_change()             # matches train_minute.py
    o["log_return"] = np.log(o["Close"] / o["Close"].shift(1))
    o["volume_change"] = o["Volume"].pct_change()
    o["volatility_5"] = o["return"].rolling(5).std()
    o["volatility_10"] = o["return"].rolling(10).std()
    o["volatility_20"] = o["return"].rolling(20).std()
    o["volatility"] = o["volatility_10"]
    o["momentum_5"] = o["Close"].pct_change(5)
    o["momentum_10"] = o["Close"].pct_change(10)
    o["momentum"] = o["Close"] - o["Close"].shift(5)
    o["VWAP"] = (o["High"] + o["Low"] + o["Close"]) / 3.0
    o["vwap_diff"] = (o["Close"] - o["VWAP"]) / o["VWAP"].replace(0, np.nan)
    o["high_low_spread"] = (o["High"] - o["Low"]) / o["Close"].replace(0, np.nan)
    o["open_close_return"] = (o["Close"] - o["Open"]) / o["Open"].replace(0, np.nan)
    o["turnover_change"] = (o["Close"] * o["Volume"]).pct_change()
    o["price_acceleration"] = o["return"].diff()
    return o


def _pick_features(n_feat: int) -> list[str]:
    if n_feat == 14:
        return FEATURES_14
    elif n_feat == 10:
        return FEATURES_10
    elif n_feat == 5:
        return FEATURES_5
    return FEATURES_14[:n_feat]


def _build_seq(df, timesteps, n_feat):
    eng = _engineer(df)
    feats = _pick_features(n_feat)
    ff = eng[feats].copy()
    # Replace inf first, then forward-fill, then zero-fill any remaining NaN
    ff = ff.replace([np.inf, -np.inf], np.nan)
    ff = ff.ffill().bfill().fillna(0)
    if len(ff) < timesteps:
        raise ValueError(f"Need {timesteps} rows, got {len(ff)}")
    # Scale all features — models were trained on StandardScaler'd data
    sc = StandardScaler()
    ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
    seq = np.expand_dims(ff.tail(timesteps).to_numpy(np.float32), 0)
    return seq, eng


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


def _period_rows(period: str, interval: str) -> int:
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
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}[interval]
        return max(60, (trading_days * 390) // minutes)
    if interval in {"60m", "1h"}:
        return max(30, trading_days * 6)
    return max(30, trading_days)


def _generate_demo_data(period: str, interval: str, seed: int = 42) -> pd.DataFrame:
    rows = _period_rows(period, interval)
    freq = _demo_freq(interval)
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    ret = rng.normal(0, 0.004 if interval in {"60m", "1h"} else 0.01, rows)
    close = 2000.0 * np.cumprod(1 + ret)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.integers(500_000, 5_000_000, rows).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=ts,
    )


def _fetch(ticker, period="6mo", interval="1d", max_retries: int = 3):
    last_err = ""
    for attempt in range(max_retries):
        try:
            df = yf.Ticker(ticker).history(
                period=period,
                interval=interval,
                auto_adjust=False,
            )
            if df is not None and not df.empty:
                df = _flatten(df)
                df.index = pd.to_datetime(df.index)
                return df, False
            last_err = "Empty response"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))

    low = last_err.lower()
    is_rate_limited = (
        "rate limit" in low
        or "too many requests" in low
        or "yfratelimiterror" in low
    )
    if is_rate_limited:
        return _generate_demo_data(period, interval), True

    raise ValueError(f"No data for {ticker}. {last_err}")


def _risk_band(prob, threshold=0.20):
    if prob >= threshold:
        return "HIGH RISK"
    elif prob >= threshold * 0.65:
        return "ELEVATED"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    models = _discover()
    result = []
    for name in models:
        try:
            m = _get_model(name)
            ts, nf = _model_sig(m)
            result.append({"name": name, "timesteps": ts, "features": nf,
                           "params": int(m.count_params())})
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        ticker = data.get("ticker", "RELIANCE.NS")
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "6mo")
        interval = data.get("interval", "1d")
        threshold = float(data.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        df, is_demo = _fetch(ticker, period, interval)
        seq, eng = _build_seq(df, ts, nf)

        prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
        band = _risk_band(prob, threshold)

        # OHLC data for chart (last 60 bars)
        chart_df = _flatten(df).tail(60)
        ohlc = []
        for idx, row in chart_df.iterrows():
            ohlc.append({
                "date": str(idx)[:10],
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        return jsonify({
            "ticker": ticker,
            "model": model_name,
            "probability": round(prob, 6),
            "risk_pct": round(prob * 100, 2),
            "band": band,
            "threshold": threshold,
            "timesteps": ts,
            "features": nf,
            "ohlc": ohlc,
            "latest_close": round(float(df["Close"].iloc[-1]), 2),
            "latest_date": str(df.index[-1])[:10],
            "source": "demo" if is_demo else "live",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/portfolio", methods=["POST"])
def portfolio():
    try:
        data = request.json
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "6mo")
        interval = data.get("interval", "1d")
        threshold = float(data.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        results = []
        for t in tickers:
            try:
                df, is_demo = _fetch(t, period, interval)
                seq, _ = _build_seq(df, ts, nf)
                prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
                results.append({
                    "ticker": t,
                    "probability": round(prob, 6),
                    "risk_pct": round(prob * 100, 2),
                    "band": _risk_band(prob, threshold),
                    "latest_close": round(float(df["Close"].iloc[-1]), 2),
                    "source": "demo" if is_demo else "live",
                })
            except Exception as e:
                results.append({"ticker": t, "error": str(e)})

        results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return jsonify({"results": results, "model": model_name, "threshold": threshold})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ── Crash Replay definitions ─────────────────────────────────────────────────
CRASH_DATA_DIR = Path(__file__).resolve().parent.parent / "nifty 50 index minute data"

CRASH_EVENTS = [
    {"id": "nifty100_jun4", "label": "June 4, 2024 — Election Results (NIFTY 100)",
     "file": "NIFTY 100_minute.csv", "date": "2024-06-04",
     "start": "09:15", "end": "12:00", "crash_time": "10:35"},
    {"id": "infra_jun4", "label": "June 4, 2024 — Infra Shock (NIFTY INFRA)",
     "file": "NIFTY INFRA_minute.csv", "date": "2024-06-04",
     "start": "09:15", "end": "12:00", "crash_time": "10:33"},
    {"id": "auto_jun4", "label": "June 4, 2024 — Auto Selloff (NIFTY AUTO)",
     "file": "NIFTY AUTO_minute.csv", "date": "2024-06-04",
     "start": "10:00", "end": "13:00", "crash_time": "11:51"},
]


@app.route("/api/crash-events", methods=["GET"])
def list_crash_events():
    return jsonify(CRASH_EVENTS)


@app.route("/api/crash-replay", methods=["POST"])
def crash_replay():
    try:
        data = request.json
        event_id = data.get("event_id", CRASH_EVENTS[0]["id"])
        model_name = data.get("model", _discover()[0])

        event = next((e for e in CRASH_EVENTS if e["id"] == event_id), None)
        if not event:
            return jsonify({"error": f"Unknown crash event: {event_id}"}), 400

        csv_path = CRASH_DATA_DIR / event["file"]
        if not csv_path.exists():
            return jsonify({"error": f"Data file not found: {event['file']}"}), 404

        model = _get_model(model_name)
        ts, nf = _model_sig(model)
        feats = _pick_features(nf)

        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)
        df["Date"] = pd.to_datetime(df["Date"])

        start_dt = pd.to_datetime(f"{event['date']} {event['start']}")
        end_dt = pd.to_datetime(f"{event['date']} {event['end']}")
        window = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
        window = window.set_index("Date")

        eng = _engineer(window)
        ff = eng[feats].dropna()
        ff = ff.replace([np.inf, -np.inf], np.nan).fillna(0)

        sc = StandardScaler()
        ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
        arr = ff.to_numpy(np.float32)

        risk_points = []
        for i in range(ts, len(arr)):
            seq = np.expand_dims(arr[i - ts:i], 0)
            p = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
            risk_points.append({
                "date": str(ff.index[i - 1]),
                "risk": round(p * 100, 2),
            })

        # OHLC for chart
        ohlc = []
        for idx, row in window.iterrows():
            ohlc.append({
                "date": str(idx),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })

        return jsonify({
            "event": event,
            "model": model_name,
            "risk_points": risk_points,
            "ohlc": ohlc,
            "peak_risk": max(p["risk"] for p in risk_points) if risk_points else 0,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        model_name = request.form.get("model", _discover()[0])
        threshold = float(request.form.get("threshold", 0.20))

        model = _get_model(model_name)
        ts, nf = _model_sig(model)

        import io
        df = pd.read_csv(io.BytesIO(f.read()))
        df = _normalize_columns(df)

        seq, eng = _build_seq(df, ts, nf)
        prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
        band = _risk_band(prob, threshold)

        # Feature snapshot — use same NaN handling as _build_seq
        feats = _pick_features(nf)
        feat_df = eng[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        if len(feat_df) > 0:
            feat_vals = feat_df.tail(1).iloc[0].to_dict()
            feat_snapshot = {k: round(float(v), 6) for k, v in feat_vals.items()}
        else:
            feat_snapshot = {}

        return jsonify({
            "probability": round(prob, 6),
            "risk_pct": round(prob * 100, 2),
            "band": band,
            "threshold": threshold,
            "model": model_name,
            "rows_loaded": len(df),
            "features": feat_snapshot,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/timeline", methods=["POST"])
def timeline():
    try:
        data = request.json
        ticker = data.get("ticker", "RELIANCE.NS")
        model_name = data.get("model", _discover()[0])
        period = data.get("period", "1y")
        interval = data.get("interval", "1d")

        model = _get_model(model_name)
        ts, nf = _model_sig(model)
        feats = _pick_features(nf)

        df, is_demo = _fetch(ticker, period, interval)
        eng = _engineer(df)
        ff = eng[feats].dropna()
        ff = ff.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale all features — models were trained on StandardScaler'd data
        sc = StandardScaler()
        ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)

        arr = ff.to_numpy(np.float32)
        step = max(1, len(arr) // 200)  # ~200 points max

        points = []
        for i in range(ts, len(arr), step):
            seq = np.expand_dims(arr[i - ts:i], 0)
            p = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
            points.append({
                "date": str(ff.index[i - 1])[:10],
                "risk": round(p * 100, 2),
            })

        return jsonify({
            "ticker": ticker,
            "model": model_name,
            "points": points,
            "source": "demo" if is_demo else "live",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("=" * 60)
    print("  Flash Crash Prediction — API Server")
    print("=" * 60)
    print(f"  Models found: {_discover()}")
    print(f"  Frontend: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
