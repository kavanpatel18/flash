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

import json, sys, traceback
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


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    o = _flatten(df.copy())
    o["return"] = np.log(o["Close"] / o["Close"].shift(1))
    o["log_return"] = o["return"]
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
    ff = eng[feats].dropna()
    if len(ff) < timesteps:
        raise ValueError(f"Need {timesteps} rows, got {len(ff)}")
    # Replace infinity values and fill remaining NaNs
    ff = ff.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Scale all features — models were trained on StandardScaler'd data
    sc = StandardScaler()
    ff = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
    seq = np.expand_dims(ff.tail(timesteps).to_numpy(np.float32), 0)
    return seq, eng


def _fetch(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = _flatten(df)
    df.index = pd.to_datetime(df.index)
    return df


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

        df = _fetch(ticker, period, interval)
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
                df = _fetch(t, period, interval)
                seq, _ = _build_seq(df, ts, nf)
                prob = float(np.clip(model.predict(seq, verbose=0).ravel()[0], 0, 1))
                results.append({
                    "ticker": t,
                    "probability": round(prob, 6),
                    "risk_pct": round(prob * 100, 2),
                    "band": _risk_band(prob, threshold),
                    "latest_close": round(float(df["Close"].iloc[-1]), 2),
                })
            except Exception as e:
                results.append({"ticker": t, "error": str(e)})

        results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return jsonify({"results": results, "model": model_name, "threshold": threshold})
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

        df = _fetch(ticker, period, interval)
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

        return jsonify({"ticker": ticker, "model": model_name, "points": points})
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
