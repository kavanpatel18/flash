---
title: GRU Flash Crash Risk App
emoji: 📉
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GRU Flash Crash Risk App

Streamlit application for early flash-crash risk prediction using GRU/LSTM time-series models.

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch app:

```bash
streamlit run dashboard.py
```

## Streamlit Community Cloud Deployment

1. Push this `github/` folder to a GitHub repository root.
2. Ensure these files stay in repo root:
   - `dashboard.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
   - your model file (for example `flash_crash_gru_model.keras`)
3. In Streamlit Community Cloud:
   - Click **New app**
   - Select your repo and branch
   - Set main file path to `dashboard.py`
   - Deploy

## Notes

- App auto-detects model input shape and supports 30x5 and 20x14 feature pipelines.
- For live mode, use symbols like `RELIANCE.NS`, `TCS.NS`, `INFY.NS`.
- If startup fails due to model loading, verify that at least one model artifact exists in the repo root.
