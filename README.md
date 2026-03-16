# Vitality Leisure V2

A Streamlit web application that predicts daily visitor numbers for a thermal spa facility using machine learning and live weather data.

## What it does

**Manager Dashboard** — 7-day visitor forecast with capacity tracking, daily breakdown, monthly outlook, historical heatmap, and year-over-year trends.

**Wellness Coach** — conversational AI chatbot powered by Cohere that takes visitor input (how they feel, what they need) and generates a personalised spa day plan grounded in the live crowd forecast and weather data.

**Plan My Visit** — filter-based recommendation tool that ranks the best days to visit based on crowd preference, weather, and day of week.

## What's new in v2

- NRW public holidays and school holidays integrated as model features and displayed in the forecast
- Numeric weather features: actual temperature (°C) and precipitation (mm) from Open-Meteo historical API
- Daily capacity usage shown as % of maximum (2,400 visitors)
- Wellness Coach: multi-turn LLM chatbot using Cohere, grounded in the 7-day ML forecast

## Model

Gradient Boosting Regressor trained on 6,182 daily records (2008–2025).
CV MAE: ~211 visitors/day · R²: 0.774

## Setup

Add your Cohere API key to Streamlit secrets:
```
COHERE_API_KEY = "your-key-here"
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `train_model.py` | Model training script (run locally) |
| `fetch_weather_history.py` | Fetches historical weather from Open-Meteo (run locally) |
| `model.joblib` | Trained model |
| `data.xlsx` | Anonymised historical visitor data |

*Data anonymised: facility renamed Vitality Leisure Park, visitor numbers adjusted.*
