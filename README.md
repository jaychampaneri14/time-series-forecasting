# Time Series Forecasting

7-day ahead sales forecasting comparing ARIMA, LSTM with attention, and naive seasonal baseline.

## Features
- Realistic synthetic sales data with trend, weekly/annual seasonality, and holiday spikes
- Seasonal decomposition (trend / seasonal / residual)
- SARIMA model via statsmodels
- LSTM with attention mechanism and early stopping
- MAE, RMSE, MAPE evaluation

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `decomposition.png` — seasonal decomposition plot
- `forecast.png` — 7-day forecast comparison
- `lstm_forecaster.pth` — saved LSTM weights
