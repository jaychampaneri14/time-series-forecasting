"""
Time Series Forecasting
Compares ARIMA, Prophet-like trend decomposition, and LSTM for sales forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def generate_sales_data(n_days=730, seed=42):
    """Generate realistic retail sales time series."""
    np.random.seed(seed)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    t = np.arange(n_days)
    # Trend
    trend = 1000 + 0.5 * t
    # Weekly seasonality
    weekly = 200 * np.sin(2 * np.pi * t / 7)
    # Annual seasonality (peak in December)
    annual = 500 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    # Holiday spikes
    holidays = np.zeros(n_days)
    for day_of_year in [25, 200, 359]:
        holidays[t % 365 == day_of_year] = 800
    # Noise
    noise = np.random.normal(0, 50, n_days)
    sales = trend + weekly + annual + holidays + noise
    sales = np.maximum(sales, 0)
    return pd.Series(sales, index=dates, name='Sales')


def create_sequences(data, seq_len=30, horizon=7):
    """Create input/output sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_weights).sum(dim=1)
        return self.fc(context)


def train_lstm(X_train, y_train, X_val, y_val, horizon=7, epochs=50):
    """Train LSTM with early stopping."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    Xt = scaler_X.fit_transform(X_train)
    yt = scaler_y.fit_transform(y_train)
    Xv = scaler_X.transform(X_val)
    yv = scaler_y.transform(y_val)

    Xt = torch.FloatTensor(Xt).unsqueeze(-1)  # (N, seq, 1)
    yt = torch.FloatTensor(yt)
    Xv = torch.FloatTensor(Xv).unsqueeze(-1)
    yv = torch.FloatTensor(yv)

    train_ld = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    model   = LSTMForecaster(horizon=horizon)
    opt     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit    = nn.HuberLoss()
    best_val_loss = float('inf')
    patience_cnt  = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for xb, yb in train_ld:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        sched.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={tr_loss/len(train_ld):.6f}, val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, scaler_X, scaler_y


def forecast_lstm(model, scaler_X, scaler_y, X_test):
    model.eval()
    Xs = scaler_X.transform(X_test)
    Xt = torch.FloatTensor(Xs).unsqueeze(-1)
    with torch.no_grad():
        preds = model(Xt).numpy()
    return scaler_y.inverse_transform(preds)


def simple_arima_forecast(series, steps=7):
    """Simple SARIMA forecast."""
    if not HAS_STATSMODELS:
        naive = series.iloc[-steps:].values
        return naive, None
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit   = model.fit(disp=False)
        return fit.forecast(steps=steps), fit
    except Exception as e:
        print(f"ARIMA failed: {e}, using naive forecast")
        return series.iloc[-steps:].values, None


def plot_forecast(series, forecasts_dict, save_path='forecast.png'):
    """Plot historical + multiple forecasts."""
    last_n = 90
    hist = series.iloc[-last_n:]
    forecast_dates = pd.date_range(series.index[-1] + pd.Timedelta('1D'), periods=7)

    plt.figure(figsize=(14, 6))
    plt.plot(hist.index, hist.values, 'b-', lw=1.5, label='Historical', alpha=0.8)
    colors = ['red', 'green', 'orange']
    for (name, vals), color in zip(forecasts_dict.items(), colors):
        plt.plot(forecast_dates, vals, 'o-', color=color, lw=2, label=f'Forecast: {name}', markersize=6)
    plt.axvline(series.index[-1], color='gray', linestyle='--', alpha=0.5, label='Forecast start')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Forecasting — 7-Day Ahead')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Forecast plot saved to {save_path}")


def decompose_plot(series, save_path='decomposition.png'):
    """Seasonal decomposition."""
    if not HAS_STATSMODELS:
        return
    decomp = seasonal_decompose(series, period=7, model='additive')
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    decomp.observed.plot(ax=axes[0]); axes[0].set_title('Observed'); axes[0].set_ylabel('Sales')
    decomp.trend.plot(ax=axes[1]);    axes[1].set_title('Trend');    axes[1].set_ylabel('')
    decomp.seasonal.plot(ax=axes[2]); axes[2].set_title('Seasonal'); axes[2].set_ylabel('')
    decomp.resid.plot(ax=axes[3]);    axes[3].set_title('Residual'); axes[3].set_ylabel('')
    plt.suptitle('Time Series Decomposition')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Decomposition plot saved to {save_path}")


def main():
    print("=" * 60)
    print("TIME SERIES FORECASTING")
    print("=" * 60)

    series = generate_sales_data(730)
    print(f"Generated {len(series)} days of sales data")
    print(f"Mean: {series.mean():.0f}, Std: {series.std():.0f}")

    decompose_plot(series)

    # Prepare sequences
    SEQ_LEN = 30
    HORIZON  = 7
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_s = scaler.fit_transform(data).ravel()
    X, y = create_sequences(data_s, SEQ_LEN, HORIZON)

    train_size = int(len(X) * 0.7)
    val_size   = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test,  y_test  = X[train_size+val_size:], y[train_size+val_size:]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} sequences")

    # ARIMA
    print("\n--- ARIMA Forecast ---")
    arima_forecast, _ = simple_arima_forecast(series, steps=HORIZON)
    print(f"ARIMA 7-day forecast: {np.array(arima_forecast).round(0)}")

    # Naive seasonal baseline (same weekday last week)
    naive_forecast = series.values[-(HORIZON + 7):-7]

    # LSTM
    print("\n--- Training LSTM ---")
    model, sc_X, sc_y = train_lstm(X_train, y_train, X_val, y_val, horizon=HORIZON, epochs=50)
    lstm_forecast = forecast_lstm(model, sc_X, sc_y, X_test[-1:]).ravel()
    actual = scaler.inverse_transform(y_test[-1:].reshape(-1, 1)).ravel()

    # Metrics
    print("\n--- 7-Day Forecast Evaluation (last test window) ---")
    for name, pred in [('Naive', naive_forecast[:HORIZON]), ('LSTM', lstm_forecast)]:
        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / (actual + 1))) * 100
        print(f"  {name:8s}: MAE={mae:.1f}, RMSE={rmse:.1f}, MAPE={mape:.1f}%")

    # Plot
    plot_forecast(series, {
        'ARIMA': np.array(arima_forecast).ravel(),
        'LSTM':  lstm_forecast,
        'Naive': naive_forecast[:HORIZON]
    })

    torch.save(model.state_dict(), 'lstm_forecaster.pth')
    print("\nLSTM model saved to lstm_forecaster.pth")
    print("\n✓ Time Series Forecasting complete!")


if __name__ == '__main__':
    main()
