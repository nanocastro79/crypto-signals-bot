import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import requests

# ========================
# CONFIG TELEGRAM
# ========================
import os
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ========================
# FUNCIONES
# ========================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def hurst_est(x):
    return np.log(x.max() - x.min() + 1e-8)

def process_symbol(symbol):

    df = yf.download(symbol, period="5y", interval="1d", interval="1d", auto_adjust=True)
    df = df.dropna().copy()

    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # Features
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_5d"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_5"] = df["return_1d"].rolling(5).std()
    df["rsi_14"] = compute_rsi(df["close"], 14)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["volume_norm"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_20"] = df["return_1d"].rolling(20).std()

    vol_threshold = df["vol_20"].median()
    df["highvol_regime"] = (df["vol_20"] > vol_threshold).astype(int)
    df["hurst"] = df["close"].rolling(20).apply(hurst_est)

    df["target_5d"] = (df["close"].shift(-5) > df["close"]).astype(int)
    df = df.dropna().copy()

    features = [
        "return_1d", "return_2d", "return_5d",
        "ma_5", "ma_20", "vol_5",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_mid", "bb_upper", "bb_lower",
        "atr_14",
        "volume_norm", "vol_20", "highvol_regime", "hurst"
    ]

    X = df[features]
    y = df["target_5d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    last_X = X.iloc[[-1]]
    last_X_scaled = scaler.transform(last_X)
    prob = model.predict_proba(last_X_scaled)[0, 1]

    return float(prob), df["close"].iloc[-1]


# ========================
#  EXEC MULTI-CRYPTO
# ========================

symbols = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD"
}

results = {}
TH = 0.65

for name, ticker in symbols.items():
    prob, price = process_symbol(ticker)
    if prob > TH:
        signal = "LONG"
    elif prob < 1 - TH:
        signal = "SHORT"
    else:
        signal = "NEUTRO"
    results[name] = (prob, signal)

# ========================
# ENVÍO A TELEGRAM
# ========================

msg = "SEÑALES DIARIAS – IA 5 DÍAS\n\n"

for name, (prob, signal) in results.items():
    msg += f"{name}: {signal} ({round(prob,3)})\n"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
requests.get(url, params={"chat_id": CHAT_ID, "text": msg})

print("Enviado:", msg)
