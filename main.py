import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import requests
import os

# ========================
# CONFIG TELEGRAM
# ========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ========================
# CONFIG ESTRATEGIA
# ========================
CAPITAL_TOTAL = 1000.0       # USD considerados por trade
RIESGO_POR_TRADE = 0.02      # 2% de riesgo
STOP_LOSS_PCT = 0.08         # 8% de SL
TAKE_PROFIT_PCT = 0.15       # 15% de TP
TH = 0.65                    # Threshold de probabilidad

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
    # Descarga datos
    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=True)
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

    # Split temporal 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic mejorado (balancea clases)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Señal para el último día
    last_X = X.iloc[[-1]]
    last_X_scaled = scaler.transform(last_X)
    prob = model.predict_proba(last_X_scaled)[0, 1]

    last_close = float(df["close"].iloc[-1])
    last_date = df.index[-1]

    return float(prob), last_close, last_date

# ========================
#  EXEC MULTI-CRYPTO
# ========================

symbols = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
}

results = {}
fecha_ref = None

riesgo_usd = CAPITAL_TOTAL * RIESGO_POR_TRADE
tamaño_usd = riesgo_usd / STOP_LOSS_PCT  # posición total sugerida por operación

def compute_strength(signal, prob):
    """
    Devuelve (score, etiqueta_fuerza) en función de la señal y probabilidad.
    score se usa para el ranking.
    """
    if signal == "LONG":
        score = prob
    elif signal == "SHORT":
        score = 1.0 - prob
    else:
        return 0.0, "NEUTRO"

    # Etiqueta de fuerza
    if score >= 0.80:
        label = "FUERTE"
    elif score >= 0.70:
        label = "MEDIA"
    elif score >= TH:  # 0.65 <= score < 0.70
        label = "DÉBIL"
    else:
        label = "NEUTRO"
    return score, label

for name, ticker in symbols.items():
    prob, price, last_date = process_symbol(ticker)
    if fecha_ref is None:
        fecha_ref = last_date

    if prob > TH:
        signal = "LONG"
        sl = price * (1 - STOP_LOSS_PCT)
        tp = price * (1 + TAKE_PROFIT_PCT)
    elif prob < 1 - TH:
        signal = "SHORT"
        sl = price * (1 + STOP_LOSS_PCT)
        tp = price * (1 - TAKE_PROFIT_PCT)
    else:
        signal = "NEUTRO"
        sl = None
        tp = None

    unidades = tamaño_usd / price if signal != "NEUTRO" else 0.0
    score, fuerza = compute_strength(signal, prob)

    results[name] = {
        "prob": prob,
        "signal": signal,
        "price": price,
        "sl": sl,
        "tp": tp,
        "size_usd": tamaño_usd,
        "units": unidades,
        "score": score,
        "fuerza": fuerza,
    }

# ========================
# ARMAR MENSAJE
# ========================

lineas = []

# Cabecera
if fecha_ref is not None:
    lineas.append(f"SEÑALES DIARIAS – IA 5 DÍAS\nFecha: {fecha_ref.date()}\n")
else:
    lineas.append("SEÑALES DIARIAS – IA 5 DÍAS\n\n")

# Resumen por cripto (con fuerza)
for name, data in results.items():
    lineas.append(
        f"{name}: {data['signal']} ({data['prob']:.3f}) "
        f"[fuerza: {data['fuerza']}]"
    )

lineas.append(
    f"\nCapital por trade considerado: {CAPITAL_TOTAL:.2f} USD | "
    f"Riesgo: {RIESGO_POR_TRADE*100:.1f}%"
)

# Ranking de oportunidades (solo LONG/SHORT, ordenadas por score)
operables = [
    (name, data) for name, data in results.items()
    if data["signal"] != "NEUTRO" and data["score"] > 0
]
operables_sorted = sorted(operables, key=lambda x: x[1]["score"], reverse=True)

if operables_sorted:
    lineas.append("\nTOP OPORTUNIDADES HOY:")
    for name, data in operables_sorted:
        lineas.append(
            f"- {name}: {data['signal']} "
            f"({data['prob']:.3f}) [fuerza: {data['fuerza']}]"
        )
else:
    lineas.append("\nTOP OPORTUNIDADES HOY: ninguna (todo en zona neutra)")

# Detalle solo para señales operables
for name, data in operables_sorted:
    lineas.append(
        f"\n{name} – Detalle:\n"
        f"Tamaño sugerido: {data['size_usd']:.2f} USD "
        f"(~{data['units']:.6f} {name})\n"
        f"Precio actual: {data['price']:.2f}\n"
        f"Stop-Loss: {data['sl']:.2f}\n"
        f"Take-Profit: {data['tp']:.2f}"
    )

mensaje = "\n".join(lineas)

print("Mensaje a enviar:")
print(mensaje)

# ========================
# GUARDAR HISTORIAL EN CSV (EXTENDIDO)
# ========================
import csv
import os

history_file = "signals_history.csv"
fieldnames = [
    "date", "symbol", "prob", "signal",
    "strength", "price", "score",
    "result", "forward_ret_5", "correct"
]

if fecha_ref is not None:
    fecha_str = str(fecha_ref.date())
else:
    from datetime import datetime
    fecha_str = datetime.utcnow().date().isoformat()

mode = "a"
file_exists = os.path.exists(history_file)

with open(history_file, mode, newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for name, data in results.items():
        writer.writerow({
            "date": fecha_str,
            "symbol": name,
            "prob": round(data["prob"], 6),
            "signal": data["signal"],
            "strength": data["fuerza"],
            "price": round(data["price"], 6),
            "score": round(data["score"], 6),
            "result": "",
            "forward_ret_5": "",
            "correct": ""
        })

# ========================
# ENVÍO A TELEGRAM
# ========================

if TELEGRAM_TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": mensaje})
    print("Enviado a Telegram.")
    # ================================
# 1B — COMPLETAR HISTORIAL (5 días después)
# ================================
import pandas as pd
import csv
import os
from datetime import datetime, timedelta

history_file = "signals_history.csv"

if os.path.exists(history_file):
    hist = pd.read_csv(history_file)

    # Convertir fecha
    hist["date"] = pd.to_datetime(hist["date"])

    # Procesar solo señales antiguas sin completar
    pending = hist[hist["result"].isna() | (hist["result"] == "")].copy()

    if not pending.empty:
        print(f"Completando {len(pending)} señales anteriores...")

        updated_rows = []

        for _, row in pending.iterrows():
            sym = row["symbol"]
            start_date = row["date"]
            end_date = start_date + timedelta(days=5)

            ticker_map = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "SOL": "SOL-USD",
                "BNB": "BNB-USD",
                "XRP": "XRP-USD",
                "ADA": "ADA-USD"
            }

            yf_symbol = ticker_map[sym]

            # Descargar precios desde la fecha original hasta 10 días más
            df_test = yf.download(
                yf_symbol,
                start=start_date,
                end=end_date + timedelta(days=2),
                interval="1d",
                auto_adjust=True
            )

            if len(df_test) < 2:
                continue

            try:
                price_start = df_test["Close"].iloc[0]
                price_end = df_test["Close"].iloc[-1]
            except Exception:
                continue

            forward_ret = (price_end / price_start) - 1

            # Determinar si la señal fue correcta
            if row["signal"] == "LONG":
                correct = forward_ret > 0
            elif row["signal"] == "SHORT":
                correct = forward_ret < 0
            else:
                correct = None

            updated_rows.append({
                "index": row.name,
                "forward_ret_5": forward_ret,
                "correct": correct,
                "result": "WIN" if correct else "LOSS"
            })

        # Actualizar el CSV
        if updated_rows:
            for upd in updated_rows:
                idx = upd["index"]
                hist.loc[idx, "forward_ret_5"] = upd["forward_ret_5"]
                hist.loc[idx, "correct"] = upd["correct"]
                hist.loc[idx, "result"] = upd["result"]

            hist.to_csv(history_file, index=False)
            print("Historial actualizado con éxito.")
        else:
            print("No había señales pendientes para completar.")

else:
    print("Faltan TELEGRAM_TOKEN o CHAT_ID en los secretos.")
