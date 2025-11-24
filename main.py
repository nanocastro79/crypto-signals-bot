import os
import requests
import pandas as pd
from signals import generate_signals

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --------------------------
# Descargar datos desde Yahoo Finance
# --------------------------
def get_market_data(symbol, days=5):
    """
    Descargar datos reales (OHLC) desde Yahoo Finance.
    Este proveedor es muy estable y funciona perfecto en GitHub Actions.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}-USD?range={days}d&interval=1h"

        r = requests.get(url)
        data = r.json()

        result = data["chart"]["result"][0]

        timestamps = result["timestamp"]
        indicators = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="s"),
            "open": indicators["open"],
            "high": indicators["high"],
            "low": indicators["low"],
            "close": indicators["close"],
            "volume": indicators["volume"]
        })

        df = df.dropna()
        return df

    except Exception as e:
        raise RuntimeError(f"Yahoo Finance Error for {symbol}: {e}")

# --------------------------
# Enviar mensaje a Telegram
# --------------------------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    requests.post(url, json=payload)

# --------------------------
# Lista de criptos a analizar
# --------------------------
CRYPTO_LIST = {
    "BTC": "BTC",
    "ETH": "ETH",
    "BNB": "BNB",
    "SOL": "SOL",
    "ADA": "ADA",
    "XRP": "XRP"
}

# --------------------------
# Flujo principal
# --------------------------
def run_bot():

    print("=== Iniciando bot ===")

    results = []

    for symbol, yahoo_symbol in CRYPTO_LIST.items():
        print(f"Descargando datos para {symbol}...")
        try:
            df = get_market_data(yahoo_symbol, days=5)
            print(f"Datos obtenidos para {symbol}: {df.shape}")
            signal, prob = generate_signals(df)
            results.append((symbol, signal, prob))
        except Exception as e:
            print(f"ERROR en {symbol}: {e}")
            results.append((symbol, "ERROR", str(e)))

    print("Generando mensaje final...")

    message = "SEÑALES DIARIAS – IA\n\n"

    for symbol, signal, prob in results:
        if signal == "ERROR":
            message += f"{symbol}: ERROR ({prob})\n"
        else:
            message += f"{symbol}: {signal} ({prob:.3f})\n"

    print("Enviando mensaje a Telegram...")
    send_telegram(message)

    print("Reporte enviado correctamente.")
    print(message)
