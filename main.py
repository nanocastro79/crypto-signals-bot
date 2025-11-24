import os
import requests
import pandas as pd
from signals import generate_signals

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --------------------------
# Descargar datos de CoinGecko
# --------------------------
def get_market_data(symbol, days=5):
    """
    Descarga precios OHLC de CoinGecko para un símbolo.
    CoinGecko no provee OHLC perfecto para todas las cryptos,
    así que usamos el precio de cierre diario.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"

    r = requests.get(url)
    data = r.json()

    # CoinGecko da precios en timestamps
    prices = data["prices"]  # [ [timestamp, price], ... ]

    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["close"].shift(1)
    df["high"] = df["close"].rolling(2).max()
    df["low"] = df["close"].rolling(2).min()
    df["volume"] = 0.0  # dummy, porque no lo necesitamos

    df = df.dropna()
    return df

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
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple"
}

# --------------------------
# Flujo principal
# --------------------------
def run_bot():

    results = []

    for symbol, coingecko_id in CRYPTO_LIST.items():
        try:
            df = get_market_data(coingecko_id, days=10)
            signal, prob = generate_signals(df)
            results.append((symbol, signal, prob))
        except Exception as e:
            results.append((symbol, "ERROR", str(e)))

    # Armar mensaje final
    message = "SEÑALES DIARIAS – IA\n\n"

    for symbol, signal, prob in results:
        message += f"{symbol}: {signal} ({prob:.3f})\n"

    # Enviar a Telegram
    send_telegram(message)

    print("Reporte enviado correctamente.")
    print(message)

# --------------------------
# Ejecutar si es script principal
# --------------------------
if __name__ == "__main__":
    run_bot()
