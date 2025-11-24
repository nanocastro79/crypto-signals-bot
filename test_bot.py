import requests
import json

BOT_TOKEN = "TU_BOT_TOKEN_AQUI"
CHAT_ID = "TU_CHAT_ID_AQUI"

def get_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    r = requests.get(url)
    data = r.json()
    return float(data["price"])

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    r = requests.post(url, json=payload)
    return r.status_code, r.text

if __name__ == "__main__":
    try:
        price = get_btc_price()
        message = f"Precio actual de BTC: {price:.2f} USD"
        status, resp = send_telegram_message(message)
        print("Mensaje enviado:", status, resp)
    except Exception as e:
        print("Error:", e)
