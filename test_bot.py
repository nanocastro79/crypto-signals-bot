import os
import requests

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def get_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    r = requests.get(url)
    data = r.json()
    return data["bitcoin"]["usd"]

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
        message = f"Precio actual de BTC (CoinGecko): {price:.2f} USD"
        status, resp = send_telegram_message(message)
        print("Mensaje enviado:", status, resp)
    except Exception as e:
        print("Error:", e)
