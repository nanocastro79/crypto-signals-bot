import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Feature Engineering
# ---------------------------
def build_features(df):
    df = df.copy()

    # Retorno logarítmico
    df["return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatilidad
    df["volatility"] = df["return"].rolling(10).std()

    # Momentum
    df["roc"] = df["close"].pct_change(5)

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(14).mean()
    avg_loss = down.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMAs
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    df["ema9_21"] = df["ema9"] - df["ema21"]
    df["ema21_50"] = df["ema21"] - df["ema50"]

    df = df.dropna()
    return df

# ---------------------------
# 2. Label Creation
# ---------------------------
def build_labels(df):
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    def labeler(x):
        if x > 0.003:   # +0.3%
            return 1    # LONG
        if x < -0.003:  # -0.3%
            return -1   # SHORT
        return 0        # NEUTRO

    df["label"] = df["future_return"].apply(labeler)
    df = df.dropna()
    return df

# ---------------------------
# 3. Train & Predict
# ---------------------------
def model_predict(df):
    features = ["return", "volatility", "roc", "rsi", "ema9_21", "ema21_50"]
    X = df[features].values
    y = df["label"].values

    # Entrenar modelo simple
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    model.fit(X[:-1], y[:-1])

    # Predicción 1 paso adelante
    pred = model.predict([X[-1]])[0]
    prob = max(model.predict_proba([X[-1]])[0])
    return pred, prob

# ---------------------------
# 4. Wrapper final para el bot
# ---------------------------
def generate_signals(df):
    df = build_features(df)
    df = build_labels(df)

    signal, prob = model_predict(df)

    mapping = {
        1: "LONG",
        0: "NEUTRO",
        -1: "SHORT"
    }

    return mapping[signal], float(prob)
