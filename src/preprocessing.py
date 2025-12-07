import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_btc_data():
    btc = yf.download("BTC-USD", start="2018-01-01", end="2024-01-01")
    btc = btc[['Close', 'Volume']]
    btc['Returns'] = np.log(btc['Close'] / btc['Close'].shift(1))
    btc = btc.dropna()
    return btc

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_sequences(data, seq_len=30):
    sequences = []
    for i in range(seq_len, len(data)):
        sequences.append(data[i-seq_len:i])
    return np.array(sequences)

if __name__ == "__main__":
    df = load_btc_data()
    scaled, scaler = scale_data(df)
    X = create_sequences(scaled, seq_len=30)
    os.makedirs("data", exist_ok=True)
    np.save("data/X.npy", X)
    print("Saved data/X.npy")
