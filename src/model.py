import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model

def build_autoencoder(timesteps, n_features, latent_dim=64):
    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(latent_dim, activation='relu')(inputs)
    encoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(encoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
