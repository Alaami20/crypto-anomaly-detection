import numpy as np
import tensorflow as tf
from model import build_autoencoder

X = np.load("data/X.npy")
train_size = int(0.7 * len(X))
X_train = X[:train_size]

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

autoencoder = build_autoencoder(timesteps, n_features)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

import os
os.makedirs("results", exist_ok=True)
autoencoder.save("results/autoencoder_model.h5")
np.save("results/history.npy", history.history)
print("Training complete.")
