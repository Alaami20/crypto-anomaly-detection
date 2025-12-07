import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

X = np.load("data/X.npy")
model = load_model("results/autoencoder_model.h5")

train_size = int(0.7 * len(X))
X_test = X[train_size:]

X_pred = model.predict(X_test)
mse = np.mean(np.power(X_test - X_pred, 2), axis=(1,2))
threshold = mse.mean() + 3 * mse.std()

plt.figure(figsize=(15,5))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.savefig("results/error_plot.png")
print("Evaluation complete. Threshold:", threshold)
