# 📈 Anomaly Detection in Bitcoin Time-Series Using LSTM Autoencoders
*A Research-Style Machine Learning Project by **Alaa Miari***

Cryptocurrency markets often behave unpredictably — crashes, pumps, abnormal volatility, and manipulation.  
This project builds a **deep learning anomaly detection system** to identify such irregular events in **Bitcoin price data** using an **LSTM Autoencoder**.

This project follows a professional, research-style ML pipeline similar to work done at **Google Research, Amazon ML, PayPal Risk, NVIDIA**, and other top companies.

---

## 🧠 Project Highlights
- Detect anomalies in BTC-USD time series  
- Deep learning using **LSTM Autoencoders**  
- Automatic anomaly threshold detection  
- End-to-end ML pipeline  
- Includes research experiments  
- Google-level project structure  

---

## 🗂 Project Structure
crypto-anomaly-detection/
│
├── data/
│ ├── X.npy
│ └── README.md
│
├── notebooks/
│ ├── 01_download_and_preprocess.ipynb
│ ├── 02_train_autoencoder.ipynb
│ ├── 03_threshold_selection.ipynb
│ ├── 04_experiments.ipynb
│ └── 05_visualizations.ipynb
│
├── src/
│ ├── model.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── results/
│ ├── autoencoder_model.h5 OR autoencoder.pt
│ ├── history.npy
│ ├── error_plot.png
│ └── README.md
│
├── requirements.txt
└── README.md



Creates:
- `results/error_plot.png`  
- anomaly threshold  
- anomaly detection summary  

---

## 🔬 Experiments Included
- Latent dimension: 16, 32, 64, 128  
- Dropout variations  
- Attention-based Autoencoder  
- GRU Autoencoder  
- 1D CNN Autoencoder  

These experiments replicate research-paper methodology.

---

## 📚 Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow **or** PyTorch  
- Matplotlib  
- Seaborn  
- yfinance  
- Jupyter Notebook  

---

## 👤 Author

**Alaa Miari**  
B.Sc. Data Science & Computer Science  
University of Haifa  

GitHub: **@Alaami20**  

---

## ⭐ Why This Project Stands Out
- Real-world time-series anomaly detection  
- Deep learning model architecture  
- Clean code structure  
---

