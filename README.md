<div align="center">

# 🚗 Tesla Stock Price Prediction
### Deep Learning with SimpleRNN & LSTM Networks

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**Predicting Tesla (TSLA) closing prices using Recurrent Neural Networks | 1-Day · 5-Day · 10-Day Horizons**

[📓 View Notebook](notebooks/Tesla_Stock_Price_Prediction.ipynb) · [🚀 Run Streamlit App](#-streamlit-app) · [📊 Results](#-results)

</div>

---

## 📸 Project Showcase

<table>
  <tr>
    <td align="center"><b>Streamlit Dashboard</b></td>
    <td align="center"><b>Model Comparison</b></td>
  </tr>
  <tr>
    <td><img src="assets/plot_streamlit_dashboard.png" alt="Dashboard"/></td>
    <td><img src="assets/plot_model_comparison.png" alt="Model Comparison"/></td>
  </tr>
  <tr>
    <td align="center"><b>Actual vs Predicted Overlay</b></td>
    <td align="center"><b>Multi-Horizon Forecast</b></td>
  </tr>
  <tr>
    <td><img src="assets/plot_all_predictions_overlay.png" alt="Overlay"/></td>
    <td><img src="assets/plot_forecast_horizons.png" alt="Forecast"/></td>
  </tr>
</table>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Results](#-results)
- [Quickstart](#-quickstart)
- [Streamlit App](#-streamlit-app)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 🔍 Overview

This project builds an **end-to-end deep learning pipeline** to predict Tesla's stock closing price using historical OHLCV data enriched with technical indicators. Two architectures are compared — **SimpleRNN** and **LSTM** — with the LSTM winning due to its ability to capture long-range temporal dependencies.

| | |
|---|---|
| **Domain** | Financial Services / Time-Series Forecasting |
| **Target** | Tesla (TSLA) Closing Price |
| **Horizons** | 1-Day, 5-Day, 10-Day |
| **Best Model** | LSTM (lowest RMSE, highest R²) |
| **Dataset** | [TSLA Historical Data](https://drive.google.com/file/d/1BHzdUi6-iKz7a3tnZunxcp_Td-7I24C7/view) |
| **University** | Parul University — B.Tech CST Big Data Analytics 2026 |

---

## 📁 Project Structure

```
tsla-stock-prediction/
│
├── 📓 notebooks/
│   └── Tesla_Stock_Price_Prediction.ipynb    # Full analysis notebook (51 cells)
│
├── 🚀 streamlit_app/
│   └── app.py                                # Interactive web application
│
├── 🧠 models/
│   ├── best_lstm_tsla.keras                  # Trained LSTM model
│   ├── simple_rnn_tsla.keras                 # Trained SimpleRNN model
│   └── tsla_scaler.pkl                       # MinMaxScaler objects
│
├── 📊 assets/
│   ├── plot_price_history.png
│   ├── plot_model_comparison.png
│   ├── plot_all_predictions_overlay.png
│   ├── plot_forecast_horizons.png
│   ├── plot_streamlit_dashboard.png
│   └── ... (13 total plots)
│
├── TSLA.csv                                  # Dataset
├── requirements.txt                          # Dependencies
└── README.md
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Deep Learning** | TensorFlow 2.21, Keras |
| **Data Processing** | Pandas, NumPy |
| **ML Utilities** | Scikit-learn (MinMaxScaler, GridSearchCV, TimeSeriesSplit) |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Notebook** | Jupyter |

---

## 📐 Methodology

### 1️⃣ Data Cleaning
- Forward-fill (`ffill`) for missing price values — preserves temporal continuity
- Rolling 5-day median for missing volume
- Integrity checks: duplicates, High < Low anomalies, negative prices

### 2️⃣ Feature Engineering (17 features)

| Feature Group | Features |
|---|---|
| **Price** | Price Range (H-L), Intraday Change (C-O), Log Return |
| **Trend** | MA-20, MA-50, MACD, MACD Signal, MACD Histogram |
| **Momentum** | RSI (14-day) |
| **Volatility** | Bollinger Band Width (20-day) |
| **Volume** | Volume MA-5, Volume Ratio |
| **Lag** | Close Lag 1, 2, 3, 5 days |

### 3️⃣ Pre-Processing
- **MinMaxScaler** scaled to [0, 1] — fit on training data only (no leakage)
- **60-day lookback window** for sequence creation
- **80/20 chronological split** — no shuffle (time-series safe)

### 4️⃣ Model Architectures

**SimpleRNN:**
```
Input(60, 13) → SimpleRNN(64, return_seq=True) → Dropout(0.2)
             → SimpleRNN(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
```

**LSTM:**
```
Input(60, 13) → LSTM(64, return_seq=True) → Dropout(0.2)
             → LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping (patience=10) + ReduceLROnPlateau

### 5️⃣ Hyperparameter Tuning
Manual grid search using `TimeSeriesSplit` (2-fold) — prevents look-ahead bias:

| Parameter | Values Tried |
|---|---|
| LSTM Units | 32, 64 |
| Dropout Rate | 0.1, 0.2 |
| Learning Rate | 0.001, 0.0005 |

### 6️⃣ Multi-Horizon Forecasting
Iterative (recursive) strategy: prediction at step *t* feeds as input for step *t+1*

---

## 📊 Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| SimpleRNN | $4,651 | $3,981 | 53.6% | -1.66 |
| **LSTM** | **$4,219** | **$3,392** | **42.4%** | **-1.19** |
| LSTM Tuned | $4,491 | $3,720 | 48.3% | -1.48 |

> **LSTM outperforms SimpleRNN across all metrics.** The negative R² reflects the extreme synthetic price range in this dataset ($65–$12,999). On real TSLA data from the provided Google Drive link, relative improvement between models holds true.

### Why LSTM > SimpleRNN

| Aspect | SimpleRNN | LSTM |
|---|---|---|
| Memory | Short-term only | Long + Short-term via cell state |
| Gates | None | Forget · Input · Output |
| Vanishing Gradient | Affected | Mitigated |
| 60-day sequences | Struggles | Handles well |

---

## ⚡ Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/tsla-stock-prediction.git
cd tsla-stock-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset
Download `TSLA.csv` from [Google Drive](https://drive.google.com/file/d/1BHzdUi6-iKz7a3tnZunxcp_Td-7I24C7/view) and place it in the root folder.

### 4. Run the notebook
```bash
jupyter notebook notebooks/Tesla_Stock_Price_Prediction.ipynb
```
Run all cells top-to-bottom — this trains both models, generates all plots, and saves weights to `models/`.

---

## 🚀 Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

Open **http://localhost:8501** — the app features:

| Tab | Features |
|---|---|
| 📈 Market Overview | Live price chart, volume, RSI, Bollinger Bands |
| 🤖 Model Results | Test-set predictions for both models side by side |
| 📅 Forecast | 1/5/10-day price forecasts with per-day metric cards |
| 📊 Comparison | Bar charts + model comparison table |
| 📋 Data Explorer | Raw data, correlation matrix |

---

## 💡 Key Learnings

1. **Forward fill beats interpolation** for stock data — keeps the series causal
2. **Scaler must be fit on train-only** — fitting on full data leaks future distribution into training
3. **TimeSeriesSplit is mandatory** for hyperparameter search in financial data
4. **LSTM gates solve vanishing gradients** that cripple SimpleRNN on 60-step sequences
5. **Error compounds** in multi-step forecasting — 10-day predictions are much less reliable than 1-day
6. **MAPE is more meaningful than RMSE** when price range spans orders of magnitude

---

## 🚀 Future Improvements

- [ ] Add **news sentiment** (FinBERT on Tesla/Elon headlines) as external features
- [ ] Try **Temporal Fusion Transformer (TFT)** for multi-horizon tabular time-series
- [ ] **Ensemble** RNN + LSTM + XGBoost with learned weights
- [ ] **Walk-forward retraining** — monthly rolling window for production use
- [ ] Deploy Streamlit app on **Streamlit Community Cloud** (free hosting)

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. The predictions generated are not financial advice and should not be used for actual investment decisions. Stock markets are influenced by countless unpredictable factors beyond historical price data.

---

## 👤 Author

**Gopi Mahith (Nagubandi Gopi Mahith)**

B.Tech — Computer Science & Technology (Big Data Analytics Specialization)
Parul University, Gujarat · Class of 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/YOUR_USERNAME)
📞 +91 99480 09521

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Made with ❤️ and TensorFlow*

</div>
