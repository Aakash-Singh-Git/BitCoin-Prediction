# Bitcoin Price Prediction Project

## 🧠 Overview
This project focuses on predicting Bitcoin price movements using machine learning (ML) and deep learning (DL) techniques. Data is collected via the Binance API and stored in a CSV file. While traditional ML models were initially used, deep learning models—particularly GRU—demonstrated superior performance.

**⚠️ Note:** The project is currently a work in progress. The GRU model has shown excellent results and has been further optimized using Keras Tuner.

---

## 📁 Project Structure

- **data.ipynb**: 
  - Connects to the Binance API
  - Downloads and stores historical BTC/USDT price data into a CSV file
  - Generates technical indicators

- **ML.ipynb**: 
  - Applies classical ML models (e.g., LightGBM)
  - Results were suboptimal, leading to the use of DL models

- **DL model.ipynb**:
  - Implements and evaluates LSTM, GRU, and Bidirectional LSTM models
  - Identifies GRU as the best-performing model

- **final.ipynb**:
  - Uses Keras AutoTuner to fine-tune the GRU model over 20 trials and 100 epochs
  - Trains the optimal GRU model for final predictions

- **Model Files**:
  - `best_gru_model.h5`: Final trained GRU model
  - Saved scalers and configuration files for prediction

---

## 📊 Data Collection

- Source: **Binance API**
- Data: BTC/USDT historical prices (open, high, low, close, volume)
- Storage: Saved in a local CSV file via `data.ipynb`
- Preprocessing: Includes normalization and technical indicator generation (RSI, MACD, moving averages, etc.)

---

## 🔍 Models Implemented

### 📌 Machine Learning
- **LightGBM**  
  - Implemented in `ML.ipynb`  
  - Did not yield competitive accuracy

### 📌 Deep Learning
- **LSTM (Long Short-Term Memory)**  
- **GRU (Gated Recurrent Unit)** – *Best performing model*  
- **Bidirectional LSTM**  
- **Optimized GRU (via Keras AutoTuner)**

---

## ✅ Final Model Evaluation

Recent Prediction Accuracy of the Tuned GRU Model:

| Metric                          | Value     |
|---------------------------------|-----------|
| Mean Squared Error (MSE)        | 929,751.44 |
| Root Mean Squared Error (RMSE)  | 964.24    |
| Mean Absolute Error (MAE)       | 707.27    |
| R-squared (R²)                  | 0.9722    |
| Mean Absolute Percentage Error (MAPE) | 0.72% |

---

## ⚙️ Features

- ✅ Automatic data collection from Binance
- ✅ Technical indicators for enhanced prediction
- ✅ Model serialization and reusability
- ✅ GRU model optimization using Keras AutoTuner
- ✅ Visualizations for prediction analysis

---

## 🏋️ Training Details

- **Window Size**: 24 time steps
- **Batch Size**: 32
- **Epochs**: 100 (best model found via tuning)
- **Optimizer**: Adam
- **Validation Split**: 80/20
- **Hyperparameter Tuning**: Keras AutoTuner (20 trials)

---

## 🚀 Usage Instructions

1. **Run `data.ipynb`**  
   → Collects and stores BTC/USDT data from Binance into CSV

2. **Run `ML.ipynb`** *(optional)*  
   → Trains traditional ML models (for comparison)

3. **Run `DL model.ipynb`**  
   → Trains LSTM, GRU, and BiLSTM models

4. **Run `final.ipynb`**  
   → Uses AutoTuner to find and train the best GRU model  
   → Predicts and evaluates performance

---

## 🧰 Requirements

- Python 3.7+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- plotly
- python-binance
- keras-tuner

---

## 🙏 Acknowledgments

- **Binance API**: For providing real-time and historical cryptocurrency data
- **TensorFlow & Keras**: For deep learning model implementations
- **Keras Tuner**: For automated hyperparameter optimization

---

## 📌 Future Work

- Early stopping and learning rate scheduling
- Real-time price prediction system
- Incorporating Transformer and TCN architectures
- More advanced feature selection and importance analysis
- Deployment as a web or desktop application

---
