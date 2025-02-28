Bitcoin Price Prediction Project

Overview

This project aims to predict Bitcoin price movements using various deep learning models. The implementation leverages the Binance API for data collection and employs multiple neural network architectures to forecast Bitcoin prices.
Note: This project is currently a work in progress. While it shows promising results, particularly with the GRU model, further fine-tuning and optimization are needed.

Project Structure

BitCoin.ipynb: Connects to Binance API, downloads the dataset, and runs LSTM, LightGBM, and ensemble models for 100 epochs
BTCUSDT.ipynb: Uses the downloaded data to train LSTM, GRU, and Bidirectional LSTM models for 500 epochs
Model files: Saved models (lstm_model.h5, gru_model.h5, bilstm_model.h5)
Utility files: Scaler and column information for future predictions

Data Collection

Historical cryptocurrency data fetched via Binance API
Automatic downloading and preprocessing of Bitcoin price data
Technical indicators added to enhance model performance

Models Implemented

LSTM (Long Short-Term Memory): Traditional LSTM model for time series prediction
GRU (Gated Recurrent Unit): Currently showing the best performance
Bidirectional LSTM: Captures patterns from both past and future states
LightGBM: Gradient boosting framework (included in BitCoin.ipynb)
Ensemble: Combined predictions from multiple models (included in BitCoin.ipynb)

Current Results
Based on the latest training runs (500 epochs), model performance metrics are:

LSTM Performance:
MSE: 521122524.2887
RMSE: 22828.1082
MAE: 10394.3484

GRU Performance:
MSE: 5230277.9246
RMSE: 2286.9801
MAE: 1945.1007

BiLSTM Performance:
MSE: 179752522.2960
RMSE: 13407.1817
MAE: 6140.6669

Key Findings: The GRU model significantly outperforms other architectures with an RMSE of approximately 2,287, compared to 22,828 for LSTM and 13,407 for Bidirectional LSTM.

Features

Automatic data collection via Binance API
Technical indicator generation (moving averages, RSI, MACD, etc.)
Window-based prediction with configurable parameters
Model serialization for future use
Visualization of predictions

Training Process

The models are trained with the following configuration:

Window size: 24 time steps
Batch size: 32
Training/validation split: 80/20
Epochs: 500
Optimization algorithm: Adam

Future Work

Hyperparameter optimization for the GRU model
Exploration of alternative architectures (Transformer, TCN)
Feature importance analysis
Implementation of early stopping and learning rate scheduling
Ensemble methods to combine strengths of multiple models
Real-time prediction system

Requirements

Python 3.7+
TensorFlow 2.x
pandas
numpy
scikit-learn
plotly
python-binance

Usage

Run BitCoin.ipynb to fetch data from Binance API
Execute BTCUSDT.ipynb to train and evaluate models
Use the saved models for making predictions on new data

Acknowledgments

Binance API for providing historical cryptocurrency data
TensorFlow and Keras for deep learning implementations
