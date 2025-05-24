import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, BayesianOptimization
import tensorflow as tf
import math
import os
import ta
import time

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load and preprocess the dataset
def load_data(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    print(data.head())
    return data

# 2. Add technical indicators - using the same indicators as the previous successful model
def add_technical_indicators(df):
    print("Adding technical indicators...")
    
    # Make sure the dataframe has the right column names
    if 'open' not in df.columns and 'OPEN' in df.columns:
        df.columns = [col.lower() for col in df.columns]
    
    # Add moving averages
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()
    
    # Add RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Add MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    
    # Add volatility measure
    df['volatility'] = df['close'].pct_change().rolling(window=14).std()
    
    # Add price momentum
    df['pct_change'] = df['close'].pct_change()
    df['pct_change_3'] = df['close'].pct_change(periods=3)
    df['pct_change_7'] = df['close'].pct_change(periods=7)
    
    # Add Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Add Average True Range (ATR)
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Add On-Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # Drop rows with NaN values (created by rolling windows)
    df.dropna(inplace=True)
    
    print(f"Dataset with indicators shape: {df.shape}")
    
    return df

# 3. Prepare data for model training
def prepare_data(data, target_col='close', n_steps=24):
    print(f"Preparing data with a sequence length of {n_steps}...")
    
    # All columns except date column or any other non-numeric columns
    features = data.select_dtypes(include=[np.number]).columns.tolist()
    data_filtered = data[features].copy()
    
    # Print features for debugging
    print(f"Using {len(features)} features: {features}")
    
    # Scale the features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(data_filtered)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i+n_steps])
        # Get the closing price only as the target
        y.append(data[target_col].values[i+n_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale the target variable
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_reshaped = y.reshape(-1, 1)
    scaler_y.fit(y_reshaped)
    y_scaled = scaler_y.transform(y_reshaped).flatten()
    
    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler_y, features

# 4. Build the GRU model for hyperparameter tuning
def build_model_for_tuning(hp, input_shape):
    model = Sequential()
    
    # Tune the number of GRU layers (2-3)
    num_gru_layers = hp.Int('num_gru_layers', min_value=2, max_value=3)
    
    # Tune the number of units in the first GRU layer
    gru1_units = hp.Int('gru1_units', min_value=32, max_value=128, step=32)
    
    # Add first GRU layer
    if num_gru_layers == 2:
        model.add(GRU(units=gru1_units, return_sequences=False, input_shape=input_shape))
    else:
        model.add(GRU(units=gru1_units, return_sequences=True, input_shape=input_shape))
    
    # Tune dropout rate for first layer
    dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout1))
    
    # Add middle GRU layer if num_gru_layers is 3
    if num_gru_layers == 3:
        gru2_units = hp.Int('gru2_units', min_value=32, max_value=128, step=32)
        model.add(GRU(units=gru2_units, return_sequences=False))
        
        # Tune dropout rate for second layer
        dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout2))
    
    # Tune output layers
    use_hidden_layer = hp.Boolean('use_hidden_layer')
    if use_hidden_layer:
        hidden_units = hp.Int('hidden_units', min_value=8, max_value=64, step=8)
        model.add(Dense(hidden_units, activation='relu'))
    
    # Output layer
    model.add(Dense(1))
    
    # Tune learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

# 5. Function to evaluate model performance
def evaluate_model(model, X_test, y_test, scaler_y):
    print("Evaluating model performance...")
    
    # Make predictions
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Bitcoin Price Prediction - GRU Model')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('gru_bitcoin_prediction.png')
    plt.show()
    
    return rmse, mae, r2, mape

# 6. Save the model
def save_model(model, model_name='gru_bitcoin_model'):
    print(f"Saving model as {model_name}...")
    model.save(f"{model_name}.h5")
    print(f"Model saved successfully as {model_name}.h5")

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    
    # File path
    file_path = r"C:\Users\Aakash\Downloads\Project\BTCUSDT_1h_20170701_to_20250427.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        # 1. Load data
        data = load_data(file_path)
        
        # Convert column names to lowercase if needed
        if 'OPEN' in data.columns and 'open' not in data.columns:
            data.columns = [col.lower() for col in data.columns]
        
        # 2. Add technical indicators
        data = add_technical_indicators(data)
        
        # 3. Prepare data - tune sequence length 
        n_steps_options = [24, 48, 72]  # Different sequence lengths to try
        best_rmse = float('inf')
        best_n_steps = 24  # Default
        
        for n_steps in n_steps_options:
            print(f"\nTrying sequence length: {n_steps}")
            X_train_temp, y_train_temp, X_test_temp, y_test_temp, scaler_y_temp, _ = prepare_data(data, n_steps=n_steps)
            
            # Simple model to test sequence length
            temp_model = Sequential()
            temp_model.add(GRU(64, input_shape=(X_train_temp.shape[1], X_train_temp.shape[2])))
            temp_model.add(Dense(1))
            temp_model.compile(optimizer='adam', loss='mse')
            
            # Quick training
            temp_model.fit(X_train_temp, y_train_temp, epochs=10, batch_size=32, verbose=0)
            
            # Evaluate
            preds = temp_model.predict(X_test_temp)
            preds_actual = scaler_y_temp.inverse_transform(preds)
            y_test_actual = scaler_y_temp.inverse_transform(y_test_temp.reshape(-1, 1))
            
            temp_rmse = math.sqrt(mean_squared_error(y_test_actual, preds_actual))
            print(f"RMSE with {n_steps} steps: {temp_rmse:.4f}")
            
            if temp_rmse < best_rmse:
                best_rmse = temp_rmse
                best_n_steps = n_steps
        
        print(f"\nUsing best sequence length: {best_n_steps}")
        X_train, y_train, X_test, y_test, scaler_y, features = prepare_data(data, n_steps=best_n_steps)
        
        # Get input shape for the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 4. Setup hyperparameter tuning - with more trials
        print("\nSetting up hyperparameter tuning...")
        tuner = BayesianOptimization(
            lambda hp: build_model_for_tuning(hp, input_shape),
            objective='val_loss',
            max_trials=20,  # Increased number of trials
            executions_per_trial=1,
            directory='hyperparameter_tuning',
            project_name='gru_bitcoin_prediction'
        )
        
        # Early stopping callback for tuning
        early_stopping_tuning = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Run hyperparameter search
        print("Running hyperparameter tuning...")
        tuner.search(
            X_train, y_train,
            epochs=100,  # More epochs for tuning
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping_tuning]
        )
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nBest hyperparameters:")
        print(f"Number of GRU layers: {best_hps.get('num_gru_layers')}")
        print(f"GRU1 units: {best_hps.get('gru1_units')}")
        print(f"Dropout1: {best_hps.get('dropout1')}")
        
        if best_hps.get('num_gru_layers') == 3:
            print(f"GRU2 units: {best_hps.get('gru2_units')}")
            print(f"Dropout2: {best_hps.get('dropout2')}")
        
        print(f"Use hidden layer: {best_hps.get('use_hidden_layer')}")
        if best_hps.get('use_hidden_layer'):
            print(f"Hidden units: {best_hps.get('hidden_units')}")
        
        print(f"Learning rate: {best_hps.get('learning_rate')}")
        
        # Build model with best hyperparameters
        best_model = tuner.hypermodel.build(best_hps)
        
        # Define callbacks for full training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,  # More patience for full training
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_gru_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Add learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model for 500 epochs
        print("\nTraining model for 500 epochs...")
        history = best_model.fit(
            X_train, y_train,
            epochs=500,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, checkpoint, lr_scheduler],
            verbose=1
        )
        
        # Load the best model
        best_model.load_weights('best_gru_model.h5')
        
        # Evaluate model performance
        print("\nFinal model evaluation:")
        rmse, mae, r2, mape = evaluate_model(best_model, X_test, y_test, scaler_y)
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('gru_training_history.png')
        plt.show()
        
        # Save the model
        save_model(best_model, 'gru_bitcoin_model_optimized')
        
        # Check if we need more epochs
        last_20_val_loss = history.history['val_loss'][-20:]
        first_of_last_20 = last_20_val_loss[0]
        last_of_last_20 = last_20_val_loss[-1]
        
        if last_of_last_20 < first_of_last_20 * 0.99:  # Still improving by at least 1%
            print("\nThe model is still improving. You might benefit from training for more epochs.")
        else:
            print("\nThe model has converged. No additional epochs are needed.")
        
        # Calculate and display total execution time
        end_time = time.time()
        execution_time_hours = (end_time - start_time) / 3600
        print(f"\nTotal execution time: {execution_time_hours:.2f} hours")
        
        print(f"Final RMSE: {rmse:.4f}")
        print("Training complete!")