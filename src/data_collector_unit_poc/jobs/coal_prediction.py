"""Coal price prediction jobs"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the paths to the data files
def get_data_path(symbol: str) -> str:
    """Get the path to the data file for a given symbol"""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "commodities")
    return os.path.join(base_path, f"{symbol}.parquet")

def load_commodity_data(symbol: str) -> pd.DataFrame:
    """Load commodity data from Parquet file"""
    file_path = get_data_path(symbol)
    try:
        df = pd.read_parquet(file_path)
        # Sort by date to ensure chronological order
        df = df.sort_values('date')
        return df
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_next_monday(from_date: Optional[datetime] = None) -> datetime:
    """Get the date of the next Monday from a given date"""
    if from_date is None:
        from_date = datetime.now()
    
    # If today is Monday, get next Monday
    days_ahead = 7 - from_date.weekday() if from_date.weekday() == 0 else (7 - from_date.weekday()) % 7
    
    # If days_ahead is 0, it means today is Monday, so we want the next Monday (7 days ahead)
    if days_ahead == 0:
        days_ahead = 7
        
    next_monday = from_date + timedelta(days=days_ahead)
    return next_monday

def prepare_data_for_prediction(df: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for prediction by creating features and target"""
    # Use closing prices
    prices = df['close'].values
    
    # Create features (previous n days' prices) and target (next day's price)
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    
    return np.array(X), np.array(y)

def prepare_combined_data(coal_df: pd.DataFrame, oil_df: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare combined coal and oil data for prediction"""
    # Merge dataframes on date
    merged_df = pd.merge(coal_df, oil_df, on='date', suffixes=('_coal', '_oil'))
    
    # Sort by date
    merged_df = merged_df.sort_values('date')
    
    # Use closing prices
    coal_prices = merged_df['close_coal'].values
    oil_prices = merged_df['close_oil'].values
    
    # Create features (previous n days' prices for both coal and oil) and target (next day's coal price)
    X, y = [], []
    for i in range(len(coal_prices) - window_size):
        # Combine coal and oil features
        features = []
        for j in range(window_size):
            features.append(coal_prices[i+j])
            features.append(oil_prices[i+j])
        X.append(features)
        y.append(coal_prices[i+window_size])
    
    return np.array(X), np.array(y)

def train_coal_only_model(coal_df: pd.DataFrame, window_size: int = 5) -> Tuple[LinearRegression, StandardScaler, StandardScaler]:
    """Train a model using only coal price data"""
    # Prepare data
    X, y = prepare_data_for_prediction(coal_df, window_size)
    
    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    return model, X_scaler, y_scaler

def train_combined_model(coal_df: pd.DataFrame, oil_df: pd.DataFrame, window_size: int = 5) -> Tuple[LinearRegression, StandardScaler, StandardScaler]:
    """Train a model using both coal and oil price data"""
    # Prepare data
    X, y = prepare_combined_data(coal_df, oil_df, window_size)
    
    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    return model, X_scaler, y_scaler

def predict_next_day(model: LinearRegression, X_scaler: StandardScaler, y_scaler: StandardScaler, 
                     recent_data: np.ndarray) -> float:
    """Predict the next day's price"""
    # Scale input data
    X_scaled = X_scaler.transform(recent_data.reshape(1, -1))
    
    # Make prediction
    y_scaled_pred = model.predict(X_scaled)
    
    # Inverse transform to get actual price
    y_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()[0]
    
    return y_pred

def backtest_model(df: pd.DataFrame, window_size: int = 5, test_size: int = 30, 
                   horizon: int = 1, is_combined: bool = False, oil_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Backtest the model on historical data
    
    Args:
        df: DataFrame with historical data
        window_size: Number of days to use for prediction
        test_size: Number of days to use for testing
        horizon: Prediction horizon (1 for today, 2 for tomorrow, 7 for next week)
        is_combined: Whether to use combined model with oil data
        oil_df: DataFrame with oil data (required if is_combined is True)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if is_combined:
        if oil_df is None:
            raise ValueError("Oil data is required for combined model")
    
    # Prepare data
    if is_combined:
        # Merge dataframes on date
        merged_df = pd.merge(df, oil_df, on='date', suffixes=('_coal', '_oil'))
        merged_df = merged_df.sort_values('date')
        
        # Get prices
        coal_prices = merged_df['close_coal'].values
        oil_prices = merged_df['close_oil'].values
        
        # We need at least window_size + test_size + horizon days of data
        if len(coal_prices) < window_size + test_size + horizon:
            raise ValueError("Not enough data for backtesting")
        
        # Split data into training and testing
        train_coal = coal_prices[:-test_size-horizon]
        train_oil = oil_prices[:-test_size-horizon]
        
        # Create training data
        X_train, y_train = [], []
        for i in range(len(train_coal) - window_size):
            features = []
            for j in range(window_size):
                features.append(train_coal[i+j])
                features.append(train_oil[i+j])
            X_train.append(features)
            y_train.append(train_coal[i+window_size])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Test data
        test_coal = coal_prices[-test_size-window_size-horizon:]
        test_oil = oil_prices[-test_size-window_size-horizon:]
        
        # Make predictions
        y_true = []
        y_pred = []
        
        for i in range(test_size):
            # Get features for current window
            current_features = []
            for j in range(window_size):
                current_features.append(test_coal[i+j])
                current_features.append(test_oil[i+j])
            current_features = np.array(current_features)
            
            # For multi-day predictions, we need to predict iteratively
            if horizon > 1:
                # Copy the current features
                future_features = current_features.copy()
                
                # Predict each day until horizon
                for h in range(horizon):
                    # Scale and predict
                    X_scaled = X_scaler.transform(future_features.reshape(1, -1))
                    y_scaled_pred = model.predict(X_scaled)
                    next_price = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()[0]
                    
                    if h < horizon - 1:
                        # Update features for next prediction
                        # Remove first coal and oil price
                        future_features = future_features[2:]
                        # Add predicted coal price and keep oil price the same
                        future_features = np.append(future_features, [next_price, test_oil[i+window_size+h]])
                
                # The last prediction is our final prediction
                predicted_price = next_price
            else:
                # For 1-day prediction, just predict directly
                X_scaled = X_scaler.transform(current_features.reshape(1, -1))
                y_scaled_pred = model.predict(X_scaled)
                predicted_price = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()[0]
            
            # Actual price is horizon days ahead
            actual_price = test_coal[i+window_size+horizon-1]
            
            y_true.append(actual_price)
            y_pred.append(predicted_price)
    else:
        # Coal-only model
        prices = df['close'].values
        
        # We need at least window_size + test_size + horizon days of data
        if len(prices) < window_size + test_size + horizon:
            raise ValueError("Not enough data for backtesting")
        
        # Split data into training and testing
        train_prices = prices[:-test_size-horizon]
        
        # Create training data
        X_train, y_train = [], []
        for i in range(len(train_prices) - window_size):
            X_train.append(train_prices[i:i+window_size])
            y_train.append(train_prices[i+window_size])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Test data
        test_prices = prices[-test_size-window_size-horizon:]
        
        # Make predictions
        y_true = []
        y_pred = []
        
        for i in range(test_size):
            # Get features for current window
            current_window = test_prices[i:i+window_size]
            
            # For multi-day predictions, we need to predict iteratively
            if horizon > 1:
                # Copy the current window
                future_window = current_window.copy()
                
                # Predict each day until horizon
                for h in range(horizon):
                    # Scale and predict
                    X_scaled = X_scaler.transform(future_window.reshape(1, -1))
                    y_scaled_pred = model.predict(X_scaled)
                    next_price = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()[0]
                    
                    if h < horizon - 1:
                        # Update window for next prediction
                        future_window = np.append(future_window[1:], next_price)
                
                # The last prediction is our final prediction
                predicted_price = next_price
            else:
                # For 1-day prediction, just predict directly
                X_scaled = X_scaler.transform(current_window.reshape(1, -1))
                y_scaled_pred = model.predict(X_scaled)
                predicted_price = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()[0]
            
            # Actual price is horizon days ahead
            actual_price = test_prices[i+window_size+horizon-1]
            
            y_true.append(actual_price)
            y_pred.append(predicted_price)
    
    # Calculate metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = float(np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def predict_coal_prices_job(cancel_event=None):
    """Job to predict coal prices using two methods and print the results"""
    try:
        print(f"Starting coal price prediction job at {datetime.now()}")
        
        # Load data
        coal_df = load_commodity_data("MTF_F")
        oil_df = load_commodity_data("CL_F")  # Using WTI crude oil
        
        if coal_df.empty:
            print("Error: Coal price data is empty")
            return
        
        if oil_df.empty:
            print("Error: Oil price data is empty")
            return
        
        # Get current date and next Monday
        today = datetime.now()
        next_monday_date = get_next_monday(today)
        
        # Calculate days until next Monday
        days_until_monday = (next_monday_date - today).days
        
        # Train models
        window_size = 5  # Use last 5 days for prediction
        
        # Backtest models on historical data
        print("\n===== MODEL BACKTESTING RESULTS =====")
        
        # Backtest for 1-day predictions (today)
        print("\nBacktesting for 1-day predictions (today):")
        coal_only_metrics_1day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=1, is_combined=False)
        combined_metrics_1day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=1, is_combined=True, oil_df=oil_df)
        
        print("Model 1 (Coal only):")
        print(f"  - Mean Absolute Error: ${coal_only_metrics_1day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${coal_only_metrics_1day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {coal_only_metrics_1day['mape']:.2f}%")
        print(f"  - R² Score: {coal_only_metrics_1day['r2']:.4f}")
        
        print("Model 2 (Coal + Oil):")
        print(f"  - Mean Absolute Error: ${combined_metrics_1day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${combined_metrics_1day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {combined_metrics_1day['mape']:.2f}%")
        print(f"  - R² Score: {combined_metrics_1day['r2']:.4f}")
        
        # Backtest for 2-day predictions (tomorrow)
        print("\nBacktesting for 2-day predictions (tomorrow):")
        coal_only_metrics_2day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=2, is_combined=False)
        combined_metrics_2day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=2, is_combined=True, oil_df=oil_df)
        
        print("Model 1 (Coal only):")
        print(f"  - Mean Absolute Error: ${coal_only_metrics_2day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${coal_only_metrics_2day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {coal_only_metrics_2day['mape']:.2f}%")
        print(f"  - R² Score: {coal_only_metrics_2day['r2']:.4f}")
        
        print("Model 2 (Coal + Oil):")
        print(f"  - Mean Absolute Error: ${combined_metrics_2day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${combined_metrics_2day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {combined_metrics_2day['mape']:.2f}%")
        print(f"  - R² Score: {combined_metrics_2day['r2']:.4f}")
        
        # Backtest for 7-day predictions (next week)
        print("\nBacktesting for 7-day predictions (next week):")
        coal_only_metrics_7day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=7, is_combined=False)
        combined_metrics_7day = backtest_model(coal_df, window_size=window_size, test_size=60, horizon=7, is_combined=True, oil_df=oil_df)
        
        print("Model 1 (Coal only):")
        print(f"  - Mean Absolute Error: ${coal_only_metrics_7day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${coal_only_metrics_7day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {coal_only_metrics_7day['mape']:.2f}%")
        print(f"  - R² Score: {coal_only_metrics_7day['r2']:.4f}")
        
        print("Model 2 (Coal + Oil):")
        print(f"  - Mean Absolute Error: ${combined_metrics_7day['mae']:.2f}")
        print(f"  - Root Mean Squared Error: ${combined_metrics_7day['rmse']:.2f}")
        print(f"  - Mean Absolute Percentage Error: {combined_metrics_7day['mape']:.2f}%")
        print(f"  - R² Score: {combined_metrics_7day['r2']:.4f}")
        
        print("===================================\n")
        
        # Train models for prediction
        # Model 1: Coal only
        coal_model, coal_X_scaler, coal_y_scaler = train_coal_only_model(coal_df, window_size)
        
        # Model 2: Coal + Oil
        combined_model, combined_X_scaler, combined_y_scaler = train_combined_model(coal_df, oil_df, window_size)
        
        # Get recent data for prediction
        recent_coal_prices = coal_df['close'].values[-window_size:]
        
        # For combined model, we need both coal and oil prices
        recent_combined_data = []
        for i in range(window_size):
            idx = -window_size + i
            recent_combined_data.append(coal_df['close'].values[idx])
            recent_combined_data.append(oil_df['close'].values[idx])
        recent_combined_data = np.array(recent_combined_data)
        
        # Predict today's and next day's price
        today_coal_only = predict_next_day(coal_model, coal_X_scaler, coal_y_scaler, recent_coal_prices)
        today_combined = predict_next_day(combined_model, combined_X_scaler, combined_y_scaler, recent_combined_data)
        
        # For tomorrow, we need to predict using today's predicted price
        tomorrow_coal_only_prices = np.append(recent_coal_prices[1:], today_coal_only)
        tomorrow_coal_only = predict_next_day(coal_model, coal_X_scaler, coal_y_scaler, tomorrow_coal_only_prices)
        
        # For combined model, we need both coal and oil prices
        tomorrow_combined_data = recent_combined_data.copy()
        # Remove first coal and oil price
        tomorrow_combined_data = tomorrow_combined_data[2:]
        # Add today's predicted coal price and keep oil price the same
        tomorrow_combined_data = np.append(tomorrow_combined_data, [today_combined, oil_df['close'].values[-1]])
        tomorrow_combined = predict_next_day(combined_model, combined_X_scaler, combined_y_scaler, tomorrow_combined_data)
        
        # For next Monday, we need to predict iteratively
        # First, predict each day until next Monday using coal-only model
        monday_coal_only_prices = np.append(recent_coal_prices[1:], today_coal_only)
        monday_coal_only_prices = np.append(monday_coal_only_prices, tomorrow_coal_only)
        
        # We already predicted 2 days ahead (today and tomorrow), so we need days_until_monday - 2 more days
        remaining_days = max(0, days_until_monday - 2)
        for _ in range(remaining_days):
            next_price = predict_next_day(coal_model, coal_X_scaler, coal_y_scaler, monday_coal_only_prices[-window_size:])
            monday_coal_only_prices = np.append(monday_coal_only_prices, next_price)
        next_monday_coal_only = monday_coal_only_prices[-1]
        
        # Now predict each day until next Monday using combined model
        # This is more complex as we need to predict both coal and oil prices
        # For simplicity, we'll assume oil prices remain constant
        monday_combined_prices = tomorrow_combined_data.copy()
        monday_combined_prices = np.append(monday_combined_prices, [tomorrow_combined, oil_df['close'].values[-1]])
        
        # We already predicted 2 days ahead (today and tomorrow), so we need days_until_monday - 2 more days
        for _ in range(remaining_days):
            next_price = predict_next_day(combined_model, combined_X_scaler, combined_y_scaler, monday_combined_prices[-window_size*2:])
            # Add predicted coal price and keep oil price the same
            monday_combined_prices = np.append(monday_combined_prices, [next_price, oil_df['close'].values[-1]])
        next_monday_combined = monday_combined_prices[-2]  # Coal price is at even indices
        
        # Print predictions
        print("\n===== COAL PRICE PREDICTIONS =====")
        print(f"Current date: {today.strftime('%Y-%m-%d')}")
        print(f"Next Monday: {next_monday_date.strftime('%Y-%m-%d')}")
        print("\nModel 1: Using only coal price data")
        print(f"  - Predicted price for today: ${today_coal_only:.2f}")
        print(f"  - Predicted price for tomorrow: ${tomorrow_coal_only:.2f}")
        print(f"  - Predicted price for next Monday: ${next_monday_coal_only:.2f}")
        print("\nModel 2: Using coal and oil price data")
        print(f"  - Predicted price for today: ${today_combined:.2f}")
        print(f"  - Predicted price for tomorrow: ${tomorrow_combined:.2f}")
        print(f"  - Predicted price for next Monday: ${next_monday_combined:.2f}")
        print("==================================\n")
        
    except Exception as e:
        print(f"Error in coal price prediction job: {str(e)}")

if __name__ == "__main__":
    predict_coal_prices_job()
