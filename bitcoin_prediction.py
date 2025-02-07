import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta

def fetch_bitcoin_data():
    """Fetch real-time Bitcoin data using yfinance."""
    btc_data = yf.download('BTC-USD', period='1y', interval='1d')
    return btc_data

def fetch_latest_bitcoin_price():
    """Fetch the latest Bitcoin price."""
    btc_data = yf.download('BTC-USD', period='1d', interval='1m')
    latest_price = btc_data['Close'].iloc[-1]  # Get the latest price from the minute data
    return latest_price

def preprocess_data(btc_data):
    """Preprocess the data for training."""
    # Create moving averages
    btc_data['MA_10'] = btc_data['Close'].rolling(window=10).mean()
    btc_data['MA_50'] = btc_data['Close'].rolling(window=50).mean()

    # Drop missing values
    btc_data.dropna(inplace=True)

    # Features and target
    X = btc_data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    y = btc_data['Close']

    return X, y, btc_data

def train_model(X, y):
    """Train a Linear Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

def predict_price(model, latest_data):
    """Predict the Bitcoin price using the trained model."""
    prediction = model.predict(latest_data)
    return prediction[0]  # Return the scalar value

def get_friday_prediction(model, btc_data):
    """Get the Bitcoin price prediction for the next Friday at 12 PM."""
    today = datetime.today()
    days_ahead = 4 - today.weekday()  # 4 means Friday
    if days_ahead <= 0:
        days_ahead += 7  # If today is Friday or later in the week, get next Friday
    
    next_friday = today + timedelta(days=days_ahead)
    friday_date = next_friday.strftime('%Y-%m-%d')

    # Add the date for Friday
    future_data = btc_data.iloc[-1:].copy()
    future_data['Date'] = pd.to_datetime(friday_date)

    latest_data = future_data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    predicted_price = predict_price(model, latest_data.values)
    
    return predicted_price, friday_date

def get_month_end_prediction(model, btc_data):
    """Get the Bitcoin price prediction for the end of the current month."""
    today = datetime.today()
    next_month = today.replace(day=28) + timedelta(days=4)  # Get the first day of the next month
    month_end = next_month - timedelta(days=next_month.day)  # Subtract the days to get the last day of the current month
    month_end_date = month_end.strftime('%Y-%m-%d')

    # Add the date for the end of the month
    future_data = btc_data.iloc[-1:].copy()
    future_data['Date'] = pd.to_datetime(month_end_date)

    latest_data = future_data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    predicted_price = predict_price(model, latest_data.values)

    return predicted_price, month_end_date

def retrain_model():
    """Retrain the model periodically with new data."""
    btc_data = fetch_bitcoin_data()
    X, y, _ = preprocess_data(btc_data)
    model = train_model(X, y)
    return model

if __name__ == "__main__":
    # Fetch and preprocess data
    btc_data = fetch_bitcoin_data()
    X, y, _ = preprocess_data(btc_data)

    # Train the model
    model = train_model(X, y)

    # Predict the latest price
    latest_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = predict_price(model, latest_data)
    print(f"Predicted Bitcoin Price: ${predicted_price:.2f}")
