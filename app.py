from flask import Flask, render_template, jsonify
import bitcoin_prediction as bp
import json
import datetime

app = Flask(__name__)

# Load the model and data
btc_data = bp.fetch_bitcoin_data()
X, y, btc_data = bp.preprocess_data(btc_data)
model = bp.train_model(X, y)

# Function to predict Friday's price
def predict_friday():
    # Get the current date and the next Friday
    today = datetime.date.today()
    days_until_friday = (4 - today.weekday()) % 7
    friday_date = today + datetime.timedelta(days=days_until_friday)
    
    # Get prediction for Friday
    latest_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = bp.predict_price(model, latest_data)
    
    return predicted_price, friday_date

# Function to predict Month-End price
def predict_month_end():
    today = datetime.date.today()
    # Find the last day of the current month
    next_month = today.replace(day=28) + datetime.timedelta(days=4)  # this will give us the 1st of next month
    month_end = next_month - datetime.timedelta(days=next_month.day)
    
    # Get prediction for the end of the month
    latest_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = bp.predict_price(model, latest_data)
    
    return predicted_price, month_end

@app.route('/')
def home():
    """Render the home page."""
    # Prepare data for the graph
    dates = btc_data.index.strftime('%Y-%m-%d').tolist()
    prices = btc_data['Close'].tolist()
    return render_template('index.html', dates=json.dumps(dates), prices=json.dumps(prices))

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Bitcoin price for Friday and month end."""
    # Get predictions for Friday and month end
    friday_price, friday_date = predict_friday()
    month_end_price, month_end_date = predict_month_end()

    # Return predictions as JSON
    return jsonify({
        'friday_price': friday_price,
        'friday_date': friday_date.strftime('%Y-%m-%d'),
        'month_end_price': month_end_price,
        'month_end_date': month_end_date.strftime('%Y-%m-%d')
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model and return the status."""
    # Retrain the model with the latest data
    btc_data = bp.fetch_bitcoin_data()
    X, y, btc_data = bp.preprocess_data(btc_data)
    model = bp.train_model(X, y)
    return jsonify({'status': 'Model retrained successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
