import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
from flask import Flask, render_template, request, jsonify

# Define a list of ticker symbols
tickerSymbols = ['AAPL', 'MSFT', 'GOOG']

# Load the trained model from the saved file
model = joblib.load('stock_model.joblib')

# Create a Flask app object
app = Flask(__name__)

# Define a route for the live dashboard
@app.route('/')
def dashboard():
    # Load the live data from a CSV file
    df = pd.read_csv('stock_data.csv')
    # Filter the data to show only the latest data point for each stock
    latest_df = df.groupby('symbol').tail(1)
    # Get the latest data instance for prediction
    instance = latest_df[['Open', 'High', 'Low', 'Volume']].values[0]
    # Predict the target variable using the trained model
    y_pred = model.predict([instance])[0]
    # Compute the accuracy of the model
    accuracy = r2_score(latest_df['Close'], model.predict(latest_df[['Open', 'High', 'Low', 'Volume']]))
    # Render the HTML template with the live data and prediction of the model
    return render_template('dashboard.html', latest_df=latest_df.to_dict('records'), y_pred=y_pred, accuracy=accuracy)

# Define a route for the prediction service
@app.route('/predict', methods=['POST'])
def predict():
    # Get the live data instance from the request JSON object
    data = request.get_json()
    instance = [data['Open'], data['High'], data['Low'], data['Volume']]
    # Predict the target variable using the trained model
    y_pred = model.predict([instance])[0]
    # Return the predicted value as a JSON response
    return jsonify({'prediction': y_pred})

if __name__ == '__main__':
    app.run(debug=True)
