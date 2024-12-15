from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (Ensure you have a pre-trained model saved as 'fraud_detection_model.pkl')
model = joblib.load('fraud_detection_model.pkl')

# Sample feature columns expected by the model
# Make sure these match the features used during model training
FEATURES = ['transaction_amount', 'time_of_day', 'user_id', 'location', 'device_type']


# Function to preprocess the input data before passing it to the model
def preprocess_data(data):
    # Convert the input JSON data to a Pandas DataFrame
    df = pd.DataFrame([data])

    # You may need additional preprocessing depending on how your model was trained
    # For example, encoding categorical variables, handling missing values, etc.
    # Example: Handle missing values, scale features, etc. (adjust as needed)

    # For now, just return the dataframe (adjust preprocessing based on your model)
    return df[FEATURES]


# API route to predict fraudulent transactions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Preprocess the data
        processed_data = preprocess_data(data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return prediction result
        result = {
            'fraudulent': bool(prediction[0]),  # If model predicts 1, return True (fraud), else False (non-fraud)
            'prediction': prediction[0]  # 1 for fraud, 0 for non-fraud
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True)