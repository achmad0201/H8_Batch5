from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained linear regression model
model = joblib.load('model.pkl')

# Load the scaler used for normalization
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        distance = float(request.form['distance'])
        name = request.form['name']
        surge_multiplier = float(request.form['surge_multiplier'])

        # Convert input to normalized data
        input_data = np.array([[distance, surge_multiplier]])
        input_data_normalized = scaler.transform(input_data)

        # Include logic to handle different service names if needed

        # Make prediction using the model
        prediction = model.predict(input_data_normalized)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
