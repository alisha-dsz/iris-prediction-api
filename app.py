# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "Iris Flower Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON input
    features = [
        data['sepal_length'], 
        data['sepal_width'], 
        data['petal_length'], 
        data['petal_width']
    ]
    prediction = model.predict([features])[0]
    classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return jsonify({'prediction': classes[prediction]})

if __name__ == "__main__":
    app.run(debug=True)
