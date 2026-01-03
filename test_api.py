import requests

# =========================
# Replace this with your actual API URL
url = "http://127.0.0.1:5000/predict"
# =========================

# Sample Iris flower features
payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

try:
    # Send POST request with JSON data
    response = requests.post(url, json=payload)

    # Print status code
    print("Status Code:", response.status_code)

    # Print raw response text
    print("Response Text:", response.text)

    # Try to parse JSON (if API returns JSON)
    try:
        data = response.json()
        print("Prediction (JSON):", data)
    except requests.exceptions.JSONDecodeError:
        print("API returned plain text, not JSON.")

except requests.exceptions.RequestException as e:
    # Handles network errors, wrong URL, etc.
    print("Error connecting to API:", e)
