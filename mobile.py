from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the trained model
model = joblib.load("mobile_price_model.pkl")

# Serve the frontend dashboard
@app.route('/', methods=['GET'])
def dashboard():
    return render_template("mob.html")   # YOUR FRONTEND FILE

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # get JSON input
    df = pd.DataFrame([data])  # convert to DataFrame

    prediction = model.predict(df)[0]

    labels = {
        0: "Low Price",
        1: "Medium Price",
        2: "High Price",
        3: "Very High Price"
    }

    return jsonify({
        "prediction": int(prediction),
        "label": labels[prediction]
    })


if __name__ == '__main__':
    app.run(debug=True)
