from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("model", "svm_model.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get user input
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])

            # Predict using the trained model
            input_features = np.array([[x1, x2]])
            predicted_class = model.predict(input_features)[0]

            prediction = f"Predicted Class: {predicted_class}"

        except ValueError:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
