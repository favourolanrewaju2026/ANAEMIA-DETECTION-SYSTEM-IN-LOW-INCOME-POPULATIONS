
# Flask Application - Anemia Detection System


# Import Required Libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd


# Initialize Flask App
app = Flask(__name__)


# Load Trained Model (Pickle)
with open("model/anemia_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        input_data = None
        try:
            input_data = pd.DataFrame({
                "Gender": [request.form["Gender"]],
                "Hemoglobin": [float(request.form["Hemoglobin"])],
                "MCH": [float(request.form["MCH"])],
                "MCHC": [float(request.form["MCHC"])],
                "MCV": [float(request.form["MCV"])]
            })

            print("INPUT DATA:")
            print(input_data)
            print(input_data.dtypes)

            result = model.predict(input_data)
            print("RAW PRED:", result)

            proba = model.predict_proba(input_data)
            print("RAW PROBA:", proba)

            prediction = "Anemic" if result[0] == 1 else "Not Anemic"
            probability = f"{proba[0][1] * 100:.2f}%"

        except Exception as e:
            prediction = "Prediction failed"
            probability = None
            print("🔥 FULL ERROR TRACE:")
            import traceback
            traceback.print_exc()

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )





# Run Flask Application
if __name__ == "__main__":
    app.run(debug=True)
