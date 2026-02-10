from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd

app = Flask(__name__)

# -------- LOAD MODEL ONCE (IMPORTANT) --------
model = joblib.load("attrition_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------- HOME ROUTE ----------------
@app.route("/", methods=["GET"])
def home():
    #return "Employee Attrition Prediction API is running"
    return render_template("index.html")

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    #data = request.get_json()

    # Convert input JSON to DataFrame
    # df = pd.DataFrame([data])
    # df_encoded = pd.get_dummies(df)
    # df_encoded = df_encoded.reindex    (columns=model_columns, fill_value=0)
    # # Keep only training-time columns
    # df_encoded = df_encoded[model_columns]
    # # Prediction
    # pred = int(model.predict(df_encoded)[0])
    # proba = model.predict_proba(df_encoded)[0][1]
    # return jsonify({
    #     "Attrition": pred,
    #     "Probability_of_Leaving": round(float(proba), 3)
    # })

    #using form submission 
    # Extract form data
    data = request.form.to_dict()

    # Convert numeric fields
    data['age'] = int(data['age'])
    data['length_of_service'] = float(data['length_of_service'])

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    pred = int(model.predict(df_encoded)[0])
    proba = float(model.predict_proba(df_encoded)[0][1])

    # Pass prediction back to template
    return render_template("index.html", prediction={"Attrition": pred, "Probability": round(proba, 3)}, form_data=data)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
