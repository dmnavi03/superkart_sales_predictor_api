# Import necessary libraries
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_sales_predictor_api = Flask("Superkart Sales Predictor")
superkart_sales_predictor_api.logger.setLevel(logging.INFO)

MODEL_PATH = Path(__file__).resolve().parent / "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


def normalize_input_frame(frame):
    """Normalize known spelling variants without inventing feature values."""
    frame = frame.copy()
    if "Product_Sugar_Content" in frame.columns:
        frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace({"reg": "Regular"})
    if "Product_Type_Category" in frame.columns:
        frame["Product_Type_Category"] = frame["Product_Type_Category"].replace({
            "Perishables": "Perishable",
            "Non Perishables": "Non-Perishable",
            "Non-Perishable ": "Non-Perishable",
        })
    return frame


def validate_features(frame):
    if frame.empty:
        return None, "Input contains no rows."
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        return None, f"Missing required feature columns: {missing}"
    # Ignore extra columns, then restore training order before calling the pipeline.
    frame = normalize_input_frame(frame.loc[:, FEATURE_COLUMNS])
    return frame, None


@superkart_sales_predictor_api.get("/")
def home():
    return "Welcome to the SuperKart Product Sales Prediction API!"


@superkart_sales_predictor_api.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@superkart_sales_predictor_api.post("/v1/superkart")
def predict_superkart_sales():
    product_data = request.get_json(silent=True)
    if not isinstance(product_data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    frame, error = validate_features(pd.DataFrame([product_data]))
    if error:
        return jsonify({"error": error}), 400

    try:
        prediction = float(model.predict(frame)[0])
        return jsonify({"Predicted Product Store Sales": round(prediction, 2)})
    except Exception:
        superkart_sales_predictor_api.logger.exception("Online prediction failed")
        return jsonify({"error": "Prediction failed. Check backend container logs for details."}), 500


@superkart_sales_predictor_api.post("/v1/superkartbatch")
def predict_superkart_sales_batch():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload a CSV using the multipart field name 'file'."}), 400

    try:
        input_data = pd.read_csv(uploaded_file)
    except Exception:
        superkart_sales_predictor_api.logger.exception("Could not parse uploaded CSV")
        return jsonify({"error": "Uploaded file could not be read as a CSV."}), 400

    frame, error = validate_features(input_data)
    if error:
        return jsonify({"error": error}), 400

    try:
        predictions = [round(float(value), 2) for value in model.predict(frame)]
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception:
        superkart_sales_predictor_api.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed. Check backend container logs for details."}), 500


if __name__ == "__main__":
    superkart_sales_predictor_api.run(host="0.0.0.0", port=7860, debug=False)
