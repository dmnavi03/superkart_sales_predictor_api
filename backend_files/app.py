# 🔴 CHANGES: hash-visible model version, training-schema validation, non-negative outputs, and safe errors.
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask("SuperKart Sales Predictor")
app.logger.setLevel(logging.INFO)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "superkart_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    raise FileNotFoundError("Model or metadata is missing. Run the corrected training/serialization cells first.")

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())
FEATURE_COLUMNS = metadata["feature_columns"]
NUMERIC_RANGES = metadata["numeric_ranges"]
CATEGORICAL_VALUES = metadata["categorical_values"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

MODEL_SHA256 = sha256_file(MODEL_PATH)
METADATA_SHA256 = sha256_file(METADATA_PATH)


def normalize_input_frame(frame):
    """Normalize known textual variants; do not silently invent categories."""
    frame = frame.copy()
    if "Product_Sugar_Content" in frame.columns:
        values = frame["Product_Sugar_Content"].astype("string").str.strip().str.casefold()
        normalized = values.map({
            "low sugar": "Low Sugar", "regular": "Regular", "reg": "Regular", "no sugar": "No Sugar"
        })
        frame["Product_Sugar_Content"] = normalized.where(normalized.notna(), frame["Product_Sugar_Content"].astype("string").str.strip())
    if "Product_Type_Category" in frame.columns:
        values = frame["Product_Type_Category"].astype("string").str.strip().str.casefold()
        normalized = values.map({
            "perishable": "Perishable", "perishables": "Perishable",
            "non-perishable": "Non-Perishable", "non perishable": "Non-Perishable",
            "non perishables": "Non-Perishable",
        })
        frame["Product_Type_Category"] = normalized.where(normalized.notna(), frame["Product_Type_Category"].astype("string").str.strip())
    for column in ("Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
    return frame


def validate_features(frame):
    if frame.empty:
        return None, "Input contains no rows."
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        return None, f"Missing required feature columns: {missing}"

    frame = normalize_input_frame(frame.loc[:, FEATURE_COLUMNS])
    issues = []

    # Numeric parsing and range validation prevents silent out-of-domain predictions.
    for column, bounds in NUMERIC_RANGES.items():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid_text = frame[column].notna() & numeric.isna()
        if invalid_text.any():
            issues.append(f"{column} contains non-numeric values at rows {frame.index[invalid_text].tolist()[:10]}")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            issues.append(f"{column} contains infinite values")
        out_of_range = numeric.notna() & ((numeric < bounds["min"]) | (numeric > bounds["max"]))
        if out_of_range.any():
            rows = frame.index[out_of_range].tolist()[:10]
            issues.append(f"{column} must be within [{bounds['min']}, {bounds['max']}]; invalid rows: {rows}")
        frame[column] = numeric

    for column, allowed_values in CATEGORICAL_VALUES.items():
        observed = set(frame[column].dropna().astype(str))
        unknown = sorted(observed - set(allowed_values))
        if unknown:
            issues.append(f"Unsupported {column} categories: {unknown}. Allowed values: {allowed_values}")

    if issues:
        return None, "Input validation failed: " + "; ".join(issues)
    return frame, None


@app.get("/")
def home():
    return "Welcome to the SuperKart Product Sales Prediction API!"


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_sha256": MODEL_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "feature_count": len(FEATURE_COLUMNS),
        "selected_model": metadata.get("selected_model"),
    })


@app.post("/v1/superkart")
def predict_superkart_sales():
    product_data = request.get_json(silent=True)
    if not isinstance(product_data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    frame, error = validate_features(pd.DataFrame([product_data]))
    if error:
        return jsonify({"error": error}), 400

    try:
        raw_prediction = float(np.asarray(model.predict(frame), dtype=float).reshape(-1)[0])
        if not np.isfinite(raw_prediction):
            raise ValueError("Model produced a non-finite prediction.")
        was_clipped = raw_prediction < 0
        prediction = max(0.0, raw_prediction)  # revenue cannot be negative
        if was_clipped:
            app.logger.warning("Negative online prediction clipped to zero; investigate model/domain shift.")
        return jsonify({
            "Predicted Product Store Sales": round(prediction, 2),
            "prediction_clipped": was_clipped,
            "model_sha256": MODEL_SHA256,
        })
    except Exception:
        app.logger.exception("Online prediction failed")
        return jsonify({"error": "Prediction failed. Check backend container logs for details."}), 500


@app.post("/v1/superkartbatch")
def predict_superkart_sales_batch():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload a CSV using the multipart field name 'file'."}), 400

    try:
        input_data = pd.read_csv(uploaded_file)
    except Exception:
        app.logger.exception("Could not parse uploaded CSV")
        return jsonify({"error": "Uploaded file could not be read as a CSV."}), 400

    frame, error = validate_features(input_data)
    if error:
        return jsonify({"error": error}), 400

    try:
        raw_predictions = np.asarray(model.predict(frame), dtype=float).reshape(-1)
        if not np.isfinite(raw_predictions).all():
            raise ValueError("Model produced non-finite predictions.")
        clipped_count = int((raw_predictions < 0).sum())
        predictions = np.maximum(raw_predictions, 0.0)
        if clipped_count:
            app.logger.warning("Clipped %s negative batch predictions; investigate model/domain shift.", clipped_count)
        return jsonify({
            "predictions": [round(float(value), 2) for value in predictions],
            "count": len(predictions),
            "clipped_negative_predictions": clipped_count,
            "model_sha256": MODEL_SHA256,
        })
    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed. Check backend container logs for details."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
