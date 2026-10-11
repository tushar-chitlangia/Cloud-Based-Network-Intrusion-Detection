import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "nids_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "nids_metadata.json"

with METADATA_PATH.open() as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]
CLASSES = metadata["classes"]
MODEL = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AI-Based Network Intrusion Detection API",
    description="Predict network traffic classes using the trained ML pipeline.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(FEATURES),
        "classes": CLASSES,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata.get("model_name", "Random Forest"),
        "feature_count": len(FEATURES),
        "feature_names": FEATURES,
        "classes": CLASSES,
        "metrics": metadata.get("metrics", {}),
    }



@app.post("/predict-batch")
def predict_batch(payload: dict):
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=422, detail="Provide a non-empty records list.")
    if len(records) > 5000:
        raise HTTPException(status_code=413, detail="Maximum batch size is 5000 records.")

    rows = []
    expected = set(FEATURES)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise HTTPException(status_code=422, detail=f"Record {index} must be an object.")

        supplied = set(record)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing or unexpected:
            raise HTTPException(
                status_code=422,
                detail={
                    "record": index,
                    "missing_features": missing,
                    "unexpected_features": unexpected,
                },
            )

        row = {}
        for name in FEATURES:
            try:
                value = float(record[name])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}, feature '{name}' must be numeric.",
                )
            if not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Record {index}, feature '{name}' must be finite.",
                )
            row[name] = value
        rows.append(row)

    try:
        frame = pd.DataFrame(rows, columns=FEATURES)
        predictions = MODEL.predict(frame)
        results = []
        probabilities = MODEL.predict_proba(frame) if hasattr(MODEL, "predict_proba") else None

        for index, prediction in enumerate(predictions):
            item = {"prediction": str(prediction)}
            if probabilities is not None:
                item["probabilities"] = {
                    str(label): round(float(prob), 6)
                    for label, prob in zip(MODEL.classes_, probabilities[index])
                }
            results.append(item)

        return {"count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
    
@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    expected = set(FEATURES)
    received = set(supplied)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature names do not match the model schema.",
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    for name, value in supplied.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be a finite number.",
            )

    row = pd.DataFrame(
        [[supplied[name] for name in FEATURES]],
        columns=FEATURES,
    )

    try:
        prediction = str(MODEL.predict(row)[0])
        result = {"prediction": prediction}

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(row)[0]
            result["probabilities"] = {
                str(label): round(float(probability), 6)
                for label, probability in zip(MODEL.classes_, probabilities)
            }

        return result

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        )
