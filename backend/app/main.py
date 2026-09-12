"""FarePulse API — used-car price prediction with continuous fine-tuning."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .model_service import service
from .schemas import (
    CarFeatures,
    FeedbackRequest,
    FeedbackResponse,
    HistoryResponse,
    ModelInfo,
    PredictionResponse,
    TrainResponse,
)

app = FastAPI(
    title="FarePulse API",
    description="Used-car market price prediction with live fine-tuning from real sale prices.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "FarePulse"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return service.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    info = service.info()
    if features.make not in info["makes"]:
        raise HTTPException(400, f"Unknown make. Choose from: {', '.join(info['makes'])}")
    models = info["models_by_make"].get(features.make, [])
    if features.model not in models:
        raise HTTPException(400, f"Unknown model for {features.make}. Choose from: {', '.join(models)}")
    if features.condition.lower() not in info["conditions"]:
        raise HTTPException(400, "Invalid condition.")
    if features.fuel_type.lower() not in info["fuel_types"]:
        raise HTTPException(400, "Invalid fuel type.")
    if features.transmission.lower() not in info["transmissions"]:
        raise HTTPException(400, "Invalid transmission.")
    if features.body_type.lower() not in info["body_types"]:
        raise HTTPException(400, "Invalid body type.")

    payload = features.model_dump()
    payload["condition"] = payload["condition"].lower()
    payload["fuel_type"] = payload["fuel_type"].lower()
    payload["transmission"] = payload["transmission"].lower()
    payload["body_type"] = payload["body_type"].lower()
    try:
        return service.predict(payload)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest):
    """Submit an actual sale/purchase price to fine-tune the model."""
    features = body.features.model_dump()
    features["condition"] = features["condition"].lower()
    features["fuel_type"] = features["fuel_type"].lower()
    features["transmission"] = features["transmission"].lower()
    features["body_type"] = features["body_type"].lower()
    try:
        return service.fine_tune(features, body.actual_price, body.notes)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/train", response_model=TrainResponse)
def train(reset_dataset: bool = Query(False, description="Regenerate base synthetic data")):
    meta = service.train(reset_dataset=reset_dataset)
    return {
        "message": "Model re-trained successfully.",
        "model_version": meta["model_version"],
        "samples_used": meta["samples_used"],
        "metrics": meta["metrics"],
    }


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(25, ge=1, le=100)):
    items, total = service.history(limit=limit)
    return {"items": items, "total": total}
