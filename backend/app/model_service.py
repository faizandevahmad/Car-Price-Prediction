"""FarePulse model training, prediction, and continuous fine-tuning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_generator import (
    BODY_TYPES,
    CONDITIONS,
    FEATURE_COLUMNS,
    FUEL_TYPES,
    MAKES_MODELS,
    TRANSMISSIONS,
    USD_TO_PKR,
    generate_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_PATH = DATA_DIR / "used_cars.csv"
FEEDBACK_PATH = DATA_DIR / "feedback_history.json"
META_PATH = MODEL_DIR / "meta.json"
PIPELINE_PATH = MODEL_DIR / "farepulse_pipeline.joblib"

CATEGORICAL = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "transmission",
    "body_type",
]
NUMERIC = ["year", "mileage", "accident_history", "owners"]
FEATURE_INPUT = CATEGORICAL + NUMERIC


class FarePulseService:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[Pipeline] = None
        self.meta: Dict[str, Any] = {
            "model_version": 0,
            "samples_used": 0,
            "metrics": {},
            "last_trained_at": None,
            "feature_names": FEATURE_INPUT,
            "status": "uninitialized",
        }
        self._bootstrap()

    def _bootstrap(self) -> None:
        if not DATASET_PATH.exists():
            df = generate_dataset(2500)
            df.to_csv(DATASET_PATH, index=False)
        if not FEEDBACK_PATH.exists():
            FEEDBACK_PATH.write_text("[]", encoding="utf-8")

        if PIPELINE_PATH.exists() and META_PATH.exists():
            self.pipeline = joblib.load(PIPELINE_PATH)
            self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            self.meta["status"] = "ready"
        else:
            self.train(reset_dataset=False)

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(DATASET_PATH)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Dataset missing column: {col}")
        df["accident_history"] = df["accident_history"].astype(int)
        return df

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL,
                ),
                ("num", "passthrough", NUMERIC),
            ]
        )
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    def train(self, reset_dataset: bool = False) -> Dict[str, Any]:
        if reset_dataset:
            df = generate_dataset(2500)
            df.to_csv(DATASET_PATH, index=False)
            FEEDBACK_PATH.write_text("[]", encoding="utf-8")
        else:
            df = self._load_dataset()

        X = df[FEATURE_INPUT]
        y = df["price"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
            "r2": round(float(r2_score(y_test, preds)), 4),
        }

        self.pipeline = pipeline
        self.meta = {
            "model_version": int(self.meta.get("model_version", 0)) + 1,
            "samples_used": int(len(df)),
            "metrics": metrics,
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": FEATURE_INPUT,
            "status": "ready",
        }
        joblib.dump(self.pipeline, PIPELINE_PATH)
        META_PATH.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.meta

    def _features_to_frame(self, features: Dict[str, Any]) -> pd.DataFrame:
        row = {
            "make": features["make"],
            "model": features["model"],
            "year": int(features["year"]),
            "mileage": int(features["mileage"]),
            "condition": features["condition"].lower(),
            "fuel_type": features["fuel_type"].lower(),
            "transmission": features["transmission"].lower(),
            "body_type": features["body_type"].lower(),
            "accident_history": int(bool(features.get("accident_history", False))),
            "owners": int(features.get("owners", 1)),
        }
        return pd.DataFrame([row])

    def _feature_importance(self) -> Dict[str, float]:
        if self.pipeline is None:
            return {}
        prep: ColumnTransformer = self.pipeline.named_steps["prep"]
        model: GradientBoostingRegressor = self.pipeline.named_steps["model"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
        except Exception:
            cat_names = [f"cat_{i}" for i in range(len(model.feature_importances_) - len(NUMERIC))]
        names = cat_names + NUMERIC
        importances = model.feature_importances_
        grouped: Dict[str, float] = {c: 0.0 for c in FEATURE_INPUT}
        for name, value in zip(names, importances):
            matched = False
            for col in CATEGORICAL:
                if name.startswith(f"{col}_") or name == col:
                    grouped[col] += float(value)
                    matched = True
                    break
            if not matched and name in NUMERIC:
                grouped[name] += float(value)
        total = sum(grouped.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(grouped.items(), key=lambda x: -x[1])}

    def _advice(self, price: float, features: Dict[str, Any], confidence: float) -> str:
        age = 2026 - int(features["year"])
        bits = [
            f"Estimated market value is about Rs {price:,.0f}.",
        ]
        if features.get("accident_history"):
            bits.append("Accident history typically pulls offers lower—price carefully.")
        if int(features["mileage"]) > 120_000:
            bits.append("High mileage: expect buyers to negotiate harder.")
        elif int(features["mileage"]) < 30_000 and age <= 3:
            bits.append("Low miles for the year—strong asking-price position.")
        if features["condition"] == "excellent":
            bits.append("Excellent condition supports pricing near the top of the range.")
        elif features["condition"] == "poor":
            bits.append("Fair/poor condition usually sells faster with a discount.")
        if confidence >= 0.75:
            bits.append("Model confidence is solid for this profile.")
        else:
            bits.append("Fewer similar cars in training—treat the range as a guide.")
        return " ".join(bits)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained yet.")
        frame = self._features_to_frame(features)
        pred = float(self.pipeline.predict(frame)[0])
        min_price = 800.0 * USD_TO_PKR
        pred = max(min_price, pred)

        # Confidence from residual MAE relative to prediction
        mae = float(self.meta.get("metrics", {}).get("mae", pred * 0.12))
        band = max(mae * 1.15, pred * 0.06)
        confidence = float(np.clip(1.0 - (mae / max(pred, 1.0)), 0.35, 0.95))

        return {
            "predicted_price": round(pred, 2),
            "price_low": round(max(500.0 * USD_TO_PKR, pred - band), 2),
            "price_high": round(pred + band, 2),
            "currency": "PKR",
            "confidence": round(confidence, 3),
            "advice": self._advice(pred, features, confidence),
            "model_version": int(self.meta.get("model_version", 1)),
            "feature_importance": self._feature_importance(),
        }

    def _append_feedback_history(
        self,
        features: Dict[str, Any],
        actual_price: float,
        predicted_price: Optional[float],
        notes: Optional[str],
    ) -> None:
        history = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        history.append(
            {
                "id": len(history) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "make": features["make"],
                "model": features["model"],
                "year": int(features["year"]),
                "predicted_price": predicted_price,
                "actual_price": float(actual_price),
                "notes": notes,
                "features": {
                    **features,
                    "accident_history": int(bool(features.get("accident_history", False))),
                },
            }
        )
        FEEDBACK_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def fine_tune(
        self,
        features: Dict[str, Any],
        actual_price: float,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        previous_metrics = dict(self.meta.get("metrics", {}))
        predicted = None
        try:
            predicted = self.predict(features)["predicted_price"]
        except Exception:
            predicted = None

        # Append labeled sample (with light augmentation for stronger local update)
        df = self._load_dataset()
        base_row = {
            "make": features["make"],
            "model": features["model"],
            "year": int(features["year"]),
            "mileage": int(features["mileage"]),
            "condition": features["condition"].lower(),
            "fuel_type": features["fuel_type"].lower(),
            "transmission": features["transmission"].lower(),
            "body_type": features["body_type"].lower(),
            "accident_history": int(bool(features.get("accident_history", False))),
            "owners": int(features.get("owners", 1)),
            "price": float(actual_price),
        }
        extras = [base_row]
        rng = np.random.default_rng(abs(hash((features["make"], features["model"], actual_price))) % (2**32))
        for _ in range(2):
            jitter = dict(base_row)
            jitter["mileage"] = int(max(0, base_row["mileage"] + int(rng.integers(-800, 800))))
            jitter["price"] = round(float(actual_price) * float(rng.normal(1.0, 0.015)), 2)
            extras.append(jitter)

        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False)
        self._append_feedback_history(features, actual_price, predicted, notes)

        meta = self.train(reset_dataset=False)
        return {
            "message": "Thanks - FarePulse absorbed your sale price and re-trained.",
            "model_version": meta["model_version"],
            "samples_used": meta["samples_used"],
            "metrics": meta["metrics"],
            "previous_metrics": previous_metrics,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "model_version": int(self.meta.get("model_version", 0)),
            "samples_used": int(self.meta.get("samples_used", 0)),
            "metrics": self.meta.get("metrics", {}),
            "makes": list(MAKES_MODELS.keys()),
            "models_by_make": MAKES_MODELS,
            "conditions": CONDITIONS,
            "fuel_types": FUEL_TYPES,
            "transmissions": TRANSMISSIONS,
            "body_types": BODY_TYPES,
            "last_trained_at": self.meta.get("last_trained_at"),
            "feature_names": FEATURE_INPUT,
            "status": self.meta.get("status", "unknown"),
        }

    def history(self, limit: int = 25) -> Tuple[List[Dict[str, Any]], int]:
        history = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        items = list(reversed(history))[:limit]
        slim = [
            {
                "id": h["id"],
                "timestamp": h["timestamp"],
                "make": h["make"],
                "model": h["model"],
                "year": h["year"],
                "predicted_price": h.get("predicted_price"),
                "actual_price": h["actual_price"],
                "notes": h.get("notes"),
            }
            for h in items
        ]
        return slim, len(history)


service = FarePulseService()
