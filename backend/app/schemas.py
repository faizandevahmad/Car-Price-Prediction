from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class CarFeatures(BaseModel):
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., ge=1995, le=2026, description="Model year")
    mileage: int = Field(..., ge=0, le=400000, description="Odometer reading in miles")
    condition: str = Field(..., description="Condition: excellent, good, fair, poor")
    fuel_type: str = Field(..., description="Fuel type: gasoline, hybrid, electric, diesel")
    transmission: str = Field(..., description="Transmission: automatic or manual")
    body_type: str = Field(..., description="Body style: sedan, suv, hatchback, coupe, truck, wagon")
    accident_history: bool = Field(False, description="Whether the car has accident history")
    owners: int = Field(1, ge=1, le=8, description="Number of previous owners")


class PredictionResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    currency: str = "USD"
    confidence: float
    advice: str
    model_version: int
    feature_importance: Dict[str, float]


class FeedbackRequest(BaseModel):
    features: CarFeatures
    actual_price: float = Field(..., ge=500, le=200000)
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    message: str
    model_version: int
    samples_used: int
    metrics: Dict[str, float]
    previous_metrics: Dict[str, float]


class ModelInfo(BaseModel):
    model_version: int
    samples_used: int
    metrics: Dict[str, float]
    makes: List[str]
    models_by_make: Dict[str, List[str]]
    conditions: List[str]
    fuel_types: List[str]
    transmissions: List[str]
    body_types: List[str]
    last_trained_at: Optional[str]
    feature_names: List[str]
    status: str


class TrainResponse(BaseModel):
    message: str
    model_version: int
    samples_used: int
    metrics: Dict[str, float]


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    make: str
    model: str
    year: int
    predicted_price: Optional[float]
    actual_price: float
    notes: Optional[str]


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
