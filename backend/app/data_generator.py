"""Synthetic used-car dataset generator for FarePulse."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAKES_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Fit"],
    "Ford": ["F-150", "Escape", "Focus", "Mustang", "Explorer"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "i4"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
    "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Ioniq 5"],
    "Chevrolet": ["Malibu", "Equinox", "Silverado", "Traverse", "Bolt"],
    "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder", "Leaf"],
}

BASE_PRICES = {
    ("Toyota", "Camry"): 28000,
    ("Toyota", "Corolla"): 24000,
    ("Toyota", "RAV4"): 32000,
    ("Toyota", "Highlander"): 40000,
    ("Toyota", "Prius"): 30000,
    ("Honda", "Civic"): 25000,
    ("Honda", "Accord"): 29000,
    ("Honda", "CR-V"): 33000,
    ("Honda", "Pilot"): 41000,
    ("Honda", "Fit"): 20000,
    ("Ford", "F-150"): 45000,
    ("Ford", "Escape"): 30000,
    ("Ford", "Focus"): 22000,
    ("Ford", "Mustang"): 38000,
    ("Ford", "Explorer"): 42000,
    ("BMW", "3 Series"): 45000,
    ("BMW", "5 Series"): 58000,
    ("BMW", "X3"): 50000,
    ("BMW", "X5"): 65000,
    ("BMW", "i4"): 56000,
    ("Tesla", "Model 3"): 42000,
    ("Tesla", "Model Y"): 48000,
    ("Tesla", "Model S"): 85000,
    ("Tesla", "Model X"): 90000,
    ("Hyundai", "Elantra"): 23000,
    ("Hyundai", "Sonata"): 27000,
    ("Hyundai", "Tucson"): 31000,
    ("Hyundai", "Santa Fe"): 36000,
    ("Hyundai", "Ioniq 5"): 44000,
    ("Chevrolet", "Malibu"): 26000,
    ("Chevrolet", "Equinox"): 30000,
    ("Chevrolet", "Silverado"): 46000,
    ("Chevrolet", "Traverse"): 38000,
    ("Chevrolet", "Bolt"): 32000,
    ("Nissan", "Altima"): 27000,
    ("Nissan", "Sentra"): 22000,
    ("Nissan", "Rogue"): 31000,
    ("Nissan", "Pathfinder"): 39000,
    ("Nissan", "Leaf"): 30000,
}

CONDITIONS = ["excellent", "good", "fair", "poor"]
CONDITION_MULT = {"excellent": 1.12, "good": 1.0, "fair": 0.82, "poor": 0.62}
FUEL_TYPES = ["gasoline", "hybrid", "electric", "diesel"]
FUEL_MULT = {"gasoline": 1.0, "hybrid": 1.08, "electric": 1.15, "diesel": 1.05}
TRANSMISSIONS = ["automatic", "manual"]
TRANS_MULT = {"automatic": 1.03, "manual": 0.95}
BODY_TYPES = ["sedan", "suv", "hatchback", "coupe", "truck", "wagon"]
BODY_MULT = {
    "sedan": 1.0,
    "suv": 1.1,
    "hatchback": 0.92,
    "coupe": 1.05,
    "truck": 1.18,
    "wagon": 0.96,
}

FEATURE_COLUMNS = [
    "make",
    "model",
    "year",
    "mileage",
    "condition",
    "fuel_type",
    "transmission",
    "body_type",
    "accident_history",
    "owners",
    "price",
]


def _likely_fuel(make: str, model: str, rng: np.random.Generator) -> str:
    if make == "Tesla" or model in {"Bolt", "Leaf", "Ioniq 5", "i4"}:
        return "electric"
    if model in {"Prius"} or rng.random() < 0.12:
        return "hybrid"
    if rng.random() < 0.08:
        return "diesel"
    return "gasoline"


def _likely_body(model: str, rng: np.random.Generator) -> str:
    suv = {
        "RAV4",
        "Highlander",
        "CR-V",
        "Pilot",
        "Escape",
        "Explorer",
        "X3",
        "X5",
        "Model Y",
        "Model X",
        "Tucson",
        "Santa Fe",
        "Equinox",
        "Traverse",
        "Rogue",
        "Pathfinder",
    }
    trucks = {"F-150", "Silverado"}
    coupes = {"Mustang"}
    hatch = {"Fit", "Focus", "Bolt", "Leaf", "Prius"}
    if model in trucks:
        return "truck"
    if model in suv:
        return "suv"
    if model in coupes:
        return "coupe"
    if model in hatch:
        return "hatchback"
    if rng.random() < 0.08:
        return "wagon"
    return "sedan"


def generate_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    make_list = list(MAKES_MODELS.keys())

    for _ in range(n_samples):
        make = rng.choice(make_list)
        model = rng.choice(MAKES_MODELS[make])
        year = int(rng.integers(2005, 2026))
        age = max(2026 - year, 0)
        mileage = int(rng.integers(2_000, 18_000) * max(age, 1) + rng.integers(0, 8000))
        mileage = int(np.clip(mileage, 500, 350_000))
        condition = str(rng.choice(CONDITIONS, p=[0.18, 0.45, 0.28, 0.09]))
        fuel_type = _likely_fuel(make, model, rng)
        transmission = "automatic" if rng.random() > 0.12 else "manual"
        if fuel_type == "electric":
            transmission = "automatic"
        body_type = _likely_body(model, rng)
        accident = bool(rng.random() < 0.18)
        owners = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.3, 0.15, 0.07, 0.03]))

        base = BASE_PRICES[(make, model)]
        age_factor = (0.92 ** age) * (1.02 if year >= 2022 else 1.0)
        mileage_factor = max(0.35, 1.0 - (mileage / 220_000) * 0.75)
        price = (
            base
            * age_factor
            * mileage_factor
            * CONDITION_MULT[condition]
            * FUEL_MULT[fuel_type]
            * TRANS_MULT[transmission]
            * BODY_MULT[body_type]
        )
        if accident:
            price *= 0.78
        price *= max(0.7, 1.0 - (owners - 1) * 0.045)
        price *= float(rng.normal(1.0, 0.06))
        price = float(np.clip(price, 1200, 160_000))

        rows.append(
            {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "condition": condition,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "body_type": body_type,
                "accident_history": int(accident),
                "owners": owners,
                "price": round(price, 2),
            }
        )

    return pd.DataFrame(rows)
