# FarePulse

Used-car market price intelligence: train on a dataset, predict from vehicle details, then **fine-tune live** when users submit real sale prices.

## What it does

1. Generates a synthetic used-car dataset (makes, models, year, mileage, condition, etc.)
2. Trains a **Gradient Boosting** regressor (scikit-learn) in Python
3. Serves predictions over a **FastAPI** backend
4. Accepts sale-price feedback and **re-trains** the model automatically
5. React frontend for estimates, fine-tuning, metrics, and feedback history

## Project layout

```
backend/          FastAPI + ML pipeline
frontend/         React + Vite UI
```

## Quick start

### 1. Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://127.0.0.1:5173  
(Vite proxies `/api` → backend `:8000`)

## Main APIs

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model version, metrics, dropdown options |
| POST | `/predict` | Price estimate from car features |
| POST | `/feedback` | Submit actual price → append data + retrain |
| POST | `/train` | Manual retrain (`?reset_dataset=true` regenerates base data) |
| GET | `/history` | Recent fine-tune feedback |

## Example predict body

```json
{
  "make": "Toyota",
  "model": "Camry",
  "year": 2019,
  "mileage": 45000,
  "condition": "good",
  "fuel_type": "gasoline",
  "transmission": "automatic",
  "body_type": "sedan",
  "accident_history": false,
  "owners": 1
}
```

## Continuous learning loop

User gets a prediction → submits actual sale price → sample is added to `backend/data/used_cars.csv` → model retrains → version/metrics update → next predictions use the improved model.
