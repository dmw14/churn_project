"""
Step 5 — FastAPI Endpoint
Mobile Churn Prediction (Maharashtra Cities)

Wraps the trained sklearn Pipeline (preprocessing + tuned Logistic
Regression, from Step 4) in a REST API. Same structure as the Employee
Management API: Pydantic request/response models, CORS enabled, docs
auto-generated at /docs, in-memory (no DB — the "database" here is the
loaded model artifact, not row storage).

Run locally:
    uvicorn main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for Swagger UI, or test with
Postman / curl against http://127.0.0.1:8000/predict-churn
"""

from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = "artifacts/churn_model_pipeline.joblib"

NUMERIC_COLS = [
    "age", "monthly_income", "tenure_months", "monthly_recharge_amount",
    "data_usage_gb", "call_minutes_used", "sms_count", "number_of_complaints",
    "avg_monthly_bill", "late_payment_count", "customer_service_calls",
    "value_added_services_count", "satisfaction_score",
    "app_usage_hours_per_week", "social_media_usage_hours",
    "number_of_family_lines", "tenure_days_from_signup",
]
CATEGORICAL_COLS = [
    "city", "gender", "plan_type", "complaint_resolved", "network_type",
    "device_brand", "device_price_range", "contract_type",
    "international_roaming_used", "streaming_service_used", "payment_method",
]

model = None 


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded OK from {MODEL_PATH}")
    except FileNotFoundError:
        model = None
        print(f"WARNING: model file not found at {MODEL_PATH} — "
              f"/predict-churn will return 503 until it's fixed")
    yield
    model = None


app = FastAPI(
    title="Mobile Churn Prediction API",
    description="Predicts whether a mobile customer is likely to churn, "
                "using a Logistic Regression pipeline trained on the "
                "Maharashtra-cities churn dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / response models

class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 34,
                "monthly_income": 42000,
                "tenure_months": 6,
                "monthly_recharge_amount": 249.0,
                "data_usage_gb": 4.2,
                "call_minutes_used": 310,
                "sms_count": 18,
                "number_of_complaints": 3,
                "avg_monthly_bill": 260.5,
                "late_payment_count": 2,
                "customer_service_calls": 4,
                "value_added_services_count": 1,
                "satisfaction_score": 2,
                "app_usage_hours_per_week": 5.5,
                "social_media_usage_hours": 3.0,
                "number_of_family_lines": 0,
                "tenure_days_from_signup": 180,
                "city": "Pune",
                "gender": "Female",
                "plan_type": "Prepaid",
                "complaint_resolved": "No",
                "network_type": "4G",
                "device_brand": "Xiaomi",
                "device_price_range": "Low (<10k)",
                "contract_type": "No Contract",
                "international_roaming_used": "No",
                "streaming_service_used": "Yes",
                "payment_method": "UPI",
            }
        }
    )

    # Numeric fields
    age: Optional[float] = Field(None, description="Customer age in years")
    monthly_income: Optional[float] = Field(None, description="Monthly income in INR")
    tenure_months: Optional[float] = Field(None, description="Months since signup (0-96)")
    monthly_recharge_amount: Optional[float] = None
    data_usage_gb: Optional[float] = Field(None, description="Monthly data usage in GB")
    call_minutes_used: Optional[float] = None
    sms_count: Optional[float] = None
    number_of_complaints: Optional[float] = None
    avg_monthly_bill: Optional[float] = Field(None, description="Average monthly bill in INR")
    late_payment_count: Optional[float] = None
    customer_service_calls: Optional[float] = None
    value_added_services_count: Optional[float] = None
    satisfaction_score: Optional[float] = Field(None, description="1 (low) to 5 (high)")
    app_usage_hours_per_week: Optional[float] = None
    social_media_usage_hours: Optional[float] = None
    number_of_family_lines: Optional[float] = None
    tenure_days_from_signup: Optional[float] = Field(
        None, description="(today - signup_date).days — computed by the caller's own DB"
    )

    # Categorical fields
    city: Optional[str] = Field(None, description="e.g. Mumbai, Pune, Thane, Nagpur...")
    gender: Optional[str] = Field(None, description="Male / Female / Other")
    plan_type: Optional[str] = Field(None, description="Prepaid / Postpaid")
    complaint_resolved: Optional[str] = Field(None, description="Yes / No / 'No Complaint Filed'")
    network_type: Optional[str] = Field(None, description="2G / 3G / 4G / 5G")
    device_brand: Optional[str] = None
    device_price_range: Optional[str] = None
    contract_type: Optional[str] = Field(None, description="With Contract / No Contract")
    international_roaming_used: Optional[str] = Field(None, description="Yes / No")
    streaming_service_used: Optional[str] = Field(None, description="Yes / No")
    payment_method: Optional[str] = Field(None, description="UPI / Credit Card / Debit Card / Cash / ...")


class ChurnPrediction(BaseModel):
    churn_probability: float = Field(..., description="Model's probability that this customer churns (0-1)")
    churn_prediction: str = Field(..., description="'Yes' or 'No' at the model's 0.5 threshold")
    risk_level: str = Field(..., description="Low / Medium / High — for prioritizing retention outreach")

# Routes

@app.get("/")
def root():
    return {
        "message": "Mobile Churn Prediction API",
        "docs": "/docs",
        "predict_endpoint": "POST /predict-churn",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict-churn", response_model=ChurnPrediction)
def predict_churn(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs")

    row = {**{c: None for c in NUMERIC_COLS + CATEGORICAL_COLS}, **customer.model_dump()}
    X = pd.DataFrame([row])[NUMERIC_COLS + CATEGORICAL_COLS]

    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    prediction = "Yes" if proba >= 0.5 else "No"
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return ChurnPrediction(churn_probability=round(proba, 4), churn_prediction=prediction, risk_level=risk)
