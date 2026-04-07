from fastapi import FastAPI, HTTPException
from app.schemas import CustomerInput, FullPredictionResponse, ChurnResponse, SegmentResponse
import joblib
import json
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load models
churn_model = joblib.load("/home/harshjoshi/Downloads/ChurnPrediction/Models/churn_model.pkl")
segmentation_model = joblib.load("/home/harshjoshi/Downloads/ChurnPrediction/Models/segmentation_model.pkl")
scaler = joblib.load("/home/harshjoshi/Downloads/ChurnPrediction/Models/scaler.pkl")

with open("/home/harshjoshi/Downloads/ChurnPrediction/Models/segment_map.json", "r") as f:
    segment_map = json.load(f)

# Retention strategies
strategies = {
    "Unsatisfied Churner": [
        "Service quality improvement offer",
        "Downgrade option to lower tier plan",
        "Priority tech support outreach"
    ],
    "Lifestyle Migrator": [
        "Pause plan option for up to 3 months",
        "Loyalty reward for long tenure",
        "Flexible contract offer"
    ],
    "Conditional Churner": [
        "Discount offer on current plan",
        "Competitor price match guarantee",
        "Shorter commitment plan option"
    ]
}

# Columns the segmentation scaler saw during training — must match exactly
SEGMENT_COLUMNS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'tenure',
    'Phone Service', 'Multiple Lines', 'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
    'Contract', 'Paperless Billing', 'Monthly Charges', 'Total Charges',
    'Internet Service_Fiber optic', 'Internet Service_No',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check', 'Payment Method_Mailed check'
]


def prepare_segment_input(df: pd.DataFrame) -> np.ndarray:
    """OHE Internet Service and Payment Method to match training columns exactly"""
    df = pd.get_dummies(df, columns=['Internet Service', 'Payment Method'])

    # Add any missing columns with 0 — handles unseen categories
    for col in SEGMENT_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches training
    df = df[SEGMENT_COLUMNS]

    return scaler.transform(df)


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict", response_model=FullPredictionResponse)
def predict(customer: CustomerInput):
    try:
        # Step 1 — Churn prediction
        churn_input = customer.to_churn_input()
        churn_prob = churn_model.predict_proba(churn_input)[0][1]
        churn_predicted = churn_prob >= 0.5

        churn_response = ChurnResponse(
            churn_predicted=bool(churn_predicted),
            churn_probability=round(float(churn_prob), 4),
            message="Customer is likely to churn." if churn_predicted else "Customer is not likely to churn."
        )

        # Step 2 — Segmentation (only if churn predicted)
        segment_response = None

        if churn_predicted:
            segment_input = customer.to_segment_input()
            segment_scaled = prepare_segment_input(segment_input)
            segment_id = segmentation_model.predict(segment_scaled)[0]
            segment_label = segment_map[str(segment_id)]
            segment_strategies = strategies[segment_label]

            segment_response = SegmentResponse(
                segment=segment_label,
                strategies=segment_strategies
            )

        return FullPredictionResponse(
            churn=churn_response,
            segmentation=segment_response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))