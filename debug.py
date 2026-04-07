import joblib
import pandas as pd

model = joblib.load('/home/harshjoshi/Downloads/ChurnPrediction/Models/churn_model.pkl')

# Simulate exactly what to_churn_input() sends
test_input = pd.DataFrame([{
    "Gender": 0,
    "Senior Citizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 1,
    "Phone Service": 0,
    "Multiple Lines": 0,
    "Internet Service": "DSL",
    "Online Security": 0,
    "Online Backup": 1,
    "Device Protection": 0,
    "Tech Support": 0,
    "Streaming TV": 0,
    "Streaming Movies": 0,
    "Contract": "Month-to-month",
    "Paperless Billing": 1,
    "Payment Method": "Electronic check",
    "Monthly Charges": 29.85,
    "Total Charges": 29.85
}])

prob = model.predict_proba(test_input)[0][1]
print(f"Churn probability: {prob:.4f}")
print(f"Predicted: {'Churn' if prob >= 0.5 else 'No Churn'}")