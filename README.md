# Customer Churn Prediction and Segmentation

A machine learning system that predicts customer churn and segments at-risk customers into behavioral groups with targeted retention strategies. Built with XGBoost, KMeans clustering, FastAPI, Pydantic, and Streamlit.

---

## Overview

The system takes customer data as input and runs it through a two-stage pipeline. In the first stage, a binary classifier determines whether a customer is likely to churn. If churn is predicted, the data is passed to a segmentation model that assigns the customer to one of three behavioral segments and recommends a set of retention strategies.

---

## Dataset

The model was trained on a telecom customer churn dataset with 7,044 records and 21 features including demographic information, subscribed services, contract type, billing method, and monthly charges. The target variable is binary churn (Yes/No) with a 73/27 class distribution.

---

## System Architecture

```
Customer Input (Streamlit UI)
        |
        v
Pydantic Validation
        |
        v
FastAPI /predict endpoint
        |
        v
Churn Prediction Model (XGBoost)
        |
   _____|_____
  |           |
No Churn    Churn Predicted
  |           |
  v           v
Return      Segmentation Model (KMeans)
Result              |
                    v
             Assign Segment
                    |
                    v
          Retention Strategy Engine
                    |
                    v
           Return Full Response
```

---

## Models

### Churn Prediction

- Algorithm: XGBoost binary classifier
- Handling class imbalance: scale_pos_weight set to ratio of negative to positive class
- Hyperparameters tuned via 5-fold GridSearchCV
- Evaluation metric: F1-score on the Churn class
- Final performance: Churn Recall 0.82, Churn F1 0.64

### Customer Segmentation

- Algorithm: KMeans with 3 clusters
- Applied only to churned customers (1,857 records after deduplication)
- Features scaled with StandardScaler before clustering
- Segments defined based on cluster profiling of tenure, monthly charges, contract type, and senior citizen status

---

## Customer Segments

| Segment | Profile | Strategies |
|---|---|---|
| Unsatisfied Churner | Short tenure, high monthly charges, month-to-month contract | Service quality improvement offer, downgrade option, priority tech support outreach |
| Lifestyle Migrator | Long tenure, highest charges, mixed contracts | Pause plan option, loyalty reward, flexible contract offer |
| Conditional Churner | Shortest tenure, lowest charges, month-to-month contract | Discount offer, competitor price match guarantee, shorter commitment plan option |

---

## Project Structure

```
churn_prediction/
├── app/
│   ├── main.py               FastAPI backend
│   ├── schemas.py            Pydantic input and output schemas
│   └── streamlit_app.py      Streamlit frontend
├── models/
│   ├── churn_model.pkl       Trained XGBoost pipeline
│   ├── segmentation_model.pkl   Trained KMeans model
│   ├── scaler.pkl            StandardScaler for segmentation input
│   └── segment_map.json      Cluster index to segment name mapping
├── notebooks/
│   ├── 01_churn_preprocessing.ipynb
│   ├── 02_churn_model.ipynb
│   └── 03_segmentation_model.ipynb
├── Churn.csv
│ 
├── train.py           
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

### Run the API

```bash
uvicorn app.main:app --reload
```

API will be available at http://127.0.0.1:8000
Interactive docs at http://127.0.0.1:8000/docs

### Run the UI

Open a second terminal with the venv activated:

```bash
streamlit run app/streamlit_app.py
```

UI will be available at http://localhost:8501

---

## API Reference

### POST /predict

Accepts a customer record and returns churn prediction and optional segmentation.

Request body:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.00,
  "total_charges": 1020.00
}
```

Response when churn is predicted:

```json
{
  "churn": {
    "churn_predicted": true,
    "churn_probability": 0.87,
    "message": "Customer is likely to churn."
  },
  "segmentation": {
    "segment": "Unsatisfied Churner",
    "strategies": [
      "Service quality improvement offer",
      "Downgrade option to lower tier plan",
      "Priority tech support outreach"
    ]
  }
}
```

Response when no churn is predicted:

```json
{
  "churn": {
    "churn_predicted": false,
    "churn_probability": 0.21,
    "message": "Customer is not likely to churn."
  },
  "segmentation": null
}
```

---



## License

MIT
