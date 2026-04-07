import pandas as pd
import numpy as np
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# ── Load & Preprocess ──────────────────────────────────────────────
df = pd.read_csv("/home/harshjoshi/Downloads/ChurnPrediction/Churn.csv")

df.drop('Customer ID', axis=1, inplace=True)

df['Total Charges'] = df['Total Charges'].str.strip()
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(df['Monthly Charges'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

binary_cols = ['Partner', 'Dependents', 'Phone Service', 'Paperless Billing', 'Gender']
df[binary_cols] = df[binary_cols].apply(
    lambda col: col.map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})
)

three_val_cols = ['Multiple Lines', 'Online Security', 'Online Backup',
                  'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
df[three_val_cols] = df[three_val_cols].apply(
    lambda col: col.map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
)

df = df.drop_duplicates().reset_index(drop=True)

# ── Churn Model ────────────────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = ['tenure', 'Monthly Charges', 'Total Charges']
categorical_cols = ['Internet Service', 'Contract', 'Payment Method']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='passthrough')

churn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        scale_pos_weight=5175/1869,
        random_state=42,
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=5,
        n_estimators=100
    ))
])

churn_pipeline.fit(X_train, y_train)
joblib.dump(churn_pipeline, '/home/harshjoshi/Downloads/ChurnPrediction/Models/churn_model.pkl')
print("✅ Churn model saved.")

# ── Segmentation Model ─────────────────────────────────────────────
churned_df = df[df['Churn'] == 1].drop(columns=['Churn'])

churned_df['Contract'] = churned_df['Contract'].map({
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
})

churned_df = pd.get_dummies(churned_df,
                             columns=['Internet Service', 'Payment Method'],
                             drop_first=True)

scaler = StandardScaler()
churned_scaled = scaler.fit_transform(churned_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(churned_scaled)

segment_map = {
    "0": "Unsatisfied Churner",
    "1": "Lifestyle Migrator",
    "2": "Conditional Churner"
}

joblib.dump(churn_pipeline, '/home/harshjoshi/Downloads/ChurnPrediction/Models/churn_model.pkl')
joblib.dump(kmeans, '/home/harshjoshi/Downloads/ChurnPrediction/Models/segmentation_model.pkl')
joblib.dump(scaler, '/home/harshjoshi/Downloads/ChurnPrediction/Models/scaler.pkl')

with open('/home/harshjoshi/Downloads/ChurnPrediction/Models/segment_map.json', 'w') as f:
    json.dump(segment_map, f)

print("✅ Segmentation model saved.")
print("✅ Scaler saved.")
print("✅ Segment map saved.")
print("\n All models retrained and saved on scikit-learn 1.8.0")