import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

st.title("Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict churn and get retention strategies.")

# ── Customer Demographics ──────────────────────────────────────────
st.subheader(" Customer Demographics")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
with col3:
    partner = st.selectbox("Partner", ["Yes", "No"])

col4, col5 = st.columns(2)
with col4:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
with col5:
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

# ── Services ──────────────────────────────────────────────────────
st.subheader(" Services")
col6, col7, col8 = st.columns(3)

with col6:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
with col7:
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
with col8:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

col9, col10, col11 = st.columns(3)
with col9:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
with col10:
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col11:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

col12, col13, col14 = st.columns(3)
with col12:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
with col13:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
with col14:
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ── Billing ───────────────────────────────────────────────────────
st.subheader("Billing")
col15, col16 = st.columns(2)

with col15:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col16:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

col17, col18 = st.columns(2)
with col17:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=29.85)
with col18:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=29.85)

# ── Predict ───────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict Churn", use_container_width=True):
    payload = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            churn = result["churn"]
            segmentation = result.get("segmentation")

            # ── Churn Result ──
            if churn["churn_predicted"]:
                st.error(f"**Churn Predicted** — Probability: {churn['churn_probability']*100:.1f}%")
            else:
                st.success(f"**No Churn Predicted** — Probability: {churn['churn_probability']*100:.1f}%")

            # ── Segmentation Result ──
            if segmentation:
                st.markdown("---")
                st.subheader("Customer Segment")
                st.info(f"**Segment:** {segmentation['segment']}")

                st.subheader("Retention Strategies")
                for strategy in segmentation["strategies"]:
                    st.markdown(f"- {strategy}")

        except Exception as e:
            st.error(f"API Error: {str(e)}")