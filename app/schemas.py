from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

    def to_churn_input(self) -> pd.DataFrame:
        """Prepares input for the churn model — Contract stays as string for pipeline OHE"""
        binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        three_val_map = {"Yes": 1, "No": 0,
                         "No internet service": 0, "No phone service": 0}

        data = {
            "Gender": binary_map[self.gender],
            "Senior Citizen": self.senior_citizen,
            "Partner": binary_map[self.partner],
            "Dependents": binary_map[self.dependents],
            "tenure": self.tenure,
            "Phone Service": binary_map[self.phone_service],
            "Multiple Lines": three_val_map[self.multiple_lines],
            "Internet Service": self.internet_service,      # string — pipeline OHEs
            "Online Security": three_val_map[self.online_security],
            "Online Backup": three_val_map[self.online_backup],
            "Device Protection": three_val_map[self.device_protection],
            "Tech Support": three_val_map[self.tech_support],
            "Streaming TV": three_val_map[self.streaming_tv],
            "Streaming Movies": three_val_map[self.streaming_movies],
            "Contract": self.contract,                      # string — pipeline OHEs
            "Paperless Billing": binary_map[self.paperless_billing],
            "Payment Method": self.payment_method,          # string — pipeline OHEs
            "Monthly Charges": self.monthly_charges,
            "Total Charges": self.total_charges,
        }

        return pd.DataFrame([data])

    def to_segment_input(self) -> pd.DataFrame:
        """Prepares input for the segmentation model — Contract encoded as 0/1/2"""
        binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        three_val_map = {"Yes": 1, "No": 0,
                         "No internet service": 0, "No phone service": 0}
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

        data = {
            "Gender": binary_map[self.gender],
            "Senior Citizen": self.senior_citizen,
            "Partner": binary_map[self.partner],
            "Dependents": binary_map[self.dependents],
            "tenure": self.tenure,
            "Phone Service": binary_map[self.phone_service],
            "Multiple Lines": three_val_map[self.multiple_lines],
            "Online Security": three_val_map[self.online_security],
            "Online Backup": three_val_map[self.online_backup],
            "Device Protection": three_val_map[self.device_protection],
            "Tech Support": three_val_map[self.tech_support],
            "Streaming TV": three_val_map[self.streaming_tv],
            "Streaming Movies": three_val_map[self.streaming_movies],
            "Contract": contract_map[self.contract],        # integer — segmentation expects this
            "Paperless Billing": binary_map[self.paperless_billing],
            "Monthly Charges": self.monthly_charges,
            "Total Charges": self.total_charges,
            "Internet Service": self.internet_service,      # string — will be OHE'd manually in main.py
            "Payment Method": self.payment_method,          # string — will be OHE'd manually in main.py
        }

        return pd.DataFrame([data])


class ChurnResponse(BaseModel):
    churn_predicted: bool
    churn_probability: float
    message: str


class SegmentResponse(BaseModel):
    segment: str
    strategies: list[str]


class FullPredictionResponse(BaseModel):
    churn: ChurnResponse
    segmentation: SegmentResponse | None  # None if no churn predicted