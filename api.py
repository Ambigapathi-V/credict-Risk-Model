from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from src.Credit_Risk_Model.logger import logger  # Ensure logger is correctly configured

# Initialize FastAPI app
app = FastAPI()

# Set CORS middleware (allow specific origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with trusted origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor paths
MODEL_PATH = "artifacts/model_training/best_model.joblib"
PREPROCESSOR_PATH = "model/preprocessor.joblib"

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading model or preprocessor")

# Define input request schema with specified columns
class PredictionRequest(BaseModel):
    age: int
    residence_type: str
    loan_purpose: str
    loan_type: str
    loan_tenure_months: int
    number_of_open_accounts: int
    loan_amount: float
    income: float
    delinquent_months: int
    total_dpd: int
    credit_utilization_ratio: float
    

# Define response model schema
class PredictionResponse(BaseModel):
    repayment_probability: float
    credit_score: int
    rating: str

# Feature engineering function for additional columns
def encoding_columns(df):
    try:
        df['loan_to_income'] = round(df['loan_amount'] / df['income'], 2)
        if 'total_loan_months' not in df.columns:
            df['total_loan_months'] = df['loan_tenure_months']
        df['delinquency_ratio'] = (df['delinquent_months'] * 100 / df['total_loan_months']).round(1)
        df['avg_dpd_per_delinquency'] = np.where(df['delinquent_months'] != 0, 
                                                  (df['total_dpd'] / df['delinquent_months']).round(1), 0)
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise HTTPException(status_code=400, detail="Error in feature engineering")

# Credit score calculation function
def calculate_credit_score(input_array, base_score=300, scale_length=600):
    try:
        # Obtain the probability of the person being a non-default (repayment probability)
        repayment_probability = model.predict_proba(input_array)[:, 0]  # Probability of the positive class (repayment)
        credit_score = base_score + repayment_probability * scale_length

        # Determine rating based on credit score
        def get_rating(score):
            if 300 <= score < 500:
                return 'Poor'
            elif 500 <= score < 650:
                return 'Average'
            elif 650 <= score < 750:
                return 'Good'
            elif 750 <= score <= 900:
                return 'Excellent'
            else:
                return 'Undefined'

        rating = get_rating(credit_score[0])
        return repayment_probability[0], int(credit_score[0]), rating
    except Exception as e:
        logger.error(f"Error in credit score calculation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating credit score")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionRequest):
    try:
        # Prepare data for prediction
        data = pd.DataFrame([input_data.dict()])
        logger.info(f"Received prediction request: {data.to_dict()}")

        # Feature engineering
        data = encoding_columns(data)

        # Ensure data is in the format expected by the preprocessor
        data = pd.DataFrame(data, columns=preprocessor.feature_names_in_)

        # Apply preprocessor and get a NumPy array
        transformed_data = preprocessor.transform(data)

        # Calculate credit score and ratings
        repayment_prob, credit_score, rating = calculate_credit_score(transformed_data)
        
        # Return prediction results
        return PredictionResponse(
            repayment_probability=repayment_prob,
            credit_score=credit_score,
            rating=rating
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")
    

# Run the app (if running the script directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
