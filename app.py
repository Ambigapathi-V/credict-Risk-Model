import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import csv
from src.Credit_Risk_Model.logger import logger  # Ensure logger is correctly configured

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
    st.error(f"Error loading model or preprocessor: {str(e)}")
    raise RuntimeError("Model or Preprocessor loading failed")

# Set the title and description of the app
st.set_page_config(page_title="Credit Risk Prediction")
st.title("Credit Risk Prediction")
st.write("""
    This tool allows you to assess an applicant's creditworthiness based on personal and financial details.
    Please fill in the details below, and the app will predict the applicant's credit score, default probability, and rating.
""")

# Input form section
st.subheader("Enter Applicant's Details")

# Using a three-column layout for the form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    residence_type = st.selectbox("Residence Type", ['Owned', 'Rented', 'Mortgage'])
    loan_purpose = st.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'])

with col2:
    loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, value=24)
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=3)

with col3:
    loan_amount = st.number_input("Loan Amount (USD)", min_value=0.0, value=10000.0)
    income = st.number_input("Income (USD)", min_value=0.0, value=5000.0)
    delinquent_months = st.number_input("Delinquent Months", min_value=0, value=1)

# Second row of input fields
col1, col2, col3 = st.columns(3)

with col1:
    total_dpd = st.number_input("Total DPD (Days Past Due)", min_value=0, value=15)

with col2:
    total_loan_months = st.number_input("Total Loan Months", min_value=0, value=12)

with col3:
    credit_utilization_ratio = st.number_input(
        "Credit Utilization Ratio",
        min_value=1,
        max_value=50,
        value=10
    )

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
        st.error("Error in feature engineering")
        raise RuntimeError("Error in feature engineering")

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
        st.error("Error in credit score calculation")
        raise RuntimeError("Error in credit score calculation")

# Function to save input and prediction results to CSV
def save_to_csv(input_data, result, filename='input_data.csv'):
    output_data = {**input_data,
                   'credit_score': result.get('credit_score'),
                   'repayment_probability': result.get('repayment_probability'),
                   'rating': result.get('rating')}
    
    file_exists = os.path.exists(filename)
    
    try:
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['age', 'residence_type', 'loan_purpose', 'loan_type', 'loan_tenure_months',
                          'number_of_open_accounts', 'loan_amount', 'income', 'delinquent_months',
                          'total_dpd', 'total_loan_months', 'credit_utilization_ratio', 'credit_score', 
                          'repayment_probability', 'rating']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(output_data)
        
        logger.info("Data saved to CSV successfully.")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        st.error(f"Error saving data: {e}")

# Prediction Button
if st.button("Predict Credit Risk"):
    # Prepare input data to send for prediction
    input_data = {
        "age": age,
        "residence_type": residence_type,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": number_of_open_accounts,
        "loan_amount": loan_amount,
        "income": income,
        "delinquent_months": delinquent_months,
        "total_dpd": total_dpd,
        "total_loan_months": total_loan_months,
        "credit_utilization_ratio": credit_utilization_ratio
    }

    try:
        # Prepare data for prediction
        data = pd.DataFrame([input_data])
        logger.info(f"Received prediction request: {data.to_dict()}")

        # Feature engineering
        data = encoding_columns(data)

        # Ensure data is in the format expected by the preprocessor
        data = pd.DataFrame(data, columns=preprocessor.feature_names_in_)

        # Apply preprocessor and get a NumPy array
        transformed_data = preprocessor.transform(data)

        # Calculate credit score and ratings
        repayment_prob, credit_score, rating = calculate_credit_score(transformed_data)
        
        # Display the results
        st.success(f"Credit Score: {credit_score}")
        st.write(f"Default Probability: {repayment_prob:.2%}")
        st.write(f"Rating: {rating}")

        # Log the prediction details
        logger.info(f"Predicted credit score: {credit_score}")
        logger.info(f"Default probability: {repayment_prob:.2%}")
        logger.info(f"Rating: {rating}")

        # Save the input and prediction details to CSV
        save_to_csv(input_data, {'credit_score': credit_score, 'repayment_probability': repayment_prob, 'rating': rating})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        st.error(f"Error during prediction: {str(e)}")
