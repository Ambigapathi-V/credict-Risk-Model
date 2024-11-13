import streamlit as st
import requests
import csv
import os
from src.Credit_Risk_Model.logger import logger  # Ensure logger is correctly configured

# URL of the FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

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

# Function to save input and prediction results to CSV
def save_to_csv(input_data, result, filename='input_data.csv'):
    # Combine input and result data into a single dictionary
    output_data = {**input_data,
                   'credit_score': result.get('credit_score'),
                   'repayment_probability': result.get('repayment_probability'),
                   'rating': result.get('rating')}
    
    # Check if the file exists and write the header if the file is new
    file_exists = os.path.exists(filename)
    
    try:
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['age', 'residence_type', 'loan_purpose', 'loan_type', 'loan_tenure_months',
                          'number_of_open_accounts', 'loan_amount', 'income', 'delinquent_months',
                          'total_dpd', 'total_loan_months', 'credit_utilization_ratio', 'credit_score', 
                          'repayment_probability', 'rating']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if the file doesn't exist or is empty
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(output_data)
        
        logger.info("Data saved to CSV successfully.")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        st.error(f"Error saving data: {e}")

# Prediction Button
if st.button("Predict Credit Risk"):
    # Prepare input data to send to FastAPI in a flat dictionary format
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
        # Send a POST request to FastAPI to get the prediction
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()

        # Display the results from the FastAPI response
        result = response.json()

        # Display the results
        st.success(f"Credit Score: {result.get('credit_score')}")
        st.write(f"Default Probability: {result.get('repayment_probability'):.2%}")
        st.write(f"Rating: {result.get('rating')}")

        # Log the prediction details
        logger.info(f"Predicted credit score: {result.get('credit_score')}")
        logger.info(f"Default probability: {result.get('repayment_probability'):.2%}")
        logger.info(f"Rating: {result.get('rating')}")
        #logger.info("Prediction request completed successfully.")

        # Save the input and prediction details to CSV
        save_to_csv(input_data, result)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching prediction: {e}")
        st.error(f"Error fetching prediction: {e}")
    except ValueError:
        logger.error("Unexpected response format received from API.")
        st.error("Unexpected response format received from API.")
