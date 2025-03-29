import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def load_model():
    return joblib.load('loan_approval_model.pkl')

def load_scaler():
    return joblib.load('scaler.pkl')

def preprocess_input(data, scaler):
    # Map categorical variables to match training phase
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3})
    
    # One-hot encoding for Property_Area
    data = pd.get_dummies(data, columns=['Property_Area'], drop_first=True)
    
    # Ensure all expected columns exist (to match model training)
    expected_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                        'Property_Area_Semiurban', 'Property_Area_Urban']
    
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default 0
    
    # Scale numeric data
    data_scaled = scaler.transform(data)
    return data_scaled

def main():
    st.title("Loan Approval Prediction App")
    st.write("Enter the details below to check loan approval status.")
    
    # Sidebar inputs
    st.sidebar.header("User Input Parameters")
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    married = st.sidebar.selectbox("Married", ['Yes', 'No'])
    dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
    education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount (in Thousands)", min_value=0)
    loan_term = st.sidebar.selectbox("Loan Amount Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360])
    credit_history = st.sidebar.selectbox("Credit History", [1, 0])
    property_area = st.sidebar.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])
    
    # Create DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    model = load_model()
    scaler = load_scaler()
    
    if st.sidebar.button("Predict"):
        processed_data = preprocess_input(input_data, scaler)
        prediction = model.predict(processed_data)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("Loan Approved (Y)")
        else:
            st.error("Loan Not Approved (N)")

if __name__ == "__main__":
    main()
