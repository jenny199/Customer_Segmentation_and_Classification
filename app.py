import pandas as pd
import joblib
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

st.title("Customer Segmentation App")
st.caption("This app predicts the customer segments based on the input")

#load the saved model and the scaler
model = joblib.load('saved_models/tuned_PCA_random_forest_classifier.pkl')
scaler = joblib.load('saved_models/standard_scaler.pkl')
# Load the saved PCA model
pca = joblib.load('saved_models/pca_model.pkl')


#get user input
BALANCE = st.number_input('BALANCE')
BALANCE_FREQUENCY = st.number_input('BALANCE_FREQUENCY')
PURCHASES = st.number_input('PURCHASES')
ONEOFF_PURCHASES = st.number_input('ONEOFF_PURCHASES')
INSTALLMENTS_PURCHASES = st.number_input('INSTALLMENTS_PURCHASES')
CASH_ADVANCE = st.number_input('CASH_ADVANCE')
PURCHASES_FREQUENCY = st.number_input('PURCHASES_FREQUENCY')
ONEOFF_PURCHASES_FREQUENCY = st.number_input('ONEOFF_PURCHASES_FREQUENCY')       
PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input('PURCHASES_INSTALLMENTS_FREQUENCY') 
CASH_ADVANCE_FREQUENCY = st.number_input('CASH_ADVANCE_FREQUENCY')          
CASH_ADVANCE_TRX = st.number_input('CASH_ADVANCE_TRX')              
PURCHASES_TRX = st.number_input('PURCHASES_TRX')                   
CREDIT_LIMIT = st.number_input('CREDIT_LIMIT')                   
PAYMENTS = st.number_input('PAYMENTS')                        
MINIMUM_PAYMENTS =st.number_input('MINIMUM_PAYMENTS')                
PRC_FULL_PAYMENT = st.number_input('PRC_FULL_PAYMENT')                
TENURE= st.number_input('TENURE')


# Create a DataFrame from the inputs
input_data = {
    'BALANCE': [BALANCE],
    'BALANCE_FREQUENCY': [BALANCE_FREQUENCY],
    'PURCHASES': [PURCHASES],
    'ONEOFF_PURCHASES': [ONEOFF_PURCHASES],
    'INSTALLMENTS_PURCHASES': [INSTALLMENTS_PURCHASES],
    'CASH_ADVANCE': [CASH_ADVANCE],
    'PURCHASES_FREQUENCY': [PURCHASES_FREQUENCY],
    'ONEOFF_PURCHASES_FREQUENCY': [ONEOFF_PURCHASES_FREQUENCY],
    'PURCHASES_INSTALLMENTS_FREQUENCY': [PURCHASES_INSTALLMENTS_FREQUENCY],
    'CASH_ADVANCE_FREQUENCY': [CASH_ADVANCE_FREQUENCY],
    'CASH_ADVANCE_TRX': [CASH_ADVANCE_TRX],
    'PURCHASES_TRX': [PURCHASES_TRX],
    'CREDIT_LIMIT': [CREDIT_LIMIT],
    'PAYMENTS': [PAYMENTS],
    'MINIMUM_PAYMENTS': [MINIMUM_PAYMENTS],
    'PRC_FULL_PAYMENT': [PRC_FULL_PAYMENT],
    'TENURE': [TENURE]
}

user_input_df = pd.DataFrame(input_data)

scaled_inputs = scaler.transform(user_input_df)
# st.dataframe(scaled_inputs)
# Apply PCA
X_pca = pca.transform(scaled_inputs)
X_pca_df = pd.DataFrame(X_pca)
# print(X_pca)
if st.button('Predict'):
    prediction = model.predict(X_pca_df)
    st.write(f'The customer belongs to cluster: {prediction}')


