import streamlit as st
import pandas as pd
import pickle as pk

# Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))



# Title
st.header('Loan Prediction App')

# Input Fields
no_of_dep = st.slider('Choose No of Dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.slider('Choose Annual Income', 0, 1000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 1000000)
Loan_Dur = st.slider('Choose Loan Duration (in years)', 0, 30)
Cibil = st.slider('Choose Cibil Score', 0, 1000)
Assets = st.slider('Choose Assets', 0, 1000000)

# Encoding Categorical Inputs
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 1 if self_emp == 'Yes' else 0

# Prediction
if st.button("Predict"):
    try:
        # Form input as DataFrame
        pred_data = pd.DataFrame(
            [[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
            columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                     'loan_amount', 'loan_term', 'cibil_score', 'Assets']
        )

        # Apply scaling
        pred_data_scaled = scaler.transform(pred_data)

        # Make prediction
        prediction = model.predict(pred_data_scaled)

        # Show result
        if prediction[0] == 1:
            st.success('✅ Loan is Approved')
        else:
            st.error('❌ Loan is Rejected')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


