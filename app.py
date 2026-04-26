import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. SET UP PAGE ---
st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('attrition_model.pkl', 'rb'))
    scaler = pickle.load(open('attrition_scaler.pkl', 'rb'))
    cols = pickle.load(open('attrition_columns.pkl', 'rb'))
    return model, scaler, cols

try:
    model, scaler, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- 3. UI DESIGN ---
st.title("👥 Employee Attrition Predictor")
st.markdown("Enter employee details to predict the likelihood of resignation.")
st.divider()

# Organize inputs into columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    distance = st.number_input("Distance From Home (km)", 1, 30, 5)
    total_years = st.number_input("Total Working Years", 0, 40, 5)

with col2:
    job_level = st.slider("Job Level", 1, 5, 2)
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    stock = st.slider("Stock Option Level", 0, 3, 0)

with col3:
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    work_life = st.slider("Work-Life Balance (1-4)", 1, 4, 3)

# Add other necessary dropdowns for the One-Hot Encoding
st.divider()
dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Attrition Risk", use_container_width=True):
    # 1. Create a dictionary with all 0s for all model columns
    input_data = {col: 0 for col in model_columns}
    
    # 2. Fill numerical values
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['DistanceFromHome'] = distance
    input_data['TotalWorkingYears'] = total_years
    input_data['JobLevel'] = job_level
    input_data['JobSatisfaction'] = job_sat
    input_data['EnvironmentSatisfaction'] = env_sat
    input_data['StockOptionLevel'] = stock
    input_data['WorkLifeBalance'] = work_life

    # 3. Handle Binary Encoding (OverTime)
    if overtime == "Yes":
        input_data['OverTime_Yes'] = 1

    # 4. Handle One-Hot Encoding manually for selects
    # Marital Status
    if f"MaritalStatus_{marital}" in input_data:
        input_data[f"MaritalStatus_{marital}"] = 1
        
    # Department
    if f"Department_{dept}" in input_data:
        input_data[f"Department_{dept}"] = 1
        
    # Job Role
    # Note: Streamlit selectbox values must match the column suffix in your dummy variables
    # We replace spaces with underscores to match pd.get_dummies behavior
    formatted_role = f"JobRole_{job_role.replace(' ', ' ')}" # Adjusting for pandas naming
    if formatted_role in input_data:
        input_data[formatted_role] = 1

    # 5. Convert to DataFrame to ensure correct order
    final_df = pd.DataFrame([input_data])[model_columns]
    
    # 6. Scale and Predict
    scaled_df = scaler.transform(final_df)
    prediction = model.predict(scaled_df)[0]
    probability = model.predict_proba(scaled_df)[0][1]

    # 7. Display Results
    st.divider()
    if prediction == 1:
        st.error(f"### High Risk of Attrition")
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.success(f"### Low Risk of Attrition")
        st.write(f"Confidence Score: {(1-probability):.2%}")
