import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("Diabetes Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# Preprocessing and model training
@st.cache_resource
def train_model():
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

model, scaler = train_model()

# Sidebar input
st.sidebar.header("Enter Patient Data")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 21, 100, 33)
    
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("Patient Input:")
st.write(input_df)

# Prediction
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.subheader("Prediction:")
    st.success(result)
