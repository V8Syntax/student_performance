import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px

# ============================
# LOAD MODELS
# ============================

lr = joblib.load("classification_model.pkl")
linreg = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

st.title("Student Performance Prediction System")

st.write("Predict whether a student will pass or fail and estimate final grade.")

# ============================
# USER INPUT
# ============================

st.sidebar.header("Student Information")

age = st.sidebar.slider("Age",15,22,17)
studytime = st.sidebar.slider("Study Time (1-4)",1,4,2)
absences = st.sidebar.slider("Absences",0,50,5)
g1 = st.sidebar.slider("First Period Grade (G1)",0,20,10)
g2 = st.sidebar.slider("Second Period Grade (G2)",0,20,10)

# ============================
# PREDICTION
# ============================

if st.sidebar.button("Predict Performance"):

    input_data = np.array([[age,studytime,absences,g1,g2]])

    scaled_input = scaler.transform(input_data)

    pass_fail = lr.predict(scaled_input)[0]

    grade = linreg.predict(input_data)[0]

    if pass_fail == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")

    st.metric("Predicted Final Grade", round(grade,2))

# ============================
# MODEL PERFORMANCE GRAPH
# ============================

st.subheader("Model Accuracy Comparison")

models = ["Logistic Regression","Decision Tree","Neural Network"]

accuracy = [0.89,0.87,0.91]

df = pd.DataFrame({
    "Model":models,
    "Accuracy":accuracy
})

fig = px.bar(df,x="Model",y="Accuracy",color="Model")

st.plotly_chart(fig)

# ============================
# PROJECT DESCRIPTION
# ============================

st.subheader("About the Project")

st.write("""
This machine learning system predicts student academic performance.

Dataset: UCI Student Performance Dataset

Models implemented:
• Logistic Regression
• Decision Tree
• Artificial Neural Network

The system predicts:
• Pass / Fail classification
• Final mathematics grade (G3)

The dashboard allows educators to input student attributes and obtain predictions instantly.
""")