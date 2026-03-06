import streamlit as st
import numpy as np
import plotly.express as px

from train_model import train_models

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

df, X, scaler, lr, linreg, lr_acc, dt_acc = train_models()

# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")

st.write("Machine Learning System for Academic Performance Analysis")

# ------------------------------------------------
# SIDEBAR INPUT
# ------------------------------------------------

st.sidebar.header("Student Information")

age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study Time", 1, 4, 2)
absences = st.sidebar.slider("Absences", 0, 30, 3)
g1 = st.sidebar.slider("G1 Grade", 0, 20, 10)
g2 = st.sidebar.slider("G2 Grade", 0, 20, 10)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.sidebar.button("Predict Performance"):

    input_data = np.zeros((1, X.shape[1]))

    input_data[0][0] = age
    input_data[0][1] = studytime
    input_data[0][2] = absences
    input_data[0][3] = g1
    input_data[0][4] = g2

    scaled = scaler.transform(input_data)

    pred = lr.predict(scaled)[0]

    grade = linreg.predict(input_data)[0]

    grade = max(0, min(20, grade))

    result = "PASS" if pred == 1 else "FAIL"

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Pass / Fail", result)
    col2.metric("Predicted Final Grade", round(grade,2))

# ------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Total Features", len(df.columns))
col3.metric("Target Variable", "G3")

st.dataframe(df.head())

# ------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------

st.subheader("Model Accuracy Comparison")

models = ["Logistic Regression", "Decision Tree"]

scores = [lr_acc*100, dt_acc*100]

fig = px.bar(
    x=models,
    y=scores,
    labels={"x":"Model","y":"Accuracy (%)"},
    title="Model Performance Comparison"
)

st.plotly_chart(fig)

# ------------------------------------------------
# PASS FAIL DISTRIBUTION
# ------------------------------------------------

st.subheader("Pass vs Fail Distribution")

pass_count = (df['pass_fail']==1).sum()
fail_count = (df['pass_fail']==0).sum()

pie = px.pie(
    values=[pass_count, fail_count],
    names=["Pass","Fail"]
)

st.plotly_chart(pie)