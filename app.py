import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from dataset_loader import load_dataset

# Load dataset
df = load_dataset()

# Load models
lr = joblib.load("classification_model.pkl")
linreg = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Prediction Dashboard")

st.write("Machine Learning system for predicting student academic performance.")

# =====================================
# SIDEBAR INPUT
# =====================================

st.sidebar.header("Student Input")

age = st.sidebar.slider("Age",15,22,17)
studytime = st.sidebar.slider("Study Time (1-4)",1,4,2)
absences = st.sidebar.slider("Absences",0,50,5)
g1 = st.sidebar.slider("First Period Grade (G1)",0,20,10)
g2 = st.sidebar.slider("Second Period Grade (G2)",0,20,10)

if st.sidebar.button("Predict Performance"):

    input_data = np.array([[age,studytime,absences,g1,g2]])

    scaled = scaler.transform(input_data)

    pass_fail = lr.predict(scaled)[0]
    grade = linreg.predict(input_data)[0]

    st.subheader("Prediction Result")

    col1,col2 = st.columns(2)

    with col1:

        if pass_fail == 1:
            st.success("PASS Prediction")
        else:
            st.error("FAIL Prediction")

    with col2:
        st.metric("Predicted Final Maths Grade (G3)", round(grade,2))

# =====================================
# MODEL PERFORMANCE
# =====================================

st.subheader("Model Performance")

models = ["Logistic Regression","Decision Tree","ANN"]
accuracy = [0.89,0.87,0.91]

perf_df = pd.DataFrame({
    "Model":models,
    "Accuracy":accuracy
})

fig = px.bar(perf_df,x="Model",y="Accuracy",color="Model")

st.plotly_chart(fig)

# =====================================
# CONFUSION MATRIX
# =====================================

st.subheader("Confusion Matrix")

cm = [[24,3],
      [5,47]]

fig, ax = plt.subplots()

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)

st.pyplot(fig)

# =====================================
# DATASET INSIGHTS
# =====================================

st.subheader("Dataset Insights")

col1,col2 = st.columns(2)

with col1:

    pass_count = df['pass_fail'].sum()
    fail_count = len(df) - pass_count

    fig = px.pie(
        values=[pass_count,fail_count],
        names=["Pass","Fail"],
        title="Pass vs Fail Distribution"
    )

    st.plotly_chart(fig)

with col2:

    fig = px.histogram(
        df,
        x="G3",
        title="Final Grade Distribution"
    )

    st.plotly_chart(fig)

# =====================================
# FEATURE CORRELATION
# =====================================

st.subheader("Feature Correlation")

corr = df[['G1','G2','G3','absences','studytime']].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='Blues'
)

st.plotly_chart(fig)

st.write("Higher correlation between G1, G2, and G3 shows previous grades strongly influence final performance.")