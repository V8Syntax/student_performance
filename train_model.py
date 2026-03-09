import pandas as pd
import numpy as np
import zipfile
import io
import requests
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

df = pd.read_csv(z.open("student-mat.csv"), sep=';')

df['pass_fail'] = (df['G3'] >= 10).astype(int)

features = df[['age','studytime','absences','G1','G2']]

X = features
y_class = df['pass_fail']
y_reg = df['G3']

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Classification model
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train_c)

# Regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train_r)

# Save models
joblib.dump(lr,"classification_model.pkl")
joblib.dump(linreg,"regression_model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Models saved successfully")