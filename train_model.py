import pandas as pd
import numpy as np
import zipfile
import io
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_models():

    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

    response = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(response.content))

    df = pd.read_csv(z.open("student-mat.csv"), sep=';')

    df['pass_fail'] = (df['G3'] >= 10).astype(int)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(['G3','pass_fail'], axis=1)

    y_class = df['pass_fail']
    y_reg = df['G3']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train_c = scaler.fit_transform(X_train_c)
    X_test_c = scaler.transform(X_test_c)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_c, y_train_c)

    lr_acc = accuracy_score(y_test_c, lr.predict(X_test_c))

    dt = DecisionTreeClassifier()
    dt.fit(X_train_c, y_train_c)

    dt_acc = accuracy_score(y_test_c, dt.predict(X_test_c))

    linreg = LinearRegression()
    linreg.fit(X_train_r, y_train_r)

    return df, X, scaler, lr, linreg, lr_acc, dt_acc