import pandas as pd
import zipfile
import io
import requests

def load_dataset():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))

    df = pd.read_csv(z.open("student-mat.csv"), sep=';')

    df['pass_fail'] = (df['G3'] >= 10).astype(int)

    return df