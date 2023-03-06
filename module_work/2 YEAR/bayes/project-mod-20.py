import arviz as az
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pymc3 as pm


raw_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education-categorical",
        "educ",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "captial-gain",
        "capital-loss",
        "hours",
        "native-country",
        "income",
    ],
)

data = raw_data[~pd.isnull(raw_data["income"])]

income = 1 * (data["income"] == " >50K")


data = data[["age", "educ", "hours","marital-status"]]


data["age"] = data["age"] / 10.0
data["age2"] = np.square(data["age"])
data["income"] = income


def replace_marital(marital):
    if 'married' in marital.lower():
        return 4
    elif 'single' in marital.lower():
        return 3
    elif 'divorced' in marital.lower():
        return 2
    elif 'unknown' in marital.lower():
        return 1


data['marital'] = data["marital-status"].apply(lambda x: replace_marital(x))
data['marital'].fillna(1, inplace=True)


with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    obs = pm.Normal('obs', mu=mu, sd=1, observed=np.random.randn(100))

    trace = pm.sample(1000, tune=500)
print(trace)