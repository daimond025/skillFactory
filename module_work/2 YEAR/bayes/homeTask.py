
import arviz as az
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pymc3 as pm


data = pd.read_csv('./data/flats.csv')



data.rename({'price_doc': 'price', 'full_sq': 'area', 'life_sq': 'life','num_room': 'room','kitch_sq': 'kitchen', }, axis=1, inplace=True)


data = data[["price","area","life","room","kitchen"]]

data.dropna(inplace=True)
print(data[data["area"] == 0])
exit()