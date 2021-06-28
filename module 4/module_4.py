import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statistics

class PrepareDate():
    # в констуркоре определим методы обработки данных
    def __init__(self, df_output):
        self.df_output = df_output

    def processData(self):
        #

data = pd.read_csv('data.csv')

df_object = PrepareDate(data)
df_preproc = df_object.processData()


