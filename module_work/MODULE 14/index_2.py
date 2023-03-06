import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import datasets
import tokenizers
import wandb
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

document = datasets.load_dataset("imdb")


vectorizer = TfidfVectorizer()
values = vectorizer.fit_transform(document)

feature_names = vectorizer.get_feature_names()
data = pd.DataFrame(values.toarray(), columns = feature_names)
print(data)



