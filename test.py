import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/patriciagoetz/Documents/DataScienceAssignment/luxury_cosmetics_popups.csv")
print("shap before dropping NA", data.shape)
print(data.isna().sum())

# ger rid of all the rows with missing values
data_cleaned = data.dropna()
print("shape after dropping NA", data_cleaned.shape)

