import numpy as np
import pandas as pd

data = pd.read_csv("/Users/patriciagoetz/Documents/DataScienceAssignment/luxury_cosmetics_popups.csv")
#print(data.head(5))
print("shape", data.shape)
print(data.isna().sum())

# ger rid of all the rows with missing values
data_cleaned = data.dropna()
print("shape after dropping NA", data_cleaned.shape)

