import pandas as pd
import torch
import numpy as np


#How to create a tensor
#A tensor act as storage containers for numerical data
apartment_tensor = torch.tensor(np.array([2000, 500, 7]), dtype=torch.int)
#print(apartment_tensor)

#Create dataframe using streeteasy csv
apartments_df = pd.read_csv('streeteasy.csv')
apartments_df = apartments_df[["rent", "size_sqft", "building_age_yrs"]]

apartments_tensor = torch.tensor(apartments_df.values, dtype=torch.float32)

#print(apartments_tensor)

#In a linear regression model, y = m1x1 + m2x2 + m3x3 + ... + b, y is the expected output, m are the weights for each x and b is the bias of the model