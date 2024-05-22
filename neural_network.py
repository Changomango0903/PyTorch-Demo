import pandas as pd
import torch
import torch.nn as nn
import numpy as np


#How to create a tensor
#A tensor act as storage containers for numerical data
apartment_tensor = torch.tensor(np.array([2000, 500, 7]), dtype=torch.int)
#print(apartment_tensor)

#Create dataframe using streeteasy csv
apartments_df = pd.read_csv('streeteasy.csv')
apartments_df = apartments_df[["rent", "size_sqft", "bedrooms", "building_age_yrs"]]

apartments_tensor = torch.tensor(apartments_df.values, dtype=torch.float32)

#print(apartments_tensor)

#In a linear regression model, y = m1x1 + m2x2 + m3x3 + ... + b, y is the expected output, m are the weights for each x and b is the bias of the model
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values
X = torch.tensor(apartments_numpy, dtype = torch.float32)

#building a sequential model with different activations functions at each layer
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

predicted_rent = model(X)
#Output doesn't make sense since model was not trained
print(predicted_rent[:5])