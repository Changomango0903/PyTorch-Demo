import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

apartments_df = pd.read_csv("streeteasy.csv")

numerical_features = ['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs',
                      'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',
                      'has_patio', 'has_gym']

X = torch.tensor(apartments_df[numerical_features].values, dtype = torch.float)
Y = torch.tensor(apartments_df['rent'].values, dtype=torch.float)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20000
# for epoch in range(num_epochs):
#     predictions = model(X)
#     MSE = loss(predictions, Y)
#     MSE.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {MSE.item()**(1/2)}')

#We can use sklearn to split our dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=2)

for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, Y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {MSE.item()**(1/2)}')

#To save the model
torch.save(model, '20000-model.pth')

#To load the model
loaded_model = torch.load('20000-model.pth')

#Model Evaluation
loaded_model.eval()
with torch.no_grad():
    predictions = loaded_model(X_test)
    test_MSE = loss(predictions, Y_test)
print('Test MSE is ' + str(test_MSE.item()))
print('Test Root MSE is ' + str(test_MSE.item()**(1/2)))

#To Plot the predictions:
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, label='Predictions', alpha=0.5, color='blue')

plt.xlabel('Actual Values (Y_test)')
plt.ylabel('Predicted Values')

plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='gray', linewidth=2,
         label="Actual Rent")
plt.legend()
plt.title('StreetEasy Dataset - Predictions vs Actual Values')
plt.show()