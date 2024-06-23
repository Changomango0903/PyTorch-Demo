import numpy as np
import torch
import torch as nn


def BCELoss(p, y):
    if y == 1:
        return -np.log(p)
    else:
        return -np.log(1-p)

p = 0.9
print("Actual Classification 1")
print("When p is " + str(p) + ", the loss is " + str(BCELoss(p, 1)))
print("Actual Classification 0")
print("When p is " + str(p) + ", the loss is " + BCELoss(p, 0))

# create an instance of BCELoss
loss = nn.BCELoss()

# create a tensor with an output probability
p = torch.tensor([0.7], dtype=torch.float)

# create a tensor with the actual classification
y = torch.tensor([0], dtype=torch.float)

# define loss_value
loss_value = loss(p, y)

# print the BCELoss
print(loss_value)

p = torch.tensor([0.5], dtype=torch.float)
y1 = torch.tensor([1], dtype=torch.float)
y0 = torch.tensor([0], dtype=torch.float)
# compute loss for p=.5 and y=1 here

loss1 = loss(p, y1)

# compute loss for p=.5 and y=0 here

loss0 = loss(p, y0)

print("Loss for y=1: " + str(loss1))
print("Loss for y=0: " + str(loss0))