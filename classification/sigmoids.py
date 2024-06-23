import math
import torch as nn

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define inputs
studies = 0.0
notes = 0.0
gpa = 4.0

# Multiply each input by the corresponding weight
weighted_studies = -5.0 * (studies)
weighted_notes = -4.0 * (notes)
weighted_last_gpa = 2.2 * (gpa)

# Calculate weighted sum
weighted_sum = weighted_studies + weighted_notes + weighted_last_gpa

# Apply the sigmoid activation function (run the setup cell if you haven't yet)
predicted_probability = sigmoid(weighted_sum)

# Determine a prediction using a threshold of .5
threshold = 0.5
classification = predicted_probability > threshold 

# Print probability and classification
print("Probability:", predicted_probability)
print("Classification:", classification)

# re-calculating the probability from the prior checkpoint for ease
predicted_probability = sigmoid(1.6)

## YOUR SOLUTION HERE ##
threshold = 0.85
classification = predicted_probability > threshold

# Print probability and classification - do not modify
print("Probability:", predicted_probability)
print("Classification:", classification)

model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid())