import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from benchmark_suite import *
# Define a simple neural network
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define another simple neural network
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(10, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()  # Add this line to match the output dimension with the label dimension


# Create inputs and labels
np.random.seed(0)  # Set random seed for reproducibility
inputs = [np.random.rand(10) for i in range(100)]
labels = [[np.sum(input)] for input in inputs]  # Labels are the sum of input values, wrapped in a list



# Define evaluation metrics
metrics = [nn.MSELoss(), nn.L1Loss()]

# Create neural networks and test them using the benchmark function
net1 = Net1()
net2 = Net2()
metric_values1 = benchmark([net1, net2], inputs, labels, metrics)

# Plot the benchmark results and save the plots
plot_benchmark(metric_values1, metrics)