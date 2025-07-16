

import torch
from hypothesis import target
from sklearn import preprocessing
import numpy as np
# design a neural network
import torch.nn as nn
# binary classifier
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 21.1 Using Autograd with PyTorch

# create torch tensor that requires a gradient
t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# perform a tensor operation simulating "forward propagation"
tensor_sum = t.sum()

# perform back propagation
tensor_sum.backward()

# view the gradients
t.grad

# 21.2 Preprocessing data for Neural Networks

# Create feature
features = np.array([[-100.1, 3240.1],
                    [-200.2, -234.1],
                    [5000.5, 150.1],
                    [6000.6, -125.1],
                    [9000.9, -673.1]])

# create a scaler
scaler = preprocessing.StandardScaler()

# convert to a tensor
features_std_tensor = torch.from_numpy(scaler.fit_transform(features))
features_std_tensor

# standardize manually if using grad

torch_features = torch.tensor([[-100.1, 3240.1],
                    [-200.2, -234.1],
                    [5000.5, 150.1],
                    [6000.6, -125.1],
                    [9000.9, -673.1]], requires_grad=True)
# compute mean & sd
mean = torch_features.mean(0, keepdim=True)
sd = torch_features.std(0, unbiased=False, keepdim=True)

# standardize
torch_features_std = torch_features - mean
torch_features_std = torch_features_std / sd

torch_features_std

# 21.3 Designing a Neural Network

# define a neural network

class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 11)

    def forward(selfself, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return x

# initialize the neural network
network = SimpleNeuralNet()

# define loss function, optimizer
loss_criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(network.parameters())

# show the network
network

# 21.4 Train a Binary Classifier

# create training and test sets
features, target = make_classification(n_classes=2, n_features=10, n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# convert data to pytorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# define neural network using sequential
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.sequential(x)
        return x

# initialize network
network = SimpleNeuralNet()

# define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# compile the model using  torch
network = torch.compile(network)

# train neural network
epochs = 3
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", epoch+1, "\tLoss: ", loss.item())

    # Evaluate neural network
    with torch.no_grad():
        output = network(x_test)
        test_loss = criterion(output, y_test)
        test_accuracy = (output.round() == y_test).float().mean()
        print("Test Loss: ", test_loss.item(), "\tTest Accuracy: ", test_accuracy.item())



