

import torch
from sklearn import preprocessing
import numpy as np
# design a neural network
import torch.nn as nn
# binary classifier
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# tune neural net

from functools import partial
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# early stopping
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# plot
from torchviz import make_dot
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
plt.ion()
matplotlib.use('TkAgg')
plt.show()

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

# 21.5 Training a multiclass classifier
N_CLASSES = 3
EPOCHS=3

# create training and test set
features, target = make_classification(n_classes=N_CLASSES, n_informative=9,
                                       n_redundant=0, n_features=10, n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# convert data to pytorch tensors
X_train = torch.from_numpy(features_train).float()
y_train = torch.nn.functional.one_hot(torch.from_numpy(target_train).long(),
                                      num_classes=N_CLASSES).float()
X_test = torch.from_numpy(features_test).float()
y_test= torch.nn.functional.one_hot(torch.from_numpy(target_test).long(),
                                      num_classes=N_CLASSES).float()

# define a neural network using sequential
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x
# initialize neural net
network = SimpleNeuralNet()

# define loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(network.parameters())

# define data loader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# compile model using torch 2.0 optimizer
network = torch.compile(network)

# train neural network
for e in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", e+1, "\tLoss: ", loss.item())

# evaluate neural net
    with torch.no_grad():
        output = network(X_test)
        test_loss = criterion(output, y_test)
        test_accuracy = (output.round() == y_test).float().mean()
        print("test loss: ", test_loss.item(), "\ttest accuracy: ", test_accuracy.item())

# 21.6 Training a Regressor

EPOCHS=5

# create training and test sets
features, target = make_regression(n_features=10, n_samples = 1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.1, random_state=1
)

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# convert data to PyTorch tensors
X_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
X_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# define a neural net using Sequential
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
    def forward(self, x):
        x = self.sequential(x)
        return x
# initialize network
network = SimpleNeuralNet()

# define loss function, optimizer
criterion = nn.MSELoss()
optimizer = RMSprop(network.parameters())

# define data loader

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# compile model using torch optimizer
network = torch.compile(network)

# train neural network
for e in range (EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", e+1, "\tLoss: ", loss.item())
# evaluate neural network
with torch.no_grad():
    output = network(X_test)
    test_loss = float(criterion(output, y_test))
    print("Test MSE: ", test_loss)


# 21.7 Making predictions

N_CLASSES = 3


# create training and test set
features, target = make_classification(n_classes=N_CLASSES, n_informative=9,
                                       n_redundant=0, n_features=10, n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# convert data to pytorch tensors
X_train = torch.from_numpy(features_train).float()
y_train = torch.nn.functional.one_hot(torch.from_numpy(target_train).long(),
                                      num_classes=N_CLASSES).float()
X_test = torch.from_numpy(features_test).float()
y_test= torch.nn.functional.one_hot(torch.from_numpy(target_test).long(),
                                      num_classes=N_CLASSES).float()

# define a neural network using sequential
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x
# initialize neural net
network = SimpleNeuralNet()

# define loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(network.parameters())

# define data loader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# compile model using torch 2.0 optimizer
network = torch.compile(network)

# train neural network
epochs=8
train_losses = []
test_losses = []

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", epoch+1, "\tLoss: ", loss.item())

# evaluate neural net
    with torch.no_grad():
        predicted_class = network.forward(X_train).round()
        train_output = network(X_train)
        train_loss = criterion(output, target)
        train_losses.append(train_loss.item())

        test_output = network(X_test)
        test_loss = criterion(test_output, y_test)
        test_losses.append(test_loss.item())
        predicted_class[0]

# visualize loss history
e = range(0, epochs)
plt.plot(e, train_losses, "r--")
plt.plot(e, test_losses, "b--")
plt.legend(["training loss", "test loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 21.9 Reducing overfitting with weight regularization

# create training and test sets

N_CLASSES = 3
EPOCHS=8

# create training and test set
features, target = make_classification(n_classes=N_CLASSES, n_informative=9,
                                       n_redundant=0, n_features=10, n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# convert data to pytorch tensors
X_train = torch.from_numpy(features_train).float()
y_train = torch.nn.functional.one_hot(torch.from_numpy(target_train).long(),
                                      num_classes=N_CLASSES).float()
X_test = torch.from_numpy(features_test).float()
y_test= torch.nn.functional.one_hot(torch.from_numpy(target_test).long(),
                                      num_classes=N_CLASSES).float()

# define a neural network using sequential
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x
# initialize neural net
network = SimpleNeuralNet()

# define loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-4, weight_decay = 1e-5)

# define data loader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# compile model using torch 2.0 optimizer
network = torch.compile(network)

# train neural network
for e in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", e+1, "\tLoss: ", loss.item())

# evaluate neural net
    with torch.no_grad():
        output = network(X_test)
        test_loss = criterion(output, y_test)
        test_accuracy = (output.round() == y_test).float().mean()
        print("test loss: ", test_loss.item(), "\ttest accuracy: ", test_accuracy.item())

# 21.10 Reducing overfitting with early stopping

# Create training and test sets
features, target = make_classification(n_classes=2, n_features=10, n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using `Sequential`
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

# Define LightningModule
class LightningNetwork(pl.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.criterion = nn.BCELoss()
        self.metric = nn.functional.binary_cross_entropy

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, target = batch
        output = self.network(data)
        loss = self.criterion(output, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Initialize neural network
network = LightningNetwork(SimpleNeuralNet())

# Train network
trainer = pl.Trainer(
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
    max_epochs=1000
)
trainer.fit(model=network, train_dataloaders=train_loader)

# 21.11 Reducing overfitting with Dropout


# Create training and test sets
features, target = make_classification(
    n_classes=2, n_features=10, n_samples=1000
)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using `Sequential`
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Dropout(0.1),  # Drop 10% of neurons
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

# Initialize neural network
network = SimpleNeuralNet()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# Define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
epochs = 3
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: ", epoch + 1, "\tLoss: ", loss.item())

# Evaluate neural network
with torch.no_grad():
    output = network(x_test)
    test_loss = criterion(output, y_test)
    test_accuracy = (output.round() == y_test).float().mean()
print("Test Loss:", test_loss.item(), "\tTest Accuracy:", test_accuracy.item())

# 21.12 Saving model training progress

# Create training and test sets
features, target = make_classification(n_classes=2, n_features=10, n_samples=1000)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using `Sequential`
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Dropout(0.1),  # Drop 10% of neurons
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

# Initialize neural network
network = SimpleNeuralNet()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# Define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
epochs = 5
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Save the model at the end of every epoch
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        "model.pt"
    )
    print("Epoch:", epoch + 1, "\tLoss:", loss.item())

# 21.13 Tuning Neural Networks

# Create training and test sets
features, target = make_classification(n_classes=2, n_features=10, n_samples=1000)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using `Sequential`
class SimpleNeuralNet(nn.Module):
    def __init__(self, layer_size_1=10, layer_size_2=10):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, layer_size_1),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size_1, layer_size_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size_2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

# Define search space for Ray Tune
config = {
    "layer_size_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "layer_size_2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
}

# Define ASHA scheduler
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=1000,
    grace_period=1,
    reduction_factor=2
)

# Define CLI reporter for progress visualization
reporter = CLIReporter(
    parameter_columns=["layer_size_1", "layer_size_2", "lr"],
    metric_columns=["loss"]
)

# Training function for Ray Tune
def train_model(config, epochs=3):
    network = SimpleNeuralNet(config["layer_size_1"], config["layer_size_2"])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(network.parameters(), lr=config["lr"], momentum=0.9)

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    # Compile the model using torch 2.0's optimizer
    network = torch.compile(network)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Report final loss to Ray Tune
    tune.report(loss=(loss.item()))

# Run hyperparameter tuning
result = tune.run(
    train_model,
    resources_per_trial={"cpu": 1},
    config=config,
    num_samples=1,
    scheduler=scheduler,
    progress_reporter=reporter
)

# Extract best trial information
best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

# Recreate the best model with optimal config
best_trained_model = SimpleNeuralNet(
    best_trial.config["layer_size_1"],
    best_trial.config["layer_size_2"]
)


# 21.14 Visualizing neural net

# Create training and test sets
features, target = make_classification(n_classes=2, n_features=10, n_samples=1000)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using Sequential
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

# Initialize neural network
network = SimpleNeuralNet()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# Define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
epochs = 3
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Visualize the model graph
make_dot(
    output.detach(),
    params=dict(list(network.named_parameters()))
).render(
    "simple_neural_network",
    format="png"
)

'simple_neural_network.png'


