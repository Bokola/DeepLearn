# scikit-learn model
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import sklearn

# TensorFlow model
import numpy as np
from tensorflow import keras

# PyTorch model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# serve scikit-learn model
from flask import Flask, request


# 23.1 Saving and loading a Scikit-learn model
# save as a pickle file

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create decision tree classifier object
classifier = RandomForestClassifier()

# train model
model = classifier.fit(features, target)

# save model as pickle file
joblib.dump(model, "model.pkl")

# load model from file
classifier = joblib.load("model.pkl")

# make predictions
new_obs = [[5.2, 3.2, 1.1, 0.1]]
classifier.predict(new_obs)

# for compatibility capture scikit-learn version
scikit_version = sklearn.__version__

# save model as pickle file
joblib.dump(model, "model_{verson}.pkl".format(verson=scikit_version))


# 23.2 Saving and loading TensorFlow model
# use TensorFlow model.save

# set random seed
np.random.seed(0)

# create model with one hidden layer
input_layer = keras.Input(shape=(10,))
hidden_layer = keras.layers.Dense(10)(input_layer)
output_layer = keras.layers.Dense(1)(input_layer)
model = keras.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss = "mean_squared_error")

# train model
X_train = np.random.random((1000, 10))
y_train = np.random.random((1000, 1))
model.fit(X_train, y_train)

# save the model to a directory called `saved_model`
model.save("saved_model.keras")

# load neural network
model = keras.models.load_model('saved_model.keras')

# 23.3 Saving and loading PyTorch model
# use torch.save and torch.load

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

    # Save the model after it's been trained
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        "model.pt"
    )

# Reinitialize neural network
network = SimpleNeuralNet()
state_dict = torch.load("model.pt", map_location=torch.device('cpu'))["model_state_dict"]
network.load_state_dict(state_dict, strict=False)
network.eval()

# 23.4 Serving scikit-learn models
# use a web server powered by Flask

# instantiate a flask app
app = Flask(__name__)

# load the model from disk
model = joblib.load("model.pkl")

## Create a predict route that takes JSON data, makes predictions, and returns them
@app.route("/predict", methods=["POST"])
def predict():
    print(request.json)
    inputs = request.json["inputs"]
    prediction = model.predict(inputs)
    return {
        "prediction": prediction.tolist()
    }

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)

# 23.5 Serving TensorFlow Models
# uses web server and docker

docker run -p 8501:8501 -p 8500:8500 \
--mount type=bind,source=$(pwd)/saved_model,target=/models/saved_model/1 \
-e MODEL_NAME=saved_model -t tensorflow/serving