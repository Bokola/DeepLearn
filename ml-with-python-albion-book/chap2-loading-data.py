# 2.1 loading sample dataset
from jedi.api.refactoring import inline
from keras.src.backend.jax.random import shuffle
from sklearn import datasets

digits = datasets.load_digits()

# create features matrix
features = digits.data
# create target vector
target = digits.target
# view first observation
features[0]

# 2.2 Creating a simulated dataset: make_regession, make_classification, make_blob
## regression dataset

from sklearn.datasets import make_regression

# generate features matrix, target vector, and true coefficients
features, target, coefficients = make_regression(n_samples=100,
                                                 n_features=3,
                                                 n_informative=3,
                                                 n_targets=1,
                                                 noise=0.0,
                                                 coef=True,
                                                 random_state=1)
# view feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# classification dataset
from sklearn.datasets import make_classification
features, target = make_classification(
    n_samples=100
    ,n_features=3
    ,n_informative=3 # no. of features used to generate target vector
    ,n_redundant=0
    ,n_classes=2
    ,weights=[.25, .75] # for imbalanced classes
    ,random_state=1
)

# clustering dataset - fro clustering techniques

from sklearn.datasets import make_blobs

features, target = make_blobs(
    n_samples=100
    ,n_features=2
    ,centers=3 #no of clusters
    ,cluster_std=0.5
    ,shuffle = True
    ,random_state=1
)

# plot clusters
import matplotlib
matplotlib.use('TkAgg') # switch backend to show plot do this before importing pyplot
import matplotlib.pyplot as plt
plt.scatter(features[:,0], features[:,1], c = target)
plt.show()

# 2.3 loading CSV file

import pandas as pd

## create url
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
## load
df = pd.read_csv(url)
## view first 2 rows
df.head(2)

# 2.4 Loading an Excel file

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.xlsx'
df = pd.read_excel(url, sheet_name=0, header=0)
df.head(2)

# 2.12 Loading unstructured data

import requests
txt_url = "https://machine-learning-python-cookbook.s3.amazonaws.com/text.txt"
## get the text file
r = requests.get(txt_url)
## write it to .txt locally
with open('txt.txt', 'wb') as f:
    f.write(r.content)
## read in the file
with open('txt.txt', 'r') as f:
    text = f.read()
## print the content
print(text)