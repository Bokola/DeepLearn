# 5.1 one-hot encode nominal categories

import numpy as np
from pyexpat import features

from hypothesis.extra.pandas import columns
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

feature = np.array([["Texas"], ["California"], ["Texas"], ["Delaware"], ["Texas"]])
# create one-hot encoder
one_hot = LabelBinarizer()
# fit
one_hot.fit_transform(feature)
# we can use the classes_ attribute to output the classes
one_hot.classes_
# reverse one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))

# you can use pandas to reverse encode

pd.get_dummies(feature[:,0])

# sklearn can encode multiclass feature
mult_feature = [("Texas", "Florida"),
("California", "Alabama"),
("Texas", "Florida"),
("Delaware", "Florida"),
("Texas", "Alabama")]

# create multi-class one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()
one_hot_multiclass.fit_transform(mult_feature)
# see classes with classes_ method
one_hot_multiclass.classes_

# 5.2 Encoding ordinal categorical feature
df = pd.DataFrame({"score": ["Low", "Medium", "Medium", "Low", "High" ]})
# create a mapper
scale_mapper = {"Low":1,
                "Medium":2,
                "High":3}
# replace feature values with scale
df["score"].replace(scale_mapper)

# 5.3 Encoding dictionaries of features using DictVectorizer

data_dict = [{"Red": 2, "Blue": 4},
{"Red": 4, "Blue": 3},
{"Red": 1, "Yellow": 2},
{"Red": 2, "Yellow": 2}]

# create a dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False) # sparse=False outputs a dense matrix
features = dictvectorizer.fit_transform(data_dict)
features
# for illustration, create a pandas df to view the output better
# get feature names
feature_names = dictvectorizer.get_feature_names_out()
pd.DataFrame(features, columns=feature_names)

# 5.4 Imputing missing class values with KNN classifier

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
[1, 1.18, 1.33],
[0, 1.22, 1.27],
[1, -0.21, -1.19]])

# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
[np.nan, -0.67, -0.22]])

# train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:, 1:], X[:,0])
# predict class of missing values
imputed_values = trained_model.predict(X_with_nan[:, 1:])
# join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:,1:]))
# join two feature matrices
np.vstack((X_with_imputed, X))

# an alternative is a simpler imputer with a feature's most frequent value

X_complete = np.vstack((X_with_nan, X))
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit_transform(X_complete)

# 5.5 Handling imbalanced classes

# uses class weight parameters, downsampling and upsampling

iris = load_iris()
# create a feature matrix
features = iris.data
# create target vector
target = iris.target
# remove first 40 observation
target = target[40:]
# create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)
# look at the imbalanced target vector
target

# create weights
weights = {0: 0.9, 1: 0.1}
# create random forest classifier with weights
RandomForestClassifier(class_weight=weights)
RandomForestClassifier(class_weight={0: 0.9, 1: 0.1})
# you can also pass balance which automatically creates weights inversely proportional
# to class frequencies
RandomForestClassifier(class_weight='balanced')
