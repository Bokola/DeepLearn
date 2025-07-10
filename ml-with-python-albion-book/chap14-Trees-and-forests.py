# train decision tree classifier
from random import random

from rich.jupyter import display
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, preprocessing

# decision tree regressor
from sklearn.tree import DecisionTreeRegressor

# visualize decision tree
import numpy as np
import matplotlib
import pydotplus
from IPython.display import Image
from sklearn import tree
from matplotlib import pyplot as plt
from IPython.display import SVG
from sqlalchemy.dialects.mssql import IMAGE

plt.ion()
matplotlib.use('TkAgg')
plt.show()
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
# random forest regressor
from sklearn.ensemble import RandomForestRegressor

# select important features
from sklearn.feature_selection import SelectFromModel

# boosting
from sklearn.ensemble import AdaBoostClassifier

# XGBoost model
import xgboost as xgb
from sklearn.metrics import classification_report
from numpy import argmax

# computationally optimized GBM
import lightgbm as lgb

# 14.1 Training a Decision Tree Classifier

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create a decision tree classifier object
decision_tree = DecisionTreeClassifier(random_state=0)

# train model
model = decision_tree.fit(features, target)

# predict
observation = [[5, 4, 3, 2]]
model.predict(observation)
# view class probabilities
model.predict_proba(observation)
# you can specify a purity measure
decision_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0)
# train model
model_entropy = decision_entropy.fit(features, target)

# 14.2 Training a Decision Tree Regressor

# load data with only 2 features
diabetes = datasets.load_diabetes()
features = diabetes.data
target = diabetes.target

# create a decision tree regressor object
dec_tree_reg = DecisionTreeRegressor(random_state=0)

# train model
model = dec_tree_reg.fit(features, target)

# predict
observation= [features[0]]
model.predict(observation)

# specify Mean Absolute Error (MAE) as a criterion
decision_tree_mae = DecisionTreeRegressor(criterion="absolute_error", random_state=0)
model_mae = decision_tree_mae.fit(features, target)

# 14.3 Visualizing a decision tree model

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# fit and train
dec_tree = DecisionTreeClassifier(random_state=0)
model = dec_tree.fit(features, target)
# create DOT data
dot_data = tree.export_graphviz(dec_tree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names
                                )
# draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# show graph - did not run on windows
Image(graph.create_png())

# 14.4 Training a Random Forest Classifier

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# train model
model = randomforest.fit(features, target)
model.score(features, target)

# 14.5 Training a Random ForestRegressor

# data
diabetes = datasets.load_diabetes()
features = diabetes.data
target = diabetes.target
# create a random forest regressor object
randomforest_reg = RandomForestRegressor(random_state=0, n_jobs=-1)
# train model
model = randomforest_reg.fit(features, target)

# 14.6 Evaluating Random forests with Out-of-Bag Errors
# evaluate random forests without using cross-validation

iris = datasets.load_iris()
features = iris.data
target = iris.target

# create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=-1, oob_score=True)
# train model
model = randomforest.fit(features, target)
# view out-of-bag error
randomforest.oob_score_

# 14.7 Identifying important features in a random forest

# data and model
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# train model
model = randomforest.fit(features, target)

# calculate feature importance
model.feature_importances_
importances = model.feature_importances_

# sort feature importance in descending order
indices = np.argsort(importances)[::-1]

# rearrange feature names so they match the sorted feature importance
names = [iris.feature_names[i] for i in indices]

# create a plot
plt.figure()

# create plot title
plt.title("Feature Importance")

# add bars
plt.bar(range(features.shape[1]), importances[indices])

# add feature names as X-axis labels
plt.xticks(range(features.shape[1]), names, rotation=75)

# 14.8 Selecting Important Features in Random Forests

# data and model
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# create object that selects features with importance
# greater than or equal to a threshold
selector = SelectFromModel(randomforest, threshold=0.3)

#create a new feature matrix using selector
features_important = selector.fit_transform(features, target)

# train model using more important features
model = randomforest.fit(features_important, target)

# 14.9 Handling Imbalanced Classes

# data and model
iris = datasets.load_iris()
features = iris.data
target = iris.target

# make class highly imbalanced by removing first 40 observations
features = features[40:,:]
target = target[40:]

# create target vector indicating if class 0, otherwise 1
target = np.where((target == 0), 0, 1)

# create a random forest classifier object
randomforest_balanced = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")

# train model
model = randomforest_balanced.fit(features, target)

# 14.10 Controlling tree size
# manually determine the structure and size of a decision tree

# data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None, #maximum depth of the tree
                                      min_samples_split=2,# minimum number of obs at a node before the node is split
                                      min_samples_leaf=1, # minimum number of observations required to be at a leaf
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)

model = decisiontree.fit(features,target)

#14.11 Improving Performance Through Boosting

# data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create adaboost tree classifier object
adaboost = AdaBoostClassifier(random_state=0)

# train model
model = adaboost.fit(features, target)


# 14.12 Training an XGBoost Model

# data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create dataset
xgb_train = xgb.DMatrix(features, label=target)

# define parameters
param = {
    'objective': 'multi:softprob',
    'num_class': 3
}



# train model
gbm = xgb.train(param, xgb_train)

# get predictions
predictions = argmax(gbm.predict(xgb_train), axis=1)

# get a classification report
print(classification_report(target, predictions))

# 14.13 Improving Real-Time Performance with LightGBM

# data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create dataset
lgb_train = lgb.Dataset(features, target)

# define parameters
param = {
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1
}

# train model
gbm = lgb.train(param, lgb_train)

# get predictions
preds = argmax(gbm.predict(features), axis=1)

# get classification report
print(classification_report(target, preds))