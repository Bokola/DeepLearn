import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# baseline model
from sklearn.dummy import DummyRegressor, DummyClassifier

# evaluate binary classifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# evaluate binary classifier threshold

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# evaluate clustering model
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# create a custom evaluation metric
from sklearn.metrics import make_scorer, r2_score

# describe classifier's performance
from sklearn.metrics import classification_report

plt.ion()
matplotlib.use('TkAgg')

# visualize a classifiers performance
import seaborn as sns
from sklearn.metrics import confusion_matrix

# visualize effect of training set size
from sklearn.model_selection import learning_curve

# visualize effect of hyperparameter values
from sklearn.model_selection import validation_curve

# 11.1 Cross-validating models
# how does a classification model generalize to unseen data

# load digits dataset
digits = datasets.load_digits()

# create feature matrix
features = digits.data

# create target vector
target = digits.target

# create standardizer
standardizer = StandardScaler()

# create logistic regression object
logit = LogisticRegression()

# create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

# create k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# conduct k-fold cross-validation
cv_results = cross_val_score(
    pipeline, #pipeleine
    features, #  feature matrix
    target, #target vector
    cv=kf, # performance metric
    scoring="accuracy", # loss function
    n_jobs=-1 # use ll cpu cores
)

# calculate mean
cv_results.mean()

# score for all 5 folds
cv_results

# pre-process training set and apply those transformations
# to training and test sets - this is automated by pipeline above

# create training and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# fit standardizer to training set
standardizer.fit(features_train)

# apply to both training and test sets which can then be used to train models
feature_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

# 11.2 Creating a baseline regression model

# load data
wine = load_wine()

# create features
features, target = wine.data, wine.target

# make test and training split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0
)

# create a dummy regressor
dummy = DummyRegressor(strategy='mean')

# 'train' dummy regressor

dummy.fit(features_train, target_train)

# get r-squared score
dummy.score(features_test, target_test)

# to compare, we train our model and evaluate the performance score:

# train simple linear regression model
ols = LinearRegression()
ols.fit(features_train, target_train)

# get r-squared score
ols.score(features_test, target_test)

# 11.3 Creating a baseline classification model

# load data
iris = load_iris()

# create target vector and feature matrix
features, target = iris.data, iris.target

# split into training and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0
)

# create a dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)

# 'train' model
dummy.fit(features_train, target_train)

# get accuracy score
dummy.score(features_test, target_test)

# compare baseline classifier to trained classifier

# create classifier
classifier = RandomForestClassifier()

# train model
classifier.fit(features_train, target_train)

# get accuracy score
classifier.score(features_test, target_test)

# 11.4 Evaluating Binary Classifier Predictions

# generate features matrix and target vector
x, y = make_classification(
    n_samples=10000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=1
)

# Create logistic regression
logit = LogisticRegression()

# cross-validate model using accuracy
cross_val_score(logit, x, y, scoring="accuracy")

# cross-validate model using precision
cross_val_score(logit, x, y, scoring="precision")

# cross-validate model using recall
cross_val_score(logit, x, y, scoring='recall')

# cross-validate model using F1
cross_val_score(logit, x, y, scoring='f1')

# if we have true y values and predicted y values we can
# calculate metrics directly

# create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=1
)

# predict values for training target vector
y_hat = logit.fit(X_train, y_train).predict(X_test)

# calculate accuracy
accuracy_score(y_test, y_hat)

# 11.5 Evaluating binary classifier thresholds
# - uses receiver operating characteristic (ROC) curve

# create feature matrix and target vector

features, target = make_classification(
    n_samples=10000,
    n_features=10,
    n_classes=2,
    n_informative=3,
    random_state=3
)

# split into training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# create classifier
logit = LogisticRegression()

# train model
logit.fit(features_train, target_train)

# get predicted probabilities
target_probabilites = logit.predict_proba(features_test)[:,1]

# create a true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(
target_test, target_probabilites
)

# plot ROC curve
# the better the model the closer it is to the solid gray line
plt.title("Receiver Operating Charactersitic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

# get predicted probabilities
logit.predict_proba(features_test)[0:1]
# get class
logit.classes_

# get tpr and fpr for threshold=0.5
print("Threshold:", threshold[124])
print("True Positive Rate:", true_positive_rate[124])
print("False Positive Rate:", false_positive_rate[124])

# the better the model the higher the area under curve
roc_auc_score(target_test, target_probabilites)

# 11.6 Evaluating Multiclass classifier predictions


# create feature matrix and target vector

features, target = make_classification(
    n_samples=10000,
    n_features=3,
    n_classes=3,
    n_informative=3,
    n_redundant=0,
    random_state=1
)

# create logistic regression
logit = LogisticRegression()

# cross-validate using accuracy
cross_val_score(logit, features, target, scoring='accuracy')

# cross-validate model using macro averaged F1 score,
# averaging the evaluation score from the classes
cross_val_score(logit, features, target, scoring='f1_macro')

# 11.7 Visualizing a classifiers performance

# load data
iris = datasets.load_iris()

# create feature matrix
features = iris.data

# create target vector
target = iris.target

# create a list of target class names
class_names = iris.target_names

# create training and test set
feature_train, feature_test, target_train, target_test = train_test_split(
    features, target, random_state=2
)

# create logistic regression
classifier = LogisticRegression()

# train model and make predictions
target_predicted = classifier.fit(feature_train, target_train).predict(feature_test)

# create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)

# create pandas dataframe
df = pd.DataFrame(matrix, index = class_names, columns = class_names)

# create heatmap
sns.heatmap(df, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")

# 11.8 Evaluating regression models
# - use mean squared error

# generate features matrix, target vector
features, target = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=1, noise=50,
                                   coef=False, random_state=1)

# create a linear regression object
ols = LinearRegression()

# cross validate the linear regression using (negative) MSE
cross_val_score(ols, features, target, scoring='neg_mean_squared_error')

# use coefficient of determination: R-squared
cross_val_score(ols, features, target, scoring='r2')


# 11.9 Evaluating clustering models

# create features matrix

features, _ = make_blobs(
    n_samples=1000, n_features=10, centers=2, cluster_std=0.5, shuffle=True, random_state=1
)

# cluster data using k-means to predict classes
model = KMeans(n_clusters=2, random_state=1).fit(features)

# get predicted classes
target_predicted = model.labels_

# evaluate model
silhouette_score(features, target_predicted)

# 11.10 Creating a custom metric
# -use a function to create a metric and convert it into a scorer
# using scikit-learn's make_scorer

# generate features matrix and target vector
features, target = make_regression(
    n_samples=100, n_features=3, random_state=1
)

# create training and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.10, random_state=1
)

# create custom metric
def custom_metric(target_test, target_predicted):
    # calculate r-squared
    r2 = r2_score(target_test, target_predicted)
    return r2

# make scorer and specify that higher scores are better
score = make_scorer(custom_metric, greater_is_better=True)

# create Ridge regression object
classifier = Ridge()

# train ridge regression model
model = classifier.fit(features_train, target_train)

# apply custom scorer
score(model, features_test, target_test)


# 11.11 Visualizing the effect of training set size

# load data
digits = load_digits()

# create feature matrix and target vector
features, target = digits.data, digits.target

# create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(), #classifier
    features, #feature matrix
    target, #target vector
    cv=10, #number of folds
    scoring='accuracy', #performance matrix
    n_jobs=-1, #use all computing cores
    train_sizes=np.linspace(0.01, 1.0, 50) # size of 50 training set
)

# create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111", label = "Training score")
plt.plot(train_sizes, test_mean, '--', color="#111111", label = "Cross-validation score")

# draw bands
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color = "#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color = "#DDDDDD")

# create plot
plt.title('Learning Curve')
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc = "best")
plt.tight_layout()
plt.show()

# 11.12 Creating text report of evaluation metrics
# e.g., describe a classifiers performance

# load data
iris = datasets.load_iris()

# create features matrix
features = iris.data

# create target vector
target = iris.target

# create list of target class names
class_names = iris.target_names

# create training and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0
)

# create logistic regression
classifier = LogisticRegression()

# train model and make predictions
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# create a classification report
print(classification_report(
    target_test, target_predicted, target_names=class_names
))

# 11.14 Visualizing the effect of hyperparameter values

# how does model performance change as values of some hyperparameters change

# load data
digits = load_digits()

# create feature matrix and target vector
features, target = digits.data, digits.target

# create a range of values for parameter
param_range = np.arange(1, 250, 2)

# calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), #classifier
    features, #feature matrix
    target, #target vector
    cv=3, #number of folds
    scoring='accuracy', #performance matrix
    n_jobs=-1, #use all computing cores
    param_name="n_estimators", #hyperparameter to examine
    param_range=param_range, #range of parameter's values
)

# create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score",
color="dimgrey")

# draw bands
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, color = "gray")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, color = "gainsboro")

# create plot
plt.title('Validation curve with Random Forests')
plt.xlabel("Number of Tress"), plt.ylabel("Accuracy Score"),
plt.legend(loc = "best")
plt.tight_layout()
plt.show()