import numpy as np
from mpmath import hyper
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# selecting best model using exhaustive search
from sklearn.model_selection import GridSearchCV, cross_val_score
# selecting best model using randomized search
from sklearn.model_selection import RandomizedSearchCV
from srsly.ruamel_yaml.comments import tag_attrib

# selecting best models from multiple learning algorithms


# 12.1 Selecting best models using exhaustive search

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# create a range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']

# create a range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)

# create a dictionary of hyperparameter candidates
hyperparameters = dict(C=C, penalty = penalty)

# create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# fit grid search
best_model = gridsearch.fit(features, target)

# show the best model
print(best_model.best_estimator_)

# predict target vector from best model
best_model.predict(features)

# 12.2 Selecting the best model using randomized search

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# create a range of candidate regularization hyperparameter values
penalty = ['l1', 'l2']

# create a distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)

# create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# create randomized search
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=1, n_jobs=-1
)

# fit randomized search
best_model = randomizedsearch.fit(features, target)

# print best model
print(best_model.best_estimator_)

# 12.3 Selecting the best models from multiple learning algorithms

# create a pipeline
pipe = Pipeline([('classifier', RandomForestClassifier())])

# create a dictionary with candidate learning algorithms and their hyperparameters
search_space = [{"classifier": [LogisticRegression(max_iter=500, solver='liblinear')],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1,2,3]}]

# create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=2)

# fit grid search
best_model = gridsearch.fit(features, target)

# print best model
print(best_model.best_estimator_)

# 12,4 Selecting best models when preprocessing

np.random.seed(0)

# create a preprocessing object that includes StandardScaler features and PCA

preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# create a pipeline
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression(max_iter=1000, solver='liblinear'))])

# create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# create grid search
clf = GridSearchCV(pipe, search_space, cv=5, verbose=1, n_jobs=-1)

# fit grid search
best_model = clf.fit(features, target)

# print best model
print(best_model.best_estimator_)

# view best n_components
best_model.best_estimator_.get_params()['preprocess__pca__n_components']

# 12.5 Speeding up model selection with parallelization
# - use all the cores: set n_jobs=-1

# create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# create a range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']

# create a range of candidate values for C
C = np.logspace(0,4, 1000)

# create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# fit grid search
best_model = gridsearch.fit(features, target)

# print best model
print(best_model.best_estimator_)


# 12.6 Speeding up model selection using algorithm specific methods
# - use scikit-learn model-specific cross-validation hyperparameter tuning

# create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100, max_iter=500, solver='liblinear')

# train model
logit.fit(features, target)

# print model
print(logit)

# 12.7 Evaluating performance after model selection
# - use nested cross-validation to avoid biased evaluation

# create logistic regression

logistic = linear_model.LogisticRegression(max_iter=500, solver="liblinear")

# create a range of 20 values for C
C = np.logspace(0, 4, 20)

# create hyperparameter options
hyperparameters = dict(C=C)

# create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# conduct nested cross-validation and output the average score
cross_val_score(gridsearch, features, target).mean()