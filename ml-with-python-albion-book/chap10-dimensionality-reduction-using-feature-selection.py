from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
# categorical features
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectPercentile
# recursively select features

import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model


# 10.1 Thresholding numerical feature variance
# - drop features with low variance

# data
iris = datasets.load_iris()

# create features and target
features = iris.data
target = iris.target

# create thresholder
thresholder = VarianceThreshold(threshold=0.5)

# create high variance feature matrix
features_high_variance = thresholder.fit_transform(features)

# view high variance matrix
features_high_variance[0:3]

# view variance of each feature
thresholder.fit(features).variances_

# variance thresholding will not work for standardized features

# standardize feature matrix
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Caculate variance of each feature
selector = VarianceThreshold()
selector.fit(features_std).variances_

# 10.2 Thresholding Binary Feature Variance

# Create feature matrix with:
# Feature 0: 80% class 0
# Feature 1: 80% class 1
# Feature 2: 60% class 0, 40% class 1
features = [[0, 1, 0],
[0, 1, 1],
[0, 1, 0],
[0, 1, 1],
[1, 0, 0]]
# Run threshold by variance
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
thresholder.fit_transform(features)

# 10.3 Handling highly correlated features

# Create feature matrix with two highly correlated features
features = np.array([[1, 1, 1],
[2, 2, 0],
[3, 3, 1],
[4, 4, 0],
[5, 5, 1],
[6, 6, 0],
[7, 7, 1],
[8, 7, 0],
[9, 7, 1]])

# convert ot df
df = pd.DataFrame(features)

# create a correlation matrix
corr_mat = df.corr().abs()

# select upper triangle of correlation matrix
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

# find index of feature columns with correlation greater than 0.95
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]

# drop features

df.drop(df.columns[to_drop], axis=1).head(3)

# 10.4 Removing irrelevant features for classification

# load date
iris = load_iris()
features = iris.data
target = iris.target

# convert to categorical data by converting data to integers
features = features.astype(int)

# select 2 features with highest chi-squared statistic
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# for quantitative features compute ANOVA f-values btwn each feature and target

# select two features with highest F-values
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# select top n percentile features instead

# select top 75% of features with highest F-values
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 10.5 Recursively eliminating features

# suppress any annoying but harmless warning
warnings.filterwarnings(action='ignore', module="scipy", message="^internal gelsd")

# Generate features matrix, target vector, amd the true coefficients
features, target = make_regression(n_samples=1000, n_features=100, n_informative=2, random_state=1)

# create a linear regression
ols = linear_model.LinearRegression()

# recursively eliminate features
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
rfecv.transform(features)

# Number of best features
rfecv.n_features_

# which categories are best to keep
rfecv.support_

# see feature rankings
rfecv.ranking_