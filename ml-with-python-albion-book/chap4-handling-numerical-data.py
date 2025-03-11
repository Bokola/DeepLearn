import numpy as np
from keras.src.layers.preprocessing.feature_space import Feature
from sklearn import preprocessing
import pandas as pd

# for normalizing vectors e.g l1, l2 etc

from sklearn.preprocessing import Normalizer

# for robust scaling - median and interquartile range

from sklearn.preprocessing import robust_scale

# for polynomials

from sklearn.preprocessing import PolynomialFeatures

# for transforming features

from sklearn.preprocessing import FunctionTransformer

# for detecting outliers

from sklearn.covariance import EllipticEnvelope

# for isotropic Gaussian blobs - Normal mixture distribution

from sklearn.datasets import make_blobs
# discretize features
from sklearn.preprocessing import Binarizer
# clustering
from sklearn.cluster import KMeans

# impute missing values

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 4.1: rescalling a feature

# create a feature

feature = np.array(
    [
        [-500.5],
        [-100.1],
        [0],
        [100.1],
        [900.9]
    ]
)
# create a scalar
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))

# scale feature
scaled_feature = minmax_scale.fit_transform(feature)
# show feature
scaled_feature

# 4.2 standardizing a feature N(0,1)
scaler = preprocessing.StandardScaler()
standardized = scaler.fit_transform(feature)
standardized

# scale using median and quantile range for data with outliers
robust_scaler = preprocessing.RobustScaler()
robust_scaler.fit_transform(feature)

# 4.3 Normalizing observations

#  rescale feature values to have a unit norm (a total length of 1)

features = np.array([[0.5, 0.5],
[1.1, 3.4],
[1.5, 20.2],
[1.63, 34.4],
[10.9, 3.3]])

# create normalizer
normalizer = Normalizer(norm="l2")
normalizer.transform(features)

# Generating polynomial interactions and features
features = np.array([
    [2, 3], [2,3], [2,3]
])
# create polynomial feature object
# degree parameter determines maximum degree of the polynomial
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
polynomial_interaction.fit_transform(features)

# restrict to only interaction features with interaction_only=True (xy)
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction.fit_transform(features)

# 4.5 Transforming features
# -> specifies return type
# see https://stackoverflow.com/questions/54962869/function-parameter-with-colon
def add_ten(x: int) -> int:
    return  x+10
# create a transformer
ten_transformer = FunctionTransformer(add_ten)
# transform feature matrix
ten_transformer.transform(features)

# using pandas instead
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
df.apply(add_ten)

# 4.6 Detecting outliers
# common approach is to assume data is normally distributed and draw an ellipse around the data

# create simulated data

features, _ = make_blobs(n_samples=10,n_features=2,centers=1,random_state=1)
# replace the first observation's values with extreme values
features[0,0] = 10000
features[0,1] = 10000

# create detector

outlier_detector = EllipticEnvelope(contamination=0.1)

# fit detector

outlier_detector.fit(features)

# predict outliers

outlier_detector.predict(features) # values of -1 are outliers, values of 1 inliers

# check extreme values in individual features

feature = features[:,0]

def indices_of_outliers(x: int) -> np.array(int):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

indices_of_outliers(feature)

# 4.8 Discretizating features

# binarizing
age = np.array([[6], [20], [12], [54]])
binarizer = Binarizer(threshold=18)
binarizer.fit_transform(age)
# using numpy

np.digitize(age, bins=[18], right=True)

# break up numerical feature according to multiple thresholds

np.digitize(age, bins = [20, 30, 50], right=True)

# 4.9 Grouping observations using clustering

# make simulated feature matrix

features, _ = make_blobs(n_samples=50,
                         n_features=2,
                         centers=3,
                         random_state=1)

# create df

df = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# make k-means cluster
clusterer = KMeans(3, random_state=0)

# fit cluster
clusterer.fit(features)
# predict
df["group"] = clusterer.predict(features)
df.head(5)

# 4.10 Dropping missing values

features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])

features[~np.isnan(features).any(axis=1)]


# using pandas

df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
df.dropna()

# 4.11 Imputing missing values

# make simulated feature matrix
features, _ = make_blobs(n_samples=1000, n_features=2, random_state=1)
# standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan
# predict missing value in the feature matrix
knn_imputer = KNNImputer(n_neighbors=5)
features_knn_imputed = knn_imputer.fit_transform(standardized_features)

# compare true and imputed values
print("true value: ", true_value)
print("imputed value: ", features_knn_imputed[0,0])
