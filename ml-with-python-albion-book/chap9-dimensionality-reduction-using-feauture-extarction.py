from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets
from sklearn.datasets import make_circles
# class separability
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# nonnegative matrix factorization (NMF)
from sklearn.decomposition import NMF
# truncated singular value decomposition (TSVD)
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np


# 9.1 Reducing features using Principal components

# load the data
digits = datasets.load_digits()
# standardize the feature matrix
features = StandardScaler().fit_transform(digits.data)
# create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True)

# conduct PCA
features_pca = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])

# 9.2 Reducing features when data is linearly inseparable
# - use extension of PCA that use kernels for non-linear dimensionality reduction

# create linearly inseparable data

features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# apply kernel PCA with radius basis function (RBF) kernel
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

# 9.3 Reducing features by maximizing class separability
# Load Iris flower dataset:
iris = datasets.load_iris()
features = iris.data
target = iris.target
# create and run LDA, then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# Print the number of features
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])
# return explained variance
lda.explained_variance_ratio_

# finding number of components

# create and run lda
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)

# create array of explained variance ratios
lda_var_ratio = lda.explained_variance_ratio_

# create function
def select_n_components(var_ratio, goal_var: float) -> int:
    # set initial variance explained
    total_variance = 0.0

    # set initial number of features
    n_components=0

    # for the explained_variance of each feature
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add 1 to the number of components
        n_components += 1

        # If we reach our goal of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components

# run function
select_n_components(lda_var_ratio, 0.95)


# 9.4 Reducing features using matrix factorization
# uses nonnegetive matrix factorization

# load the dataset
digits = datasets.load_digits()

# load feature matrix
features = digits.data

# create, fit and apply NMF
nmf = NMF(n_components=10, random_state=4)
features_nmf = nmf.fit_transform(features)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])


# 9.5 Reducing features on sparse data

# Load the data
digits = datasets.load_digits()
# Standardize feature matrix
features = StandardScaler().fit_transform(digits.data)
# Make sparse matrix
features_sparse = csr_matrix(features)
# Create a TSVD
tsvd = TruncatedSVD(n_components=10)
# Conduct TSVD on sparse matrix
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
# Show results
print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])

# sum of first three components' explained variance ratios
tsvd.explained_variance_ratio_[0:3].sum()

# Automate getting n_components

#  create and run a TSVD with one less than the number of features
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

# list of explained variances
tsvd_var_ratios = tsvd.explained_variance_ratio_

# create a function

def select_n_components(var_ratio, goal_var):
    # set initial variance explained so far
    total_variance = 0.0

    # set initial number of features
    n_components = 0

    # For the explained variance of each feature
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add 1 to the number of components
        n_components += 1

        # if we reach our goal of explained variance
        if total_variance >= goal_var:
            # End  the loop
            break
    # return n_components
    return n_components

# run function
select_n_components(tsvd_var_ratios, 0.95)
