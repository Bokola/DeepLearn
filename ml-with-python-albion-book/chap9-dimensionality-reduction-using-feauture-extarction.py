from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets
from sklearn.datasets import make_circles


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