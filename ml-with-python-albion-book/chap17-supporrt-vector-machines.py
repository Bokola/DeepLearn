# linear classifier
from sklearn.svm import LinearSVC, SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# plot
from matplotlib.colors import ListedColormap
import matplotlib
from matplotlib import pyplot as plt
plt.ion()
matplotlib.use('TkAgg')
plt.show()

# 17.1 Training a Linear Classifier

# data with 2 classes and 2 features
iris = datasets.load_iris()
features = iris.data[:100, :2]
target = iris.target[:100]

# standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# create a support vector classifier
svc = LinearSVC(C=1.0)

# Train model
model = svc.fit(features_standardized, target)

# visualize results
color = ["black" if c == 0 else "lightgray" for c in target]
plt.scatter(features_standardized[:,0], features_standardized[:,1], c=color)
# create hyperplane
w = svc.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0])/ w[1]

# plot the hyperplane
plt.plot(xx, yy)
plt.axis("off")

# predict
new_obs = [[-2, 3]]
svc.predict(new_obs)

# 17.2 Handling Linearly Inseparable Classes using Kernels

# set randomization seed
np.random.seed(0)

# generate 2 features
features = np.random.randn(200, 2)
# use XOR gate to generate linearly inseparable classes
target_xor = np.logical_xor(features[:, 0 ]>0, features[:, 1] > 0)
target = np.where(target_xor, 0, 1)

# create support vector machine with radial basis function kernel
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# train the classifier
model = svc.fit(features, target)

# plot decision regions

def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "blue"))
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker="+", label=cl)

# Create support vector classifier with a linear kernel
svc_linear = SVC(kernel="linear", random_state=0, C=1)
# Train model
svc_linear.fit(features, target)
SVC(C=1, kernel='linear', random_state=0)

# Plot observations and hyperplane
plot_decision_regions(features, target, classifier=svc_linear)
plt.axis("off"), plt.show();

# Create a support vector machine with a radial basis function kernel
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)
# Train the classifier
model = svc.fit(features, target)

# Plot observations and hyperplane
plot_decision_regions(features, target, classifier=svc)
plt.axis("off"), plt.show();

# 17.3 Creating predicted probabilities

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# standardize
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# create support vector classifier object
svc = SVC(kernel="linear", probability=True, random_state=0)

# train classifier
model = svc.fit(features_standardized, target)

# predict
new_obs = [[0.4, 0.4, 0.4, 0.4]]
model.predict_proba(new_obs)

# 17.4 Identifying Support Vectors
# which observations are the support vectors of the decision hyperplane?
# view features of the support vectors
model.support_vectors_

# view indices of the support vectors
model.support_

# view number of support vectors for each class
model.n_support_