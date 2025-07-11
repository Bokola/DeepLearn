from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# k-neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
# best neighborhood size, k
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
# radius-based nearest neighbor
from sklearn.neighbors import RadiusNeighborsClassifier



# 15.1 Find an observation's nearest neighbours

# load data
iris = datasets.load_iris()
features = iris.data

# create a standardizer
standardizer = StandardScaler()

# standardize features
features_standardized = standardizer.fit_transform(features)

# Two nearest neighbors
knn = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# create an observation
new_obs = [1, 1, 1, 1]

# find distance and indices of the observation's nearest neighbors
distances, indices = knn.kneighbors([new_obs])

# view the nearest neighbors
features_standardized[indices]

# use kneighbors_graph to create a matrix of each observation's nearest neighbors
# including itself

nearest_neighbors_euclidean = NearestNeighbors(
    n_neighbors=3, metric="euclidean"
).fit(features_standardized)

# list of lists indicating each observation's 3 nearest neighbors
# (including itself)
nearest_neighbors_with_self = nearest_neighbors_euclidean.kneighbors_graph(features_standardized).toarray()

# remove 1s marking an observation is a nearest neighbor to itself
for i, x in enumerate(nearest_neighbors_with_self):
    x[i] = 0
# view first observation's nearest neighbors
nearest_neighbors_with_self[0]

# 15.2 Creating K-Nearest Neighbors Classifier
# if the data not very large, use KNeighborsClassifier
# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a standardizer
standardizer = StandardScaler()

# standardize features
X_std = standardizer.fit_transform(X)

# train KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

# create 2 obs
new_obs = [[0.75, 0.75, 0.75, 0.75], [1, 1, 1, 1]]

# predict the class of 2 obs
knn.predict(new_obs)

# view prediction probabilities
knn.predict_proba(new_obs)

# 15.3 Identifying the Best Neighborhood Size, k

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create standardizer
standardizer = StandardScaler()

# create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# create a pipeline
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

# create grid search
classifier = GridSearchCV(
    pipe, search_space, cv=5, verbose=0
).fit(features, target)

classifier = GridSearchCV(
pipe, search_space, cv=5, verbose=0).fit(features, target)

# best neighborhood size (k)
classifier.best_estimator_.get_params()["knn__n_neighbors"]

# 15.4 Creating a radius-based nearest neighbors

# train a radius neighbors classifier
rnn = RadiusNeighborsClassifier(radius=.5, n_jobs=-1).fit(standardizer.fit_transform(features), target)

# predict class of new obs
new_obs = [[1, 1, 1, 1]]
rnn.predict(new_obs)
