from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# mini-batch k-means to speed up k-means clustering
from sklearn.cluster import MiniBatchKMeans

# using mean shift
from sklearn.cluster import MeanShift

# using DBSCAN to group obs into clusters of high density
from sklearn.cluster import DBSCAN

# using Hierarchical merging
from sklearn.cluster import AgglomerativeClustering

# load data
iris = datasets.load_iris()
features = iris.data

# standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# create K-Means object
cluster = KMeans(n_clusters=3, random_state=0, n_init="auto")

# train model
model = cluster.fit(features_standardized)

# view predicted classes
model.labels_

# predict cluster of new observation
new_obs = [[0.8, 0.8, 0.8, 0.8]]
model.predict(new_obs)

# view cluster centers
model.cluster_centers_

# 19.2 Speeding Up K-Means Clustering

cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100, n_init="auto")
model = cluster.fit(features_standardized)

# 19.3 Clustering using mean shift
# do not assume number of clusters

# create mean shift object
cluster = MeanShift(n_jobs=-1)

# train model
model = cluster.fit(features_standardized)

# 19.4 Clustering using DBSCAN
cluster = DBSCAN(n_jobs=-1)
model = cluster.fit(features_standardized)

# 19.5 Clustering using Hierarchical merging
cluster = AgglomerativeClustering(n_clusters=3)
model = cluster.fit(features_standardized)