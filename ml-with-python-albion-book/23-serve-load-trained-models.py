import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

from loo import features

# 23.1 Saving and Loading a scikit-learn Model

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create decision tree classifier object
classifier = RandomForestClassifier()

# train model
model = classifier.fit(features, target)

# save model as a pickle file
joblib.dump(model, 'model.pkl')

# load and use saved model
classifier = joblib.load('model.pkl')
new_obs = [[5.2, 3.2, 1.1, 0.1]]
classifier.predict(new_obs)

# 23.2 Saving and Loading TensorFlow Model
