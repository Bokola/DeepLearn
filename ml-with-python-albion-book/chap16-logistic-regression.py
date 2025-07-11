# binary classifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# regularization to reduce variance
from sklearn.linear_model import LogisticRegressionCV

# 16.1 Training a Binary Classifier

# load data with only 2 classes
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

# standardize
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# create logistic regression object
logistic_regression = LogisticRegression(random_state=0)

# train model
model = logistic_regression.fit(features_standardized, target)

# predict class of new observations
new_obs = [[.5, 0.3, 0.5, 0.6]]
model.predict(new_obs)

# view predicted probabilities
model.predict_proba(new_obs)

# 16.2 Training a Multiclass Classifier
# uses one-vs-rest

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

features_standardized = scaler.fit_transform(features)

# create one-vs-rest logistic regression object
logistic_regression = LogisticRegression(random_state=0)#, multi_class="ovr")

# train model
model = OneVsRestClassifier(logistic_regression.fit(features_standardized, target))

# 16.3 Reducing Variance through Regularization
# create decision tree regression object
logistic_regression = LogisticRegressionCV(penalty='l2', Cs=10, random_state=0, n_jobs=-1)

# train model
model = logistic_regression.fit(features_standardized, target)

# 16.4 Training a Classifier over very Large Data
# -use stochastic average gradient (SAG) solver

logistic_sag = LogisticRegression(random_state=0, solver="sag")

model = logistic_sag.fit(features_standardized, target)