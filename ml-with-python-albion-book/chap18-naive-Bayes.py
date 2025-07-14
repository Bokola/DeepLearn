# continuous feature
from pyexpat import features

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# discrete/count data
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# binary features
from sklearn.naive_bayes import BernoulliNB

# calibrate probabilities
from sklearn.calibration import CalibratedClassifierCV


# 18.1 Training a Classifier for Continuous Features

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create Bayesian naive Bayes object
classifier  = GaussianNB()

# train model
model = classifier.fit(features, target)

# predict
new_obs = [[4, 4, 4, 0.4]]
# these are not calibrated and should not be trusted
model.predict_proba(new_obs)

# you can assign priors:
clf = GaussianNB(priors=[0.25, 0.25, 0.5])
model_clf = clf.fit(features, target)
model_clf.predict_proba(new_obs)

# 18.2 Training a Classifier for Discrete and Count Features

# create text data
text_data = np.array(['I love Brazil. Brazil!','Brazil is best','Germany beats both'])

# create a bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# create feature matrix
features = bag_of_words.toarray()

# create target vector
target = np.array([0, 0, 1])

# create a multinomial naive Bayes object with prior probabilities of each class
classifier = MultinomialNB(class_prior=[0.25, 0.5])

# train model
model = classifier.fit(features, target)

# 18.3 Training Naive Bayes Classifier for Binary Features

# create 3 binary features
features = np.random.randint(2, size = (100, 3))

# create a binary target vector
target = np.random.randint(2, size=(100, 1)).ravel()

# Bernoulli niave Bayes object
classifier = BernoulliNB(class_prior=[0.25, 0.5])

# train model
model = classifier.fit(features, target)

# 18.4 Calibrating Predicted Probabilities

# load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# create a Gaussian naive Bayes object
classifier = GaussianNB()

# create a calibrated cross-validation with sigmoid calibration
classifier_sigmoid = CalibratedClassifierCV(classifier, cv=2, method='sigmoid')

# calibrate probabilities
classifier_sigmoid.fit(features, target)

# create new observation
new_obs = [[2.6, 2.6, 2.6, 0.4]]

# view calibrated probabilities
classifier_sigmoid.predict_proba(new_obs)

