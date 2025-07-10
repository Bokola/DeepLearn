from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# regularize to reduce variance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# interactive effects
from sklearn.preprocessing import PolynomialFeatures
# get ideal alpha value for ridge regression
from sklearn.linear_model import RidgeCV
# reduce features with Lasso regression
from sklearn.linear_model import Lasso

# 13.1 Fitting a line with LinearRegression

# generate feature matrix and target vector
features, target = make_regression(n_samples=1000,
                                   n_features=3,
                                   n_informative=2,
                                   n_targets=1,
                                   noise=0.2,
                                   coef=False,
                                   random_state=1
                                   )
# create linear regression
lm = LinearRegression()

# fit linear regression
model = lm.fit(features, target)

# view coefs and intercept
model.coef_
model.intercept_

# first value in the target vector
target[0]

# predict target value of the first observation
model.predict(features)[0]

# print score of the model on the training data
print(model.score(features, target))

# 13.2 Handling Interactive effects

# generate feature matrix and target vector
features, target = make_regression(n_samples=100,
                                   n_features=2,
                                   n_informative=2,
                                   n_targets=1,
                                   noise=0.2,
                                   coef=False,
                                   random_state=1
                                   )
# create interaction term
interaction = PolynomialFeatures(degree=3,
                                 include_bias=False
                                 ,interaction_only=True)
features_interaction = interaction.fit_transform(features)

# create linear regression
regression = LinearRegression()

# fit the linear regression
model = regression.fit(features_interaction, target)

# create interaction by multiplying features
interaction_term = np.multiply(features[:, 0], features[:, 1])
interaction_term[0]


# 13.3 fit non-linear relationships by including polynomials

polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# fit
reg = LinearRegression()
model = reg.fit(features_polynomial, target)

# 13.4 Reducing variance with regularization

# generate feature matrix and target vector
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   n_informative=2,
                                   n_targets=1,
                                   noise=0.2,
                                   coef=False,
                                   random_state=1
                                   )

# standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# create a ridge regression with an alpha value
reg = Ridge(alpha=0.5)

# fit the linear regression
model = reg.fit(features_standardized, target)

# create a ridge regression with 3 alpha values
regr_Cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# fit the linear regression
model_cv = regr_Cv.fit(features_standardized, target)

# view best model's alpha value
model_cv.alpha_


# 13.5 Reducing Features with Lasso Regression

# standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# create Lasso regression with alpha value
lass_reg = Lasso(alpha=0.5)
# fit the linear regression
model = lass_reg.fit(features_standardized, target)

# lasso shrinks some coefficients to zero
model.coef_