from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, add_dummy_feature

"""
Linear Regression (my old friend)
"""
# Codeless section


"""
The Normal Equation ((XTX-1)XTy, another old friend)
"""
np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)  # column vector
y = 4 + 3 * X + np.random.randn(m, 1)  # column vector

X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y


"""
Computational Complexity
"""
# Codeless section


"""
Gradient Descent
"""
# Codeless section


"""
Batch Gradient Descent
"""
eta = 0.1  # learning rate
n_epochs = 100
m = len(X_b)  # number of instances
np.random.seed(42)
theta = np.random.randn(2, 1)  # randomly initialized model parameters
for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients


"""
Stochastic Gradient Descent
"""
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization
for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients


sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,
    eta0=0.01,
    n_iter_no_change=100,
    random_state=42,
)
sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets

sgd_reg.intercept_, sgd_reg.coef_


"""
Mini-Batch Gradient Descent
"""
# This one can leverage the GPU!
# Codeless section


"""
Polynomial Regression
"""
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_


"""
Learning Curves
"""


train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(),
    X,
    y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error",
)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

# plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
# plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
# plt.legend()
# plt.show()


polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False), LinearRegression()
)
train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression,
    X,
    y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error",
)

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)


# plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
# plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
# ax = plt.gca()
# ax.set_ylim([0, 2.5])
# plt.legend()
# plt.show()


"""
Ridge Regression
"""
ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])


sgd_reg = SGDRegressor(
    penalty="l1", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42
)

sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
sgd_reg.predict([[1.5]])


"""
Lasso Regression
"""

lasso_reg = Lasso(alpha=0.1)  # Similar as AS SGDRegressor(plenalty = 'l1', alpha = 0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])


"""
Elastic Net Regression (AKA Ridge-Lasso Regression)
"""
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])


"""
Early Stopping
"""
train_set, test_set = train_test_split(list(zip(X, y)), test_size=0.2, random_state=42)

X_train, y_train = zip(*train_set)
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1)
X_valid, y_valid = zip(*test_set)
X_valid, y_valid = np.array(X_valid), np.array(y_valid).reshape(-1)


preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False), StandardScaler()
)
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 5000
best_valid_rmse = float("inf")
# This algorythm does not really stop searching,
# but it does save the best model so we can rollback to it
for epoch in range(n_epochs):
    _ = sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
        final_epoch = epoch.copy()

"""
Logistic Regression
"""
iris = load_iris(as_frame=True)
iris.data.head(3)
iris.target.head(3)  # note that the instances are not shuffled
iris.target_names

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to get a column vector
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot(
    [decision_boundary, decision_boundary],
    [0, 1],
    "k:",
    linewidth=2,
    label="Decision boundary",
)
plt.legend()
plt.show()


"""
Softmax Regression
"""

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict_proba([[5, 2]]).round(2)
