import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

"""
Linear SVM Classification
"""
# Codeless section

"""
Soft Margin Classification
"""

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 2  # Iris virginica
svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)


X_new = [[5.5, 1.7], [5.0, 1.5]]
svm_clf.predict(X_new)

svm_clf.decision_function(X_new)

"""
Nonlinear SVM Classification
"""

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42),
)
polynomial_svm_clf.fit(X, y)

"""
Polynomial Kernel
"""
poly_kernel_svm_clf = make_pipeline(
    StandardScaler(), SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_svm_clf.fit(X, y)


"""
Similarity Features
"""
# Codeless section

"""
Gaussian RBF Kernel
"""

rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(), SVC(kernel="rbf", gamma=5, C=0.001)
)
rbf_kernel_svm_clf.fit(X, y)

"""
SVM Classes and Computational Complexity
"""
# Codeless chapter

"""
SVM Regression
"""
# Linear
X, y = [...]  # a linear dataset

np.random.seed(42)
m = 1000
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X + 2 + np.random.randn(m, 1)

svm_reg = make_pipeline(StandardScaler(), LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X, y)

svm_reg.predict(X[1].reshape(-1, 1))

x = X[1]
0.5 * x + 2

# Quadratic
X, y = [...]  # a quadratic dataset
m = 1000
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

svm_poly_reg = make_pipeline(
    StandardScaler(), SVR(kernel="poly", degree=2, C=0.1, epsilon=0.1)
)
svm_poly_reg.fit(X, y)


svm_poly_reg.predict(X[0].reshape(1, -1))

x = X[0]
0.5 * x**2 + x + 2

"""
Under the Hood of Linear SVM Classifiers
"""
# Codeless chapter

"""
The Dual Problem
"""
# Codeless chapter


"""
Kernelized SVMs
"""
# Codeless chapter
