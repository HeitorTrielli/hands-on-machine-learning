import numpy as np
from graphviz import Digraph, Source
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

"""
Training and Visualizing a Decision Tree
"""
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_iris, y_iris)

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
)

s = Source.from_file("iris_tree.dot")
s.view()

"""
Making Predictions
"""
# Codeless section


"""
Estimating Class Probabilities
"""

tree_clf.predict_proba([[5, 1.5]]).round(3)
tree_clf.predict([[5, 1.5]])


"""
The CART Training Algorithm
"""
# Codeless section


"""
Computational Complexity
"""
# Codeless section


"""
Gini Impurity or Entropy?
"""
# Codeless section


"""
Regularization Hyperparameters
"""

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)

X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=42)

tree_clf1.score(X_moons_test, y_moons_test)
tree_clf2.score(X_moons_test, y_moons_test)


"""
Regression
"""
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5  # a single random input feature
y_quad = X_quad**2 + 0.025 * np.random.randn(200, 1)
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)


"""
Sensitivity to Axis Orientation
"""
pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)


"""
Decision Trees Have a High Variance
"""
