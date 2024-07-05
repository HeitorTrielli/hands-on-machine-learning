import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_moons
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from chapter_2.data_handling import load_housing_data

"""
Voting Classifiers
"""
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(random_state=42)),
    ]
)
voting_clf.fit(X_train, y_train)

for name, clf in voting_clf.named_estimators_.items():
    print(f"{name} = {clf.score(X_test, y_test)}")

voting_clf.score(X_test, y_test)  # It's better :)

voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)  # Even better :O


"""
Bagging and Pasting
"""
# Codeless section


"""
Bagging and Pasting in Scikit-Learn
"""
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    random_state=42,
)
bag_clf.fit(X_train, y_train)


"""
Out-of-Bag Evaluation
"""
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    oob_score=True,
    random_state=42,
)

bag_clf.fit(X_train, y_train)
bag_clf.oob_score_


y_pred = bag_clf.predict(X_test)

# OOB was pessimistic
accuracy_score(y_test, y_pred)

bag_clf.oob_decision_function_[:3]


"""
Random Patches and Random Subspaces
"""
# Codeless section


"""
Random Forests
"""
rnd_clf = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=10, random_state=42
)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

accuracy_score(y_test, y_pred_rf)

# Equivalent bagging_clf
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500,
    n_jobs=10,
    random_state=42,
)


"""
Extra-Trees
"""
# Codeless section


"""
Feature Importance
"""

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)


"""
Boosting
"""
# Codeless section

"""
AdaBoosting
"""

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=30,
    learning_rate=0.5,
    random_state=42,
    algorithm="SAMME",  # sklearn was yelling at me for using the SAMME.R algo :(
)
ada_clf.fit(X_train, y_train)


"""
Gradient Boosting
"""
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)  # y = 3x² + Gaussian noise
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

X_new = np.array([[-0.4], [0.0], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

# Same ensamble as the last one:
gbrt = GradientBoostingRegressor(
    max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42
)
gbrt.fit(X, y)

# With early stopping
gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=500,
    n_iter_no_change=10,
    random_state=42,
)
gbrt_best.fit(X, y)
gbrt_best.n_estimators_

# if we set subsample, we'll train each tree in smaller sets, trading
# higher bias for lowe variance. This is stochastic gradient boosting


"""
Histogram-Based Gradient Boosting
"""
hgb_reg = make_pipeline(
    make_column_transformer(
        (OrdinalEncoder(), ["ocean_proximity"]), remainder="passthrough"
    ),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42),
)

housing = load_housing_data()
housing_labels = housing.median_house_value
housing = housing.drop("median_house_value", axis=1)

hgb_reg.fit(housing, housing_labels)

# RMSE
np.sqrt((np.sum((hgb_reg.predict(housing) - housing_labels) ** 2)) / len(housing))


"""
Stacking
"""
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5,  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
