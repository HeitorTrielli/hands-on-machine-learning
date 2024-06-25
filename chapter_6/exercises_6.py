import numpy as np

"""
1. What is the approximate depth of a decision tree trained (without restrictions)
on a training set with one million instances?
"""
# It depends on how easily separatable the training set is. Might be two, might be one million


# Correction: I got it all wrong.
# Generally, decision trees are balanced and the depth is approximately log_2(m). Thus, if we
# have one million instances, we should have around

np.ceil(np.log2(1_000_000))

# layers

"""
2. Is a node's Gini impurity generally lower or higher than its parent's? Is it generally
lower/higher, or always lower/higher?
"""

# It is generally higher, since the higher sets had more freedom to select the best separation
# to reduce the gini, while the lower ones had to work with what they got

# Got it all wrong again (I think I interpreted wrong the question, which is the worst kind of
# error). Here's the correction:
# It's generally LOWER. The CART algorithm specifically works to minimize the GINI of the child
# node, although if one of the child nodes can have a significant reduction on the GINI, the
# other might end up having a higher GINI than the father node, since the algorithm does not care
# for individual child nodes, but the sum of the GINI of them all

"""
3. If a decision tree is overfitting the training set, is it a good idea to try decreasing
max_depth?
"""

# Yes! If it is overfitting, you should reduce its freedom, and capping the depth is a way to
# restrict the model

"""
4. If a decision tree is underfitting the training set, is it a good idea to try scaling the
input features?
"""

# Decision trees should be fairly unaffected by scaling, so it shouldn't matter much

"""
5. If it takes one hour to train a decision tree on a training set containing one
million instances, roughly how much time will it take to train another decision
tree on a training set containing ten million instances? Hint: consider the CART
algorithm's computational complexity.
"""
import numpy as np

# If it takes 1h for 1m
np.exp(1) / np.exp(1)

# It takes 8100 hours for 10m, oof
np.exp(10) / np.exp(1)


# Another correction: I was using the complexity of finding the BEST tree, not a reasonably
# good tree. Here is the revised maths:

# If it takes 1h for 1m:
m0 = 1_000_000

normalizer = m0 * np.log2(m0)

time_0 = m0 * np.log2(m0) / normalizer

# Then it takes around 11.6 hours to train 10m

m1 = 10_000_000

time_1 = m1 * np.log2(m1) / normalizer

"""
6. If it takes one hour to train a decision tree on a given training set, roughly how
much time will it take if you double the number of features?
"""

# it will take e times more

np.exp(2) / np.exp(1)

# Got it wrong for the same reason of exercise 5. Here's the revised math:

m0 = 2

for m0 in list(range(2, 300000, 100000)):
    normalizer = m0 * np.log2(m0)
    time_0 = m0 * np.log2(m0) / normalizer

    # It takes around 2.1 times more
    m1 = 2 * m0
    time_1 = m1 * np.log2(m1) / normalizer

    print(time_1 / time_0)
    print(2 * np.log2(m1) / np.log2(m0))

# For smaller datasets it takes a little bit longer, but it shouldn't take
# much more than 2.1 times.

# It will take 2*log2(m1)/log2(m0) times longer, if you want a closed form


"""
7. Train and fine-tune a decision tree for the moons dataset by following these
steps:
a. Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset.
b. Use train_test_split() to split the dataset into a training set and a test set.
c. Use grid search with cross-validation (with the help of the GridSearchCV
class) to find good hyperparameter values for a DecisionTreeClassifier.
Hint: try various values for max_leaf_nodes.
d. Train it on the full training set using these hyperparameters, and measure
your model's performance on the test set. You should get roughly 85% to 87%
accuracy.
"""
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

feats, targets = moons[0], moons[1]
train_feats, test_feats, train_target, test_target = train_test_split(
    feats, targets, random_state=42
)

dtc = DecisionTreeClassifier()

np.log2(10000)

param_grid = {
    "max_depth": [1, 2, 4, 7, 8, 9, 10, 15, None],
    "max_leaf_nodes": range(2, 50),
    "min_samples_split": [2, 10, 100, 400, 1000],
}

grid_search = GridSearchCV(
    dtc, param_grid=param_grid, scoring="accuracy", verbose=5, n_jobs=10
)

grid_search.fit(train_feats, train_target)

best_estimator = grid_search.best_estimator_

accuracy_score(test_target, best_estimator.predict(test_feats))


"""
8. Grow a forest by following these steps:
a. Continuing the previous exercise, generate 1,000 subsets of the training set,
each containing 100 instances selected randomly. Hint: you can use ScikitLearn's 
ShuffleSplit class for this.
b. Train one decision tree on each subset, using the best hyperparameter values
found in the previous exercise. Evaluate these 1,000 decision trees on the test
set. Since they were trained on smaller sets, these decision trees will likely
perform worse than the first decision tree, achieving only about 80% accuracy.
c. Now comes the magic. For each test set instance, generate the predictions of
the 1,000 decision trees, and keep only the most frequent prediction (you can
use SciPy's mode() function for this). This approach gives you majority-vote
predictions over the test set.
d. Evaluate these predictions on the test set: you should obtain a slightly higher
accuracy than your first model (about 0.5 to 1.5% higher). Congratulations,
you have trained a random forest classifier!
"""
# How will I split the train test into 1000 sets of 100, if the full set has
# 100_000 observations? We won't have a test set! Or can it have replacement?
# I'll do it with replacement

from scipy.stats import mode
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

indexes = ShuffleSplit(1000, train_size=100, random_state=42)

predictions = list()
accuracies = list()

train_idxs, test_idxs = zip(*[(x, y) for x, y in indexes.split(train_target)])

mini_train_feats = [train_feats[idxs] for idxs in train_idxs]
mini_train_target = [train_target[idxs] for idxs in train_idxs]

forest = [clone(grid_search.best_estimator_) for _ in range(len(mini_train_feats))]

for tree, feats, targets in zip(forest, mini_train_feats, mini_train_target):
    tree.fit(feats, targets)

    prediction = tree.predict(test_feats)
    predictions.append(prediction)

    accuracies.append(accuracy_score(test_target, prediction))


full_prediction, n_votes = mode(predictions)

# Not that far of the best estimator, but I did search a lot
accuracy_score(test_target, full_prediction)
