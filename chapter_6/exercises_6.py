"""
1. What is the approximate depth of a decision tree trained (without restrictions)
on a training set with one million instances?
"""

"""
2. Is a node’s Gini impurity generally lower or higher than its parent’s? Is it generally
lower/higher, or always lower/higher?
"""
"""
3. If a decision tree is overfitting the training set, is it a good idea to try decreasing
max_depth?
"""
"""
4. If a decision tree is underfitting the training set, is it a good idea to try scaling the
input features?
"""
"""
5. If it takes one hour to train a decision tree on a training set containing one
million instances, roughly how much time will it take to train another decision
tree on a training set containing ten million instances? Hint: consider the CART
algorithm’s computational complexity.
"""
"""
6. If it takes one hour to train a decision tree on a given training set, roughly how
much time will it take if you double the number of features?
"""
"""
7. Train and fine-tune a decision tree for the moons dataset by following these
steps:
a. Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset.
b. Use train_test_split() to split the dataset into a training set and a test set.
c. Use grid search with cross-validation (with the help of the GridSearchCV
class) to find good hyperparameter values for a DecisionTreeClassifier.
Hint: try various values for max_leaf_nodes.
d. Train it on the full training set using these hyperparameters, and measure
your model’s performance on the test set. You should get roughly 85% to 87%
accuracy.
"""
"""
8. Grow a forest by following these steps:
a. Continuing the previous exercise, generate 1,000 subsets of the training set,
each containing 100 instances selected randomly. Hint: you can use ScikitLearn’s ShuffleSplit class for this.
b. Train one decision tree on each subset, using the best hyperparameter values
found in the previous exercise. Evaluate these 1,000 decision trees on the test
set. Since they were trained on smaller sets, these decision trees will likely
perform worse than the first decision tree, achieving only about 80% accuracy.
c. Now comes the magic. For each test set instance, generate the predictions of
the 1,000 decision trees, and keep only the most frequent prediction (you can
use SciPy’s mode() function for this). This approach gives you majority-vote
predictions over the test set.
d. Evaluate these predictions on the test set: you should obtain a slightly higher
accuracy than your first model (about 0.5 to 1.5% higher). Congratulations,
you have trained a random forest classifier!
"""
