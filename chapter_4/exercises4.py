"""
1. Which linear regression training algorithm can you use if you have a training set
with millions of features?
"""

# Stochastic Gradient Descent


"""
2. Suppose the features in your training set have very different scales. Which algo‐
rithms might suffer from this, and how? What can you do about it?
"""

# Pretty much any alogrithim with regularization, because the regularization
# thetas scale with the feature. For example, a theta would be estimated as 1 or 1000
# if we chose to use a variable measured in units or in thousands.
# This would not be very good for scaling, since in the first example it would be
# wouldn't be that penalized in regularization, while in the second example it would.
# You can solve this by scalling the model.


"""
3. Can gradient descent get stuck in a local minimum when training a logistic
regression model?
"""


"""
4. Do all gradient descent algorithms lead to the same model, provided you let them
run long enough?
"""


"""
5. Suppose you use batch gradient descent and you plot the validation error at every
epoch. If you notice that the validation error consistently goes up, what is likely
going on? How can you fix this?
"""


"""
6. Is it a good idea to stop mini-batch gradient descent immediately when the
validation error goes up?
"""


"""
7. Which gradient descent algorithm (among those we discussed) will reach the
vicinity of the optimal solution the fastest? Which will actually converge? How
can you make the others converge as well?
"""


"""
8. Suppose you are using polynomial regression. You plot the learning curves and
you notice that there is a large gap between the training error and the validation
error. What is happening? What are three ways to solve this?
"""


"""
9. Suppose you are using ridge regression and you notice that the training error
and the validation error are almost equal and fairly high. Would you say that
the model suffers from high bias or high variance? Should you increase the
regularization hyperparameter α or reduce it?
"""


"""
10. Why would you want to use:
a. Ridge regression instead of plain linear regression (i.e., without any
regularization)?
b. Lasso instead of ridge regression?
c. Elastic net instead of lasso regression?
"""


"""
11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime.
Should you implement two logistic regression classifiers or one softmax regres‐
sion classifier?
"""


"""
12. Implement batch gradient descent with early stopping for softmax regression
without using Scikit-Learn, only NumPy. Use it on a classification task such as
the iris dataset.
"""
