#
"""
1. Which linear regression training algorithm can you use if you have a training set
with millions of features?
"""
# Stochastic Gradient Descent.

"""
2. Suppose the features in your training set have very different scales. Which algorithms 
might suffer from this, and how? What can you do about it?
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
# Nope, the function is convex :)

"""
4. Do all gradient descent algorithms lead to the same model, provided you let them
run long enough?
"""
# No. If you run a gradient descent on a cost function that has multiple local minimum,
# two runs might get stuck in different parameters, even if you use the same gradient
# descent algorythm.


"""
5. Suppose you use batch gradient descent and you plot the validation error at every
epoch. If you notice that the validation error consistently goes up, what is likely
going on? How can you fix this?
"""
# You might be overfitting the data. You should try to reduce the features that
# you are using, because there probably is a feature that helps fit the train
# data in a mathematical way, but takes the estimates away from the true function
# that generated the data


"""
6. Is it a good idea to stop mini-batch gradient descent immediately when the
validation error goes up?
"""
# Only if you are sure that you got in a global minium, such as the case when the
# cost function is knownly convex


"""
7. Which gradient descent algorithm (among those we discussed) will reach the
vicinity of the optimal solution the fastest? Which will actually converge? How
can you make the others converge as well?
"""
# Faster => Stochastic GD; Converge => Regular GD (if convex function)
# You can help the other algorithms to converge if you reduce their step size
# as the algorithm goes further


"""
8. Suppose you are using polynomial regression. You plot the learning curves and
you notice that there is a large gap between the training error and the validation
error. What is happening? What are three ways to solve this?
"""
# Most likely its overfitting. You can gather more data, remove some useless features
# or pray. You can always pray. You could also use lasso to reduce the dimensionality,
# or use some other form of regularization


"""
9. Suppose you are using ridge regression and you notice that the training error
and the validation error are almost equal and fairly high. Would you say that
the model suffers from high bias or high variance? Should you increase the
regularization hyperparameter \alpha or reduce it?
"""
# It suffers from high bias. The assumptions are probably wrong and you are
# getting results that are far away from the true Data Generating Process
# Increasing the regularization hyperparameter will help reduze the unnecessary
# weights that might be overfitting your data, so it is a good idea to increase
# it.


"""
10. Why would you want to use:
a. Ridge regression instead of plain linear regression (i.e., without any
regularization)?
b. Lasso instead of ridge regression?
c. Elastic net instead of lasso regression?
"""
# a.
# Ridge regression will help you reduce the overfitting of the data, in case of a
# wrong specification in the linear model
# b.
#  Lasso will treat those unnecessary features even more hardly, taking them to zero
# while Ridge will permit some leeway for them to exist
# c.
# Elastic Net gives us the best of both worlds, since it is a simple mixture of
# Ridge and Lasso. It will be able to find the sweetspot between the two models,
# giving some space for the smoothness of the Ridge regression and the bluntness
# of the Lasso regression

"""
11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime.
Should you implement two logistic regression classifiers or one softmax regression 
classifier?
"""
# If you want both outputs at the same time, it cannot be done via logistic regression,
# which means you can only use softmax

"""
12. Implement batch gradient descent with early stopping for softmax regression
without using Scikit-Learn, only NumPy. Use it on a classification task such as
the iris dataset.
"""
import numpy as np
