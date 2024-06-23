"""
1. What is the fundamental idea behind support vector machines?
"""

# The idea is that you might classify data by separating them with a hyperplane.
# If you can find a hyperplane that on one side is all the data from one class and
# on the other side the data from another class, then you can use this hyperplane as
# the baseline for prediction. You just preddict the class of a new datapoint as the
# class of the other datapoints that were on its side of the hyperplane

# PS: it is also important to chose the hyperplane that is as far as possible from
# each class. This means to maximize the 'street' that separates both classes, to
# reduce the chances of missclassification when you use the model on new data

"""
2. What is a support vector?
"""

# A support vector are the closest points of each class to the hyperplane. They
# determine where the hyperplane will be (right in the middle of these points)

"""
3. Why is it important to scale the inputs when using SVMs?
"""

# SVMs are very sensitive to the scale. If you have a variable that is way larger than
# the other variables, you will get a SVM that gets way too narrow, which will make the
# classification harder

"""
4. Can an SVM classifier output a confidence score when it classifies an instance?
What about a probability?
"""

# You can have the decision function of the classification and use it to understand
# how confident it is based on the distance to the separating hyperplane. It might
# also have a predict_proba method if you use the SVC class with the probability
# input set to True, but it won't have it by default

"""
5. How can you choose between LinearSVC, SVC, and SGDClassifier?
"""

# If your data is linearly separable, easy: LinearSVC (unless you need to run the model
# out of the core, then SGDClassifier will be your only possible model). Otherwise, you
# sould go SVC or SGDClassifier. It will again depend if you need to run the classification
# out of the core and also on the number of observations, since the SVC does not scale
# well with many observations

"""
6. Say you've trained an SVM classifier with an RBF kernel, but it seems to underfit
the training set. Should you increase or decrease gamma? What about C?
"""

# If it is underfitting the training set, you sould increase both gamma and C, since
# they both act like regularizing parameters. The C parameter contrtols the 'size of
# the street'. The bigger the C, the smaller the size, so we get more restrict in our
# classification, fitting the data more closely. The gamma parameter serves as a way
# to 'unlinearize' the model, permitting the regression to understand how far a single
# instance of a class can be related (in a somewhat 'distance' measure) to the next
# instance. It creates a 'zone of influence' for each data point and using the agregate
# zone of influence of all points, it permits to unite the classes using these zones.
# If the gamma paramter is too big, every observation will be in every other observation
# zone of influence, so the RBF Kernel wont be doing much. In contrast, if the gamma
# parameter is lower, you get more clear relation between the zones of influence, and
# then you can get a nonlinear understanding of the relations between each observation
# and its classes, and thus get a better fit

"""
7. What does it mean for a model to be ϵ-insensitive?
"""

# The SVM Regression is said to be epsilon-insensitive since adding new training
# observations within the margin (i.e. within an epsilon radius of the separating
# hyperplane), you do not change the predictions, thus being epsilon-insensitive


"""
8. What is the point of using the kernel trick?
"""

# The kernel trick is used to make the model work as if it had more dimensions, but
# without actually increasing the dimensions. This permits you to apply some interesting
# transformations to your data and not make it much harder to train, since the number
# of dimensions did not increase. One such transformation is the RBF Kernel, which, if
# applied to a linear model, will make it be not linear, without adding the extra
# computations of a higher order polynomial.

"""
9. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a
SGDClassifier on the same dataset. See if you can get them to produce roughly
the same model.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

np.random.seed(27)
sample_size = 100
zeros = np.zeros(sample_size)
x0 = np.random.uniform(-1, 1, sample_size)
y0 = np.random.uniform(-1, 1, sample_size)
data_0 = np.column_stack((x0, y0, zeros))

ones = np.ones(sample_size)
x1 = np.random.uniform(1, 3, sample_size)
y1 = np.random.uniform(1, 3, sample_size)
data_1 = np.column_stack((x1, y1, ones))


data = np.row_stack((data_0, data_1))
train_set, test_set = train_test_split(data)

"""
plt.scatter(data[:, 0], data[:, 1])
plt.show()
"""

train_set_feats, train_set_label = train_set[:, :2], train_set[:, 2]
test_set_feats, test_set_label = test_set[:, :2], test_set[:, 2]


lin_svc = LinearSVC()
lin_svc.fit(train_set_feats, train_set_label)

normal_svc = SVC()
normal_svc.fit(train_set_feats, train_set_label)

sgdc = SGDClassifier()
sgdc.fit(train_set_feats, train_set_label)


# All got one
accuracy_score(test_set_label, lin_svc.predict(test_set_feats))
accuracy_score(test_set_label, normal_svc.predict(test_set_feats))
accuracy_score(test_set_label, sgdc.predict(test_set_feats))


DecisionBoundaryDisplay.from_estimator(
    lin_svc, train_set_feats, plot_method="contour", response_method="predict"
)
DecisionBoundaryDisplay.from_estimator(
    normal_svc, train_set_feats, plot_method="contour", response_method="predict"
)
DecisionBoundaryDisplay.from_estimator(
    sgdc, train_set_feats, plot_method="contour", response_method="predict"
)
plt.show()

"""
10. Train an SVM classifier on the wine dataset, which you can load using
sklearn.datasets.load_wine(). This dataset contains the chemical analyses
of 178 wine samples produced by 3 different cultivators: the goal is to train
a classification model capable of predicting the cultivator based on the wine's
chemical analysis. Since SVM classifiers are binary classifiers, you will need to
use one-versus-all to classify all three classes. What accuracy can you reach?
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)

wine_df = wine.frame

train_set, test_set = train_test_split(wine_df)

train_feats, train_label = (
    np.array(train_set.drop(columns="target")),
    np.array(train_set.target),
)

test_feats, test_label = (
    np.array(test_set.drop(columns="target")),
    np.array(test_set.target),
)

svc_model_ovr = OneVsRestClassifier(LinearSVC(kernel="rbf"))

pipeline = make_pipeline(StandardScaler(), svc_model_ovr)

pipeline.fit(train_feats, train_label)

# Poly and RBF sometimes gets 1 accuraccy
cross_val_score(pipeline, train_feats, train_label, cv=6, scoring="accuracy").mean()


"""
11. Train and fine-tune an SVM regressor on the California housing dataset. You can
use the original dataset rather than the tweaked version we used in Chapter 2,
which you can load using sklearn.datasets.fetch_california_housing().
The targets represent hundreds of thousands of dollars. Since there are over
20,000 instances, SVMs can be slow, so for hyperparameter tuning you should
use far fewer instances (e.g., 2,000) to test many more hyperparameter combinations. 
What is your best model's RMSE?
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from chapter_2.pipelines import log_pipeline

calhouse = fetch_california_housing(as_frame=True)

data, target = calhouse.data, calhouse.target

train_feats, test_feats, train_target, test_target = train_test_split(
    data, target, train_size=2000
)

# No nulls, so we dont have to impute
train_feats.loc[train_feats.isna().any(axis=1)]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=4, gamma=1.0, random_state=27)

preprocessing = ColumnTransformer(
    [
        (
            "log",
            log_pipeline,
            [
                "Population",
                "MedInc",
            ],
        ),
        ("geo", cluster_simil, ["Latitude", "Longitude"]),
    ],
    remainder=StandardScaler(),
)  # one column remaining: housing_median_age


svr = SVR()

preprocessing.fit_transform(data).shape
svr_pipe = Pipeline([("preprocessing", preprocessing), ("svr", svr)])


param_grid = [
    {
        "preprocessing__geo__n_clusters": [2, 4, 8, 32, 64, 80, 100, 128],
        "preprocessing__geo__gamma": [0.1, 0.5, 1],
        "svr__kernel": ["linear", "rbf", "poly", "sigmoid"],
        "svr__C": [0.1, 1],
    },
]

grid_search = GridSearchCV(
    svr_pipe,
    param_grid=param_grid,
    n_jobs=14,
    verbose=10,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

grid_search.fit(train_feats, train_target)

final_model = grid_search.best_estimator_

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results.sort_values("mean_fit_time", ascending=False).to_excel("cv_result.xlsx")

final_model.predict(test_feats)
