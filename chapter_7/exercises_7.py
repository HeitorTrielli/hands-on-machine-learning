"""
1. If you have trained five different models on the exact same training data, and
they all achieve 95% precision, is there any chance that you can combine these
models to get better results? If so, how? If not, why?
"""

# Yes! You could do an ensemble model, that combines the prediction of all five
# models and use this new prediction as the result. For example, if you have a
# classifier, you could use the most predicted class across the 5 models as the
# overall prediction

"""
2. What is the difference between hard and soft voting classifiers?
"""

# Hard voting classifiers take the mode of the prediction of all models, while
# soft voting weight the vote by the probability that each model assesses to
# each class. This means that soft voting can leverage on the models confidence
# in the classification, making more confident classifications have a higher
# weight in the overall voting

"""
3. Is it possible to speed up training of a bagging ensemble by distributing it across
multiple servers? What about pasting ensembles, boosting ensembles, random
forests, or stacking ensembles?
"""

# Yes. You can do Bagging in parallel, not only in the CPU level, but also server level
# if you want to. You can also do this with pasting. For boosting, you unfortunately cannot
# paralelize, since each round depends on the result of the last round. Random forests are
# just another way of bagging, hence they can be distributed. Stacking ensambles can also be
# paralelized, since each model is independent from the last one

# PS: for the stacking ensembles, the layers have to be sequential, so you can only keep
# training the model after all the servers end their computation

"""
4. What is the benefit of out-of-bag evaluation?
"""

# The benefit is that you don't need to separate part of your set for validation, since the
# OOB sample had not been used for training

"""
5. What makes extra-trees ensembles more random than regular random forests?
How can this extra randomness help? Are extra-trees classifiers slower or faster
than regular random forests?
"""

# Extra-trees gain exra randomness because the threshold that will split the predictions is
# decided randomly. This means that we can skip the costly step of finding the best threshold.
# Since we skip the optimization on the threshold, it ends up being faster than regular random
# forests

"""
6. If your AdaBoost ensemble underfits the training data, which hyperparameters
should you tweak, and how?
"""

# You could tweak the learning rate. If you increase it, you should get a better fit,
# since the model will give more weight to the features that didn't perform as well in
# the last iteration, hence making it more likely to fit better.
# You could also increase the

"""
7. If your gradient boosting ensemble overfits the training set, should you increase
or decrease the learning rate?
"""

# You should decrease it. The higher the training rate, the more weight the model gives
# to the error regressions, that are what increase the fit. Thus, if you are overfitting,
# reduce the learning rate.

"""
8. Load the MNIST dataset (introduced in Chapter 3), and split it into a training
set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000
for validation, and 10,000 for testing). Then train various classifiers, such as a
random forest classifier, an extra-trees classifier, and an SVM classifier. Next, try
to combine them into an ensemble that outperforms each individual classifier
on the validation set, using soft or hard voting. Once you have found one, try
it on the test set. How much better does it perform compared to the individual
classifiers?
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC

mnist = fetch_openml("mnist_784")

feats = mnist.data
target = mnist.target

full_set = pd.concat([feats, target], axis=1)

shuffled = full_set.sample(frac=1)

train_set = shuffled[:50000]
test_set = shuffled[50000:60000]
validate_set = shuffled[60000:]


forest_clf = RandomForestClassifier()
extra_clf = ExtraTreesClassifier()
svc_clf = SVC(
    probability=True
)  # SVC will take sooooo long to train in 50k observations :s

models = [forest_clf, extra_clf, svc_clf]

performances = np.empty(3)
for index, model in enumerate(models):
    model.fit(train_set.drop(columns="class"), train_set["class"])
    performances[index] = model.score(
        validate_set.drop(columns="class"), validate_set["class"]
    )

named_models = [("forest", forest_clf), ("extra", extra_clf), ("svc", svc_clf)]

stack = StackingClassifier(named_models, n_jobs=3)

stack.fit(train_set.drop(columns="class"), train_set["class"])

stack.score(validate_set.drop(columns="class"), validate_set["class"])


"""
9. Run the individual classifiers from the previous exercise to make predictions on
the validation set, and create a new training set with the resulting predictions:
each training instance is a vector containing the set of predictions from all your
classifiers for an image, and the target is the image's class. Train a classifier
on this new training set. Congratulations—you have just trained a blender, and
together with the classifiers it forms a stacking ensemble! Now evaluate the
ensemble on the test set. For each image in the test set, make predictions with all
your classifiers, then feed the predictions to the blender to get the ensemble's 
predictions. How does it compare to the voting classifier you trained earlier? Now
try again using a StackingClassifier instead. Do you get better performance? If
so, why?
"""
predictions = [model.predict(validate_set.drop(columns="class")) for model in models]
new_train_set = pd.concat(
    [
        pd.DataFrame(np.array(predictions).T),
        validate_set["class"].reset_index(drop=True),
    ],
    axis=1,
)

blender_forest = RandomForestClassifier()
blender_forest.fit(new_train_set.drop(columns="class"), new_train_set["class"])

new_test_predictions = [
    model.predict(test_set.drop(columns="class")) for model in models
]

blender_forest.score(np.array(new_test_predictions).T, test_set["class"])

# Stacking CLF was better, since it used predict_proba to have a more nuanced version
# of the problem. It also did kfold cross validation, so it had a better chance to
# select a more generizable model
