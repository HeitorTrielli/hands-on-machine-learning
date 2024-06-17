import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""
Introduction
"""
mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist.data, mnist.target


def plot_digit(image_data: np.ndarray):
    """
    Plots the pixel data represented by the image_data variable
    """
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# index = 12638 (the most 5-ish 5 according to the scoring system of the cross val predict score for the binary classifier)
index = 0
y[index]
digit = X[index]
# plot_digit(digit)
# plt.show()

# The set is already shuffeled for us, which is nice, no need for shuffling
train_size = 60000
X_train, X_test, y_train, y_test = (
    X[:train_size],
    X[train_size:],
    y[:train_size],
    y[train_size:],
)


"""
Training a binary classifier
"""
# Finding the 5
y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

# Stochastic Gradient Descent classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Searching for false negatives/positives
false_neg = []
false_pos = []
for i in range(len(X_test)):
    if not sgd_clf.predict([X_test[i]])[0] and y_test[i] == "5":
        false_neg.append(i)
    elif sgd_clf.predict([X_test[i]])[0] and y_test[i] != "5":
        false_pos.append(i)


# IMO false negatives are reasonable confusing, while false positives are not.
# there are some very clearly not 5's in the false positives
"""
plot_digit(X_test[false_neg[5]])
plt.show()
"""


"""
Performance Performance measures
"""
# Accuracy with cross validation
cv_socre_sgd = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))

cv_socre_dummy = cross_val_score(
    dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"
)


# Confusion Matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

cm = confusion_matrix(y_train_5, y_train_pred)

# x_1_1 = true negatives; x_1_2 = false negatives; x_2_1 = false positives; x_2_2 = true positives
print(cm)

perfect_cm = confusion_matrix(y_train_5, y_train_5)

# Precision and recall

# Proportion of true predicted positives in the world of all predicted positives
# => How likely you are to get a real positive if you see a predicted positive
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
# Proportion of true predicted positives in the world of all true positives
# => How likely you are to classify a positive correctly
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

precision_sklearn = precision_score(y_train_5, y_train_pred)
recall_sklearn = recall_score(y_train_5, y_train_pred)

precision_sklearn == precision
recall_sklearn == recall

my_f1_score = 2 * precision * recall / (precision + recall)
scklearn_f1_score = f1_score(y_train_5, y_train_pred)

np.isclose(my_f1_score, scklearn_f1_score)

# Tradeoff between precision and recall

y_scores = [sgd_clf.decision_function([digit]) for digit in X_train]
low_treshold = 0
low_treshold_pred = [(score > low_treshold)[0] for score in y_scores]
high_treshold = 3000
high_treshold_pred = [(score > high_treshold)[0] for score in y_scores]

low_treshold_precision = precision_score(y_train_5, low_treshold_pred)
low_treshold_recall = recall_score(y_train_5, low_treshold_pred)

high_treshold_precision = precision_score(y_train_5, high_treshold_pred)
high_treshold_recall = recall_score(y_train_5, high_treshold_pred)

low_treshold_precision > high_treshold_precision
low_treshold_recall > high_treshold_recall

y_scores_sklearn = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

# Here, precision can decrease because when you increase the threshold, you sometimes will
# remove a true positive from the mix, reducing the numerator and the denominator at the same time while
# there still is a false positive in the predictions, which will reduce the precision score
# recalls never increase with threshold because the denominator is fixed and the numerator
# reduce
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_sklearn)

"""
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(low_treshold, 0, 1.0, "k", "dotted", label="low_threshold")
plt.vlines(high_treshold, 0, 1.0, "k", "dashed", label="high_threshold")
plt.legend()
plt.show()


plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.vlines(low_treshold_recall, 0, 1.0, "k", "dotted", label="low_threshold")
plt.vlines(high_treshold_recall, 0, 1.0, "k", "dashed", label="high_threshold")
plt.legend()
plt.show()
"""

idx_for_90_precision = (
    precisions >= 0.90
).argmax()  # Returns the first index that obbeys the condition
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

# The ROC Curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

"""
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.legend()
plt.show()
"""

roc_auc_score(y_train_5, y_scores)

# Creating a RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probs_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probs_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest
)

"""
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.legend()
plt.show()
"""

y_train_pred_forest = y_probs_forest[:, 1] >= 0.5
f1_score(y_train_5, y_train_pred_forest)
roc_auc_score(y_train_5, y_scores_forest)


"""
Multiclass Classification
"""
# Using SVC for multiclass classification
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
svm_clf.predict([digit])
some_digit_scores = svm_clf.decision_function([digit])
svm_clf.classes_

# Running One vs Rest
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
ovr_clf.predict([digit])
digit_scores_ovr = ovr_clf.decision_function([digit])

# Using SGD for muldiclass classification
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([digit])

sgd_clf.decision_function([digit]).round()
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# Scaling helps
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


"""
Error Analysis
"""
# Classification probabilities
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred, normalize="true", values_format=".0%"
)
plt.show()

# Probability of a true value be missclassified as the column value, given that the prediction is an error
sample_weight = y_train_pred != y_train
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    sample_weight=sample_weight,
    normalize="true",
    values_format=".0%",
)
plt.show()

# Given a wrong classification, the probabilities of being of each true value
sample_weight = y_train_pred != y_train
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    sample_weight=sample_weight,
    normalize="pred",
    values_format=".0%",
)
plt.show()


"""
Multilabel Classification
"""
y_train_large = y_train >= "7"
y_train_odd = y_train.astype("int8") % 2 == 1
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="weighted")

# Chains the classifier that is not made to support multilabel classification
# in a way that the prediction of previous cv runs are used as input in the
# following cv, so that the model can get a sense of dependency on the labels
chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([digit])


"""
Multioutput Classification
"""
np.random.seed(42)  # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# Cleaning the digit
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
