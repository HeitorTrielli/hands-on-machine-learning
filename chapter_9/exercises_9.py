import numpy as np

"""
1. How would you define clustering? Can you name a few clustering algorithms?
"""
# Clustering is to identificate patterns on data that many observations share
# and classifying them according to these patterns. Each set of patterns is a
# cluster.
#
# Some clustering algorithms are K-Means, DBSCAN, Agglomerative Clustering, BIRCH
# Mean-Shift, Affinity-Propagation and Spectral Clustering


"""
2. What are some of the main applications of clustering algorithms?
"""
# Clustering can be used for many reasons. Some of the applications are:
# Consumer segmentation, anomaly detection, density estimation and dimensionality
# reduction


"""
3. Describe two techniques to select the right number of clusters when using
k-means.
"""
# You could grid search some clusters and use the one with the best silhouette score
# or at least one with a silhouette score as good as the best, but with each cluster
# having about the same size. You could also use the inertia score, but that will
# have a higher chance of choosing a high value for k.
#
# PS (from the solutions): to find the beset inertia score, it is not correct to
# just pick the lowest one, since it always reduces with k, but the first k that reduces
# the rate of decreasing the inertia

"""
4. What is label propagation? Why would you implement it, and how?
"""
# Label propagation is when you have a cluster that has one (or more) observations that
# are actually properly labeled, and then you just say that every single observation in
# the cluster follows the same label and feed this fully labeled set to fuel another
# model, such as a Random Tree. This might be useful if you want to run some predictions
# of new observations, since a supervised model depends on having labels on all entries


"""
5. Can you name two clustering algorithms that can scale to large datasets? And two
that look for regions of high density?
"""
# Two good clustering algorithms that can scale are BIRCH (as long as you don't have
# many features), and Agglomerative Clustering (as long as you provide a connectivity
# matrix). => PS (from the solutions): also K-Means
#
# Two algorithms that search for regions of high density are DBSCAN and Mean-shift


"""
6. Can you think of a use case where active learning would be useful? How would
you implement it?
"""
# You could use it for agricultural planning. For example, you have a drone that take
# some pictures of a farmland, and have a model determine what spots might be ready
# to be harvested, or maybe what spots might be struck by some desease. When the model
# finds a spot that it is not sure, it might be a new desease or some different reaction
# to the weather. Then a specialist could either see the image and classify or go on-site
# to identify what is going on. Then, with its new knowledge, the next time something like
# this happens, the model will be better suited to take the decision on its own.
#
# In practice, you would have the model to set a null or some arbitrary prediction to any
# observation that does not pass a threshold, and then a specialist fetches all of these
# predictions and switch the value to the propper one. After that you feed the model with
# the new label and re-train it.


"""
7. What is the difference between anomaly detection and novelty detection?
"""
# Anomally detection is trained in the full data-set and points out the entries that are
# out of line with the 'normal' set. That is, it points out the outliers at the moment of
# training.
#
# Novelty detection, you train the model on the 'normal' set and then, when new instances
# that does not fit the expected values of the model, it is treated as a novelty.


"""
8. What is a Gaussian mixture? What tasks can you use it for?
"""
# A Gaussian Mixture is a model that treats the clusters of the dataset as being sampled from
# many Gaussian distributions. It can be used to find clusters that are elipsoidal, even if
# each elipses are of different shapes and orientations.
#
# PS (from the solutions) (I forgot to answer the second part xD): It is useful for density
# estimation, clustering and anomaly detection


"""
9. Can you name two techniques to find the right number of clusters when using a
Gaussian mixture model?
"""
# To select the correct number of clusters, you could, just like in k-means, run the model
# with a varied number of clusters and then select the best number according to a specific
# score. For exemple, you could use ether the AIC (Akaike Information Criterion) or the
# BIC (Bayesian Information Criterion)
#
# PS (from the solutions): you could also use a Bayesian Gaussian Mixture model, that selects
# the number of clusters automatically


"""
10. The classic Olivetti faces dataset contains 400 grayscale 64 x 64-pixel images of
faces. Each image is flattened to a 1D vector of size 4,096. Forty different people
were photographed (10 times each), and the usual task is to train a model that
can predict which person is represented in each picture. Load the dataset using
the sklearn.datasets.fetch_olivetti_faces() function, then split it into a
training set, a validation set, and a test set (note that the dataset is already scaled
between 0 and 1). Since the dataset is quite small, you will probably want to
use stratified sampling to ensure that there are the same number of images per
person in each set. Next, cluster the images using k-means, and ensure that you
have a good number of clusters (using one of the techniques discussed in this
chapter). Visualize the clusters: do you see similar faces in each cluster?
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit

# Separating
faces = fetch_olivetti_faces()

feats = faces.data
labels = faces.target

train_test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, test_val_idx = list(train_test_splitter.split(feats, labels))[0]

train_feats = feats[train_idx]
train_labels = labels[train_idx]

test_val_feats = feats[test_val_idx]
test_val_labels = labels[test_val_idx]

test_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
test_idx, val_idx = list(test_val_splitter.split(test_val_feats, test_val_labels))[0]

test_feats = feats[test_idx]
test_labels = labels[test_idx]

val_feats = feats[val_idx]
val_labels = labels[val_idx]

# K-Means-ing


kmean_models = [KMeans(n_clusters=x) for x in range(5, 150, 5)]

model_score_pair = []
for kmean in kmean_models:
    kmean.fit(train_feats, train_labels)
    sil_score = silhouette_score(train_feats, kmean.labels_)
    model_score_pair.append((kmean, sil_score))

scores = [x[1] for x in model_score_pair]

plt.plot(scores)
plt.show()

best_model, score = model_score_pair[np.argmax(scores)]


# Coppied from the solutions because lazy to code plotting
def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()


"""
for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_ == cluster_id
    faces = train_feats[in_cluster]
    labels = train_labels[in_cluster]
    plot_faces(faces, labels)
"""

####

"""
11. Continuing with the Olivetti faces dataset, train a classifier to predict which
person is represented in each picture, and evaluate it on the validation set. Next,
use k-means as a dimensionality reduction tool, and train a classifier on the
reduced set. Search for the number of clusters that allows the classifier to get
the best performance: what performance can you reach? What if you append the
features from the reduced set to the original features (again, searching for the
best number of clusters)?
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest_clf = RandomForestClassifier()
forest_clf.fit(train_feats, train_labels)
accuracy_score(val_labels, forest_clf.predict(val_feats))

reduced_train_feats = best_model.transform(train_feats)
forest_clf.fit(reduced_train_feats, train_labels)
accuracy_score(val_labels, forest_clf.predict(best_model.transform(val_feats)))
# Not great

reduced_model_accuracy_pair = []
for model in kmean_models:
    forest_clf = RandomForestClassifier()
    reduced_train_feats = model.transform(train_feats)
    forest_clf.fit(reduced_train_feats, train_labels)
    accuracy = accuracy_score(
        val_labels, forest_clf.predict(model.transform(val_feats))
    )

    reduced_model_accuracy_pair.append((model, accuracy))


accuracies = [x[1] for x in reduced_model_accuracy_pair]

best_reduced_model, best_reduced_accuracy = reduced_model_accuracy_pair[
    np.argmax(accuracies)
]


enhanced_model_accuracy_pair = []
for model in kmean_models:
    forest_clf = RandomForestClassifier()

    reduced_train_feats = model.transform(train_feats)
    enhanced_train_feats = np.concatenate([train_feats, reduced_train_feats], axis=1)

    reduced_val_feats = model.transform(val_feats)
    enhanced_val_feats = np.concatenate([val_feats, reduced_val_feats], axis=1)

    forest_clf.fit(enhanced_train_feats, train_labels)
    accuracy = accuracy_score(val_labels, forest_clf.predict(enhanced_val_feats))

    enhanced_model_accuracy_pair.append((model, accuracy))

accuracies = [x[1] for x in enhanced_model_accuracy_pair]
best_enhanced_model, best_enhanced_accuracy = enhanced_model_accuracy_pair[
    np.argmax(accuracies)
]


"""
12. Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the
algorithm, you should probably reduce the dataset's dimensionality (e.g., use
PCA, preserving 99% of the variance). Use the model to generate some new
faces (using the sample() method), and visualize them (if you used PCA, you
will need to use its inverse_transform() method). Try to modify some images
(e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e.,
compare the output of the score_samples() method for normal images and for
anomalies).
"""
from copy import deepcopy

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=40)
y_pred = gm.fit_predict(train_feats)

sample_feats, sample_labels = gm.sample(40)


plot_faces(sample_feats, sample_labels)


type(sample_feats[0][0])
type(train_feats[0][0])

best_model.predict(sample_feats.astype(np.float32))

change_indexes = np.random.choice(range(sample_feats.shape[0]), 9, replace=False)

num_divide = 3

rotate_index, dim_index, flip_index = [
    change_indexes[num_divide * i : num_divide * i + num_divide]
    for i in range(num_divide)
]


rotated = np.array(
    [x.T for x in sample_feats[rotate_index].reshape(-1, 64, 64)]
).reshape(-1, 64 * 64)
dimmed = sample_feats[dim_index]
dimmed[:, 1:-1] *= 0.2
flipped = np.flip(sample_feats[flip_index].reshape(-1, 64, 64)).reshape(-1, 64 * 64)

tampered_sample_feats = deepcopy(sample_feats)

tampered_sample_feats[rotate_index] = rotated
tampered_sample_feats[dim_index] = dimmed
tampered_sample_feats[flip_index] = flipped

plot_faces(tampered_sample_feats[change_indexes], sample_labels[change_indexes])

tampered_feats = gm.score_samples(tampered_sample_feats)

tampered_feats[dim_index].mean()
tampered_feats[flip_index].mean()
tampered_feats[rotate_index].mean()
gm.score_samples(sample_feats).mean()
gm.score_samples(train_feats).mean()


"""
13. Some dimensionality reduction techniques can also be used for anomaly detection. 
For example, take the Olivetti faces dataset and reduce it with PCA, preserving 
99% of the variance. Then compute the reconstruction error for each image.
Next, take some of the modified images you built in the previous exercise and
look at their reconstruction error: notice how much larger it is. If you plot a
reconstructed image, you will see why: it tries to reconstruct a normal face.
"""
from sklearn.decomposition import PCA

pca = PCA(0.99)
pca_feats = pca.fit_transform(train_feats)

tampered_pca_feats = pca.transform(tampered_sample_feats[change_indexes])

reconstructed_feats = pca.inverse_transform(pca_feats)
reconstructed_tampered_feats = pca.inverse_transform(tampered_pca_feats)

np.square(reconstructed_feats - train_feats).mean()
np.square(reconstructed_tampered_feats - train_feats[change_indexes]).mean()

plot_faces(
    np.concatenate([reconstructed_tampered_feats, reconstructed_feats[change_indexes]]),
    np.concatenate([train_labels[change_indexes], train_labels[change_indexes]]),
)
