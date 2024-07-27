import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

"""
Unsupervised Learning Techniques
"""


"""
Clustering Algorithms: k-means and DBSCAN
"""
"""
K-means
"""
blob_centers = np.array(
    [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
)
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(
    n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=42
)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)


"""
The k-means algorithm
"""
# Codeless section


"""
Centroid initialization methods
"""
# Method 1: clairvoyance. Just know some good inits.
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
kmeans.inertia_

# Method 2: random selection + prayer. Do it many times, get the lowest inertia

# Method 3: kmeans++ (smarter random selecion + less prayer (still pray tho))

"""
Accelerated k-means and mini-batch k-means
"""
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)


"""
Finding the optimal number of clusters
"""
# Codeless section


"""
Limits of k-means
"""
# Codeless section


"""
Using Clustering for Image Segmentation
"""
homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
filename = "ladybug.png"
filepath = Path() / filename
if not filepath.is_file():
    print("Downloading", filename)
    url = f"{homl3_root}/images/unsupervised_learning/{filename}"
    urllib.request.urlretrieve(url, filepath)

image = np.asarray(PIL.Image.open(filepath))

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)


"""
Using Clustering for Semi-Supervised Learning
"""
X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]


# If we only have 50 labeled images and try to run a model logistic
# regression, we'd have some trouble...
n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)

log_reg_full = LogisticRegression(max_iter=10_000)
log_reg_full.fit(X_train, y_train)
log_reg_full.score(X_test, y_test)

# Let's try improving our model with clustering
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

"""
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(
        X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear"
    )
    plt.axis("off")


plt.show()
"""

""" Commented so black formating does not make this long
y_representative_digits = np.array(
    [8, 0, 1, 3, 6, 7, 5, 4, 2, 8, 2, 3, 9, 5, 3, 9, 1, 7, 9, 1, 4, 6, 9, 7, 5, 2, 2, 1, 3, 3, 6, 0, 4, 9, 8, 1, 8, 4, 2, 4, 2, 3, 9, 7, 8, 9, 6, 5, 6, 4]
)
"""

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
# The score of training on the 50 most 'heavy' digits is way better!

y_train_propagated = np.empty(len(X_train), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
# Got even better if we use this representative digits as the label of all
# the other observations in the same group

percentile_closest = 99
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = kmeans.labels_ == i
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = X_cluster_dist > cutoff_distance
    X_cluster_dist[in_cluster & above_cutoff] = -1
partially_propagated = X_cluster_dist != -1
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)
# Excluding the outliers did not work for my example, but it might help
# Since I had to write the numbers manually, I will not try again xD


"""
DBSCAN
"""
X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])


X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)
knn.predict_proba(X_new)

y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()


"""
Other Clustering Algorithms
"""
# Codeless section

# Examples:

# # Agglomerative clustering
# BIRCH (good for huge datasets with few (<20) features)
# Mean-shift
# Affinity propagation
# Spectral Clustering

"""
Gaussian Mixtures
"""
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

gm.weights_
gm.means_
gm.covariances_

gm.converged_
gm.n_iter_

gm.predict(X)
gm.predict_proba(X).round(3)

X_new, y_new = gm.sample(6)

gm.score_samples(X).round(2)


"""
Using Gaussian Mixtures for Anomaly Detection
"""
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]


"""
Selecting the Number of Clusters
"""
gm.bic(X)
gm.aic(X)


"""
Bayesian Gaussian Mixture Models
"""
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
bgm.weights_.round(2)


"""
Other Algorithms for Anomaly and Novelty Detection
"""
# Codeless section
#
# Algorithms that can deal with arbitrarily shaped clusters
#
# Fast-MCD (Minimum Covariance Determinant)
# (good for finding outliers)
#
# Isolation Forest
# (also good for finding outliers, especially in high dimensional datasets)
#
# Local Outlier Factor
# (guess what it is good for)
#
# One-class SVM
# (good for novelty detection)
#
# PCA and other dimensionality reduction techniques with an inverse_transform() method
