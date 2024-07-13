import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import (
    GaussianRandomProjection,
    johnson_lindenstrauss_min_dim,
)

"""
Dimensionality Reduction
"""

"""
The Curse of Dimensionality
"""
# Codeless section

"""
Main Approaches for Dimensionality Reduction
"""
"""
Projection
"""
# Codeless section


"""
Manifold Learning
"""
# Codeless section


"""
PCA
"""
"""
Preserving the Variance
"""
# Codeles section


"""
Principal Components
"""
a = 1
b = 2
c = 3

# Generate random x and y values
num_points = 100
x = np.random.uniform(-10, 10, num_points)
y = np.random.uniform(-10, 10, num_points)
z = a * x + b * y + c

X = np.array([x, y, z])
X_centered = X - X.mean(axis=0)  # Very important
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]


"""
Projecting Down to d Dimensions
"""
# Projecting down to 2 dimensions
W2 = Vt[:2].T
X2D = X_centered @ W2


"""
Using Scikit-learn
"""
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)


"""
Explained Variance Ratio
"""
pca.explained_variance_ratio_.sum()


"""
Choosing the Right Number of Dimensions
"""
mnist = fetch_openml("mnist_784", as_frame=False)

X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1  # d equals 154

pca = PCA(n_components=0.95)  # Already sets 95% of variance
X_reduced = pca.fit_transform(X_train)

# n_components can be used as a normal hyperparameter

clf = make_pipeline(PCA(random_state=42), RandomForestClassifier(random_state=42))
param_distrib = {
    "pca__n_components": np.arange(10, 80),
    "randomforestclassifier__n_estimators": np.arange(50, 500),
}
rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3, random_state=42)
rnd_search.fit(X_train[:1000], y_train[:1000])
rnd_search.best_params_


"""
PCA for Compression
"""
# Codeless section


"""
Randomized PCA
"""
rnd_pca = PCA(n_components=154, svd_solver="full", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)

rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)


"""
Incremental PCA
"""
n_batches = 100

inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype="float32", mode="write", shape=X_train.shape)
X_mmap[:] = X_train  # could be a loop instead, saving the data chunk by chunk
X_mmap.flush()

X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)


"""
Random Projection
"""
m, ε = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=ε)

n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d)  # std dev = square root of variance
X = np.random.randn(m, n)  # generate a fake dataset
X_reduced = X @ P.T

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)  # same result as above

# Recomposing the matrix
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered = X_reduced @ components_pinv.T


"""
Locally Linear Embedding (LLE)
"""
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)


"""
Other Dimensionality Reduction Techniques
"""
# sklearn.manifold.MDS => if you have low dimensional data use this instead of random projection
# sklearn.manifold.Isomap
# sklearn.manifold.TSNE => good for visualization of clusters in high dimensions
# sklearn.discriminant_analysis.LinearDiscriminantAnalysis => its a classification algorithm that
#### can be used as preprocessing for another classification algorithm
