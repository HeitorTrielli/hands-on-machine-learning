"""
1. What are the main motivations for reducing a dataset's dimensionality? What are
the main drawbacks?
"""

# The main reasons for reducing the dimensionality is to 1. reduce the time spent on
# training, and 2. reduce the chance of overfitting
# The main drawback of dimensionality reduction is losing precision. When you reduce
# the dimensionality, you lose some information. This information might be important
# for the model to predict accurately


# PS (from the solutions):
# more reasons => to better vizualise the data, to save disk/memory space
# more drawbacks => might be computationaly expensive, adds extra complexity to ML
# pipelines, transformed features are harder to interpret

"""
2. What is the curse of dimensionality?
"""
# The curse of dimensionality is when you have too many features available, such that
# the number of features surpasses the number of observations, making it hard to create
# a generizable model

"""
3. Once a dataset's dimensionality has been reduced, is it possible to reverse the
operation? If so, how? If not, why?
"""
# It is, although some information will be lost. Depending on how you do the dimensionality
# reduction, you can do the inverse steps to get back the original data. The information
# that you threw away when reducing the dimensionality is lost, though.


"""
4. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
"""
# Yes, you can use PCA to reduce dimensionality of anything. If it is a good idea, thats
# another question xD. If it is highly nonlinear, there may be some better methods, such
# as LLE


"""
5. Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained
variance ratio to 95%. How many dimensions will the resulting dataset have?
"""
# It only depends on how the variation is distributed in this 1000-dimensional dataset.
# if it is all on a single axis, it would be only one dimension. This is an extreme
# scenario and there probably will be more than one dimension

# PS (from the solutions): it shouldn't pass 950 dimensions though4

"""
6. In what cases would you use regular PCA, incremental PCA, randomized PCA,
or random projection?
"""
# Incremental PCA => when you need to do the dimensionality reduciton in batches to fit
# the memory
# Randomized PCA => if you're ok with approximizing the number of dimmensions needed
# for the PCA to get a speed boost
# Normal PCA => when you don't need to run in batches and is not willing to approximize the
# number of dimensions needed
# Random Projection => when you want to reduce the dimensionality but keep the distances
# between each observation fairly preserved. It works when the data is not linear and also
# is good for vizualising the data

"""
7. How can you evaluate the performance of a dimensionality reduction algorithm
on your dataset?
"""
# You can check how much of the original variance of the dataset is preserved. The more
# variance you can squeeze per dimension left, the better, since you're throwing away
# multiple dimensions, making the problem easier to train, without throwing away much
# information


"""
8. Does it make any sense to chain two different dimensionality reduction
algorithms?
"""
# You might. For example, if you can unroll a swiss roll from a very large dimensional
# dataset to a still large dimensional dataset , you might still want to reduce further
# the dimensionality with another algorithm.


"""
9. Load the MNIST dataset (introduced in Chapter 3) and split it into a training
set and a test set (take the first 60,000 instances for training, and the remaining
10,000 for testing). Train a random forest classifier on the dataset and time how
long it takes, then evaluate the resulting model on the test set. Next, use PCA
to reduce the dataset's dimensionality, with an explained variance ratio of 95%.
Train a new random forest classifier on the reduced dataset and see how long it
takes. Was training much faster? Next, evaluate the classifier on the test set. How
does it compare to the previous classifier? Try again with an SGDClassifier.
How much does PCA help now?
"""
from time import process_time

from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10_000)

random_forest = RandomForestClassifier()
sgd_clf = SGDClassifier()


def pca_vs_full_clf_train(clf, X_train, y_train):
    clf_full = clone(clf)
    clf_pca = clone(clf)

    t_full = process_time()
    clf_full.fit(X_train, y_train)
    elapsed_full = process_time() - t_full

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)

    t_pca = process_time()
    clf_pca.fit(X_train_pca, y_train)
    elapsed_pca = process_time() - t_pca

    return elapsed_pca, elapsed_full, clf_full, clf_pca


rf_elapsed_pca, rf_elapsed_full, rf_full, rf_pca = pca_vs_full_clf_train(
    random_forest, X_train, y_train
)
sgd_elapsed_pca, sgd_elapsed_full, sgd_full, sgd_pca = pca_vs_full_clf_train(
    sgd_clf, X_train, y_train
)

rf_prop = rf_elapsed_pca / rf_elapsed_full - 1
sgd_prop = sgd_elapsed_pca / sgd_elapsed_full - 1
# it took ~20% more to do the PCA for RF and took ~32% less to do the PCA for the SGD


pca = PCA(n_components=0.95)
pca.fit(X_train)

print("Default Random Forest score:", rf_full.score(X_test, y_test))
print("PCA Random Forest score:", rf_pca.score(pca.transform(X_test), y_test))
print("Default SGD score:", sgd_full.score(X_test, y_test))
print("PCA SGD score:", sgd_pca.score(pca.transform(X_test), y_test))

# The default RF was better than PCA RF
# The PCA SGD was better than the default SGD
# RF was always better than SGD

"""
10. Use t-SNE to reduce the first 5,000 images of the MNIST dataset down to 2
dimensions and plot the result using Matplotlib. You can use a scatterplot using
10 different colors to represent each image's target class. Alternatively, you can
replace each dot in the scatterplot with the corresponding instance's class (a digit
from 0 to 9), or even plot scaled-down versions of the digit images themselves
(if you plot all digits the visualization will be too cluttered, so you should either
draw a random sample or plot an instance only if no other instance has already
been plotted at a close distance). You should get a nice visualization with 
wellseparated clusters of digits. Try using other dimensionality reduction algorithms,
such as PCA, LLE, or MDS, and compare the resulting visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding

tsne = TSNE()

reduced_X = X_train[:5000]
reduced_y = y_train[:5000]
X_2d_tsne = tsne.fit_transform(reduced_X)

pca_2d = PCA(n_components=2)
X_2d_pca = pca_2d.fit_transform(reduced_X)

lle = LocallyLinearEmbedding()
X_2d_lle = lle.fit_transform(reduced_X)

mds = MDS(n_jobs=4, verbose=1)
X_2d_mds = mds.fit_transform(reduced_X)

cmap = plt.get_cmap("tab10")


def plot_2d_classes(X_2d):
    for class_ in range(np.unique(reduced_y).shape[0]):
        plt.scatter(
            X_2d[reduced_y == str(class_), 0],
            X_2d[reduced_y == str(class_), 1],
            color=cmap(class_),
            label=f"{class_}",
            marker=f"${str(class_)}$",
        )

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Scatter plot with 10 classes")
    plt.show()


plot_2d_classes(X_2d_tsne)
plot_2d_classes(X_2d_pca)
plot_2d_classes(X_2d_lle)
plot_2d_classes(X_2d_mds)
