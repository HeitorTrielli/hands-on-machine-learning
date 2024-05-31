import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from chapter_2.data_handling import load_housing_data

log_transformer = FunctionTransformer(np.log, np.exp, feature_names_out="one-to-one")


num_pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("standardize", StandardScaler())]
)


log_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_transformer),
        ("standardize", StandardScaler()),
    ]
)

cat_pipelinne = Pipeline(
    [("impute", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder())]
)


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


cluster_similarity = ClusterSimilarity(random_state=27)
