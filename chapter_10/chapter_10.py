"""
Introduction to Artificial Neural
Networks with Keras
"""

"""
From Biological to Artificial Neurons
"""
# Codeless section

"""
Biological Neurons
"""
# Codeless section


"""
Logical Computations with Neurons
"""
# Codeless section


"""
The Perceptron
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 0  # Iris setosa
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)


"""
The Multilayer Perceptron and Backpropagation
"""
# Codeless section


"""
Regression MLPs
"""
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)


"""
Classification MLPs
"""
# Codeless section


"""
Implementing MLPs with Keras
"""
"""
Building an Image Classifier Using the Sequential API
"""
"""
Using Keras to load the dataset
"""
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train, X_valid, X_test = X_train / 255, X_valid / 255, X_test / 255

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


import tensorflow as tf

print("Is TensorFlow built with CUDA support?", tf.test.is_built_with_cuda())
print("Is TensorFlow built with GPU support?", tf.test.is_built_with_gpu_support())

import os
import sys

sys.path
