import os
from datetime import datetime

import tensorflow as tf

"""
1. What is the problem that Glorot initialization and He initialization aim to fix?
"""
# Glorot and He initialization aim to fix the vanishing/exploding gradients


# After reading solutions: they do this by making the output standard deviation closer
# to the input standard deviation


"""
2. Is it OK to initialize all the weights to the same value as long as that value is
selected randomly using He initialization?
"""
# No, they need to be initialized randomly


# After reading solutions: if they all are initiated with the same weights, backpropagation
# will not work, since the gradient will not have any variation


"""
3. Is it OK to initialize the bias terms to 0?
"""
# Yes?


# From solutions: yes.

"""
4. In which cases would you want to use each of the activation functions we
discussed in this chapter?
"""
# For solving the vanishing/exploding gradients: Leaky ReLU, RReLU, PReLU, ELU,
# SELU (many conditions for the normalization to work, be careful), GELU, SiLU,
# Swish and Mish (pretty much all of them)

# For other purposes:
# 1. Regularization: RReLU,
# 2. Large Image datasets: PReLU (might overfit on smaller datasets)
# 3. Faster training time: ELU (slower to compute individually, though) and SELU
# 4. Better overall performance: GELU, Swish and Mish
# 5. Low runtime latency: Leaky ReLU for simple tasks, PReLU for complex tasks, but
# ReLU is ok as well, and also RReLU (benefit of some reagularization)


# Extra from solutions: ReLU might be good when you need exactly zero outputs (
# I wouldn't know, its intel from chapter 17 hah).
# For overall large neural nets: GLU, Swish and Mish will give better model, but
# they have a higher computational cost.

# On solutions they also talked about activations on other chapters, so here it goes:
# Sigmoid => estimating probabilities (only for last layer, or for the 'coding layer of
# variational autoencoders, also intel from chapter 17); tanh is useful if you
# need output between -1 and 1 (used only in recurrent nets, apparently). Softplus activation
# is useful to ensure output positivity. Softmax is the same as sigmoid, but for predicting
# multiple mutually exclusive classes, but you knew that ;)

"""
5. What may happen if you set the momentum hyperparameter too close to 1 (e.g.,
0.99999) when using an SGD optimizer?
"""
# With such a high momentum, the optimizer might take a long time to stabilize in the
# optimal value, since it will be overshooting a lot

"""
6. Name three ways you can produce a sparse model.
"""
# Lasso, l1 regularization, custom models (TensorFlow Model Optimization Toolkit)


# From solutions: you could also set to 0 the tiny weights yourself

"""
7. Does dropout slow down training? Does it slow down inference (i.e., making
predictions on new instances)? What about MC dropout?
"""
# Dropout does not slow down wall-time training. It also does not slow-down making
# predictions.

# MC dropout will slow down inference, since it needs to simulate a bunch of results before
# giving the answer


# From solutions: I got a bunch wrong. Dropout >will< slow down training, but it will result
# in a better model overall. It does not affect inference speed though. MC Dropout is the
# same in training, but it will slow down the inference, and it is defined by the number of
# samples you will generate.

"""
8. Practice training a deep neural network on the CIFAR10 image dataset:
a. Build a DNN with 20 hidden layers of 100 neurons each (that's too many, but
it's the point of this exercise). Use He initialization and the Swish activation
function.
b. Using Nadam optimization and early stopping, train the network on the
CIFAR10 dataset. You can load it with tf.keras.datasets.cifar10.load_
data(). The dataset is composed of 60,000 32 x 32-pixel color images (50,000
for training, 10,000 for testing) with 10 classes, so you'll need a softmax
output layer with 10 neurons. Remember to search for the right learning rate
each time you change the model's architecture or hyperparameters.
c. Now try adding batch normalization and compare the learning curves: is it
converging faster than before? Does it produce a better model? How does it
affect training speed?
d. Try replacing batch normalization with SELU, and make the necessary adjustments to 
ensure the network self-normalizes (i.e., standardize the input features, use LeCun 
normal initialization, make sure the DNN contains only asequence of dense layers, etc.).
e. Try regularizing the model with alpha dropout. Then, without retraining your
model, see if you can achieve better accuracy using MC dropout.
f. Retrain your model using 1cycle scheduling and see if it improves training
speed and model accuracy.
"""

# a.
cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()

cifar_train_feats, cifar_train_label = cifar_train
cifar_test_feats, cifar_test_label = cifar_test

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=cifar_train_feats.shape[1:]),
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="silu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


# b.


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(piecewise_constant_fn)


# Tensorboard
def get_run_logdir(base_path="my_logs"):
    return os.path.join(
        base_path, datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")
    )


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Nadam(),
    metrics=["accuracy"],
)

history = model.fit(
    cifar_train_feats,
    cifar_train_label,
    epochs=30,
    validation_data=(cifar_test_feats, cifar_test_label),
    callbacks=[lr_scheduler, tensorboard_cb],
)

import os

log_dir = "my_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
