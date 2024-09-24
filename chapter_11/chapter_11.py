import math
from functools import partial

import numpy as np
import tensorflow as tf

"""
Training Deep Neural Networks
"""
"""
The Vanishing/Exploding Gradients Problems
"""
# Codeless section


"""
Glorot and He Initialization
"""
dense = tf.keras.layers.Dense(50, activation="relu", kernel_initializer="he_normal")

he_avg_init = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_avg", distribution="uniform"
)

dense = tf.keras.layers.Dense(50, activation="sigmoid", kernel_initializer=he_avg_init)


"""
Better Activation Functions
"""
"""
Leaky ReLU
"""
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)  # defaults to alpha=0.3
dense = tf.keras.layers.Dense(50, activation=leaky_relu, kernel_initializer="he_normal")


model = tf.keras.models.Sequential(
    [
        [...],  # more layers
        tf.keras.layers.Dense(50, kernel_initializer="he_normal"),  # no activation
        tf.keras.layers.LeakyReLU(alpha=0.2),  # activation as a separate layer
        [...],  # more layers
    ]
)

"""
ELU and SELU
"""
# Codeless section

"""
GELU, Swish, and Mish
"""
# Codeless section


"""
Batch Normalization
"""
# Codelesse ction

"""
Implementing batch normalization with Keras
"""
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


# BN before activation
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


"""
Gradient Clipping
"""
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile([...], optimizer=optimizer)


"""
Reusing Pretrained Layers
"""
"""
Transfer Learning with Keras
"""
[...]  # Assuming model A was already trained and saved to "my_model_A"
model_A = tf.keras.models.load_model("my_model_A")
model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Cloning just clones the architecture, so you need to set the weights to actualy clone the model A performance
model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())


for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)  # Always compile after freezing or unfreezing the model


# Setting this values to stop linting
(X_train_B, y_train_B, X_valid_B, y_valid_B, X_test_B, y_test_B) = None

history = model_B_on_A.fit(
    X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B)
)
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
history = model_B_on_A.fit(
    X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B)
)

model_B_on_A.evaluate(X_test_B, y_test_B)
# On small dense networks this shouldn't work very well, but in large ones, it might help alot.


"""
Pretraining on an Auxiliary Task
"""
# Codeless section


"""
Faster optimizers
"""
"""
Momentum
"""
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)


"""
Nesterov Accelerated Gradient
"""
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)


"""
AdaGrad
"""
# Codeless section


"""
RMSProp
"""
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)


"""
Adam
"""
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)


"""
AdaMax
"""
# Codeless section


"""
Nadam
"""
# Codeless section


"""
AdamW
"""
# Codeless section


"""
Learning Rate Scheduling
"""
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)


def exponential_decay_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

X_train, y_train = None, None  # Set only for linting

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])


def exponential_decay_fn(epoch, lr):
    return lr * 0.1 ** (1 / 20)


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])


batch_size = 32
n_epochs = 25
n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1
)
optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)


"""
Avoiding Overfittint Through Regularization
"""

"""
l1 and l2 Regularization
"""
layer = tf.keras.layers.Dense(
    100,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
)

RegularizedDense = partial(
    tf.keras.layers.Dense,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        RegularizedDense(100),
        RegularizedDense(100),
        RegularizedDense(10, activation="softmax"),
    ]
)

# ℓ2 regularization is fine when using SGD, momentum optimization,
# and Nesterov momentum optimization, but not with Adam and its
# variants. If you want to use Adam with weight decay, then do not
# use ℓ2 regularization: use AdamW instead.


"""
Dropout
"""
# Dropout is so coooooool!
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
[...]  # compile and train the model


"""
Monte Carlo (MC) Dropout
"""
# MC Dropout is so cooooooool as well!
X_test = ...  # for linting


# This is the whole implementation :O
y_probas = np.stack([model(X_test, training=True) for sample in range(100)])
y_proba = y_probas.mean(axis=0)


# Use this instead of regular dropout in the architecture if you're already using regular dropout
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)


"""
Max-Norm Regularization
"""

dense = tf.keras.layers.Dense(
    100,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_constraint=tf.keras.constraints.max_norm(1.0),
)
