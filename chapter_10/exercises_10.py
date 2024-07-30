"""
1. The TensorFlow playground is a handy neural network simulator built by the
TensorFlow team. In this exercise, you will train several binary classifiers in just
a few clicks, and tweak the model's architecture and its hyperparameters to gain
some intuition on how neural networks work and what their hyperparameters
do. Take some time to explore the following:
a. The patterns learned by a neural net. Try training the default neural network
by clicking the Run button (top left). Notice how it quickly finds a good
solution for the classification task. The neurons in the first hidden layer have
learned simple patterns, while the neurons in the second hidden layer have
learned to combine the simple patterns of the first hidden layer into more
complex patterns. In general, the more layers there are, the more complex the
patterns can be.
b. Activation functions. Try replacing the tanh activation function with a ReLU
activation function, and train the network again. Notice that it finds a solution
even faster, but this time the boundaries are linear. This is due to the shape of
the ReLU function.
c. The risk of local minima. Modify the network architecture to have just one
hidden layer with three neurons. Train it multiple times (to reset the network
weights, click the Reset button next to the Play button). Notice that the training 
time varies a lot, and sometimes it even gets stuck in a local minimum.
d. What happens when neural nets are too small. Remove one neuron to keep
just two. Notice that the neural network is now incapable of finding a good
solution, even if you try multiple times. The model has too few parameters
and systematically underfits the training set.
e. What happens when neural nets are large enough. Set the number of neurons
to eight, and train the network several times. Notice that it is now consistently
fast and never gets stuck. This highlights an important finding in neural
network theory: large neural networks rarely get stuck in local minima, and
even when they do these local optima are often almost as good as the global
optimum. However, they can still get stuck on long plateaus for a long time.
f. The risk of vanishing gradients in deep networks. Select the spiral dataset (the
bottom-right dataset under “DATA”), and change the network architecture to
have four hidden layers with eight neurons each. Notice that training takes
much longer and often gets stuck on plateaus for long periods of time. Also
notice that the neurons in the highest layers (on the right) tend to evolve
faster than the neurons in the lowest layers (on the left). This problem, called
the vanishing gradients problem, can be alleviated with better weight initialization
and other techniques, better optimizers (such as AdaGrad or Adam), or
batch normalization (discussed in Chapter 11).
g. Go further. Take an hour or so to play around with other parameters and
get a feel for what they do, to build an intuitive understanding about neural
networks.
"""

# Done. Didn't write it down here because it was too much trouble to keep alt-tabbing

"""
2. Draw an ANN using the original artificial neurons (like the ones in Figure 10-3)
that computes A ⊕ B (where ⊕ represents the XOR operation). Hint: A ⊕ B =
(A ∧ ¬ B) V (¬ A ∧ B).
"""

"""
3. Why is it generally preferable to use a logistic regression classifier rather than a
classic perceptron (i.e., a single layer of threshold logic units trained using the
perceptron training algorithm)? How can you tweak a perceptron to make it
equivalent to a logistic regression classifier?
"""

"""
4. Why was the sigmoid activation function a key ingredient in training the first
MLPs?
"""

"""
5. Name three popular activation functions. Can you draw them?
"""

"""
6. Suppose you have an MLP composed of one input layer with 10 passthrough
neurons, followed by one hidden layer with 50 artificial neurons, and finally
one output layer with 3 artificial neurons. All artificial neurons use the ReLU
activation function.
a. What is the shape of the input matrix X?
b. ?
c. What are the shapes of the output layer's weight matrix Wo
 and bias vector bo c. ?
d. What is the shape of the network's output matrix Y?
e. Write the equation that computes the network's output matrix Y as a function
of X, Wh
, bh
, Wo
, and bo
.
"""

"""
7. How many neurons do you need in the output layer if you want to classify email
into spam or ham? What activation function should you use in the output layer?
If instead you want to tackle MNIST, how many neurons do you need in the
output layer, and which activation function should you use? What about for
getting your network to predict housing prices, as in Chapter 2?
"""

"""
8. What is backpropagation and how does it work? What is the difference between
backpropagation and reverse-mode autodiff?
"""

"""
9. Can you list all the hyperparameters you can tweak in a basic MLP? If the MLP
overfits the training data, how could you tweak these hyperparameters to try to
solve the problem?
"""

"""
10. Train a deep MLP on the MNIST dataset (you can load it using tf.keras.
datasets.mnist.load_data()). See if you can get over 98% accuracy by manually 
tuning the hyperparameters. Try searching for the optimal learning rate by
using the approach presented in this chapter (i.e., by growing the learning rate
exponentially, plotting the loss, and finding the point where the loss shoots up).
Next, try tuning the hyperparameters using Keras Tuner with all the bells and
whistles—save checkpoints, use early stopping, and plot learning curves using
TensorBoard.
"""
