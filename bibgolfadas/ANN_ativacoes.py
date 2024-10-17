import numpy as np

# Common activation functions for conventional neural networks
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Common activation functions for LSTMs
def hard_sigmoid(x):
    return np.clip((x + 1) / 2, 0, 1)

def softplus(x):
    return np.log(1 + np.exp(x))

def swish(x):
    return x * sigmoid(x)
