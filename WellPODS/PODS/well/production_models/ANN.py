import numpy as np

class ANN:

    def __init__(self, layers, learning_rate=0.01, epochs=10000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            net_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self._sigmoid(net_input)
            activations.append(activation)
        return activations

    def _backward_propagation(self, activations, y):
        deltas = [y - activations[-1]]
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self._sigmoid_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()
        return deltas

    def _update_weights(self, activations, deltas):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            activations = self._forward_propagation(X)
            deltas = self._backward_propagation(activations, y)
            self._update_weights(activations, deltas)

    def predict(self, X):
        activations = self._forward_propagation(X)
        return activations[-1]