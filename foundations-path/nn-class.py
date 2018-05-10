import numpy as np


class NeuralNetwork(object):
    def __init__(self, Lambda=0):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # regularization hyper parameter
        self.Lambda = Lambda

    def forward_propagation(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot([self.a2], self.W2)
        y_hat = self.sigmoid(self.z3)
        return y_hat

    def sigmoid(self, z):
        # Apply sigmoid function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoid_gradient(self, z):
        # Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def cost_function(self, X, y):
        self.y_hat = self.forward_propagation(X)

        # Error is here
        J = 0.5 * np.sum((y - self.y_hat) ** 2) / X.shape[0] + (self.Lambda/2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return J

    def cost_function_prime(self, X, y):
        self.y_hat = self.forward_propagation(X)
        delta3 = np.multiply(-(y-self.y_hat), self.sigmoid_gradient(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_gradient(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda * self.W1

        return dJdW1, dJdW2

    def get_params(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()), axis=0)
        return params

    def set_params(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()), axis=0)