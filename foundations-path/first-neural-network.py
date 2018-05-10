import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import *
from mpl_toolkits.mplot3d import Axes3D


# New complete class, with changes:
class NeuralNetwork(object):
    def __init__(self, Lambda=0):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # Regularization Parameter:
        self.Lambda = Lambda

    def forward_propagation(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def cost_function(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward_propagation(X)
        J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (
        np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return J

    def cost_function_prime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward_propagation(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoid_gradient(self.z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_gradient(self.z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda * self.W1

        return dJdW1, dJdW2

    # Helper functions for interacting with other methods/classes
    def get_params(self):
        # Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        # Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],
                             (self.hiddenLayerSize, self.outputLayerSize))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Trainer(object):
    def __init__(self, N):
        self.N = N

    def call_back_fn(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))
        self.testJ.append(self.N.cost_function(self.testX, self.testY))
        self.W.append(params)

    def cost_function_wrapper(self, params, X, y):
        self.N.set_params(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)

        return cost, grad

    def train(self, trainX, trainY, testX, testY):
        self.X = trainX
        self.y = trainY
        self.testX = testX
        self.testY = testY

        self.J = []
        self.testJ = []
        self.W = []

        params0 = self.N.get_params()

        options = {'maxiter': 200, 'disp': True }

        _res = minimize(self.cost_function_wrapper, params0, jac=True, method='BFGS', args=(trainX, trainY),
                        options=options, callback=self.call_back_fn)

        self.N.set_params(_res.x)
        self.optimization_results = _res


def compute_gradients_check(N, X, y):
    params_initial = N.get_params()
    chkgrad = np.zeros(params_initial.shape)
    perturb = np.zeros(params_initial.shape)
    e = 1e-4

    for p in range(len(params_initial)):
        perturb[p] = e
        N.set_params(params_initial + perturb)
        loss2 = N.cost_function(X, y)

        N.set_params(params_initial - perturb)
        loss1 = N.cost_function(X, y)

        chkgrad[p] = (loss2 - loss1) / (2*e)

        perturb[p] = 0

    N.set_params(params_initial)

    return chkgrad


NN = NeuralNetwork(Lambda=.0000001)
trainX = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
trainY = np.array(([75], [82], [93], [70]), dtype=float)

# Testing data:
testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

# Normalize:
trainX /= np.amax(trainX, axis=0)
trainY /= 100

# Normalize by max of training data:
testX /= np.amax(trainX, axis=0)
testY /= 100

# Train the model
T = Trainer(NN)
T.train(trainX, trainY, testX, testY)

plt.plot(T.W)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Weights (W)')
plt.show()



'''
# Plot a trained model after reducing the over-fitting and applying regularization
plt.plot(T.J)
plt.plot(T.testJ)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost(J)')
plt.legend(['J', 'Test J'])
plt.show()
'''

'''
# Training data:
trainX = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
trainY = np.array(([75], [82], [93], [70]), dtype=float)

# Testing data:
testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

# Normalize:
trainX /= np.amax(trainX, axis=0)
trainY /= 100

# Normalize by max of training data:
testX /= np.amax(trainX, axis=0)
testY /= 100

NN = NeuralNetwork()
T = Trainer(NN)
T.train(trainX, trainY, testX, testY)

plt.plot(T.J)
plt.plot(T.testJ)
plt.xlabel('Iterations')
plt.ylabel('Cost(J)')
plt.legend(['J', 'testJ'])
plt.show()
'''

'''
NN = NeuralNetwork()
X = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
y = np.array(([75], [82], [93], [70]), dtype=float)

X /= np.amax(X, axis=0)
y /= 100

T = Trainer(NN)
T.train(X, y)

# Test Code
hours_sleep = np.linspace(0, 10, 100)
hours_study = np.linspace(0, 5, 100)

hours_sleep_norm = hours_sleep/10
hours_study_norm = hours_study/5

a, b = np.meshgrid(hours_sleep_norm, hours_study_norm)

all_inputs = np.zeros((a.size, 2))
all_inputs[:, 0] = a.ravel()
all_inputs[:, 1] = b.ravel()

all_outputs = NN.forward_propogation(all_inputs)

# Plot the test output
yy = np.dot(hours_study.reshape(100, 1), np.ones((1, 100)))
xx = np.dot(hours_sleep.reshape(100, 1), np.ones((1, 100))).T

'''

'''
# 2D Plot
cs = plt.contour(xx, yy, 100*all_outputs.reshape(100, 100))
plt.clabel(cs, inline=1, fontsize=10)
plt.xlabel('Hours Sleep')
plt.ylabel('Hours Study')
plt.show()
'''

'''
# Test Code and 3D plot after overfitting regularization
hours_sleep = np.linspace(0, 10, 100)
hours_study = np.linspace(0, 5, 100)

hours_sleep_norm = hours_sleep/10
hours_study_norm = hours_study/5

a, b = np.meshgrid(hours_sleep_norm, hours_study_norm)

all_inputs = np.zeros((a.size, 2))
all_inputs[:, 0] = a.ravel()
all_inputs[:, 1] = b.ravel()

all_outputs = NN.forward_propagation(all_inputs)

# Plot the test output
yy = np.dot(hours_study.reshape(100, 1), np.ones((1, 100)))
xx = np.dot(hours_sleep.reshape(100, 1), np.ones((1, 100))).T

# 3D Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(10*trainX[:, 0], 5*trainX[:, 1], 100*trainY, c='k', alpha=1, s=30)

surf = ax.plot_surface(xx, yy, 100*all_outputs.reshape(100, 100), alpha=0.5)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')
plt.show()
'''

'''
# Plot cost function before training
plt.plot(T.J)
plt.xlabel('Iterations')
plt.ylabel('Cost (J)')
plt.show()
'''

'''
# Compare gradients
X = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
y = np.array(([75], [82], [93], [70]), dtype=float)

X /= np.amax(X, axis=0)
y /= 100

# Compare gradients
chk_grad = compute_gradients_check(NN, X, y)
grad = NN.compute_gradients(X, y)

print(pb.norm(grad-chk_grad)/pb.norm(grad+chk_grad))
'''

'''
# Test Code for Gradient
f = lambda x: x**2

delta = 1e-4
x = 1.5
numerical_grad = (f(x+delta)-f(x-delta)) / (2 * delta)

print(numerical_grad, 2*x)
'''

'''
cost1 = NN.cost_function(X, y)

dJdW1_dummy, dJdW2_dummy = NN.cost_function_prime(X, y)
learning_rate = 3
NN.W1 = NN.W1 + learning_rate * dJdW1_dummy
NN.W2 = NN.W2 + learning_rate * dJdW2_dummy
cost2 = NN.cost_function(X, y)

NN.W1 = NN.W1 - learning_rate * dJdW1_dummy
NN.W2 = NN.W2 - learning_rate * dJdW2_dummy
cost3 = NN.cost_function(X, y)

print("Cost 1: ", cost1)
print("Cost 2: ", cost2)
print("Cost 3: ", cost3)

# print(dJdW1_dummy)
# print(dJdW2_dummy)
'''

'''
sig_test_values = np.arange(-5, 5, 0.01)
plt.plot(sig_test_values, NN.sigmoid(sig_test_values), linewidth=2)
plt.plot(sig_test_values, NN.sigmoid_gradient(sig_test_values), linewidth=2)
plt.legend(['f', "f'"])
plt.show()
'''

'''
weight_to_try = np.linspace(-5, 5, 1000)
costs = np.zeros(1000)
start_time = time.clock()
for i in range(1000):
    NN.W1[0, 0] = weight_to_try[i]
    y_hat = NN.forward_propogation(X)
    costs[i] = 0.5 * np.sum((y-y_hat)**2)
end_time = time.clock()

elapsed_time = end_time - start_time
print(elapsed_time)
'''

'''
plt.plot(weight_to_try, costs)
plt.ylabel('Cost')
plt.xlabel('Weight')
plt.show()
'''

'''
bar_width = 0.35
for i in range(len(y)):
    plt.bar(i, y[i], width=0.35, color='b', alpha=0.8)
    plt.bar(i+bar_width, y_hat[0][i], width=0.35, color='r', alpha=0.8)
    plt.legend(['y', 'y_hat'])
plt.show()
'''

'''
# To find solution to overfitting problem
A simple rule of thumb is that we should have at least 10 times as many examples as the degrees for freedom in our
model. For us, since we have 9 weights that can change, we would need 90 observations, which we certainly don't have.
'''