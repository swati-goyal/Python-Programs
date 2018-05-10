import numpy as np
from matplotlib import pyplot as plt


def func(scores):
    y = 0.5 * scores + 0.2
    return scores / np.sum(np.exp(y))

scores = np.array([3.0, 1.0, 0.2])

print(func(scores).T)
plt.bar([0,1,2], func(scores).T, width=0.5, alpha=0.8)
plt.show()