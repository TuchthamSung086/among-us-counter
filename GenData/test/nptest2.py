import numpy as np


# x = np.array([1, 2, 4, 7, 1, 4, 6, 8, 1, 8, 2, 5])

# y = np.unique(x, return_counts=True)

# print(y)

x = np.array([[0, 0, 0],
              [1, 1, 1],
              [2, 2, 2]])

y = np.array([[0, 3, 0],
              [1, 1, 1],
              [2, 5, 2]])

z = np.unique(x, return_counts=True)
print(z)
