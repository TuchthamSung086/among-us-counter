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

y = np.array([[[0, 0], [3, 3], [0, 0]],
              [[1, 1], [1, 1], [1, 1]],
              [[2, 2], [5, 5], [2, 2]]])

y = np.array([[[0, 0], [4, 3], [1, 0]],
              [[1, 1], [1, 2], [1, 5]],
              [[2, 4], [5, 2], [2, 2]]])

z = np.array(y == [4])
print(z)

x = np.array([1, 2, 3])
