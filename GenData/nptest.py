import numpy as np

x = np.array([[0, 0, 0],
              [1, 1, 1],
              [2, 2, 2]])

y = np.array([[0, 3, 0],
              [1, 1, 1],
              [2, 5, 2]])

z = np.zeros(x.shape, dtype='uint8')
z[x < y] = 99
print(z)
print((x < y))
print((x < y).all())
print((x < y).any())
print(x.sum(axis=1))
