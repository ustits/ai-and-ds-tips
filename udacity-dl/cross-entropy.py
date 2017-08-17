import numpy as np


def cross_entropy(Y, P):
  sum = 0
  for y, p in zip(Y, P):
    sum -= y * np.log(p) + (1 - y) * np.log(1 - p)
  return sum


Y = [1, 0, 1, 1]
P = [0.4, 0.6, 0.1, 0.5]

result = cross_entropy(Y, P)
print(result)
