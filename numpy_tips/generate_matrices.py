import numpy as np

Z = np.zeros(10)
print(Z)
Z = np.zeros((10, 10))
print(Z)

O = np.ones(10)
print(O)

R = np.random.random((10, 10))
print(R)

G = np.random.rand(10, 10)
G.mean()
G.var()