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

random_ints = np.random.randint(10, size=12)
print(random_ints)

int_range = np.arange(3)
float_range = np.arange(3.0)
print(int_range, float_range)
int_with_step = np.arange(100, 200, step=42)
print(int_with_step)
