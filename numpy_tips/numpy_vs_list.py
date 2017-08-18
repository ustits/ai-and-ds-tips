import numpy as np

# creating arrays
L = [1, 2, 3]
A = np.array([1, 2, 3])

# adding elements for simple arrays
L.append(4)
L = L + [5]
print(L)

# operations on arrays
L2 = []
for e in L:
    L2.append(e + e)

print(L2)

print(A + A)
print(A * 3)
print(A ** 2)
print(np.log(A))
print(np.sqrt(A))
print(np.exp(A))


# dot product
a = np.array([1, 2])
b = np.array([2, 1])

dot = 0
for e, f in zip(a, b):
    dot += e * f
print(dot)

print(np.sum(a * b))
print((a * b).sum())
print(np.dot(a, b))
print(a.dot(b))

# "length" of vector
a_len = np.sqrt(a.dot(a))
print(a_len)

a_len = np.linalg.norm(a)
(print(a_len))
# angle between two vectors
cos = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cos)
print(angle)

