import numpy as np

M = np.array([[1, 2],
              [3, 4]])
L = [[1, 2], [3, 4]]

M2 = np.matrix([[1, 2], [3, 4]])

A = np.array(M2)

print(A.T)

# Invertible matrix
Minv = np.linalg.inv(M)
print(Minv)

print(M.dot(Minv))

# determinant
det = np.linalg.det(M)
print(det)

# get the diagonal
diag = np.diag(M)
print(diag)

inner = np.inner([1, 2], [3, 4])
print(inner)
# same as dot product
print(np.dot([1, 2], [3, 4]))

outer = np.outer([1, 2], [3, 4])
print(outer)

# sum of the diagonal
diag_sum = np.diag(M).sum()
print(diag_sum)

diag_sum = np.trace(M)
print(diag_sum)

# eigenvectors and eigenvalues

samples = np.random.rand(100, 3)
print(samples.shape)

cov = np.cov(samples.T)
print(cov)

eig = np.linalg.eig(cov)
print(eig)
eigh = np.linalg.eigh(cov)
print(eigh)
