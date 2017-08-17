import numpy as np
import csv

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
  if t >= 0:
    return 1
  return 0


def prediction(X, W, b):
  return stepFunction((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
  # Fill in code
  for row in X:
    result = prediction(X, W, b)
    if result != row[2]:
      if result == 0:
        W[0] += learn_rate * row[0]
        W[1] += learn_rate * row[1]
        b += learn_rate
      else:
        W[0] -= learn_rate * row[0]
        W[1] -= learn_rate * row[1]
        b -= learn_rate
  return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
  x_min, x_max = min(X.T[0]), max(X.T[0])
  y_min, y_max = min(X.T[1]), max(X.T[1])
  W = np.array(np.random.rand(2, 1))
  b = np.random.rand(1)[0] + x_max
  # These are the solution lines that get plotted below.
  boundary_lines = []
  for i in range(num_epochs):
    # In each epoch, we apply the perceptron step.
    W, b = perceptronStep(X, y, W, b, learn_rate)
    boundary_lines.append((-W[0] / W[1], -b / W[1]))
  return boundary_lines


tmp_array = []

with open('test.csv', 'rt') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for row in spamreader:
    tmp_array.append(row)


X = np.array(tmp_array)
print(X.T)