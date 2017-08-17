import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
  expArray = np.exp(L)
  expSum = np.sum(expArray)
  result = []
  for expI in expArray:
    result.append(expI / expSum)
  return result