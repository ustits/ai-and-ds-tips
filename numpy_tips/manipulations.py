import numpy as np

a = np.random.random((20, 20))

part1, part2 = np.split(a, 2)
print(len(part1), len(part2))

"""
throws exception if can't split to equals parts:
  part1, part2 = np.split(a, 3)
  
for that cases array_split can be used
"""

part1, part2, part3 = np.array_split(a, 3)
print(len(part1), len(part2), len(part3))
