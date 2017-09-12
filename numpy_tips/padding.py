import numpy as np

a = np.random.randint(10, size=3)

# pad with two zeros on both sides sides
print(np.lib.pad(a, 2, 'constant'))

# pad with two zeros on the left
print(np.lib.pad(a, (2, 0), 'constant'))

# pad with 42 on the right
print(np.lib.pad(a, (0, 3), 'constant', constant_values=42))

# pad with the maximum value
print(np.lib.pad(a, 5, 'maximum'))
