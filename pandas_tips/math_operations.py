import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=np.arange(0, 6), columns=list('ABCD'))

print(df)
print(df.describe())


print(df['A'].apply(np.mean))

tests = [[1, 2], [3, 4]]

for test in tests:
  print(test[0], test[1])