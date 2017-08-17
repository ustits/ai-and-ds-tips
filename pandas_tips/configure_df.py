import numpy as np
import pandas as pd

df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=range(0, 3), columns=['A', 'B', 'C'])
print(df)

df.set_index('C')


