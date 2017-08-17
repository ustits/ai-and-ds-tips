import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,5), index=np.arange(0, 6), columns=list('ABCDF'))
print(df)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
print(df1)
df1.loc[dates[0]:dates[1],'E'] = 1

print(df1)

filled_df = df1.fillna(value=42)
print(filled_df)

removed_df = df1.dropna(how='any')
print(removed_df)

# verify nulls
print(pd.isnull(df1))