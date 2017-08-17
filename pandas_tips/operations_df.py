import numpy as np
import pandas as pd

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))

print(df.shape)
print(len(df))

df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=range(0, 3), columns=['A', 'B', 'C'])
print(df)

print(df.iloc[0])
print(df.loc[:, 'A'])

# Getting DataFrame data
print('-------------')
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2, 'A', 4], columns=[48, 49, 50])

print(df)
# Search by label
print(df.loc[4])
# print(df.loc['A'])

# Search by index
print(df.iloc[0])

# if all labels are integers - will search by label
# otherwise - search by index
print(df.ix[0])

# Changing DataFrame rows
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])

# Can't find index with label 2, so we are changing by index
df.ix[2] = [60, 50, 40]
print(df)

# if finds label - changes data
# otherwise - creates new row
df.loc[2] = [10, 20, 30]
df.loc[2.5] = [60, 50, 40]
print(df)

# add index as a column
df['D'] = df.index
print(df)

# append column
df['E'] = [1, 2, 3, 4]
print(df)

df.loc[:, 'HEY'] = [6, 3, 5, 6]
print(df)

print('---------')

# dropping indexes
kek = df.reset_index()
print(kek)

df = df.reset_index(level=0, drop=True)
print(df)

df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [40, 50, 60], [23, 35, 37]]),
                  index=[2.5, 12.6, 4.8, 4.8, 2.5],
                  columns=[48, 49, 50])

print(df)

df = df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
print(df)

print('------')
# dropping columns and rows
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=range(0, 3), columns=['A', 'B', 'C'])
print(df)


df.drop('A', axis=1, inplace=True)
# inplace parameter allows to make an operation on the current DataFrame
print(df)
df.drop(0, axis=0, inplace=True)
print(df)
# axis = 0 - rows, axis = 1 - columns

df.drop(df.columns[[1]], axis=1, inplace=True)
print(df)