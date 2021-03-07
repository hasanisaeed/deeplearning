import pandas as pd

df = pd.read_csv('../../data/test.csv')

df = df.iloc[:, 0]
df.to_csv('../weights/class.csv', mode='w', header=True)

print('>> Done!')