import pandas as pd

df = pd.read_csv('../../data/test.csv')
df = df.drop([df.columns[0]], axis=1)
df.to_csv('../raw_signal.csv')

print('>> Done!')