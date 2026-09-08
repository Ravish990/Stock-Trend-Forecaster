import pandas as pd

df = pd.read_csv('../SP500_Historical_Data.csv')


# we have to clean the data and remove any rows with missing values
df = df.dropna()
print(df.head())