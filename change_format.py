import pandas as pd

df = pd.read_csv('MSCI_Data_Factset/daily_prices.csv', index_col=0, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['id'] = df['id'].astype(str)#.apply(lambda x: x.encode('utf-8'))
df['symbol'] = df['symbol'].astype(str)#.apply(lambda x: x.encode('utf-8'))
print(df.info())
df.to_parquet('MSCI_Data_Factset/daily_prices.parquet')