import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go


YEAR = 252
def calc_rolling_vol(df:pd.DataFrame, period_days:int=6*30, calc:str='std'):
    """
    df: Price pd.DataFrame. Index must be datetime series
    period_days: Number of days for calculation. Default 6*30
    calc: What calculation to perform. Default 'std'
    scale_by_price: Whether to calculate the percentagewise calculation (divided by price)
    """
    df_rets = df.pct_change()
    df_vol = df_rets.rolling(window=f'{period_days}D').agg(calc)
    df_vol = df_vol[df_vol.index >= df_vol.index[0] + timedelta(days=period_days)]
    df_vol = df_vol.dropna()
    return df_vol

df_histvol = pd.read_parquet(f'MSCI_Data_Factset/daily_prices.parquet')
df_histvol = df_histvol.rename(columns={
    'date':'Date',
    'id':'Id',
    'price':'p_price',
    'symbol':'stock',
})
df_histvol['Date'] = pd.to_datetime(df_histvol['Date'], format='%Y-%m-%d')
df_histvol = df_histvol[df_histvol['Date']>='2003-01-01']
df_histvol = df_histvol[~df_histvol['p_price'].isna()]
df_histvol['Id'] = df_histvol['stock'].str.replace('-', '_')
df_histvol = df_histvol.pivot(index='Date', columns='Id', values='p_price')
df_histvol = df_histvol.sort_index()
df_histvol = df_histvol.dropna(axis=1)
df_histvol = calc_rolling_vol(df_histvol, period_days=YEAR//12, calc='std')
df_histvol = df_histvol*100*((1/(YEAR//12))**.5)

df_sp = pd.read_excel('MSCI_Data_Factset/Price_MSCI_Emerging_2024.xlsx',skiprows=2)
df_sp = df_sp[['Date','Price']]
df_sp = df_sp.sort_values(by='Date',ascending=True)
df_sp = df_sp.rename(columns={
    'Price':'price',
})
df_sp.index = df_sp['Date']
df_sp = df_sp.drop('Date',axis=1)

df_sp_vol = calc_rolling_vol(df_sp, period_days=YEAR//12)
df_sp_vol = df_sp_vol[df_sp_vol.index.isin(set(df_histvol.index))]*100*((1/(YEAR//12))**.5)
df_hist_minus_sp = pd.DataFrame(df_histvol)
for c in df_hist_minus_sp.columns:
    df_hist_minus_sp[c] = df_hist_minus_sp[c] - df_sp_vol['price']
df_hist_minus_sp = df_hist_minus_sp.dropna()
df_hist_minus_sp.to_csv('1mo_rolling_hist_stock_minus_msci_vol.csv')