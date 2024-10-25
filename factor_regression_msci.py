import pandas as pd
import numpy as np
import statsmodels.api as sm
# import getFamaFrenchFactors as gff
# import pandas_datareader
# from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import warnings


import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

df_original = pd.read_parquet('MSCI_Data_Factset/daily_prices.parquet')
df_original = df_original.rename(columns={
    'date':'Date',
    'id':'Id',
    'price':'p_price',
})
df_original.head()

# Define the window size and step size of the pairs construction
YEAR = 252 # a month has approximately 252 trading days (252 samples)
window_size = 3 * YEAR  # 3 years - 756 samples
step_size = YEAR // 2  # 6 months - 126 samples 
df_original['Date'] = pd.to_datetime(df_original['Date'], format='%Y-%m-%d') # Convert column date to datetime
df_original = df_original[df_original['Date']>='2005-09-01']    # filter the df so it only mantains the desired dates (we have to take one day prior to the first to do the estimations)
df_original = df_original[~df_original['p_price'].isna()]  # Eliminate rows that have no stock prices
df_original['Id'] = df_original['symbol'].str.replace('-', '_') # Change name of the columns to avoid calculation issues
series01_orig = df_original.pivot(index='Date', columns='Id', values='p_price')
series01_orig = series01_orig.sort_index()
series01_orig = series01_orig.dropna(axis=1)
df_orig = series01_orig

df_rets = df_orig.pct_change()
df_rets = df_rets.dropna()

factors = pd.read_csv('factors_msci.csv', index_col=0, parse_dates=True)

risk_free = pd.read_csv('Pairs_SP500_FPT/DTB6.csv', index_col=0, parse_dates=True)
risk_free = risk_free.rename(columns={
    'DTB6': 'rf'
})
risk_free['rf'] = pd.to_numeric(risk_free['rf'], errors='coerce')
risk_free = risk_free.dropna()
risk_free['rf'] = risk_free['rf']/100/365
risk_free.head()

market_spread_over_sp = pd.read_csv('1mo_rolling_hist_stock_minus_msci_vol.csv', index_col=0, parse_dates=True)
market_spread_over_sp /= 100
melted_market_spread_over_sp = market_spread_over_sp.reset_index().melt(id_vars='Date', var_name='stock', value_name='spread_over_sp')
melted_market_spread_over_sp=melted_market_spread_over_sp.rename(columns={'Date':'date'})

df_exc_rets = df_rets.merge(risk_free, left_index=True, right_index=True, how='inner')
for c in df_exc_rets.columns:
    df_exc_rets[c] = df_exc_rets[c]-df_exc_rets['rf']
df_exc_rets = df_exc_rets.drop('rf',axis=1)
df_exc_rets = df_exc_rets.reset_index().rename(columns={'index':'date'})
df_exc_rets = pd.melt(df_exc_rets, id_vars='date',value_vars=df_exc_rets.columns[1:],var_name='stock').rename(columns={'value':'exc_rets'})

factors['mkt-rf'] = factors['mkt-rf'].astype(np.float32)
factors['smb'] = factors['smb'].astype(np.float32)
factors['hml'] = factors['hml'].astype(np.float32)
factors['mom'] = factors['mom'].astype(np.float32)
factors = factors.reset_index().rename(columns={'index':'date'})
df_exc_rets['exc_rets'] = df_exc_rets['exc_rets'].astype(np.float32)
melted_market_spread_over_sp['spread_over_sp'] = melted_market_spread_over_sp['spread_over_sp'].astype(np.float32)

full_dataset = df_exc_rets.merge(melted_market_spread_over_sp,on=['date','stock'], how='inner')
full_dataset = full_dataset.merge(factors, on='date', how='left')
full_dataset = full_dataset.dropna()

full_dataset = full_dataset.reset_index(drop=True)
full_dataset = full_dataset.set_index(['stock', 'date'])
Y = full_dataset['exc_rets']
factors_list = list(set(full_dataset.columns) - set(['exc_rets']))
X = full_dataset[factors_list]
model = PanelOLS(Y, sm.tools.add_constant(X), entity_effects=True)
result = model.fit()
print(result)