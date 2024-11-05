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

def agg_rets(df_orig, ret_col_list, factor_col, stock_col='stock', date_col='date'):
    df = df_orig.copy()
    df = df.sort_values(by=date_col,ascending=True).reset_index(drop=True)
    first_valid_index = df[factor_col].first_valid_index()
    df = df.loc[first_valid_index:,:]
    df = df.sort_values(by=[stock_col,date_col],ascending=[True,True]).reset_index(drop=True)
    for r in ret_col_list:
        df.loc[:,'new_'+r] = pd.NA
    prev_stock = None
    cum_rets_list = [1 for _ in ret_col_list]

    for i, row in df.iterrows():
        if row[stock_col] != prev_stock and not pd.isna(row[factor_col]):
            prev_stock = row[stock_col]
            cum_rets_list = [1 for _ in ret_col_list]
            continue
        elif row[stock_col] == prev_stock and not pd.isna(row[factor_col]):
            for j,r in enumerate(ret_col_list):
                df.loc[i,'new_'+r] = cum_rets_list[j]*(1+row[r]) - 1
            cum_rets_list = [1 for _ in ret_col_list]
            continue
        for j,r in enumerate(ret_col_list):
            cum_rets_list[j] *= (1+row[r])
    for r in ret_col_list:
        df[r] = df['new_'+r]
        df = df.drop('new_'+r,axis=1)

    df = df.dropna().reset_index(drop=True)
        
    return df


def agg_vols(df_orig, vol_col_list, factor_col, stock_col='stock', date_col='date'):
    df = df_orig.copy()
    df = df.sort_values(by=date_col,ascending=True).reset_index(drop=True)
    first_valid_index = df[factor_col].first_valid_index()
    df = df.loc[first_valid_index:,:]
    df = df.sort_values(by=[stock_col,date_col],ascending=[True,True]).reset_index(drop=True)
    for r in vol_col_list:
        df.loc[:,'new_'+r] = pd.NA
    prev_stock = None
    cum_vols_list = [1 for _ in vol_col_list]

    for i, row in df.iterrows():
        if row[stock_col] != prev_stock and not pd.isna(row[factor_col]):
            prev_stock = row[stock_col]
            cum_vols_list = [0 for _ in vol_col_list]
            continue
        elif row[stock_col] == prev_stock and not pd.isna(row[factor_col]):
            for j,r in enumerate(vol_col_list):
                df.loc[i,'new_'+r] = (cum_vols_list[j]+row[r]**2)**.5
            cum_vols_list = [0 for _ in vol_col_list]
            continue
        for j,r in enumerate(vol_col_list):
            cum_vols_list[j] += row[r]**2
    for r in vol_col_list:
        df[r] = df['new_'+r]
        df = df.drop('new_'+r,axis=1)

    df = df.dropna().reset_index(drop=True)
        
    return df


MONTH = 12//12
IEP_LAGS = MONTH * 1

market_spread_over_msci = pd.read_csv('1mo_rolling_hist_stock_minus_msci_vol.csv', index_col=0, parse_dates=True)
market_spread_over_msci /= 100
melted_market_spread_over_msci = market_spread_over_msci.reset_index().melt(id_vars='Date', var_name='stock', value_name='iep')
melted_market_spread_over_msci = melted_market_spread_over_msci.rename(columns={'Date':'date'})


df_esg = pd.read_csv('MSCI_Data_Factset/esg.csv')
df_esg = df_esg.rename(columns={
    'all_categories':'esg',
    'symbol':'stock',
})
df_esg = df_esg[['date','stock','esg']]
df_esg['stock'] = df_esg['stock'].str.replace('-', '_') 
df_esg['date'] = pd.to_datetime(df_esg['date'])

full_dataset = melted_market_spread_over_msci.copy()
full_dataset = full_dataset.loc[full_dataset['date']>=min(df_esg['date']),:]
full_dataset = full_dataset.merge(df_esg,on=['date','stock'],how='left')
full_dataset = agg_vols(full_dataset.reset_index(drop=True),['iep'],'esg')

full_dataset['esg'] = full_dataset['esg'].astype(np.float32)
full_dataset['iep'] = full_dataset['iep'].astype(np.float32)

full_dataset['iep'] = full_dataset['iep'].shift(IEP_LAGS)
full_dataset = full_dataset.dropna()

full_dataset = full_dataset.reset_index(drop=True)
full_dataset = full_dataset.set_index(['stock', 'date'])
Y = full_dataset['iep']
factors_list = list(set(full_dataset.columns) - set(['iep']))
X = full_dataset[factors_list]
model = PanelOLS(Y, sm.tools.add_constant(X), entity_effects=True)
result = model.fit()
print(result)
