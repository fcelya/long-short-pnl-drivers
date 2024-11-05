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

def agg_rets(df_orig, ret_col_list, factor_col, stock_col='stock'):
    df = df_orig.copy()
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


def agg_vols(df_orig, vol_col_list, factor_col, stock_col='stock'):
    df = df_orig.copy()
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

market_spread_over_msci = pd.read_csv('1mo_rolling_hist_stock_minus_msci_vol.csv', index_col=0, parse_dates=True)
market_spread_over_msci /= 100
melted_market_spread_over_msci = market_spread_over_msci.reset_index().melt(id_vars='Date', var_name='stock', value_name='iep')
melted_market_spread_over_msci = melted_market_spread_over_msci.rename(columns={'Date':'date'})


df_democracy = pd.read_csv('MSCI_Data_Factset/democracy-index-eiu.csv')
df_democracy = df_democracy.rename(columns={
    'Code':'country_code',
    'Democracy score':'dem_score'})

melted_market_spread_over_msci['country_code'] = melted_market_spread_over_msci['stock'].str.split('_').str[1]
melted_market_spread_over_msci['Year'] = melted_market_spread_over_msci['date'].dt.year

full_dataset = melted_market_spread_over_msci.merge(df_democracy, on=['Year','country_code'],how='left')
