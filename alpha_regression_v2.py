import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import pickle as pkl
from scipy.stats import gmean
import gc

warnings.filterwarnings('ignore')

rets = pd.read_csv('returns_spx_v11.csv', index_col=0, parse_dates=True)
rets = rets.rename(columns={
    'Daily_Return': 'rets'
})
rets_long = pd.read_csv('returns_long_spx_v11.csv', index_col=0, parse_dates=True)
rets_long = rets_long.rename(columns={
    'Daily_Return': 'rets'
})
rets_short = pd.read_csv('returns_short_spx_v11.csv', index_col=0, parse_dates=True)
rets_short = rets_short.rename(columns={
    'Daily_Return': 'rets'
})
benchmark = pd.read_csv('returns_sp500.csv', index_col=0, parse_dates=True)
benchmark = benchmark[(benchmark.index >= rets.index[0])&(benchmark.index <= rets.index[-1])]

open_pos = pd.read_csv('active_positions_spx_v11.csv', index_col=0, parse_dates=True)
open_pos = open_pos.rename(columns={
    'Daily_Return': 'open_positions'
})

YEAR = 252

def read_factors(model='ff3'):
    
    if model == 'ff3':
        factors = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', index_col=0, parse_dates=True)
        factors = factors.drop('RF', axis=1)
        factors.columns = [c.lower().strip() for c in factors.columns]
        factors_list = list(factors.columns)
        for f in factors_list:
            factors[f] = pd.to_numeric(factors[f], errors='coerce')/100
        factors = factors.dropna()
    elif model == 'ff5':
        factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.CSV', index_col=0, parse_dates=True)
        factors = factors.drop('RF', axis=1)
        factors.columns = [c.lower().strip() for c in factors.columns]
        factors_list = list(factors.columns)
        for f in factors_list:
            factors[f] = pd.to_numeric(factors[f], errors='coerce')/100
        factors = factors.dropna()
    elif model == 'ffc4':
        factors_1, _ = read_factors(model='ff3')
        factors_2 = pd.read_csv('F-F_Momentum_Factor_daily.CSV', index_col=0, parse_dates=True)
        factors_2.columns = [c.lower().strip() for c in factors_2.columns]
        for f in factors_2.columns:
            factors_2[f] = pd.to_numeric(factors_2[f], errors='coerce')/100
        factors = factors_1.merge(factors_2, left_index=True, right_index=True, how='inner')
        factors_list = list(factors.columns)
        factors = factors.dropna()
    else:
        raise ValueError("The model must be one of 'ff3', 'ff5' or 'ffc4'")

    return factors, factors_list
factors, factors_list = read_factors(model='ffc4')

risk_free = pd.read_csv('Pairs_SP500_FPT/DTB6.csv', index_col=0, parse_dates=True)
risk_free = risk_free.rename(columns={
    'DTB6': 'rf'
})
risk_free['rf'] = pd.to_numeric(risk_free['rf'], errors='coerce')
risk_free = risk_free.dropna()
risk_free['rf'] = risk_free['rf']/100/365

vol_spread = pd.read_csv('Pairs_SP500_FPT/spx_impl_minus_hist_vol.csv', index_col=0, parse_dates=True)
vol_spread['vol_spread'] /= 100

vol_premium = pd.read_csv('Pairs_SP500_FPT/1mo_rolling_impl_minus_hist_vol.csv', index_col=0, parse_dates=True)
vol_premium /= 100
melted_vol_premium = vol_premium.reset_index().melt(id_vars='Date', var_name='stock', value_name='vol_premium')

market_spread_over_sp = pd.read_csv('Pairs_SP500_FPT/1mo_rolling_hist_stock_minus_sp_vol.csv', index_col=0, parse_dates=True)
market_spread_over_sp /= 100
melted_market_spread_over_sp = market_spread_over_sp.reset_index().melt(id_vars='Date', var_name='stock', value_name='spread_over_sp')

factors['mkt-rf'] = factors['mkt-rf'].astype(np.float32)
factors['smb'] = factors['smb'].astype(np.float32)
factors['hml'] = factors['hml'].astype(np.float32)
factors['mom'] = factors['mom'].astype(np.float32)
rets['rets'] = rets['rets'].astype(np.float32)
risk_free['rf'] = risk_free['rf'].astype(np.float32)
vol_spread['vol_spread'] = vol_spread['vol_spread'].astype(np.float32)
melted_market_spread_over_sp['stock'] = melted_market_spread_over_sp['stock'].astype('category')
melted_market_spread_over_sp['spread_over_sp'] = melted_market_spread_over_sp['spread_over_sp'].astype(np.float32)
melted_vol_premium['stock'] = melted_vol_premium['stock'].astype('category')
melted_vol_premium['vol_premium'] = melted_vol_premium['vol_premium'].astype(np.float32)

full_dataset = rets.merge(factors, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(risk_free, left_index=True, right_index=True, how='inner')
# full_dataset = full_dataset.merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(melted_market_spread_over_sp, left_index=True, right_on='Date', how='left')
# full_dataset = full_dataset.merge(melted_vol_premium, left_on=['Date', 'stock'], right_on=['Date', 'stock'], how='left')
# full_dataset = benchmark.merge(factors, left_index=True, right_index=True, how='inner').merge(risk_free, left_index=True, right_index=True, how='inner').merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset['exc_rets'] = full_dataset['rets'] - full_dataset['rf']
# full_dataset = full_dataset.dropna()
full_dataset.index = full_dataset['Date']
full_dataset = full_dataset.drop(['Date'], axis=1)
full_dataset = full_dataset.dropna()

full_dataset = full_dataset.reset_index()
full_dataset = full_dataset.set_index(['stock', 'Date'])

df = pd.read_excel('20240529/ratios.xlsx', index_col=0, parse_dates=True)
# dates = pd.unique(df['date'])
dates = pd.unique(df[df['id']==df['id'][0]]['date'])
dates.sort()
del df
gc.collect()
dates_df = pd.DataFrame({'end':dates})
dates_df = dates_df.sort_values(by='end',ascending=True)
dates_df['start'] = dates_df['end'].shift(1)
dates_df = dates_df.dropna()
dates_df = dates_df[(dates_df['start']>=full_dataset.index[0][1])&((dates_df['end']<=full_dataset.index[-1][1]))]
dates_df = dates_df.dropna()
dates_df = dates_df.reset_index(drop=True)
dates_df.loc[:,'alpha'] = 0
dates_df.loc[:,'mean_exc_strat_rets'] = 0
dates_df.loc[:,'mean_exc_mkt_rets'] = 0


period = YEAR//4
n_stocks = len(set([i[0] for i in full_dataset.index]))
total = len(full_dataset)//n_stocks
for i in range(len(dates_df)):
    if i%10==0:
        print(f'{i}/{len(dates_df)}')
    start = dates_df.iloc[i]['start']
    end = dates_df.iloc[i]['end']
    dataset = full_dataset.reset_index()
    dataset = dataset[(dataset['Date']>=start)&(dataset['Date']<end)]
    dataset = dataset.set_index(['stock','Date'])
    Y = dataset['exc_rets']
    factors_list = list(set(dataset.columns) - set(['rets','exc_rets','rf']))
    X = dataset[factors_list]
    model = PanelOLS(Y, sm.tools.add_constant(X), entity_effects=True)
    result = model.fit()
    dates_df.loc[i,'alpha'] = result.params.const
    dates_df.loc[i,'mean_exc_strat_rets'] = gmean(1+Y)-1
    dates_df.loc[i,'mean_exc_mkt_rets'] = gmean(1+X['mkt-rf'])-1

dates_df.to_csv('alpha_regression_v3_total.csv')


rets_long['rets'] = rets_long['rets'].astype(np.float32)
full_dataset = rets_long.merge(factors, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(risk_free, left_index=True, right_index=True, how='inner')
# full_dataset = full_dataset.merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(melted_market_spread_over_sp, left_index=True, right_on='Date', how='left')
# full_dataset = full_dataset.merge(melted_vol_premium, left_on=['Date', 'stock'], right_on=['Date', 'stock'], how='left')
# full_dataset = benchmark.merge(factors, left_index=True, right_index=True, how='inner').merge(risk_free, left_index=True, right_index=True, how='inner').merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset['exc_rets'] = full_dataset['rets'] - full_dataset['rf']
# full_dataset = full_dataset.dropna()
full_dataset.index = full_dataset['Date']
full_dataset = full_dataset.drop(['Date'], axis=1)
full_dataset = full_dataset.dropna()
full_dataset = full_dataset.reset_index()
full_dataset = full_dataset.set_index(['stock', 'Date'])
dates_df = pd.DataFrame({'end':dates})
dates_df = dates_df.sort_values(by='end',ascending=True)
dates_df['start'] = dates_df['end'].shift(1)
dates_df = dates_df.dropna()
dates_df = dates_df[(dates_df['start']>=full_dataset.index[0][1])&((dates_df['end']<=full_dataset.index[-1][1]))]
dates_df = dates_df.dropna()
dates_df.loc[:,'alpha'] = 0
dates_df.loc[:,'mean_exc_strat_rets'] = 0
dates_df.loc[:,'mean_exc_mkt_rets'] = 0


period = YEAR//4
n_stocks = len(set([i[0] for i in full_dataset.index]))
total = len(full_dataset)//n_stocks
for i in range(len(dates_df)):
    if i%10==0:
        print(f'{i}/{len(dates_df)}')
    start = dates_df.iloc[i]['start']
    end = dates_df.iloc[i]['end']
    dataset = full_dataset.reset_index()
    dataset = dataset[(dataset['Date']>=start)&(dataset['Date']<end)]
    dataset = dataset.set_index(['stock','Date'])
    Y = dataset['exc_rets']
    factors_list = list(set(dataset.columns) - set(['rets','exc_rets','rf']))
    X = dataset[factors_list]
    model = PanelOLS(Y, sm.tools.add_constant(X), entity_effects=True)
    result = model.fit()
    dates_df.loc[i,'alpha'] = result.params.const
    dates_df.loc[i,'mean_exc_strat_rets'] = gmean(1+Y)-1
    dates_df.loc[i,'mean_exc_mkt_rets'] = gmean(1+X['mkt-rf'])-1

dates_df.to_csv('alpha_regression_v3_long.csv')


rets_short['rets'] = rets_short['rets'].astype(np.float32)
full_dataset = rets_short.merge(factors, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(risk_free, left_index=True, right_index=True, how='inner')
# full_dataset = full_dataset.merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset = full_dataset.merge(melted_market_spread_over_sp, left_index=True, right_on='Date', how='left')
# full_dataset = full_dataset.merge(melted_vol_premium, left_on=['Date', 'stock'], right_on=['Date', 'stock'], how='left')
# full_dataset = benchmark.merge(factors, left_index=True, right_index=True, how='inner').merge(risk_free, left_index=True, right_index=True, how='inner').merge(vol_spread, left_index=True, right_index=True, how='inner')
full_dataset['exc_rets'] = full_dataset['rets'] - full_dataset['rf']
# full_dataset = full_dataset.dropna()
full_dataset.index = full_dataset['Date']
full_dataset = full_dataset.drop(['Date'], axis=1)
full_dataset = full_dataset.dropna()
full_dataset = full_dataset.reset_index()
full_dataset = full_dataset.set_index(['stock', 'Date'])
dates_df = pd.DataFrame({'end':dates})
dates_df = dates_df.sort_values(by='end',ascending=True)
dates_df['start'] = dates_df['end'].shift(1)
dates_df = dates_df.dropna()
dates_df = dates_df[(dates_df['start']>=full_dataset.index[0][1])&((dates_df['end']<=full_dataset.index[-1][1]))]
dates_df = dates_df.dropna()
dates_df.loc[:,'alpha'] = 0
dates_df.loc[:,'mean_exc_strat_rets'] = 0
dates_df.loc[:,'mean_exc_mkt_rets'] = 0


period = YEAR//4
n_stocks = len(set([i[0] for i in full_dataset.index]))
total = len(full_dataset)//n_stocks
for i in range(len(dates_df)):
    if i%10==0:
        print(f'{i}/{len(dates_df)}')
    start = dates_df.iloc[i]['start']
    end = dates_df.iloc[i]['end']
    dataset = full_dataset.reset_index()
    dataset = dataset[(dataset['Date']>=start)&(dataset['Date']<end)]
    dataset = dataset.set_index(['stock','Date'])
    Y = dataset['exc_rets']
    factors_list = list(set(dataset.columns) - set(['rets','exc_rets','rf']))
    X = dataset[factors_list]
    model = PanelOLS(Y, sm.tools.add_constant(X), entity_effects=True)
    result = model.fit()
    dates_df.loc[i,'alpha'] = result.params.const
    dates_df.loc[i,'mean_exc_strat_rets'] = gmean(1+Y)-1
    dates_df.loc[i,'mean_exc_mkt_rets'] = gmean(1+X['mkt-rf'])-1

dates_df.to_csv('alpha_regression_v3_short.csv')