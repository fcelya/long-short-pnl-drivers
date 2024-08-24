# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import norm
from scipy.stats import skew

from datetime import datetime, timedelta

import itertools
import multiprocessing
from joblib import Parallel, delayed
import pickle
import sys


def process_pair_johansen(pair, series01, window_size, step_size, cointegration_confidence_level=95):
    stock_y, stock_x = pair
    parameters_list = []

    lprices1 = np.log(series01[stock_y])
    lprices2 = np.log(series01[stock_x])

    num_windows = (len(lprices1) - window_size + 1) // step_size

    confidence_level_cols = {
        90: 0,
        95: 1,
        99: 2
    }
    confidence_level_col = confidence_level_cols[cointegration_confidence_level]

    for i in range(num_windows):
        start_idx = i * step_size + 1
        end_idx = start_idx + window_size

        window_lprices1 = lprices1.iloc[start_idx-1:end_idx]  # we pass one more sample to do it as in the MLE
        window_lprices2 = lprices2.iloc[start_idx-1:end_idx]
        
        lprices = np.array([window_lprices1, window_lprices2]).T
        result = coint_johansen(lprices, 0, 1)
        trace_crit_value = result.cvt[:, confidence_level_col]
        if result.eig[0] >= result.eig[1]:
            max_idx = 0
        else:
            max_idx = 1
        w = result.evec[max_idx]

        if result.lr1[0] >= trace_crit_value[0] and w[0]*w[1]<0:
            comb_lprices = lprices@w
            parameters = {
                'Stock_y': stock_y,
                'Stock_x': stock_x,
                'Start_Date': window_lprices1.index[1],
                'End_Date': window_lprices1.index[-1],
                'lambda_y': w[0],
                'lambda_x': w[1],
                'mean': comb_lprices.mean(),
                'std': comb_lprices.std()
            }
            parameters_list.append(parameters)

    return parameters_list

def perform_parallel_processing(all_pairs, series01, window_size, step_size):
    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count() #- 1

    # Perform parallel processing
    results = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(process_pair_johansen)(pair, series01, window_size, step_size) for pair in all_pairs
    )

    # Flatten the results
    parameters_list = [param for sublist in results for param in sublist]

    # Create a DataFrame from the list of parameters
    pairs_df = pd.DataFrame(parameters_list)

    return pairs_df

def calculate_position_sizes(price_y, price_x, trigger, lambda_y, lambda_x):
    # https://epchan.blogspot.com/2013/11/cointegration-trading-with-log-prices.html
    y_pos = np.array(trigger)
    x_pos = -y_pos*price_y/price_x
    return y_pos, x_pos

def calculate_returns_abs_booksize(price_y, price_x, rfr, lambda_y, lambda_x, trigger):
    # # Apply the position_sizes function to create new column for the position size
    # # https://quant.stackexchange.com/questions/32513/calculating-the-returns-of-a-long-short-strategy
    # y_pos, x_pos = calculate_position_sizes(price_y, price_x,trigger)
    # pos_val = np.array([y_pos, x_pos]) * np.array([price_y, price_x])
    # # returns = np.diff(pos_val, axis=1).sum(axis=0)/np.abs(pos_val[:,:-1]).sum(axis=0)
    # returns = np.diff(pos_val, axis=1)/pos_val[:,:-1]
    # # weights = np.array([y_pos, -y_pos])#/(y_pos+x_pos)
    # # combined_returns = (weights[:,1:]*returns).sum(axis=0)
    # combined_returns = returns.sum(axis=0)
    # combined_returns[np.isnan(combined_returns)] = 0
    # pair_returns = pd.Series(combined_returns)

    y_pos, x_pos = calculate_position_sizes(price_y, price_x, trigger, lambda_y, lambda_x)
    prices = np.array([price_y, price_x])
    raw_returns = np.diff(prices, axis=1)/prices[:,:-1]
    raw_returns = np.vstack((raw_returns,rfr[:-1].to_numpy().reshape((1,step_size-1))))
    pos_val = np.array([y_pos, x_pos]) * np.array([price_y, price_x])
    pos_val = pos_val/np.abs(pos_val[0])
    weights = np.array([pos_val[0,1:], pos_val[1,1:], (1-pos_val[0]-pos_val[1])[1:]])
    returns = (weights*raw_returns).sum(axis=0)
    returns[np.isnan(returns)] = 0
    pair_returns = pd.Series(returns)

    return pair_returns

def process_pair_oos(pair_row, series01_filtered, rates_filtered):
    
    stock_y = pair_row['Stock_y']
    stock_x = pair_row['Stock_x']

    # Get the prices of the stock pair
    lprice_y = np.log(series01_filtered[stock_y])
    lprice_x = np.log(series01_filtered[stock_x])
    lprice = np.array([lprice_y, lprice_x])
    price = np.array([series01_filtered[stock_y], series01_filtered[stock_x]])
    w = np.array([pair_row['lambda_y'],pair_row['lambda_x']])
    comb_lprice = w@lprice
    sup = pair_row['mean'] + 2*pair_row['std']
    inf = pair_row['mean'] - 2*pair_row['std']
    mean = pair_row['mean']

    # Compute the trigger
    trigger = np.zeros(len(comb_lprice))
    state = np.nan
    for i in range(len(trigger)):
        if comb_lprice[i] >= sup:
            state = -1
        elif comb_lprice[i] <= inf:
            state = 1
        elif (comb_lprice[i] <= mean and state == -1) or (comb_lprice[i] >= mean and state == 1):
            state = np.nan
        trigger[i] = -state

    # Check if all triggers are NaN
    if ~np.isnan(trigger).all():
        # Calculate the pair returns
        returns = calculate_returns_abs_booksize(price[0], price[1], rates_filtered/100/YEAR, w[0], w[1], trigger)
        # Set the date as the index
        returns.index = lprice_y.index[1:]
        # Output the returns of the pair
        return returns
    # return nothing if the pair is not triggered
    return None


UNIVERSE = 'spx'
# UNIVERSE = 'djiak'

RECALC_PAIRS = True
RECALC_RETURNS = True

gettrace = getattr(sys, 'gettrace', None)

# if gettrace is None:
#     DEBUG = False
#     print('Not debugging')
# elif gettrace():
#     DEBUG = True
#     print('Debugging')
# else:
#     DEBUG = False
#     print('Not debugging')
# DEBUG=False
# RECALC_PAIRS &= ~DEBUG

# Step 1: Load the SPX price data
df_original = pd.read_parquet(f'Pairs_SP500_FPT/{UNIVERSE}_impl_vols_new.parquet')


# Define the window size and step size of the pairs construction
YEAR = 252 # a month has approximately 252 trading days (252 samples)
window_size = 3 * YEAR  # 3 years - 756 samples
step_size = YEAR // 4  # 6 months - 126 samples 

# Step 2: process the df
df_original['Date'] = pd.to_datetime(df_original['Date'], format='%Y-%m-%d') # Convert column date to datetime
df_original = df_original[df_original['Date']>='2005-09-01']    # filter the df so it only mantains the desired dates (we have to take one day prior to the first to do the estimations)

df_original = df_original[~df_original['p_price'].isna()]  # Eliminate rows that have no stock prices
df_original['Id'] = df_original['Id'].str.replace('-', '_') # Change name of the columns to avoid calculation issues


# Step 3: Transform the dataset to obtain the stock prices in the desired format
series01 = df_original.pivot(index='Date', columns='Id', values='p_price')
series01 = series01.sort_index()

# Drop columns with missing values
series01 = series01.dropna(axis=1)

if UNIVERSE == 'spx':
    series01 = series01.drop('EBAY_US', axis=1)


# Step 3.2: load the risk-free rates df (6-month treasury rates)
rates_df_orig = pd.read_csv('Pairs_SP500_FPT/DTB6.csv')

# Process the rates df
rates_df_orig['DATE'] = pd.to_datetime(rates_df_orig['DATE'])
# Process the rates
rates_df_orig['DTB6'] = pd.to_numeric(rates_df_orig['DTB6'], errors='coerce')
# interpolate to get rid of the rates NAs
rates_df_orig['DTB6'] = rates_df_orig['DTB6'].interpolate()
# Maintain only the trading period in the dataset

# Count all Possible Pairs
all_pairs = list(itertools.combinations(series01.columns, 2))
# all_pairs += [(y,x) for x,y in all_pairs]

COMPUTING_BATCH = 10_000
N_BATCHES = len(all_pairs)//COMPUTING_BATCH + 1
for i in range(N_BATCHES):
    pairs_file = f'pkl_data/{UNIVERSE}_pairs_johansen_log_v8_{i}.pkl'

    if RECALC_PAIRS:
        print(f"####### Calculating batch {i+1}/{N_BATCHES} of {COMPUTING_BATCH} pairs")
        
        # Perform the estimation with the MLE method
        pairs_df = perform_parallel_processing(all_pairs[i*COMPUTING_BATCH:(i+1)*COMPUTING_BATCH], series01, window_size, step_size)
        # with open(pairs_file, 'wb') as f:
        #     pickle.dump(pairs_df, f)
        pairs_df.to_pickle(pairs_file)

    # with open(pairs_file, 'rb') as f:
    #         pairs_df = pickle.load(f)
    pairs_df = pd.read_pickle(pairs_file)
    if len(pairs_df):
        unique_end_dates = pairs_df["End_Date"].unique()
        first_trading_day = np.min(unique_end_dates)

        df = pd.DataFrame(df_original)
        df = df[df['Date'] >= first_trading_day]

        # Check number of NAs per stock in implied_vols
        df[df['p_opt_atmiv_mkt'].isna()]["Id"].value_counts()

        # interpolate to get rid of the impl vols NAs
        df['p_opt_atmiv_mkt'] = df['p_opt_atmiv_mkt'].interpolate()

        series02 = df.pivot(index='Date', columns='Id', values='p_opt_atmiv_mkt')
        series02 = series02.sort_index()
        series02 = series02.dropna(axis=1)
        
        series01 = df.pivot(index='Date', columns='Id', values='p_price')
        series01 = series01.sort_index()

        series01 = series01.dropna(axis=1)
        series01 = series01[first_trading_day:]
        rates_df = pd.DataFrame(rates_df_orig)
        rates_df = rates_df[rates_df['DATE'] >= first_trading_day]
        rates_df = rates_df[rates_df['DATE'].isin(set(series01.index))]
        rates_df.set_index('DATE', inplace=True)

        returns_dict = {}

        num_cores = multiprocessing.cpu_count() #- 1

        if RECALC_RETURNS or RECALC_PAIRS:
            print(f"Will calculate {len(pairs_df)} returns")
            # Perform parallel processing
            results = Parallel(n_jobs=num_cores, verbose=10)(
                delayed(process_pair_oos)(pair_row, 
                                        series01[pair_row["End_Date"]:].head(step_size),
                                        rates_df[pair_row["End_Date"]:].head(step_size))
                for _, pair_row in pairs_df.iterrows()
            )
            results_lean = []
            for s in results:
                if s is not None:
                    results_lean.append(s)
            df = pd.concat(results_lean)
            df = df.dropna(how='all')
            df = df.groupby(df.index).agg(['sum','count'])
            df.to_pickle(f'pkl_data/results_johansen_log_v8_{UNIVERSE}_{i}_lean.pkl')
            print(f'Saved pkl_data/results_johansen_log_v8_{UNIVERSE}_{i}_lean.pkl')
    else:
        print(f"Empty pairs in batch {i+1}/{N_BATCHES}")

dfs = []

for i in range(N_BATCHES):
    try:
        df = pd.read_pickle(f'pkl_data/results_johansen_log_v8_{UNIVERSE}_{i}_lean.pkl')
        dfs.append(df)
    except Exception as e:
        print(f"No file found for {i+1}/{N_BATCHES}. {e}")

df = pd.concat(dfs)
df = df.groupby(df.index).agg(func={
    'sum':'sum',
    'count':'sum',
})

df = df['sum'] / df['count']
df = df.to_frame(name='Daily_Return')
returns_file = f'returns_johansen_log_{UNIVERSE}_v8.csv'
df.to_csv(returns_file, index=True) 
print(f'Saved returns to {returns_file}')