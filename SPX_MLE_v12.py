# Libraries
import pandas as pd
import numpy as np
from scipy.stats import norm

import itertools
import multiprocessing
from joblib import Parallel, delayed


def estimate_parameters(y, x, Delta_t=1):
    """
    Estimate the parameters of the continuous mispricing model.

    Args:
        y (ndarray): Prices of asset y.
        x (ndarray): Prices of asset x.
        Delta_t (float): Time interval between observations (usually 1 day)

    Returns:
        tuple: Estimated parameters (mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy).
    """

    # Calculate the logarithmic prices
    Y = np.log(y)
    X = np.log(x)

    # Calculate the differences and spread
    A = Y[1:] - Y[:-1]
    B = X[1:] - X[:-1]
    z = Y[:-1] - X[:-1]

    # Obtain the number of observations
    n = len(z)
    
    # Estimate lambda_1 and lambda_2
    denominator = Delta_t * np.sum((z - np.mean(z))**2)
    lambda_1 = 2 * np.sum((A - np.mean(A)) * (z - np.mean(z))) / denominator
    lambda_2 = 2 * np.sum((B - np.mean(B)) * (z - np.mean(z))) / denominator
    
    # Estimate sigma_y and sigma_x
    sigma_y = np.sqrt(np.sum(((A - np.mean(A)) - lambda_1 * Delta_t * (z - np.mean(z)))**2) / (n * Delta_t))
    sigma_x = np.sqrt(np.sum(((B - np.mean(B)) - lambda_2 * Delta_t * (z - np.mean(z)))**2) / (n * Delta_t))

    # Estimate mu_y and mu_x
    mu_y = np.sum(A / Delta_t + lambda_1 * z + 0.5 * sigma_y**2) / n
    mu_x = np.sum(B / Delta_t - lambda_2 * z + 0.5 * sigma_x**2) / n
    
    # Calculate Z_y and Z_x
    Z_y = (A - (mu_y - lambda_1 * z - 0.5 * sigma_y**2) * Delta_t) / (sigma_y * np.sqrt(Delta_t))
    Z_x = (B - (mu_x + lambda_2 * z - 0.5 * sigma_x**2) * Delta_t) / (sigma_x * np.sqrt(Delta_t))
    
    # Calculate rho_xy
    rho_xy = np.mean(Z_y * Z_x)
    
    return mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy

def process_pair(pair, series01, window_size, step_size):
    stock_y, stock_x = pair
    valid_parameters_list = []

    prices1 = series01[stock_y]
    prices2 = series01[stock_x]

    num_windows = (len(prices1) - window_size + 1) // step_size

    for i in range(num_windows):
        start_idx = i * step_size + 1
        end_idx = start_idx + window_size

        window_prices1 = prices1.iloc[start_idx-1:end_idx]  # we pass one more sample to do the price diff calculations
        window_prices2 = prices2.iloc[start_idx-1:end_idx]

        mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy = estimate_parameters(
            window_prices1.values, window_prices2.values, 1
        )

        stability_condition = lambda_1 + lambda_2 > 0 
        # cointegration_condition = lambda_1 > 0 and lambda_2 < 0   # opposite signs
        cointegration_condition = lambda_1*lambda_2 < 0   # opposite signs

        if stability_condition and cointegration_condition:
            valid_parameters = {
                'Stock_y': stock_y,
                'Stock_x': stock_x,
                'Start_Date': window_prices1.index[1],
                'End_Date': window_prices1.index[-1],
                'mu_y': mu_y,
                'mu_x': mu_x,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'sigma_y': sigma_y,
                'sigma_x': sigma_x,
                'rho_xy': rho_xy
            }

            valid_parameters_list.append(valid_parameters)

    return valid_parameters_list

def perform_parallel_processing(all_pairs, series01, window_size, step_size):
    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count() #- 1
    # Perform parallel processing
    with Parallel(n_jobs=num_cores, verbose=10) as parallel:
        results = parallel(
            delayed(process_pair)(pair, series01, window_size, step_size) for pair in all_pairs
        )

    # Flatten the results
    parameters_list = [param for sublist in results for param in sublist]

    # Create a DataFrame from the list of parameters
    pairs_df = pd.DataFrame(parameters_list)

    return pairs_df

def position_sizes(delta_y, delta_x, trigger):
    """
    In this one, trigger = 1 is in position and trigger=nan is out of position
    """
    return np.array(trigger*delta_y), np.array(-trigger*delta_x)

def calculate_returns(price_y, price_x, rfr, delta_y, delta_x, trigger):
    y_pos, x_pos = position_sizes(delta_y, delta_x, trigger)
    prices = np.array([price_y, price_x])
    raw_returns = np.diff(prices, axis=1)/prices[:,:-1]
    raw_returns = np.vstack((raw_returns,rfr[1:]))
    ## This would be taken the deltas as position sizes
    # weights = np.array([y_pos[:-1], x_pos[:-1], (1-y_pos-x_pos)[:-1]])
    ###################################
    # This would be taking the deltas as share sizes
    pos_val = np.array([y_pos, x_pos]) * np.array([price_y, price_x])
    # Here the positions are scaled by the initial y position to 
    # pos_val = pos_val/np.abs(pos_val[0])
    pos_val[0,:] = pos_val[0,:]/np.abs(price_y)
    pos_val[1,:] = pos_val[1,:]/np.abs(price_y)
    weights = np.array([pos_val[0,1:], pos_val[1,1:], (1-pos_val[0]-pos_val[1])[1:]])
    ###################################
    returns = weights*raw_returns
    long_returns = returns[0,:]
    long_returns[np.isnan(long_returns)] = 0
    long_returns = pd.Series(long_returns)
    short_returns = returns[1,:]    
    short_returns[np.isnan(short_returns)] = 0
    short_returns = pd.Series(short_returns)
    returns = returns.sum(axis=0)
    returns[np.isnan(returns)] = 0
    pair_returns = pd.Series(returns)
    return pair_returns, long_returns, short_returns

def process_pair_oos(pair_row, series01_filtered, series02_filtered, rates_filtered):
    stock_y = pair_row['Stock_y'] 
    stock_x = pair_row['Stock_x']

    # Get the prices of the stock pair
    price_y = series01_filtered[stock_y]
    price_x = series01_filtered[stock_x]

    # Get the implied volatilities
    sigma_y = series02_filtered[stock_y] / 100
    sigma_x = series02_filtered[stock_x] / 100

    # Compute the log spread of the pair
    z_spread = np.log(price_y / price_x)

    # Compute the volatilities for the stock spread with the ATM vols
    sigmaRN_z = np.sqrt(sigma_y ** 2 + sigma_x ** 2 - 2 * sigma_y * sigma_x * pair_row["rho_xy"])

    # proportion of days until the next maturity date in years
    k = np.arange(len(z_spread), 0, -1) / YEAR ### Changed this
    # k = np.ones(len(z_spread))

    # Help variable that is used a lot
    sigma_T = sigmaRN_z * np.sqrt(k)
    
    # Calculate d1 and d2 based on the formulas
    # d1 = (z_spread + (rates_filtered/100 + 0.5*sigmaRN_z**2) * k) / sigma_T ## Why add the rates filtered there? It doesn't appear in Appendix D: An Option to Exchange One Asset for Another
    d1 = (z_spread + (0.5*sigmaRN_z**2) * k) / sigma_T ## Why add the rates filtered there? It doesn't appear in Appendix D: An Option to Exchange One Asset for Another
    d2 = d1 - sigma_T                                                            ## Why use absolute value for the z spread?

    # Calculate delta_y and delta_x using the norm.cdf function
    delta_y = norm.cdf(d1)
    delta_x = norm.cdf(d2)

    # # Compute the gammas
    # gamma_y = norm.pdf(d1) / (price_y * sigma_T) # Page 18 within Hedging Pressure and Spread Option Decomposition
    # gamma_x = norm.pdf(d2) / (price_x * sigma_T)

    # Compute the determinant of the Jacobi matrix
    # det_j = (gamma_x ** 2 / gamma_y ** 2) - price_y / price_x

    # # TRIGGER TEST 2
    # trigger = np.where(det_j <= 0, np.nan, np.where(gamma_y<gamma_x, 1, -1))

    # # TRIGGER TEST
    # trigger = np.where(gamma_y < gamma_x, 1, -1)

    # # TRIGGER BEST
    open_val = (1 - pair_row["lambda_1"] - pair_row["lambda_2"]) * sigmaRN_z
    trigger = np.where(z_spread > open_val, 1, np.nan)

    # Check if all triggers are NaN
    if not np.isnan(trigger).all():
        # Calculate the pair returns
        returns, long_returns, short_returns = calculate_returns(price_y, price_x, rates_filtered/100/YEAR, delta_y, delta_x, trigger)
        # Set the date as the index
        returns.index = z_spread.index[:-1]
        long_returns.index = z_spread.index[:-1]
        short_returns.index = z_spread.index[:-1]
        # Output the returns of the pair
        cum_rets = (1+returns).cumprod()
        cum_long_rets = (1+long_returns).cumprod()
        cum_short_rets = (1+short_returns).cumprod()
        cum_rates = (1+rates_filtered/100/YEAR).cumprod()

        if cum_rets.iloc[-1] >= cum_rates.iloc[-1]:
            positive_returns = 1
        else:
            positive_returns = 0

        if cum_long_rets.iloc[-1] >= cum_rates.iloc[-1]:
            positive_long_returns = 1
        else:
            positive_long_returns = 0

        if cum_short_rets.iloc[-1] >= cum_rates.iloc[-1]:
            positive_short_returns = 1
        else:
            positive_short_returns = 0

        trigger[np.isnan(trigger)] = 0
        return (returns, long_returns, short_returns), (positive_returns, positive_long_returns, positive_short_returns), (pair_row['End_Date'], trigger[1:])
    # return nothing if the pair is not triggered
    return None

UNIVERSE = 'spx'
# UNIVERSE = 'djiak'

RECALC_PAIRS = True
RECALC_RETURNS = True

df_original = pd.read_parquet(f'Pairs_SP500_FPT/{UNIVERSE}_impl_vols_new.parquet')

# Define the window size and step size of the pairs construction
YEAR = 252 # a month has approximately 252 trading days (252 samples)
window_size = 3 * YEAR  # 3 years - 756 samples
step_size = YEAR // 2  # 6 months - 126 samples 

# Step 2: process the df
df_original['Date'] = pd.to_datetime(df_original['Date'], format='%Y-%m-%d') # Convert column date to datetime
df_original = df_original[df_original['Date']>='2005-09-01']    # filter the df so it only mantains the desired dates (we have to take one day prior to the first to do the estimations)

df_original = df_original[~df_original['p_price'].isna()]  # Eliminate rows that have no stock prices
df_original['Id'] = df_original['Id'].str.replace('-', '_') # Change name of the columns to avoid calculation issues


# Step 3: Transform the dataset to obtain the stock prices in the desired format
series01_orig = df_original.pivot(index='Date', columns='Id', values='p_price')
series01_orig = series01_orig.sort_index()

# Drop columns with missing values
series01_orig = series01_orig.dropna(axis=1)

# if UNIVERSE == 'spx':
#     series01_orig = series01_orig.drop('EBAY_US', axis=1)

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
all_pairs = list(itertools.combinations(series01_orig.columns, 2))
all_pairs += [(y,x) for x,y in all_pairs]

COMPUTING_BATCH = 2_000
N_BATCHES = len(all_pairs)//COMPUTING_BATCH + 1
total_pos_returns = [0,0]
total_pos_long_returns = 0
total_pos_short_returns = 0
active_positions = {}
for i in range(N_BATCHES):
    pairs_file = f'pkl_data/{UNIVERSE}_pairs_MLE_v12_{i}.parquet'
    print(f"####### Calculating batch {i+1}/{N_BATCHES} of {COMPUTING_BATCH} pairs")

    if RECALC_PAIRS:
        # Perform the estimation with the MLE method
        pairs_df = perform_parallel_processing(all_pairs[i*COMPUTING_BATCH:(i+1)*COMPUTING_BATCH], series01_orig, window_size, step_size)
        if len(pairs_df) > 0:      
            pairs_df.to_parquet(pairs_file)
            del pairs_df
    try:
        pairs_df = pd.read_parquet(pairs_file)
    except Exception as e:
        print(f"File not found: {pairs_file}. {e}")

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
                                        series02[pair_row["End_Date"]:].head(step_size), 
                                        rates_df[pair_row["End_Date"]:]["DTB6"].head(step_size))
                for _, pair_row in pairs_df.iterrows()
            )

            # results = []
            # for _, pair_row in pairs_df.iterrows():
            #     results.append(process_pair_oos(pair_row, 
            #                             series01[pair_row["End_Date"]:].head(step_size), 
            #                             series02[pair_row["End_Date"]:].head(step_size), 
            #                             rates_df[pair_row["End_Date"]:]["DTB6"].head(step_size)))

            results_lean = []
            long_results_lean = []
            short_results_lean = []
            for s in results:
                if s is not None:
                    results_lean.append(s[0][0])
                    total_pos_returns[0] += s[1][0]
                    long_results_lean.append(s[0][1])
                    total_pos_long_returns += s[1][1]
                    short_results_lean.append(s[0][2])
                    total_pos_short_returns += s[1][2]
                    total_pos_returns[1] += 1
                    date = s[2][0]
                    if date in active_positions.keys():
                        active_positions[date] += s[2][1]
                    else:
                        active_positions[date] = s[2][1]
            df = pd.concat(results_lean)
            df = df.dropna(how='all')
            df = df.groupby(df.index).agg(['sum','count'])
            df.to_parquet(f'pkl_data/results_v12_{UNIVERSE}_{i}_lean.parquet')
            print(f'Saved pkl_data/results_v12_{UNIVERSE}_{i}_lean.parquet')
            df = pd.concat(long_results_lean)
            df = df.dropna(how='all')
            df = df.groupby(df.index).agg(['sum','count'])
            df.to_parquet(f'pkl_data/results_v12_long_{UNIVERSE}_{i}_lean.parquet')
            print(f'Saved pkl_data/results_v12_long_{UNIVERSE}_{i}_lean.parquet')
            df = pd.concat(short_results_lean)
            df = df.dropna(how='all')
            df = df.groupby(df.index).agg(['sum','count'])
            df.to_parquet(f'pkl_data/results_v12_short_{UNIVERSE}_{i}_lean.parquet')
            print(f'Saved pkl_data/results_v12_short_{UNIVERSE}_{i}_lean.parquet')
    else:
        print(f"Empty pairs in batch {i+1}/{N_BATCHES}")

dfs = []

for i in range(N_BATCHES):
    try:
        df = pd.read_parquet(f'pkl_data/results_v12_{UNIVERSE}_{i}_lean.parquet')
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
returns_file = f'returns_{UNIVERSE}_v12.csv'
df.to_csv(returns_file, index=True) 
print(f'Saved returns to {returns_file}')
print(f'Achieved positive results in {total_pos_returns[0]}/{total_pos_returns[1]} samples, {total_pos_returns[0]/total_pos_returns[1]*100:.4f} %')


for i in range(N_BATCHES):
    try:
        df = pd.read_parquet(f'pkl_data/results_v12_long_{UNIVERSE}_{i}_lean.parquet')
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
returns_file = f'returns_long_{UNIVERSE}_v12.csv'
df.to_csv(returns_file, index=True) 
print(f'Saved returns to {returns_file}')
print(f'Achieved positive results in {total_pos_long_returns}/{total_pos_returns[1]} samples, {total_pos_long_returns/total_pos_returns[1]*100:.4f} %')


for i in range(N_BATCHES):
    try:
        df = pd.read_parquet(f'pkl_data/results_v12_short_{UNIVERSE}_{i}_lean.parquet')
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
returns_file = f'returns_short_{UNIVERSE}_v12.csv'
df.to_csv(returns_file, index=True) 
print(f'Saved returns to {returns_file}')
print(f'Achieved positive results in {total_pos_short_returns}/{total_pos_returns[1]} samples, {total_pos_short_returns/total_pos_returns[1]*100:.4f} %')

start_dates = list(active_positions.keys())
start_dates.sort()
df_active_positions = pd.DataFrame(df)

i = 0
for d in start_dates:
    for t in active_positions[d]:
        df_active_positions.iloc[i] = t
        i+=1

positions_file = f'active_positions_{UNIVERSE}_v12.csv'
df_active_positions.to_csv(positions_file, index=True)
print(f'Saved active positions to {positions_file}')