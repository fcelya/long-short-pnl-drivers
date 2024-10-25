import pandas as pd

df_original = pd.read_parquet(f'MSCI_Data_Factset/daily_prices.parquet')
df_original = df_original.rename(columns={
    'date':'Date',
    'id':'Id',
    'price':'p_price',
})
df_original.head()

print("read df")

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
df_orig.head()
print("transformed df")

df_reset = df_orig.reset_index()
df_reset['Date'] = pd.to_datetime(df_reset['Date'])
melted_df = pd.melt(df_reset, id_vars='Date', var_name='symbol', value_name='price')
melted_df
print("melted df")

dprices = pd.read_csv('MSCI_Data_Factset/daily_prices.csv', index_col=0)
dprices['symbol'] = dprices['symbol'].str.replace('-', '_')
dprices['date'] = pd.to_datetime(dprices['date'])
df = melted_df.merge(dprices[['id','date','symbol','market_cap']], how='left',left_on=['Date','symbol'],right_on=['date','symbol'])
df
ratios = pd.read_excel('MSCI_Data_Factset/ratios.xlsx', index_col=0)
ratios['date'] = pd.to_datetime(ratios['date'])
df = df.merge(ratios[['date','id','book_value_per_share']], how='left',left_on=['Date','id'],right_on=['date','id'])
df
print("created combined df")

df['book_value_per_share'] = df.groupby('symbol')['book_value_per_share'].ffill()
df = df.drop(['date_x','date_y'],axis=1)
df.loc[:,'return'] = df.groupby('symbol')['price'].pct_change()
df = df.dropna()
df['btm'] = df['book_value_per_share']/df['price']
df.info()
print("created final df")

df.loc[:,'size_group'] = df.groupby('Date')['market_cap'].transform(
    lambda x: pd.qcut(x, 2, labels=['Small', 'Big'])
)

# Step 2: Sort by btm_ratio within each size group and create High, Neutral, Low portfolios
def assign_btm_group(x):
    return pd.qcut(x, [0, 0.3, 0.7, 1], labels=['Low', 'Neutral', 'High'])

df.loc[:,'btm_group'] = df.groupby(['Date', 'size_group'])['btm'].transform(assign_btm_group)

# Step 3: Calculate the average return for each of the 6 portfolios
portfolios = df.groupby(['Date', 'size_group', 'btm_group'])['return'].mean().unstack(['size_group', 'btm_group'])

small = (portfolios[('Small', 'High')] + portfolios[('Small', 'Neutral')] + portfolios[('Small', 'Low')]) / 3
big = (portfolios[('Big', 'High')] + portfolios[('Big', 'Neutral')] + portfolios[('Big', 'Low')]) / 3
smb = small - big

high = (portfolios[('Small', 'High')] + portfolios[('Big', 'High')]) / 2
low = (portfolios[('Small', 'Low')] + portfolios[('Big', 'Low')]) / 2
hml = high - low
print("created hml")

def clean_groupby(x):
    indeces = [y[1] for y in x.index]
    values = x.values
    df = pd.DataFrame({'indeces':indeces, 'values':values})
    df = df.sort_values(by='indeces').reset_index(drop=True)
    return df['values']

df = df.sort_values(by=['symbol', 'Date']).reset_index(drop=True)

# Step 1: Calculate the cumulative returns over the past 11 months (excluding the last month)
# Define the window size: 11 months of trading days minus 1 month (approx 21 trading days)
window_size = 252
refresh_size = 1
df.loc[:,'prev_price'] = df['price'].shift(refresh_size)

a = df.groupby('symbol')['prev_price'].pct_change(periods=window_size, fill_method=None)
a = (1+a)**(1/window_size)-1
df.loc[:,'past_return'] = a
df = df.reset_index(drop=True)
df = df.loc[~df['past_return'].isna(),:]
df = df.reset_index(drop=True)
# # Step 2: Rank stocks based on their past return
# df['momentum_rank'] = df.groupby('Date')['past_return'].transform(lambda x: x.rank(method='first'))

# # Step 3: Identify Winner (Top 10%) and Loser (Bottom 10%) portfolios
# df['momentum_group'] = pd.qcut(df['momentum_rank'], 10, labels=False)

def assign_mom_group(x):
    return pd.qcut(x, 10, labels=False)
# def assign_mom_group(x):
#     # Ensure the series has enough unique values for qcut
#     if len(x) < 2:  # Can't split less than 2 elements into quantiles
#         return pd.Series([pd.NA] * len(x), index=x.index)
    
#     # Check if there are enough distinct values
#     unique_values = x.nunique()
#     if unique_values < 10:
#         return pd.qcut(x, unique_values, labels=False, duplicates='drop')
#     else:
#         return pd.qcut(x, 10, labels=False, duplicates='drop')
a = df.groupby('Date')['past_return'].apply(assign_mom_group)
a = clean_groupby(a)
df.loc[:,'mom_group'] = a

# df['momentum_group'] = pd.qcut(df.groupby('Date')['past_return'], 10, labels=False)

# Winners: momentum_group = 9 (top 10%)
# Losers: momentum_group = 0 (bottom 10%)
winners = df[df['mom_group'] == 9]
losers = df[df['mom_group'] == 0]

# Calculate average returns for Winner and Loser portfolios
winners_avg_return = winners.groupby('Date')['return'].mean()
losers_avg_return = losers.groupby('Date')['return'].mean()

# Step 5: Calculate the Momentum Factor (MOM)
mom = winners_avg_return - losers_avg_return

print("created mom")

df_smb = pd.DataFrame({'smb':smb})
df_hml = pd.DataFrame({'hml':hml})
df_mom = pd.DataFrame({'mom':mom})
df_factors = df_smb.merge(df_hml,how='outer',left_index=True,right_index=True).merge(df_mom,how='outer',left_index=True,right_index=True)
df_factors = df_factors.dropna()
df_factors.to_csv('factors_msci.csv')