from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go

#print(torch.__version__)



# In[2]:


import streamlit as st
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



#from scripts.treasury_utils import sortino


# In[3]:


import sys
print(sys.executable)


# In[4]:


random.seed(42)
np.random.seed(42)


# # Data Cleaning & Models

# ## Historical

# In[5]:

# Historical
tbill_path = '../data/csv/3_month_t_bill.csv'
tbill = pd.read_csv(tbill_path)
tbill['decimal'] = tbill['TB3MS'] / 100
current_risk_free = tbill['decimal'].iloc[-1]

assets_path = '../data/csv/asset_prices2.csv'
assets = pd.read_csv(assets_path)
assets.dropna(inplace=True)
all_assets = assets.copy()
all_assets = all_assets[all_assets['DAY'] >= '2023-01-22']
assets = assets[assets['DAY'] >= '2023-03-22']

all_assets_pivot = all_assets.pivot(index='DAY', columns='SYMBOL', values='PRICE').reset_index()
all_assets_pivot.set_index('DAY', inplace=True)
all_assets_pivot.columns = [f'DAILY_PRICE_{col}' for col in all_assets_pivot.columns]

panamadao_path = '../data/csv/panamadao_returns.csv'
panamadao_returns = pd.read_csv(panamadao_path)
panamadao_returns.dropna(inplace=True)

# **Insert the new code here**

# Convert 'DAY' columns to datetime
panamadao_returns['DAY'] = pd.to_datetime(panamadao_returns['DAY'])
assets['DAY'] = pd.to_datetime(assets['DAY'])

# Pivot the assets data to have symbols as columns
pivot_assets = assets.pivot(index='DAY', columns='SYMBOL', values='PRICE').reset_index()

# Rename columns for merging
pivot_assets.columns = ['DAY'] + [f'DAILY_PRICE_{col}' for col in pivot_assets.columns if col != 'DAY']

# Merge assets data into panamadao_returns
panamadao_returns = panamadao_returns.merge(pivot_assets, on='DAY', how='left')

# Replace ETH and WETH DAILY_PRICE with WETH price
panamadao_returns.loc[panamadao_returns['SYMBOL'] == 'ETH', 'DAILY_PRICE'] = panamadao_returns['DAILY_PRICE_WETH']
panamadao_returns.loc[panamadao_returns['SYMBOL'] == 'WETH', 'DAILY_PRICE'] = panamadao_returns['DAILY_PRICE_WETH']

# Drop the extra DAILY_PRICE columns, if they exist
columns_to_drop = [f'DAILY_PRICE_{col}' for col in pivot_assets.columns if col != 'DAY' and f'DAILY_PRICE_{col}' in panamadao_returns.columns]
panamadao_returns.drop(columns=columns_to_drop, inplace=True)

# Save the updated DataFrame to a CSV file (optional)
# panamadao_returns.to_csv('data/updated_panamadao_returns.csv', index=False)

# **Continue with the rest of your existing code**

cowdao_path = '../data/csv/cowdao_returns.csv'

cowdao_returns = pd.read_csv(cowdao_path)
cowdao_returns.dropna(inplace=True)

# **Insert the new code here**

# Convert 'DAY' columns to datetime
cowdao_returns['DAY'] = pd.to_datetime(cowdao_returns['DAY'])

# Pivot the assets data to have symbols as columns

# Rename columns for mergin

# Merge assets data into panamadao_returns
cowdao_returns = cowdao_returns.merge(pivot_assets, on='DAY', how='inner')

# Replace ETH and WETH DAILY_PRICE with WETH price
cowdao_returns.loc[cowdao_returns['SYMBOL'] == 'ETH', 'DAILY_PRICE'] = cowdao_returns['DAILY_PRICE_WETH']
cowdao_returns.loc[cowdao_returns['SYMBOL'] == 'WETH', 'DAILY_PRICE'] = cowdao_returns['DAILY_PRICE_WETH']

# Drop the extra DAILY_PRICE columns, if they exist
columns_to_drop = [f'DAILY_PRICE_{col}' for col in pivot_assets.columns if col != 'DAY' and f'DAILY_PRICE_{col}' in cowdao_returns.columns]
cowdao_returns.drop(columns=columns_to_drop, inplace=True)


panama_dao_assets = panamadao_returns['SYMBOL'].unique()
cowdao_assets = cowdao_returns['SYMBOL'].unique()

def calculate_historical_returns(panamadao_returns):
    # Convert 'DAY' to datetime
    panamadao_returns['DAY'] = pd.to_datetime(panamadao_returns['DAY'])

    # Sort by 'DAY' and 'SYMBOL'
    panamadao_returns = panamadao_returns.sort_values(by=['DAY', 'SYMBOL'])

    # Calculate previous day price
    panamadao_returns['prev_day_price'] = panamadao_returns.groupby('SYMBOL')['DAILY_PRICE'].shift(1)

    # Set first day's previous day price to current day price to get a log return of 0
    panamadao_returns.loc[panamadao_returns['prev_day_price'].isna(), 'prev_day_price'] = panamadao_returns['DAILY_PRICE']

    # Calculate daily log return
    panamadao_returns['daily_log_return'] = np.log(panamadao_returns['DAILY_PRICE'] / panamadao_returns['prev_day_price']).fillna(0)

    # Calculate weighted daily log return
    panamadao_returns['weighted_daily_return'] = (panamadao_returns['daily_log_return'] * panamadao_returns['COMPOSITION']).fillna(0)

    # Calculate weighted daily log return per day
    weighted_daily_log_returns = panamadao_returns.groupby('DAY').apply(lambda x: x['weighted_daily_return'].sum()).reset_index(name='weighted_daily_return')

    # Calculate cumulative log returns
    weighted_daily_log_returns['cumulative_log_return'] = weighted_daily_log_returns['weighted_daily_return'].cumsum().fillna(0)

    # Calculate cumulative returns from cumulative log returns
    weighted_daily_log_returns['cumulative_return'] = np.exp(weighted_daily_log_returns['cumulative_log_return']) - 1

    historical_returns = weighted_daily_log_returns[['DAY', 'weighted_daily_return']]
    historical_cumulative_return = weighted_daily_log_returns[['DAY', 'cumulative_return']]
    
    return historical_returns, historical_cumulative_return

cowdao_historical_returns, cowdao_historical_cumulative_return = calculate_historical_returns(cowdao_returns)

base_return = cowdao_historical_cumulative_return.copy()
base_return = base_return.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})

combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')

first_value = combined['base_cumulative_return'].iloc[0]
combined['CowDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))

historical_normalized_returns = combined[['DAY', 'CowDAO_treasury_return']]

cowdao_historical_cumulative_return.set_index('DAY', inplace=True)
historical_normalized_returns.set_index('DAY', inplace=True)

cow_dao_pivot_data = cowdao_returns.pivot(index='DAY', columns='SYMBOL', values=['TOTAL_FILLED_BALANCE', 'DAILY_PRICE', 'TOTAL_VALUE_IN_USD', 'COMPOSITION'])
cow_dao_pivot_data.columns = ['_'.join(col).strip() for col in cow_dao_pivot_data.columns.values]
cow_dao_pivot_data = cow_dao_pivot_data.reset_index()

cow_dao_pivot_data.set_index('DAY', inplace=True)

filtered_columns = [col for col in cow_dao_pivot_data.columns if 'COMPOSITION_' in col or 'DAILY_PRICE_' in col]
cow_pivot_data_filtered = cow_dao_pivot_data[filtered_columns]
cow_pivot_data_filtered

assets_pivot = assets.pivot(index='DAY', columns='SYMBOL', values='PRICE')

# Rename columns to match the format in pivot_data_filtered
assets_pivot.columns = [f'DAILY_PRICE_{col}' for col in assets_pivot.columns]
assets_pivot.index

cow_pivot_data_filtered.index = pd.to_datetime(cow_pivot_data_filtered.index)


# In[24]:


all_assets_pivot.index = pd.to_datetime(all_assets_pivot.index)

cow_pivot_data_filtered_no_price = cow_pivot_data_filtered.drop(columns=['DAILY_PRICE_ETH',  'DAILY_PRICE_WETH'])

cow_dao_combined_all_assets = all_assets_pivot.join(cow_pivot_data_filtered_no_price,  how='left')

cow_dao_combined_all_assets = cow_dao_combined_all_assets[cow_dao_combined_all_assets.index <= '2024-05-15']

for asset in assets['SYMBOL'].unique():
    comp_col = f'COMPOSITION_{asset}'
    if comp_col not in cow_dao_combined_all_assets.columns:
        cow_dao_combined_all_assets[comp_col] = 0.0
