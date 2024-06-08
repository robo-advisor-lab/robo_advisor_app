import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go

#print(torch.__version__)
print(PPO)
print(gym.__version__)


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

#from scripts.utils import calculate_sortino_ratio
#from scripts.data_processing import current_risk_free

class mvo_model():
    def __init__(self, current_risk_free, start_date, threshold=0):
        self.threshold = threshold 
        self.current_risk_free = current_risk_free
        self.start_date = start_date
    
    def calculate_sortino_ratio(self, returns):
        risk_free=self.current_risk_free
        #print("from", returns.index.min())
        #print('through', returns.index.max())
        #print('returns', returns.tail())
        # Calculate daily risk-free rate
        daily_risk_free_rate = (1 + risk_free) ** (1/365) - 1
    
        # Calculate excess returns
        excess_returns = returns - daily_risk_free_rate
    
        # Calculate downside returns
        downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
        daily_downside_deviation = np.sqrt(downside_returns.mean())
    
        # Handle NaN downside risk
        if np.isnan(daily_downside_deviation):
            daily_downside_deviation = 0.0
    
        # Compounding returns and annualizing based on the actual days with data
        active_days = returns.notna().sum()  # Using actual days with returns
        annual_factor = 365 / active_days
        compounding_return = (1 + excess_returns).prod() ** annual_factor - 1
    
        # Annual downside deviation
        annual_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
        # Calculate Sortino ratios
        sortino_ratio = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else 0.0
        print('sortino ratio', sortino_ratio)
    
        return sortino_ratio
    
    def mvo_sortino(self, returns, sortino_ratios, eth_index):
        n = returns.shape[1]
        weights = cp.Variable(n)
        portfolio_return = returns @ weights
        sortino_matrix = np.tile(sortino_ratios.values.reshape(1, -1), (len(returns), 1))
        portfolio_risk = cp.norm(returns - cp.multiply(sortino_matrix, cp.reshape(weights, (1, n))), 'fro')
        objective = cp.Maximize(cp.sum(portfolio_return))
        
        # Set constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= 0.3
        ]
        
        # Ensure at least 20% allocation to ETH
        constraints.append(weights[eth_index] >= 0.2)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        return weights.value
    
    def rebalance_portfolio(data, weights):
        compositions = np.outer(np.ones(len(data)), weights)
        return compositions
    
    def calculate_daily_portfolio_returns(self, data, all_assets):
        all_assets = np.array(all_assets)
        log_returns = np.log(data[[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
        composition_columns = [f'COMPOSITION_{asset}' for asset in all_assets]
        log_returns.columns = composition_columns
        daily_portfolio_returns = (log_returns * data[composition_columns].shift(1).fillna(0)).sum(axis=1)
        return daily_portfolio_returns
    
    def calculate_cumulative_return(self, daily_portfolio_returns):
        daily_portfolio_returns = daily_portfolio_returns[daily_portfolio_returns.index >= self.start_date]
        cumulative_return = np.exp(np.log1p(daily_portfolio_returns).cumsum()) - 1
        return cumulative_return
    
    def rebalance(self, data, all_assets, rebalancing_frequency=7):
        all_assets = np.array(all_assets)
        data = data.sort_index()
        rebalanced_data = data.copy()
        
        initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
        current_composition = initial_composition.copy()
    
        start_date = self.start_date
    
        # Find the index of the start date in the data
        start_index = data.index.get_loc(start_date)
    
        # Find the index of ETH in the all_assets array
        eth_index = np.where(all_assets == 'ETH')[0][0]
    
        for start in range(start_index, len(data)):
            end = start + 1
            period_data = data[start:end]
            
            if start % rebalancing_frequency == 0 and start != start_index:
                historical_returns = np.log(data[:start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
    
                if historical_returns.shape[0] == 0 or historical_returns.shape[1] == 0:
                    print(f"No returns available for historical period up to {start}")
                    continue
    
                sortino_ratios = historical_returns.apply(self.calculate_sortino_ratio)
    
                if sortino_ratios.isnull().any():
                    print(f"Sortino ratios contain NaN for historical period up to {start}")
                    continue
    
                optimal_weights = self.mvo_sortino(historical_returns.values, sortino_ratios, eth_index)
    
                # Convert weights to standard notation for readability
                current_composition = np.round(current_composition, decimals=10)
                optimal_weights = np.round(optimal_weights, decimals=10)
    
                # Apply threshold to significant changes only
                weight_changes = np.abs(current_composition - optimal_weights)
                significant_changes = weight_changes >= self.threshold
    
                print(f"Period {start} to {end}:")
                print(f"  Current composition: {[f'{weight:.10f}' for weight in current_composition]}")
                print(f"  Optimal weights: {[f'{weight:.10f}' for weight in optimal_weights]}")
                print(f"  Weight changes: {[f'{change:.10f}' for change in weight_changes]}")
    
    
                if np.any(significant_changes):
                    current_composition[significant_changes] = optimal_weights[significant_changes]
    
                    # Normalize weights to ensure they sum to 1
                    current_composition /= current_composition.sum()
    
            else:
                # Apply price changes to the current composition
                current_prices = data.iloc[start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                previous_prices = data.iloc[start-1][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                log_returns = np.log(current_prices / previous_prices)
                current_composition = (current_composition * np.exp(log_returns))
                print('current_comp w/ price change', current_composition)
                current_composition /= current_composition.sum()
                print('normalized current_comp w/ price change', current_composition)
    
            rebalanced_data.loc[period_data.index, [f'COMPOSITION_{asset}' for asset in all_assets]] = current_composition
        
        rebalanced_data = rebalanced_data.iloc[:-1]
        
        return rebalanced_data
"""
# Assuming combined_all_assets is already loaded and processed
all_assets = mvo_combined_assets

# Set rebalancing frequency (e.g., 1 for daily, 7 for weekly, 30 for monthly)
rebalancing_frequency = 1  # Change this to the desired frequency

# Perform rebalancing using MVO with Sortino ratio target
rebalanced_data = rebalance(combined_all_assets, all_assets, rebalancing_frequency)

# Calculate daily portfolio returns with rebalancing costs
mvo_daily_portfolio_returns = calculate_daily_portfolio_returns(rebalanced_data, all_assets)

# Calculate cumulative return for the rebalanced data from the panama_dao_start_date
cumulative_return = calculate_cumulative_return(mvo_daily_portfolio_returns, panama_dao_start_date)

# Print the first few rows of the rebalanced data and cumulative return to verify
print(rebalanced_data.head())
print(rebalanced_data.tail())

# Prepare base return
base_return = cumulative_return.reset_index()
base_return.columns = ['DAY', 'base_cumulative_return']

# Normalize returns
first_value = base_return['base_cumulative_return'].iloc[0]
base_return['PanamaDAO_treasury_return'] = 100 + (100 * (base_return['base_cumulative_return'] - first_value))

# Final output
normalized_returns = base_return[['DAY', 'PanamaDAO_treasury_return']]
normalized_returns.set_index('DAY', inplace=True)

# Print the first few rows of normalized returns
print(normalized_returns.head())


# In[64]:


rebalanced_data
# Define a threshold
threshold = 1e-5

# Replace values below the threshold with 0
cleaned_rebalanced_data = rebalanced_data.applymap(lambda x: 0 if abs(x) < threshold else x)
cleaned_rebalanced_data


# In[65]:


mvo_daily_portfolio_returns


# In[66]:


mvo_daily_portfolio_returns = mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= panama_dao_start_date]


# In[67]:


mvo_daily_portfolio_returns.max()


# In[68]:


sortino(mvo_daily_portfolio_returns)


# In[69]:


print("Cumulative Return:\n", cumulative_return.head())
print("Cumulative Return:\n", cumulative_return.tail())
cumulative_return.plot()

mvo_cumulative_return = cumulative_return.copy()


# In[70]:


print("Normalized Returns:\n", normalized_returns.head())
print("Normalized Returns:\n", normalized_returns.tail())
normalized_returns.plot()

mvo_normalized_returns = normalized_returns.copy()


# In[71]:


composition_columns = [f'COMPOSITION_{asset}' for asset in mvo_combined_assets]
mvo_comp = rebalanced_data[composition_columns]
mvo_comp


# In[72]:


# Define a threshold
threshold = 1e-5

# Replace values below the threshold with 0
mvo_comp = mvo_comp.applymap(lambda x: 0 if abs(x) < threshold else x)
mvo_comp


# In[73]:


mvo_comp.tail(50)
print(mvo_comp.columns)


# In[74]:


# Plot the latest composition as a bar chart


# Plot the stacked bar chart
# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 8))

mvo_comp.plot(kind='bar', stacked=True, ax=ax)

# Improve the x-axis labels
plt.title('MVO Portfolio Composition Over Time')
plt.xlabel('Date')
plt.ylabel('Composition')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(ticks=range(0, len(pivot_data_filtered), 30), rotation=45)  # Show x-axis labels every 30 days
plt.tight_layout()
plt.show()


# In[75]:


latest_comp_mvo = mvo_comp.iloc[-1]
# Plot the pie chart

def mvo_composition(): 
    plt.figure(figsize=(10, 7))
    latest_comp_mvo.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Composition')
    plt.ylabel('')  # Hide the y-label
    return plt.show()
    
mvo_composition()


# In[76]:


comp_columns = [col for col in pivot_data_filtered.columns if col.startswith('COMPOSITION_')]


latest_historical_comp = pivot_data_filtered[comp_columns].iloc[-2]

def historical_composition():
    plt.figure(figsize=(10, 7))
    latest_historical_comp.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Composition')
    plt.ylabel('')  # Hide the y-label
    return plt.show()
"""