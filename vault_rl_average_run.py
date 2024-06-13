import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta

# Machine learning tools
from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import mutual_info_regression

# Deep Learning tools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
import torch

# Additional tools
from scipy import signal
from scipy.optimize import minimize
from itertools import combinations, product

# External data and APIs
import yfinance as yf
#from dune_client.client import DuneClient
import requests
import streamlit as st

from scripts.vault_data_processing import test_data as test_data_copy, test_data, targets, features, temporals, dai_ceilings
from models.vault_rl_agent import RlAgent
from scripts.vault_utils import historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return, mvo 
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

# Disable scientific notation globally
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = False
mpl.rcParams['axes.formatter.limits'] = (-5, 6)

# Function to apply ScalarFormatter globally
def set_global_scalar_formatter():
    for axis in ['x', 'y']:
        plt.gca().ticklabel_format(axis=axis, style='plain', useOffset=False)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

pd.options.display.float_format = '{:.0f}'.format
# Register the formatter with a default plot

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


sim_cutoff = pd.to_datetime('2024-03-20 00:00:00').tz_localize(None)
test_data.index = pd.to_datetime(test_data.index).tz_localize(None)

historical_sim = test_data[test_data.index <= sim_cutoff]
historical_sim = historical_sim[targets].merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')

bounds =  {
            'BTC Vault_collateral_usd': (0.1, 0.3),
            'ETH Vault_collateral_usd': (0.1, 0.5),
            'stETH Vault_collateral_usd': (0.2, 0.3),
            'Stablecoin Vault_collateral_usd': (0.0, 0.05),
            'Altcoin Vault_collateral_usd': (0.0, 0.05),
            'LP Vault_collateral_usd': (0.05, 0.05),
            'RWA Vault_collateral_usd': (0.05, 0.05),
            'PSM Vault_collateral_usd': (0.05, 0.2)
        }

historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_sim, bounds)



# In[912]:


historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)


# In[913]:


historical_returns = calc_cumulative_return(historical_portfolio_daily_returns)





def run_simulation(seed, bounds):
    set_random_seed(seed)
    simulation_data = test_data_copy
    
    portfolio_mvo_weights, portfolio_returns, portfolio_composition, total_portfolio_value = mvo(simulation_data, bounds)
    
    
    # In[882]:
    
    
    # Assuming you have these vault names as keys in optimized_weight_dict
    vault_names = [
        'BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 'Stablecoin Vault_collateral_usd', 
        'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd'
    ]
    
    # Map from the detailed keys to simplified keys used in optimized_weight_dict
    key_mapping = {
        'BTC Vault_collateral_usd',
        'ETH Vault_collateral_usd',
        'stETH Vault_collateral_usd',
        'Stablecoin Vault_collateral_usd'
        'Altcoin Vault_collateral_usd',
        'LP Vault_collateral_usd',
        'RWA Vault_collateral_usd',
        'PSM Vault_collateral_usd'
    }
    
    # Apply this mapping to the initial_weights to align with optimized_weight_dict
    initial_weights = dict(portfolio_composition.loc['2022-05-20'].to_dict())
    
    optimized_weight_dict = dict(zip(vault_names, portfolio_mvo_weights))
    
    # Now both dictionaries use the same keys:
    print("Initial Weights on 2022-05-20:", initial_weights)
    print("Optimized Weights:", optimized_weight_dict)
    
    # Your RlAgent can now be initialized and used with these dictionaries
    
    
    # In[883]:
    
    
    
    
    
    # ### Sets Up Action Space: for Some Reason, need to run this first for decent predictions on first try: think related to how some variables are initialized
    
    # In[884]:
    
    
    str(simulation_data.index.max())
    
    
    # test_ceilings = test_data[dai_ceilings]
    # 
    # 
    # start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
    # historical_cutoff = start_date_dt - pd.DateOffset(days=1)
    # test_historical_data = test_ceilings[test_ceilings.index <= historical_cutoff]
    # 
    # test_historical_data
    # last_dai_ceiling = test_historical_data.iloc[-1]
    # last_dai_ceiling
    
    # In[885]:
    
    
    #test_data_copy[['mcap_total_volume']].plot()
    
    
    # ### Non Q Sim
    
    # In[886]:
    
    """
vault_action_ranges = {
        'stETH Vault_dai_ceiling': [-0.5, 0.5],
        'ETH Vault_dai_ceiling': [-0.5, 0.5],
        'BTC Vault_dai_ceiling': [-0.5, 0.5], # can try -1 for BTC
        'Altcoin Vault_dai_ceiling': [-0.25, 0.25],
        'Stablecoin Vault_dai_ceiling': [-0.25, 0.25],
        'LP Vault_dai_ceiling': [-0.15, 0.15],
        'RWA Vault_dai_ceiling': [0, 0],  # If no changes are allowed
        'PSM Vault_dai_ceiling': [-0.5, 0.5]
    }
    """
    
    
    
    
    # Define action space as a list of dictionaries
    vault_action_ranges = {
        'stETH Vault_dai_ceiling': [-0.5, 0.5],
        'ETH Vault_dai_ceiling': [-1, 0.5],
        'BTC Vault_dai_ceiling': [-1, 0.5],
        'Altcoin Vault_dai_ceiling': [-0.15,0.15],
        'Stablecoin Vault_dai_ceiling': [-0.1, 0.1],
        'LP Vault_dai_ceiling': [-0.1,0.1],
        'RWA Vault_dai_ceiling': [0, 0],
        'PSM Vault_dai_ceiling': [-1, 0.5]
    }
    
    action_space = generate_action_space(vault_action_ranges)
    
    # Assuming `optimized_weight_dict` and `initial_weights` are defined elsewhere in your script
    agent = RlAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
    initial_action = agent.initial_strategy(initial_weights)
    
    print("Initial Action Computed:", initial_action)
    
    
    simulation_data = test_data_copy  
    simulation_data.index = simulation_data.index.tz_localize(None)  # Remove timezone information
    #05-20-2022
    start_date = '2022-05-20'
    end_date = '2024-03-20'
    #end_date = '2022-07-20'
    
    
    simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=453000000, minimum_value_factor=0.008, volatility_window=250)
    simulator.train_model()
    #simulator.set_parameters()
    
    environment = SimulationEnvironment(simulator, start_date, end_date, agent, bounds= bounds)
    environment.reset()
    
    state, reward, done, info = environment.run()
    
    #simulator.plot_simulation_results()
    action_df = environment.generate_action_dataframe()
    
    
    # ### DAI Ceilings
    # 
    
    # In[888]:
    
    
    sim_dai_ceilings = simulator.dai_ceilings_history
   
    
    
    # In[889]:
    
    

    
    
    # ### Actions Log
    
    # In[890]:
    historical = test_data_copy[targets]
    
    start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
    historical_cutoff = start_date_dt - pd.DateOffset(days=1)
    historical_data = historical[historical.index <= historical_cutoff]
    
    test_data.index = pd.to_datetime(test_data.index).tz_localize(None)
    
    
    historical_data = historical_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
    historical_data
    
    historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_data, bounds)
    historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)
    
    
    # In[891]:
    
    
    action_df.set_index('date', inplace=True)
    sortino_timeseries = action_df['current sortino ratio'].dropna()
    
    
    # In[892]:
    
    
    
    
    
    # In[898]:
    
    
    
    # In[900]:
    
    
    rl_sim_results = simulator.results
    #evaluate_predictions(rl_sim_results, historical)
    
    
    # ### Data Prep for MVO
    
    # In[901]:
    
    
    start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
    historical_cutoff = start_date_dt - pd.DateOffset(days=1)
    
    
    # In[902]:
    
    
    historical_ceilings = test_data_copy[dai_ceilings]
    ceiling_historical_cutoff = historical_cutoff.tz_localize(None)
    
    historical_ceilings_for_sim =  historical_ceilings[historical_ceilings.index <= ceiling_historical_cutoff]
    historical_ceilings_for_sim
    combined_ceiling_data = pd.concat([historical_ceilings_for_sim, sim_dai_ceilings])
   
    
    
    # In[906]:
    
    
    historical_data = historical[historical.index <= historical_cutoff]
    rl_combined_data = pd.concat([historical_data, rl_sim_results])
    #mvo_combined_data = pd.concat([historical_data, mvo_sim_results])
    #print(mvo_combined_data)
    print(rl_combined_data)
    
    
    # In[907]:
    
    
    rl_combined_data.index = rl_combined_data.index.tz_localize(None)
    #mvo_combined_data.index = mvo_combined_data.index.tz_localize(None)
    
    test_data['RWA Vault_collateral_usd'].index = test_data['RWA Vault_collateral_usd'].index.tz_localize(None)
    
    
    #Since RWA is not a target, we need to add back in for MVO calculations
    rl_sim_w_RWA = rl_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
    #mvo_sim_w_RWA = mvo_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
    
    # Optional: Sort the DataFrame by index if it's not already sorted
    rl_sim_w_RWA.sort_index(inplace=True)
    #mvo_sim_w_RWA.sort_index(inplace=True)
    
    

    
    
    # In[909]:
    
    
    sim_cutoff = rl_sim_w_RWA.index.max()
    sim_cutoff
    
    
    # In[910]:
    
    
    # Assuming 'test_data' is the DataFrame with the timezone-aware index
    #test_data.index = pd.to_datetime(test_data.index).tz_localize(None)
    
    # Now perform the merge
    historical_sim = test_data[test_data.index <= sim_cutoff]
    historical_sim = historical_sim[targets].merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
 
    
    
    historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_sim, bounds)
  
    
    
    # In[912]:
    
    
    historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)
    print(f'historical tvl as of {historical_total_portfolio_value.index.max()} : {historical_total_portfolio_value.iloc[-1]}')
    print(f'historical soritno ratio as of {historical_total_portfolio_value.index.max()} : {historical_sortino_ratio}')
    
    
    
    # In[913]:
    
    
    historical_returns = calc_cumulative_return(historical_portfolio_daily_returns)
    print(f'historical cumulative return as of {historical_returns.index.max()} : {historical_returns.iloc[-1]}')
    
    
    rl_portfolio_mvo_weights, rl_portfolio_returns, rl_portfolio_composition, rl_total_portfolio_value = mvo(rl_sim_w_RWA, bounds)
    
    
    rl_sim_portfolio_daily_returns,  rl_sim_downside_returns, rl_sim_excess_returns, rl_sim_sortino_ratio = historical_sortino(rl_portfolio_returns,rl_portfolio_composition)
    
    
    # In[918]:
    
    
    rl_optimized_returns = calc_cumulative_return(rl_sim_portfolio_daily_returns)
    



    
    # Drop duplicates, keeping the first occurrence
    rl_optimized_returns = rl_optimized_returns[~rl_optimized_returns.index.duplicated(keep='first')]
    #
    return rl_optimized_returns, rl_total_portfolio_value, rl_sim_sortino_ratio, sortino_timeseries

def main():
    num_runs = 10
    seeds = [5, 10, 15, 20, 40, 100, 200, 300, 500, 800]
    cumulative_returns = []
    tvls = []
    sortino_ratios = []
    sortino_timeseries_list = []

    for seed in seeds[:num_runs]:
        cumulative_return, tvl, sortino_ratio, sortino_timeseries = run_simulation(seed, bounds)
        cumulative_returns.append(cumulative_return)
        tvls.append(tvl)
        sortino_ratios.append(sortino_ratio)
        sortino_timeseries_list.append(sortino_timeseries)
        print('seed used', seed)
    
    avg_cumulative_return = np.mean(cumulative_returns)
    avg_tvl = np.mean(tvls)
    avg_sortino_ratio = np.mean(sortino_ratios)

    print(f"num of runs:{num_runs}")
    print(f"Average Cumulative Return: {avg_cumulative_return}")
    print(f"Average TVL: {avg_tvl}")
    print(f"Average Sortino Ratio: {avg_sortino_ratio}")
    
    results = {
        "num runs": num_runs,
        "seeds": seeds[:num_runs],
        "avg rl cumulative return": avg_cumulative_return,
        "average rl TVL": avg_tvl,
        "average rl sortino ratio": avg_sortino_ratio
    }
    
    # Convert to DataFrame
    results_df = pd.DataFrame([results])
    
    # Save to CSV
    results_df.to_csv('data/csv/vault_rl_average_run_results.csv', index=False) 

    # Plot the results using Plotly
    colors = sns.color_palette("husl", num_runs)

    # Plot cumulative returns
    traces = []
    for i in range(num_runs):
        traces.append(go.Scatter(
            x=cumulative_returns[i].index,
            y=cumulative_returns[i].values,
            mode='lines+markers',
            name=f'Run {i + 1}',
            line=dict(color=f'rgb{colors[i]}')
        ))
    traces.append(go.Scatter(
        x=historical_returns.index,
        y=historical_returns.values,
        mode='lines',
        name='Historical Cumulative Return',
        line=dict(color='black', dash='dash')
    ))
    
    layout = go.Layout(
        title='Cumulative Returns per Run',
        xaxis=dict(title='Run'),
        yaxis=dict(title='Cumulative Return'),
        legend=dict(x=0.1, y=0.9)
    )
    fig = go.Figure(data=traces, layout=layout)
    pyo.iplot(fig)

    # Plot TVL
    traces = []
    for i in range(num_runs):
        traces.append(go.Scatter(
            x=tvls[i].index,
            y=tvls[i].values,
            mode='lines+markers',
            name=f'Run {i + 1}',
            line=dict(color=f'rgb{colors[i]}')
        ))
    traces.append(go.Scatter(
        x=historical_total_portfolio_value.index,
        y=historical_total_portfolio_value.values,
        mode='lines',
        name='Historical TVL',
        line=dict(color='black', dash='dash')
    ))
    
    layout = go.Layout(
        title='TVL per Run',
        xaxis=dict(title='Run'),
        yaxis=dict(title='TVL'),
        legend=dict(x=0.1, y=0.9)
    )
    fig = go.Figure(data=traces, layout=layout)
    pyo.iplot(fig)

    # Plot Sortino ratios
    traces = []
    for i in range(num_runs):
        traces.append(go.Scatter(
            x=sortino_timeseries_list[i].index,
            y=sortino_timeseries_list[i].values,
            mode='lines+markers',
            name=f'Run {i + 1}',
            line=dict(color=f'rgb{colors[i]}')
        ))
    
    layout = go.Layout(
        title='Sortino Ratio Timeseries per Run',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Sortino Ratio'),
        legend=dict(x=0.1, y=0.9)
    )
    fig = go.Figure(data=traces, layout=layout)
    pyo.iplot(fig)

if __name__ == "__main__":
    main()