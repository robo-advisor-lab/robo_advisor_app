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

# Additional tools
from scipy import signal
from scipy.optimize import minimize
from itertools import combinations, product

# External data and APIs
import yfinance as yf
import requests
import streamlit as st

from scripts.vault_utils import mvo, historical_sortino, optimized_sortino, visualize_mvo_results, calc_cumulative_return

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

# Register the formatter with a default plot
plt.figure()
set_global_scalar_formatter()
plt.close()
"""
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"""
dai_ceilings = [
            'ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 
    'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','RWA Vault_dai_ceiling','PSM Vault_dai_ceiling'
        ]

class SimulationEnvironment:
    def __init__(self, simulator, start_date, end_date, agent=None, bounds=None):
        self.simulator = simulator
        self.start_date = pd.to_datetime(start_date).tz_localize(None)
        self.end_date = pd.to_datetime(end_date).tz_localize(None)
        self.current_date = self.start_date 
        self.agent = agent
        self.action_log = []
        self.dai_ceiling = None
        self.previous_total_portfolio_value = None  # Initialize previous total portfolio value
        self.value_change_factor = 1
        self.bounds = bounds
        
    def run(self, predefined_actions=None):
        state = self.reset()  # Ensures starting from a clean state every time
        done = False
        reward = 0
        while self.current_date <= self.end_date:
            next_cycle_date = min(self.current_date + pd.DateOffset(days=24), self.end_date)
            print(f"Running simulation from {self.current_date} to {next_cycle_date}")
            
            if predefined_actions:
                action = predefined_actions
            else:
                action = self.agent.agent_policy(state, self.dai_ceiling) if self.agent else None
    
            if action:
                print(f"Action at {self.current_date}: {action}")
                self.action_log.append({'date': self.current_date, 'action': action})
            else:
                print(f"No action taken at {self.current_date}")
    
            self.simulator.run_simulation(self.current_date.strftime('%Y-%m-%d'), action)
            state = self.get_state(self.current_date)
            print('state in run as of self.current_date', state)
            reward, current_weights, optimized_weights = self.reward_function()  # Get the updated optimized weights
            print(f"Reward at the end of the cycle on {next_cycle_date}: {reward}")
    
            # Update target_weights for the agent
            if self.agent:
                self.agent.target_weights = optimized_weights
    
            self.current_date = next_cycle_date + pd.DateOffset(days=1)
            if self.current_date > self.end_date:
                done = True
                print(f"Reached or passed end date: stopping simulation at {self.current_date}")
                break
            
            print(f"Completed cycle up to {next_cycle_date}")
        return state, reward, done, {}




    def reset(self):
        self.simulator.reset()
        self.current_date = self.start_date
        print("Environment and simulator reset.")
        self.action_log.append({'date': self.current_date, 'env reset start date': self.start_date})
        self.simulator.current_date = self.current_date
        return self.get_state(self.current_date)

    def get_state(self, simulation_date):
        print('self date', self.current_date)
        print('self.simulator date', self.simulator.current_date)
        self.current_date = self.simulator.current_date
        print('new self date', self.current_date)
        simulation_date = self.simulator.current_date

        if simulation_date in self.simulator.data.index:
            state_data = self.simulator.data.loc[simulation_date, self.simulator.targets]
        else:
            previous_dates = self.simulator.data.index[self.simulator.data.index < simulation_date]
            if not previous_dates.empty:
                last_available_date = previous_dates[-1]
                state_data = self.simulator.data.loc[last_available_date, self.simulator.targets]
            else:
                state_data = self.simulator.data[self.simulator.targets].iloc[-1]

        total_value = state_data.sum()
        self.action_log.append({'date': self.current_date, 'get state calculated portfolio value': total_value})
        relative_weights = state_data / total_value if total_value != 0 else state_data
        print('relative weights:', relative_weights)
        print(f"Current state fetched for date {simulation_date}: {relative_weights}")
        return relative_weights.to_dict()

    def run_daily_simulation(self, simulation_date):
        volatilities = self.simulator.calculate_historical_volatility()
        if simulation_date in self.simulator.data.index:
            X_test = self.simulator.data.loc[[simulation_date], self.simulator.features]
        else:
            X_test = self.simulator.data.loc[self.simulator.data.index < simulation_date, self.simulator.features].iloc[-1].to_frame().T

        predictions = self.simulator.forecast(X_test, volatilities)
        self.simulator.update_state(pd.DatetimeIndex([simulation_date]), predictions)
   
    def step(self, action=None):
        reward = 0
        if action:
            print('action', action)
            self.simulator.apply_action(action)
            self.action_log.append({'date': self.current_date, 'action': action})

        if not hasattr(self, 'current_simulation_date') or self.current_simulation_date < self.start_date:
            self.current_simulation_date = self.start_date
        
        self.current_simulation_date += pd.DateOffset(days=1)
        
        if self.current_simulation_date <= self.end_date:
            self.run_daily_simulation(self.current_simulation_date)
            done = False
        else:
            done = True
            reward, current_weights = self.reward_function()

        current_state = self.get_state(self.current_simulation_date)
        return current_state, reward, done, {}

    def apply_action(self, action):
        for vault_name, adjustment in action.items():
            ...

        self.action_log.append({
            'date': self.current_date,
            'action': action,
            'additional_info': {}
        })

    def generate_action_dataframe(self):
        return pd.DataFrame(self.action_log)

    def reward_function(self):
        print('Simulator data for MVO:', self.simulator.data[self.simulator.targets])
        optimized_weights, returns, composition, total_portfolio_value = mvo(self.simulator.data, self.bounds)
        self.action_log.append({'date': self.current_date, 'mvo calculated current portfolio value': total_portfolio_value.iloc[-1]})
        self.action_log.append({'date': self.current_date, 'calculated optimized weights': optimized_weights})
        self.action_log.append({'date': self.current_date, 'current composition': composition.iloc[-1]})
    
        current_weights = composition.iloc[-1].to_dict()
        target_weights = {k: v for k, v in zip(composition.columns, optimized_weights)}
        print('Target weights (no agent):', target_weights)
    
        if self.agent is not None:
            self.agent.target_weights = target_weights
            self.action_log.append({'date': self.current_date, 'agent weights updated to': self.agent.target_weights})
            print('Agent weights updated:', self.agent.target_weights)
    
        current_daily_returns, current_downside_returns, current_excess_returns, current_sortino_ratio = historical_sortino(returns, composition)
        self.action_log.append({'date': self.current_date, 'current sortino ratio': current_sortino_ratio})
        target_daily_returns, target_downside_returns, target_excess_returns, target_sortino_ratio = optimized_sortino(returns, optimized_weights)
        self.action_log.append({'date': self.current_date, 'target sortino ratio': target_sortino_ratio})
        self.action_log.append({'date': self.current_date, 'current weights': current_weights})
        self.action_log.append({'date': self.current_date, 'target weights': target_weights})
        print('Current Financials:')
        cumulative_return = calc_cumulative_return(current_daily_returns)
        self.action_log.append({'date': self.current_date, 'current cumulative return': cumulative_return.iloc[-1]})
    
        max_distance = sum(abs(1 - value) for value in target_weights.values())
        distance_penalty = sum(abs(current_weights.get(key, 0) - value) for key, value in target_weights.items()) / max_distance if max_distance != 0 else 0
        self.action_log.append({'date': self.current_date, 'Distance penalty': distance_penalty})
    
        sortino_scale = 100000
        scaled_sortino_diff = (target_sortino_ratio - current_sortino_ratio) / sortino_scale
        print('Sortino ratio:', current_sortino_ratio)
        print('Scaled Sortino diff:', scaled_sortino_diff)
        self.action_log.append({'date': self.current_date, 'Scaled Sortino diff': scaled_sortino_diff})
    
        # Calculate the percentage change in total portfolio value
        if self.previous_total_portfolio_value is not None:
            portfolio_value_change_pct = (total_portfolio_value.iloc[-1] - self.previous_total_portfolio_value) / self.previous_total_portfolio_value
        else:
            portfolio_value_change_pct = 0  # No change if there is no previous value
        
        # Scale the portfolio value change percentage
        scaled_portfolio_value_change = portfolio_value_change_pct * self.value_change_factor
        print('port change pct', portfolio_value_change_pct)
        print('scaled port change', scaled_portfolio_value_change)
        
        # Update the previous total portfolio value
        self.previous_total_portfolio_value = total_portfolio_value.iloc[-1]
        
        # Incorporate the scaled percentage change in total portfolio value into the reward
        reward = scaled_sortino_diff - distance_penalty + scaled_portfolio_value_change
        reward_no_scale = current_sortino_ratio - distance_penalty + scaled_portfolio_value_change
        
        self.action_log.append({'date': self.current_date, 'Reward': reward})
        self.action_log.append({'date': self.current_date, 'Reward no scale': reward_no_scale})
    
        self.dai_ceiling = self.simulator.data[dai_ceilings]
        self.action_log.append({'date': self.current_date, 'dai ceilings': self.dai_ceiling})
        print('reward', reward)
        
        return reward_no_scale, current_weights, target_weights


"""






test_data_copy.index = test_data_copy.index.tz_localize(None)

# Assuming the necessary variables (simulation_data, features, targets, temporals) are defined and correct
simulator_simulation_data = test_data_copy  # Assuming this is correctly defined with your actual data

start_date = '2022-05-20'
end_date = '2022-07-01'
actions = {'stETH Vault_dai_ceiling': 5, 'ETH Vault_dai_ceiling': -5}  # Example action
 
test_simulator = RL_VaultSimulator(simulation_data, simulation_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)

#test_simulator.set_parameters(scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
test_simulator.train_model()

test_environment = SimulationEnvironment(test_simulator, start_date, end_date)
test_environment.run(actions)

test_simulator.plot_simulation_results()


# ### Results of Sim

# In[861]:


test_data_copy[['mcap_total_volume']]


# In[862]:


sim_results = test_simulator.results
sim_results.describe()


# In[863]:


sim_results.index.duplicated()


# ### Backtesting to Historical Data 

# In[864]:


evaluate_predictions(sim_results, historical)


# ### MVO Comparison cleaning

# In[865]:


test_data['RWA Vault_collateral_usd']


# In[866]:


start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
historical_cutoff = start_date_dt - pd.DateOffset(days=1)
historical_cutoff


# In[867]:


historical_data = historical[historical.index <= historical_cutoff]
combined_data = pd.concat([historical_data, sim_results])
print(combined_data)


# In[868]:


combined_data.index = combined_data.index.tz_localize(None)
test_data['RWA Vault_collateral_usd'].index = test_data['RWA Vault_collateral_usd'].index.tz_localize(None)


#Since RWA is not a target, we need to add back in for MVO calculations
sim_w_RWA = combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
# Optional: Sort the DataFrame by index if it's not already sorted
sim_w_RWA.sort_index(inplace=True)

# Now 'combined_data' contains both historical and simulation data in one DataFrame
sim_w_RWA.plot()
sim_w_RWA


# In[869]:


sim_cutoff = sim_w_RWA.index.max()
sim_cutoff


# In[870]:


# Assuming 'test_data' is the DataFrame with the timezone-aware index
test_data.index = pd.to_datetime(test_data.index).tz_localize(None)

# Now perform the merge
historical_sim = test_data[test_data.index <= sim_cutoff]
historical_sim = historical_sim[targets].merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
historical_sim.plot()
historical_sim


# ### Simulation MVO Scores

# In[871]:


# Optimized Weights for Simulation

portfolio_mvo_weights, portfolio_returns, portfolio_composition, total_portfolio_value = mvo(sim_w_RWA)
print('optimized weights:', portfolio_mvo_weights)
print('current composition:', portfolio_composition.iloc[-1])
print(f'current portfolio value: ${total_portfolio_value.iloc[-1]:,.2f}')


# In[872]:


sim_portfolio_daily_returns,  sim_downside_returns, sim_excess_returns, sim_sortino_ratio = historical_sortino(portfolio_returns,portfolio_composition)


# In[873]:


optimized_returns = visualize_mvo_results(sim_portfolio_daily_returns, sim_downside_returns, sim_excess_returns)


# In[874]:


optimized_returns.plot()
print('sim cumulative return', optimized_returns.iloc[-1])


# ### Historical MVO Scores

# In[875]:


historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_sim)
print('average daily return per vault:', historical_portfolio_returns.mean())
print('current composition:', historical_portfolio_composition.iloc[-1])
print(f'current portfolio value: ${historical_total_portfolio_value.iloc[-1]:,.2f}')


# In[876]:


historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)


# In[877]:


historical_returns = visualize_mvo_results(historical_portfolio_daily_returns, historical_downside_returns, historical_excess_returns)


# In[878]:


historical_returns.plot()
print('historical cumulative return', historical_returns.iloc[-1])
"""