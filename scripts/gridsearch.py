import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta
from datetime import datetime

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

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
import plotly.subplots as sp
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
import plotly.subplots as sp
import plotly.graph_objs as go

from sklearn.model_selection import KFold, GridSearchCV


# Disable scientific notation globally
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = False
mpl.rcParams['axes.formatter.limits'] = (-5, 6)

def apply_scalar_formatter(ax):
    for axis in [ax.xaxis, ax.yaxis]:
        if isinstance(axis.get_major_formatter(), ScalarFormatter):
            axis.set_major_formatter(ScalarFormatter())
            axis.get_major_formatter().set_scientific(False)
            axis.get_major_formatter().set_useOffset(False)

#from vault_data_processing import test_data, targets, features, temporals, dai_ceilings, vault_names, key_mapping
from vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return, evaluate_multi_predictions, abbreviate_number

test_data = pd.read_csv('test_data.csv')
# Convert index to datetime and ensure no timezone information


test_data.set_index('day', inplace=True)
test_data.index = pd.to_datetime(test_data.index).tz_localize(None)


targets = ['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']
features = ['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',

'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo','mcap_total_volume','defi_apy_medianAPY', 'defi_apy_avg7day', 'dpi_market_Volume',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 
'RWA Vault_dai_ceiling',
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma',
'btc_market_Volume_30d_ma','LP Vault_dai_floor_90d_ma_pct_change'

       ]
temporals = [
 'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 
'stETH Vault_collateral_usd % of Total_7d_ma', 
'BTC Vault_collateral_usd % of Total_7d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 
'LP Vault_collateral_usd % of Total_7d_ma', 
'PSM Vault_collateral_usd % of Total_7d_ma', 
'RWA Vault_collateral_usd % of Total_7d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling'
]
dai_ceilings = [
            'ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 
    'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','RWA Vault_dai_ceiling','PSM Vault_dai_ceiling'
        ]

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


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class RL_VaultSimulator:
    def __init__(self, data, initial_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250, alpha=100, lp_volatility_scale=1.0):
        self.data = data[data.index <= pd.to_datetime(start_date).tz_localize(None)]
        self.features = features
        self.targets = targets
        self.alpha = alpha
        self.model = None
        self.temporals = temporals
        self.results = pd.DataFrame()
        self.initial_data = initial_data
        self.start_date = pd.to_datetime(start_date).tz_localize(None)
        self.end_date = pd.to_datetime(end_date).tz_localize(None)
        self.current_date = self.start_date
        self.dai_ceilings_history = pd.DataFrame()
        self.volatility_window = volatility_window
        self.scale_factor = scale_factor
        self.minimum_value_factor = minimum_value_factor
        self.lp_volatility_scale = lp_volatility_scale  # Add new parameter


    def get_latest_data(self):
        # Return the latest data that the environment will use
        return self.data
        
    def reset(self):
        self.data = self.initial_data[self.initial_data.index <= self.start_date]
        self.results = pd.DataFrame()
        self.train_model()
        print('sim reset current date', self.current_date)
        print("Simulation reset and model retrained.")

    def train_model(self):
        X = self.initial_data[self.features]
        y = self.initial_data[self.targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        self.model = MultiOutputRegressor(Ridge(alpha=self.alpha))
        self.model.fit(X_train, y_train)
        print("Model trained.")
        
    def update_dai_ceilings(self):
        # Extract the current DAI ceilings
        current_ceilings = self.data.loc[self.current_date, ['ETH Vault_dai_ceiling', 'BTC Vault_dai_ceiling', 'stETH Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling', 'Stablecoin Vault_dai_ceiling', 'LP Vault_dai_ceiling','RWA Vault_dai_ceiling','PSM Vault_dai_ceiling']]
        current_ceilings['timestamp'] = self.current_date  # Adding a timestamp for reference

        # Append to the historical DataFrame
        self.dai_ceilings_history = pd.concat([self.dai_ceilings_history, current_ceilings.to_frame().T.set_index('timestamp')])



    def run_simulation(self, simulation_date, action=None):
        # Ensure the date is timezone-aware. Localize if it is naive.
        #if pd.to_datetime(simulation_date).tzinfo is None:
           # self.current_date = pd.to_datetime(simulation_date).tz_localize('UTC')
        #else:
            #self.current_date = pd.to_datetime(simulation_date).tz_convert('UTC')
    
        cycle_start_date = self.current_date
        end_date = min(cycle_start_date + timedelta(days=24), self.end_date)
    
        while self.current_date <= end_date and self.current_date <= self.initial_data.index.max():
            if self.current_date in self.data.index:
                X_test = self.data.loc[[self.current_date], self.features]
            else:
                X_test = self.data.tail(1)[self.features]
    
            volatilities = self.calculate_historical_volatility()
            predictions = self.forecast(X_test, volatilities)
            print('predictions',predictions)
            #print('predcitions nan', predictions.isna().sum().sum())
            future_index = pd.DatetimeIndex([self.current_date])
            print('future index', future_index)
            self.update_state(future_index, predictions)
    
            # Update the DAI ceilings history right after updating the state
            self.update_dai_ceilings()
    
            print('current state', self.data.iloc[-1])
            if action:
                self.apply_action(action)
                print(f"Action applied on: {self.current_date}")
    
            print(f"Day completed: {self.current_date}")
            self.current_date += timedelta(days=1)
    
        print(f"Cycle completed up to: {self.current_date - timedelta(days=1)}")
        if self.current_date > self.end_date:
            print(f"Simulation completed up to: {self.end_date}")
        else:
            print(f"Simulation completed up to: {self.current_date - timedelta(days=1)}")




    def apply_action(self, action):
        base_value_if_zero = 10000  # Base value to set if the initial DAI ceiling is zero
        if action:
            for vault, percentage_change in action.items():
                # Append the suffix '_dai_ceiling' to the vault name to match the DataFrame columns
                dai_ceiling_key = vault.replace('_collateral_usd', '_dai_ceiling') 
                print('vault', vault)
                print('Applying action to:', dai_ceiling_key)
                if dai_ceiling_key in self.data.columns:
                    original_value = self.data[dai_ceiling_key].iloc[-1]
                    if original_value == 0:
                        # If the original value is 0, initialize it with the base value
                        new_value = base_value_if_zero * (1 + percentage_change / 100)
                        print(f"Initialized and adjusted {dai_ceiling_key} from 0 to {new_value}")
                    else:
                        new_value = original_value * (1 + percentage_change / 100)
                        print(f"Adjusted {dai_ceiling_key} by {percentage_change}% from {original_value} to {new_value}")

                    self.data.at[self.data.index[-1], dai_ceiling_key] = new_value
                else:
                    print(f"No 'dai_ceiling' column found for {dai_ceiling_key}, no action applied.")
        else:
            print("No action provided; no adjustments made.")


    def forecast(self, X, volatilities):
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)

        scale_factor = self.scale_factor
        noise = np.random.normal(0, volatilities * scale_factor, predictions.shape)
        minimum_value = self.minimum_value_factor * self.initial_data[self.targets].mean()
        adjusted_predictions = np.maximum(predictions + noise, minimum_value)

        # Apply specific adjustments for the LP vault
        lp_vault_index = self.targets.index('LP Vault_collateral_usd')
        if lp_vault_index is not None:
            lp_volatility_cap = volatilities[lp_vault_index] * scale_factor * self.lp_volatility_scale  # Use lp_volatility_scale
            lp_noise = np.random.normal(0, lp_volatility_cap, adjusted_predictions[:, lp_vault_index].shape)
            adjusted_predictions[:, lp_vault_index] = np.maximum(predictions[:, lp_vault_index] + lp_noise, minimum_value[lp_vault_index])

        return adjusted_predictions


        # try scale 453000000, min val .12 or .1, window 25 or 15


    def calculate_historical_volatility(self):
        window = self.volatility_window
    
        # Assuming daily data, calculate percentage change
        daily_returns = self.data[self.targets].pct_change()
    
        # Handling possible NaN values in daily returns
        daily_returns = daily_returns.dropna()
    
        # Calculate volatility as the standard deviation of returns
        volatility = daily_returns.rolling(window=window, min_periods=1).std()
        print('all vol before', volatility.describe())
    
        # Adjust volatility for the LP vault
        lp_vault = 'LP Vault_collateral_usd'
        print('lp vol before', volatility[lp_vault])
        if lp_vault in volatility.columns:
            # Apply the lp_volatility_scale factor for the LP vault
            volatility[lp_vault] *= self.lp_volatility_scale
            print('lp vol after', volatility[lp_vault])
    
        print('all vol after', volatility.describe())
    
        # Return the average volatility over the window
        return volatility.mean(axis=0)  # Use axis=0 to average volatilities across columns if neededis=0 to average volatilities across columns if needed



    
    def update_state(self, indices, predictions):
        #print('Current state', self.data[self.targets].iloc[-1])
        #print('Current temporal', self.data[self.temporals].iloc[-1])
        # Create a new DataFrame for the predictions
        new_data = pd.DataFrame(predictions, index=indices, columns=self.targets)
        new_data = new_data.clip(lower=0)
        #print('New data', new_data)
        self.results = pd.concat([self.results, new_data])  # Append new data to results
        self.data.update(new_data)
        # Append new data if the index does not already exist
        if not self.data.index.isin(indices).any():
            # Directly assign the new data to the respective index positions
            self.data = self.data.reindex(self.data.index.union(new_data.index), method='nearest')
            for column in new_data.columns:
                self.data.loc[new_data.index, column] = new_data[column]
            self.data.sort_index(inplace=True)  # Ensure the index is sorted
        else:
            # If indices overlap, directly update the values
            self.data.update(new_data)
        #print('new state update:',self.data[self.targets].iloc[-1],self.data[self.temporals].iloc[-1])
    
        # Recalculate temporal features right after updating the state
        self.recalculate_temporal_features(indices)


    

    def recalculate_temporal_features(self, start_index):
        vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault', 'RWA Vault']
        total_usd_col = 'Vaults Total USD Value'
        self.data[total_usd_col] = self.data[[f'{vault}_collateral_usd' for vault in vault_names]].sum(axis=1)
    
        for vault in vault_names:
            usd_col = f'{vault}_collateral_usd'
            pct_col = f'{vault}_collateral_usd % of Total'
            self.data[pct_col] = self.data[usd_col] / self.data[total_usd_col]  # Update the percentage column
            # Calculate the 7-day moving average for the USD collateral
            #ma_col_usd_7d = f'{usd_col}_7d_ma'
            #self.data[ma_col_usd_7d] = self.data[usd_col].rolling(window=7, min_periods=1).mean()
            ma_col_usd_30d = f'{usd_col}_30d_ma'
            self.data[ma_col_usd_30d] = self.data[usd_col].rolling(window=30, min_periods=1).mean()
            # Calculate the 7-day and 30-day moving averages for the percentage of total
            for window in [7,30]:
                ma_col_pct = f'{pct_col}_{window}d_ma'
                self.data[ma_col_pct] = self.data[pct_col].rolling(window=window, min_periods=1).mean()
             
            dai_ceiling_col = f'{vault}_dai_ceiling'
            if dai_ceiling_col in self.data.columns:
                prev_dai_ceiling_col = f'{vault}_prev_dai_ceiling'
                self.data[prev_dai_ceiling_col] = self.data[dai_ceiling_col].shift(1)
    def print_summary_statistics(self, pre_data):
        for column in self.data.columns:
            if column not in pre_data.columns:
                pre_data[column] = pd.NA  # Handle missing column in pre_data
            pre_stats = pre_data.describe()
            post_stats = self.data.describe()
            print(f"--- {column} ---")
            print("Pre-Simulation:\n", pre_stats)
            print("Post-Simulation:\n", post_stats, "\n")
            
    def plot_vault_data(self, column):
        vault_usd_col = f'{column}_collateral_usd'
        vault_dai_col = f'{column}_dai_ceiling'
        
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
        
        # Plot USD balance on the primary y-axis
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data[vault_usd_col], name=f'{column} USD Balance', line=dict(color='blue')),
            secondary_y=False
        )
        
        # Plot DAI ceiling on the secondary y-axis
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data[vault_dai_col], name=f'{column} DAI Ceiling', line=dict(color='orange')),
            secondary_y=True
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="USD Balance", secondary_y=False, tickcolor='blue')
        fig.update_yaxes(title_text="DAI Ceiling", secondary_y=True, tickcolor='orange')
        
        # Update layout
        fig.update_layout(
            title_text=f"Time Series for {column}",
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
        )
        
        return fig
        
        
    def plot_simulation_results(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        for target in self.targets:
            ax.plot(self.results.index, self.results[target], label=target)
        ax.set_title("Simulation Results")
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True)
        #ax.ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(ax)
        plt.show()

    def plot_dai_ceilings_and_usd_balances(self, start_simulation_date, vault_names):
        if isinstance(start_simulation_date, str):
            start_simulation_date = pd.to_datetime(start_simulation_date)
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        
        for vault in vault_names:
            axes[0].plot(self.data.index, self.data[f'{vault} Vault_dai_ceiling'], label=f'{vault} Dai Ceiling')
        axes[0].axvline(x=start_simulation_date, color='r', linestyle='--', label='Start of Simulation')
        axes[0].set_title('Dai Ceilings Over Time')
        axes[0].set_ylabel('Dai Ceiling')
        axes[0].legend()
        #axes[0].ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(axes[0])
        
        for vault in vault_names:
            axes[1].plot(self.data.index, self.data[f'{vault} Vault_collateral_usd'], label=f'{vault} USD Balance')
        axes[1].axvline(x=start_simulation_date, color='r', linestyle='--', label='Start of Simulation')
        axes[1].set_title('USD Balances Per Vault Over Time')
        axes[1].set_ylabel('USD Balance')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        #axes[1].ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(axes[1])
        
        plt.show()

    def calculate_error_metrics(self, actual_data):
        vault_names = ['ETH', 'stETH', 'BTC', 'Altcoin', 'Stablecoin', 'LP', 'PSM']
        for vault in vault_names:
            column = f'{vault} Vault_collateral_usd'
            try:
                mse = mean_squared_error(actual_data[column], self.data[column])
                mae = mean_absolute_error(actual_data[column], self.data[column])
                rmse = sqrt(mse)
                print(f"--- Metrics for {vault} Vault ---")
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
                print(f"RMSE: {rmse}\n")
            except KeyError:
                print(f"Data for {vault} Vault not available in the dataset.")


test_data_copy = test_data.copy()

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import numpy as np

def tune_model_hyperparameters(simulator, param_grid):
    X = simulator.initial_data[simulator.features]
    y = simulator.initial_data[simulator.targets]

    model = MultiOutputRegressor(Ridge())

    # Define grid search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    
    # Fit grid search
    grid_search.fit(X, y)

    # Update simulator model with best parameters
    simulator.model = grid_search.best_estimator_
    simulator.alpha = grid_search.best_params_['estimator__alpha']
    best_alpha = grid_search.best_params_
    return best_alpha

    print("Best parameters found for the model: ", grid_search.best_params_)

# Example usage
model_param_grid = {
    'estimator__alpha': [100, 300, 500],
}

beginning_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


simulation_data = test_data_copy  # Assuming this is defined with your actual data
start_date = '2022-05-20'
end_date = '2024-03-20'  # Define an end date

start_date = pd.to_datetime(start_date).tz_localize(None)
end_date = pd.to_datetime(end_date).tz_localize(None)



simulation = RL_VaultSimulator(simulation_data, simulation_data, features, targets, temporals, start_date, end_date)
simulation.train_model()
best_alpha = tune_model_hyperparameters(simulation, model_param_grid)

# Manually set simulator-specific parameters
simulation.scale_factor = 300000000  # Example value
simulation.minimum_value_factor = 0.05  # Example value
simulation.volatility_window = 250  # Example value

# Run the simulation
simulation.run_simulation(start_date)
#simulation.plot_simulation_results()

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def custom_grid_search(simulator, param_grid):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_params = None
    best_score = float('inf')

    for scale_factor in param_grid['scale_factor']:
        for minimum_value_factor in param_grid['minimum_value_factor']:
            for window in param_grid['volatility_window']:
                for lp_volatility_scale in param_grid['lp_volatility_scale']:  # New parameter for LP volatility scale
                    scores = []

                    for train_index, test_index in kf.split(simulator.initial_data):
                        train_data = simulator.initial_data.iloc[train_index]
                        test_data = simulator.initial_data.iloc[test_index]

                        temp_simulator = RL_VaultSimulator(
                            train_data,
                            simulator.initial_data,
                            simulator.features,
                            simulator.targets,
                            simulator.temporals,
                            simulator.start_date,
                            simulator.end_date,
                            scale_factor=scale_factor,
                            minimum_value_factor=minimum_value_factor,
                            volatility_window=window,
                            lp_volatility_scale=lp_volatility_scale  # Pass the new parameter
                        )

                        temp_simulator.train_model()
                        temp_simulator.run_simulation(simulator.start_date)

                        # Reindex results to match test_data index
                        y_pred = temp_simulator.results.reindex(test_data.index)
                        y_true = test_data[simulator.targets]

                        # Drop rows with NaN values that might have been introduced by reindexing
                        common_index = y_pred.dropna().index
                        y_pred = y_pred.loc[common_index]
                        y_true = y_true.loc[common_index]

                        score = mean_squared_error(y_true, y_pred)
                        scores.append(score)

                    mean_score = np.mean(scores)
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = {
                            'scale_factor': scale_factor,
                            'minimum_value_factor': minimum_value_factor,
                            'volatility_window': window,
                            'lp_volatility_scale': lp_volatility_scale  # Include new parameter in best_params
                        }

    return best_params, best_score


# Usage example
simulation_data = test_data_copy
if simulation_data.index.tz is not None:
    simulation_data.index = simulation_data.index.tz_convert(None)
else:
    simulation_data.index = pd.to_datetime(simulation_data.index)

# Now remove timezone information
simulation_data.index = simulation_data.index.tz_localize(None)

# Define the parameter grid
param_grid = {
    'scale_factor': [10000000, 50000000, 100000000, 300000000, 453000000],
    'minimum_value_factor': [0.001, 0.008, 0.05, 0.1, 0.12],
    'volatility_window': [60, 150, 180, 250, 280, 360, 380],
    'lp_volatility_scale': [0.5, 0.65, 0.7, 0.75, 1.0]  # Add lp_volatility_scale options here
}


# Usage example
start_date = '2022-05-20'
end_date = '2024-03-20'

simulation = RL_VaultSimulator(simulation_data, simulation_data, features, targets, temporals, start_date, end_date)
simulation.train_model()
best_params, best_score = custom_grid_search(simulation, param_grid)

finish_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Assuming grid_search is the object from GridSearchCV
results = {
    "best_params": best_params,
    "best_score": best_score,
    "best_alpha": best_alpha,
    "beginning_timestamp": beginning_timestamp,
    "finish_timestamp": finish_timestamp
}

# Convert to DataFrame
results_df = pd.DataFrame([results])

# Save to CSV
results_df.to_csv('grid_search_results.csv', index=False)


print("Best parameters found: ", best_params), print("Best score: ", best_score), print("Best gridsearch params", best_alpha)