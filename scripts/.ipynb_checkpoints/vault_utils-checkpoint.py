import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta
import plotly.graph_objs as go


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

def mvo(data, bounds, annual_risk_free_rate=0.05, penalty_factor=10000):  # Increased penalty factor
    print(f"Bounds in mvo: {bounds}")  # Debug print
    portfolio = data[['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 
                      'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 
                      'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']]
    log_returns = np.log(portfolio / portfolio.shift(1))
    log_returns.fillna(0, inplace=True)
    returns = log_returns
    
    total_portfolio_value = portfolio.sum(axis=1)
    composition = portfolio.divide(total_portfolio_value, axis=0)
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    excess_returns = returns - daily_risk_free_rate
    downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
    daily_downside_deviation = np.sqrt(downside_returns.mean())
    active_days = returns.notna().sum().max()
    annual_factor = 365 / active_days
    compounding_return = (1 + excess_returns).prod() ** annual_factor - 1
    annual_downside_deviation = daily_downside_deviation * np.sqrt(365)
    sortino_ratios = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else np.inf
    
    print("Individual Sortino Ratios:", sortino_ratios)

    def sortino_ratio(weights):
        portfolio_returns = np.sum(weights * excess_returns, axis=1)
        excess_portfolio_returns = portfolio_returns - daily_risk_free_rate
        annualized_returns = (1 + excess_portfolio_returns).prod() ** (365 / len(excess_portfolio_returns)) - 1
        downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - daily_risk_free_rate) ** 2)) * np.sqrt(365)
        
        lower_bounds = np.array([bounds[vault][0] for vault in portfolio.columns])
        upper_bounds = np.array([bounds[vault][1] for vault in portfolio.columns])
        penalties = penalty_factor * (np.sum(np.maximum(0, lower_bounds - weights) ** 2) +
                                     np.sum(np.maximum(0, weights - upper_bounds) ** 2))
        
        return -(annualized_returns / downside_deviation) + penalties if downside_deviation != 0 else -np.inf

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.clip(composition.iloc[-1].values, a_min=[bounds[vault][0] for vault in portfolio.columns], 
                              a_max=[bounds[vault][1] for vault in portfolio.columns])

    result = minimize(sortino_ratio, initial_weights, method='SLSQP', bounds=[bounds[vault] for vault in portfolio.columns], constraints=constraints)
    if result.success:
        optimized_weights = result.x
        print('Optimized weights:', optimized_weights)
        return optimized_weights, log_returns, composition, total_portfolio_value
    else:
        raise Exception('Optimization did not converge')


        

def optimized_sortino(returns_df, weights, annual_risk_free_rate=0.05):
    """
    Calculate the Sortino Ratio for a given set of returns and weights.
    
    Parameters:
        returns_df (DataFrame): DataFrame of returns.
        weights (array): Asset weights in the portfolio.
        annual_risk_free_rate (float): Annual risk-free rate, default is 5%.
        
    Returns:
        tuple: Contains daily returns, downside returns, excess returns, and the Sortino Ratio.
    """
    # Calculate daily risk-free rate
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    
    # Portfolio daily returns
    daily_returns = returns_df.dot(weights)
    
    # Excess returns over the risk-free rate
    excess_returns = daily_returns - daily_risk_free_rate
    
    # Calculate downside returns
    downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
    
    # Daily and annualized downside deviation
    daily_downside_deviation = np.sqrt(downside_returns.mean())
    annualized_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
    # Annualized excess return calculation
    if len(excess_returns) > 0:
        annualized_excess_return = ((1 + excess_returns).prod())**(365 / len(excess_returns)) - 1
    else:
        annualized_excess_return = -1  # Handle case with no returns
    
    # Calculate Sortino Ratio
    target_sortino_ratio = (annualized_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else np.inf)
    
    # Debugging prints can be commented out in production
    print('Daily downside deviation:', daily_downside_deviation)
    print('Annualized downside deviation:', annualized_downside_deviation)
    print('Annualized excess return:', annualized_excess_return)
    print("Sortino Ratio:", target_sortino_ratio)

    return daily_returns, downside_returns, excess_returns, target_sortino_ratio


# In[805]:


def historical_sortino(returns, composition, annual_risk_free_rate=0.05):
    """
    Calculate the Sortino Ratio for a portfolio based on historical returns and asset composition.
    
    Parameters:
        returns (DataFrame): DataFrame of returns.
        composition (DataFrame): DataFrame of asset weights in the portfolio.
        annual_risk_free_rate (float): Annual risk-free rate, default is 5%.
        
    Returns:
        tuple: Contains portfolio daily returns, downside returns, excess returns, and the Sortino Ratio.
    """
    # Calculate daily risk-free rate
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    
    # Portfolio daily returns
    portfolio_daily_returns = (returns * composition).sum(axis=1)
    
    # Excess returns over the risk-free rate
    excess_returns = portfolio_daily_returns - daily_risk_free_rate
    
    # Calculate downside returns
    downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
    
    # Daily and annualized downside deviation
    daily_downside_deviation = np.sqrt(downside_returns.mean())
    annualized_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
    # Annualized excess return calculation
    if len(excess_returns) > 0:
        annualized_excess_return = ((1 + excess_returns).prod())**(365 / len(excess_returns)) - 1
    else:
        annualized_excess_return = -1  # Handle case with no returns
    
    # Calculate Sortino Ratio
    sortino_ratio = annualized_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else np.inf
    
    # Debugging prints can be commented out in production
    print('Daily downside deviation:', daily_downside_deviation)
    print('Annualized downside deviation:', annualized_downside_deviation)
    print('Annualized excess return:', annualized_excess_return)
    print("Portfolio Sortino Ratio:", sortino_ratio)

    return portfolio_daily_returns, downside_returns, excess_returns, sortino_ratio


# In[806]:


def visualize_mvo_results(daily_returns, downside_returns, excess_returns):

    # 1. Time Series Plot of Portfolio Returns
    plt.figure(figsize=(10, 6))
    daily_returns.plot(title='Daily Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True)
    plt.show()
    
    # 2. Histogram of Portfolio Returns
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, kde=True, bins=30)
    plt.title('Histogram of Portfolio Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()
    
    # 3. Cumulative Returns Plot
    optimized_cumulative_returns = (1 + daily_returns).cumprod()
    plt.figure(figsize=(10, 6))
    optimized_cumulative_returns.plot(title='Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

    downside_returns = pd.Series(downside_returns, index=daily_returns.index)
    
    # 4. Downside Returns Plot
    plt.figure(figsize=(10, 6))
    downside_returns.plot(style='ro', title='Downside Returns Below MAR')
    plt.axhline(0, color='k', linestyle='--')  # Add a line at 0 for reference
    plt.xlabel('Date')
    plt.ylabel('Downside Returns')
    plt.grid(True)
    plt.show()
    
    excess_returns = pd.Series(excess_returns, index=daily_returns.index)

    # 5. Excess Returns Over MAR
    excess_returns.plot(style='go', title='Excess Returns Over MAR')
    plt.axhline(0, color='k', linestyle='--')  # Add a line at MAR for reference
    plt.xlabel('Date')
    plt.ylabel('Excess Returns')
    plt.grid(True)
    plt.show()

    return optimized_cumulative_returns

def calc_cumulative_return(daily_returns):
    optimized_cumulative_returns = (1 + daily_returns).cumprod() - 1
    return optimized_cumulative_returns

def evaluate_predictions(predictions, historical):
    # Ensure indexes are properly aligned
    predictions.index = pd.to_datetime(predictions.index).tz_localize(None)
    historical.index = pd.to_datetime(historical.index).tz_localize(None)
    
    if not predictions.index.equals(historical.index):
        print("Warning: Indexes do not match, aligning them...")
        # Align the data by index
        combined = predictions.join(historical, lsuffix='_pred', rsuffix='_hist', how='inner')
    else:
        combined = predictions.join(historical, lsuffix='_pred', rsuffix='_hist')
    
    vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault']
    metrics = {}

    
    
    for vault in vault_names:
        pred_col = f'{vault}_collateral_usd_pred'
        hist_col = f'{vault}_collateral_usd_hist'
        
        if pred_col in combined.columns and hist_col in combined.columns:
            mse = mean_squared_error(combined[hist_col], combined[pred_col])
            mae = mean_absolute_error(combined[hist_col], combined[pred_col])
            rmse = np.sqrt(mse)
            r2 = r2_score(combined[hist_col], combined[pred_col])
            metrics[vault] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(combined.index, combined[hist_col], label='Historical', marker='o')
            plt.plot(combined.index, combined[pred_col], label='Predicted', linestyle='--', marker='x')
            plt.title(f'{vault} Collateral USD Comparison')
            plt.xlabel('Date')
            plt.ylabel('Collateral USD')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"Missing columns for {vault}")
            metrics[vault] = 'Missing data'
    
    return metrics

def evaluate_multi_predictions(rl_results, mvo_results, historical):
    # Ensure indexes are properly aligned
    rl_results.index = pd.to_datetime(rl_results.index).tz_localize(None)
    mvo_results.index = pd.to_datetime(mvo_results.index).tz_localize(None)
    historical.index = pd.to_datetime(historical.index).tz_localize(None)
    
    # Align the data by index
    combined_rl = rl_results.join(historical, lsuffix='_pred', rsuffix='_hist', how='inner')
    combined_mvo = mvo_results.join(historical, lsuffix='_pred', rsuffix='_hist', how='inner')
    
    vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault']
    metrics = {'RL': {}, 'MVO': {}}
    figures = {}

    for vault in vault_names:
        pred_col_rl = f'{vault}_collateral_usd_pred'
        pred_col_mvo = f'{vault}_collateral_usd_pred'
        hist_col = f'{vault}_collateral_usd_hist'
        
        if pred_col_rl in combined_rl.columns and hist_col in combined_rl.columns:
            mse_rl = mean_squared_error(combined_rl[hist_col], combined_rl[pred_col_rl])
            mae_rl = mean_absolute_error(combined_rl[hist_col], combined_rl[pred_col_rl])
            rmse_rl = np.sqrt(mse_rl)
            r2_rl = r2_score(combined_rl[hist_col], combined_rl[pred_col_rl])
            metrics['RL'][vault] = {'MSE': mse_rl, 'MAE': mae_rl, 'RMSE': rmse_rl, 'R2': r2_rl}
        
        if pred_col_mvo in combined_mvo.columns and hist_col in combined_mvo.columns:
            mse_mvo = mean_squared_error(combined_mvo[hist_col], combined_mvo[pred_col_mvo])
            mae_mvo = mean_absolute_error(combined_mvo[hist_col], combined_mvo[pred_col_mvo])
            rmse_mvo = np.sqrt(mse_mvo)
            r2_mvo = r2_score(combined_mvo[hist_col], combined_mvo[pred_col_mvo])
            metrics['MVO'][vault] = {'MSE': mse_mvo, 'MAE': mae_mvo, 'RMSE': rmse_mvo, 'R2': r2_mvo}

            # Plotting with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined_rl.index, y=combined_rl[hist_col], mode='lines+markers', name='Historical', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=combined_rl.index, y=combined_rl[pred_col_rl], mode='lines+markers', name='RL Model', line=dict(color='blue', dash='dash')))
            fig.add_trace(go.Scatter(x=combined_mvo.index, y=combined_mvo[pred_col_mvo], mode='lines+markers', name='MVO Model', line=dict(color='green', dash='dot')))
            fig.update_layout(title=f'{vault} Collateral USD Comparison', xaxis_title='Date', yaxis_title='Collateral USD')
            figures[vault] = fig
        else:
            metrics['RL'][vault] = 'Missing data'
            metrics['MVO'][vault] = 'Missing data'
    
    return metrics, figures

def generate_action_space(vault_action_ranges):
    action_space = {}
    for vault, limits in vault_action_ranges.items():
        if isinstance(limits, list):  # If specific steps are provided, use them
            action_space[vault] = limits
        else:  # Generate a range between min and max with the given step
            min_val, max_val, step = limits
            action_space[vault] = list(np.arange(min_val, max_val + step, step))
    return action_space

def plot_cv_results(x_st, y_st, model, cv, title_base="CV Fold"):
    fig, axes = plt.subplots(nrows=cv.get_n_splits(), ncols=1, figsize=(10, 20), sharex=True)
    
    for idx, (train_index, test_index) in enumerate(cv.split(x_st)):
        X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
        y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

        # Fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot
        ax = axes[idx]
        ax.plot(X_test.index, y_test, label='Actual', color='blue', marker='o')
        ax.plot(X_test.index, y_pred, label='Predicted', linestyle='--', color='red', marker='x')
        ax.set_title(f"{title_base} {idx+1} - MAE: {mean_absolute_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_multioutput_cv_results(X, y, n_splits=5, alpha=100, title="CV Results with Ridge Regularization"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fig, axes = plt.subplots(nrows=n_splits, ncols=1, figsize=(15, 3 * n_splits))

    # Using Ridge with regularization
    for idx, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Multioutput with Ridge regression
        model = MultiOutputRegressor(Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')

        ax = axes[idx]
        # Assuming the first output for simplicity in plotting
        ax.plot(y_test.index, y_test.iloc[:, 0], label='Actual', marker='o', linestyle='-', color='blue')
        ax.plot(y_test.index, y_pred[:, 0], label='Predicted', marker='x', linestyle='--', color='red')
        ax.set_title(f"{title} {idx+1} - MAE: {mae[0]:.2f}, R²: {r2[0]:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, title_base):
    # Assuming y_train and y_test are DataFrame with multiple columns for each target
    num_targets = y_train.shape[1]  # Number of target variables
    
    for i in range(num_targets):
        # Create DataFrames for plotting
        target_name = y_train.columns[i]
        train_df = pd.DataFrame({
            'Actual': y_train.iloc[:, i],
            'Predicted': y_train_pred[:, i]
        }, index=X_train.index)
        
        test_df = pd.DataFrame({
            'Actual': y_test.iloc[:, i],
            'Predicted': y_test_pred[:, i]
        }, index=X_test.index)
        
        # Combine and sort the DataFrames
        combined_df = pd.concat([train_df, test_df])
        combined_df.sort_index(inplace=True)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(combined_df.index, combined_df['Actual'], label='Actual Values', color='blue')
        plt.plot(combined_df.index, combined_df['Predicted'], label='Predicted Values', linestyle='--', color='red')
        plt.title(f"{title_base} for {target_name}")
        plt.xlabel('Date')
        plt.ylabel('Collateral in USD')
        plt.legend()
        plt.show()



# In[700]:


def plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, title):
    y_train_df = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred}, index=X_train.index)
    y_test_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}, index=X_test.index)
    
    # Combine and sort the DataFrames
    combined_df = pd.concat([y_train_df, y_test_df])
    combined_df.sort_index(inplace=True)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df.index, combined_df['Actual'], label='Actual Values', color='blue')
    plt.plot(combined_df.index, combined_df['Predicted'], label='Predicted Values', linestyle='--', color='orange')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Collateral in USD')
    plt.legend()
    plt.show()

def abbreviate_number(num):
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.2f} K"
    else:
        return f"{num:.2f}"