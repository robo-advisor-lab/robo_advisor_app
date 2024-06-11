import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scripts.treasury_utils import calculate_sortino_ratio, mvo, calculate_log_returns

class mvo_model():
    def __init__(self, eth_bound, current_risk_free, start_date, end_date, threshold=0):
        self.threshold = threshold
        self.current_risk_free = current_risk_free
        self.start_date = start_date
        self.end_date = end_date 
        self.eth_bound = eth_bound
    
    def calculate_sortino_ratio(self, returns):
        risk_free = self.current_risk_free
        daily_risk_free_rate = (1 + risk_free) ** (1/365) - 1
        excess_returns = returns - daily_risk_free_rate
        downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
        daily_downside_deviation = np.sqrt(downside_returns.mean())
    
        if np.isnan(daily_downside_deviation):
            daily_downside_deviation = 0.0
    
        active_days = returns.notna().sum()
        annual_factor = 365 / active_days
        compounding_return = (1 + excess_returns).prod() ** annual_factor - 1
        annual_downside_deviation = daily_downside_deviation * np.sqrt(365)
        sortino_ratio = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else 0.0
        print('sortino ratio', sortino_ratio)
    
        return sortino_ratio
    
    def mvo_sortino(self, returns, sortino_ratios, eth_index, seed):
        n = returns.shape[1]
        random.seed(seed)
        np.random.seed(seed)
        print('seed', seed)
        
        # Perturb sortino ratios slightly to introduce randomness
        perturbation = np.random.normal(loc=0, scale=0.01, size=sortino_ratios.shape)
        print('perturbation', perturbation)
        perturbed_sortino_ratios = sortino_ratios + perturbation
        
        weights = cp.Variable(n)
        portfolio_return = returns @ weights
        sortino_matrix = np.tile(perturbed_sortino_ratios.values.reshape(1, -1), (len(returns), 1))
        
        portfolio_risk = cp.norm(returns - cp.multiply(sortino_matrix, cp.reshape(weights, (1, n))), 'fro')
        objective = cp.Maximize(cp.sum(portfolio_return) - 0.001 * portfolio_risk)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        if self.eth_bound > 0:
            constraints.append(weights[eth_index] >= self.eth_bound)
        
        for i in range(n):
            if i != eth_index:
                constraints.append(weights[i] <= 0.3)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        
        if weights.value is not None:
            return weights.value
        else:
            return None



    def rebalance_portfolio(self, data, weights):
        compositions = np.outer(np.ones(len(data)), weights)
        return compositions
    
    def calculate_daily_portfolio_returns(self, data, all_assets):
        all_assets = np.array(all_assets)
        current_prices = data[[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
        print('current_prices', current_prices)
        prev_prices = data[[f'DAILY_PRICE_{asset}' for asset in all_assets]].shift(1).fillna(method='ffill').values
        print('prev_prices', prev_prices)
        log_returns = np.log(current_prices / prev_prices)
        print('log returns', log_returns)
        composition_columns = [f'COMPOSITION_{asset}' for asset in all_assets]
        daily_portfolio_returns = (log_returns * data[composition_columns].shift(1).fillna(0)).sum(axis=1)
        return daily_portfolio_returns
    
    def calculate_cumulative_return(self, daily_portfolio_returns):
        daily_portfolio_returns = daily_portfolio_returns[daily_portfolio_returns.index >= self.start_date]
        cumulative_return = np.exp(np.log1p(daily_portfolio_returns).cumsum()) - 1
        return cumulative_return
    
    def rebalance(self, data, all_assets, rebalancing_frequency=7, seed=42):
        all_assets = np.array(all_assets)
        data_start = data.index.min()
        data = data.sort_index()
        data = data.loc[(data.index >= data_start) & (data.index <= self.end_date)]
        rebalanced_data = data.copy()
    
        # Calculate the initial optimal weights
        historical_returns = np.log(data[:data.index.get_loc(self.start_date)][[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
        if historical_returns.shape[0] > 0 and historical_returns.shape[1] > 0:
            sortino_ratios = historical_returns.apply(self.calculate_sortino_ratio)
            if not sortino_ratios.isnull().any():
                eth_index = np.where(all_assets == 'ETH')[0][0]
                initial_optimal_weights = self.mvo_sortino(historical_returns.values, sortino_ratios, eth_index, seed)
                if initial_optimal_weights is not None:
                    initial_composition = initial_optimal_weights
                else:
                    initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
            else:
                initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
        else:
            initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
    
        current_composition = initial_composition.copy()
        start_index = data.index.get_loc(self.start_date)
    
        for start in range(start_index, len(data)):
            end = start + 1
            period_data = data[start:end]
            if (start - start_index) % rebalancing_frequency == 0 and start != start_index:
                historical_returns = np.log(data[:start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
                if historical_returns.shape[0] == 0 or historical_returns.shape[1] == 0:
                    continue
                sortino_ratios = historical_returns.apply(self.calculate_sortino_ratio)
                if sortino_ratios.isnull().any():
                    continue
                optimal_weights = self.mvo_sortino(historical_returns.values, sortino_ratios, eth_index, seed)
                if optimal_weights is None:
                    continue
    
                current_composition = np.round(current_composition, decimals=10)
                optimal_weights = np.round(optimal_weights, decimals=10)
    
                weight_changes = np.abs(current_composition - optimal_weights)
                significant_changes = weight_changes >= self.threshold
    
                if np.any(significant_changes):
                    current_composition[significant_changes] = optimal_weights[significant_changes]
                    current_composition /= current_composition.sum()
            else:
                current_prices = data.iloc[start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                previous_prices = data.iloc[start-1][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                log_returns = np.log(current_prices / previous_prices)
                current_composition = (current_composition * np.exp(log_returns))
                current_composition /= current_composition.sum()
    
    
        rebalanced_data = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[:-1]
    
        return rebalanced_data

