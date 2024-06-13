from scripts.treasury_data_processing import combined_all_assets, current_risk_free

import numpy as np
import pandas as pd
import cvxpy as cp



class PortfolioOptimizer:
    def __init__(self, eth_bound=0):
        self.eth_bound = eth_bound
        self.threshold = 1e-5
        self.current_risk_free = current_risk_free

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

    def mvo_sortino(self, returns, sortino_ratios, eth_index):
        n = returns.shape[1]
        print('n for weights', n)
        
        # Validate the shape and content of returns and sortino_ratios
        print(f'Returns shape: {returns.shape}')
        print(f'Returns head:\n{returns[:5]}')
        print(f'Sortino ratios shape: {sortino_ratios.shape}')
        print(f'Sortino ratios head:\n{sortino_ratios[:5]}')
        print(f'ETH index: {eth_index}')
        
        weights = cp.Variable(n)
        portfolio_return = returns @ weights
        sortino_matrix = np.tile(sortino_ratios.values.reshape(1, -1), (len(returns), 1))
        
        # Log the sortino_matrix and reshaped weights for debugging
        print(f'Sortino matrix shape: {sortino_matrix.shape}')
        print(f'Sortino matrix head:\n{sortino_matrix[:5]}')
        
        portfolio_risk = cp.norm(returns - cp.multiply(sortino_matrix, cp.reshape(weights, (1, n))), 'fro')
        objective = cp.Maximize(cp.sum(portfolio_return) - 0.001 * portfolio_risk)  # Adding regularization term
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= 0.3  # Setting an upper bound for each weight
        ]
        
        # Add the constraint for ETH only if eth_bound is greater than 0
        if self.eth_bound > 0:
            constraints.append(weights[eth_index] >= self.eth_bound)

        # Log constraints for debugging
        print(f'Constraints: {constraints}')
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)  # Using Clarabel as suggested in the warning
        
        # Log problem status and weights
        print(f'Problem status: {problem.status}')
        print(f'Weights value: {weights.value}')
        
        if weights.value is not None:
            return weights.value
        else:
            return None

    def rebalance(self, data, all_assets, rebalancing_frequency=7):
        all_assets = np.array(all_assets)
        data_start = data.index.min()
        data = data.sort_index()
        data = data.loc[(data.index >= data_start) & (data.index <= self.end_date)]
        rebalanced_data = data.copy()
    
        # Calculate the initial optimal weights
        historical_returns = np.log(data[:data.index.get_loc(self.start_date)][[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
        print(f"Initial historical_returns index start {historical_returns.index.min()} through {historical_returns.index.max()}")
    
        if historical_returns.shape[0] > 0 and historical_returns.shape[1] > 0:
            sortino_ratios = historical_returns.apply(self.calculate_sortino_ratio)
            print("Initial sortino ratios", sortino_ratios)
    
            if not sortino_ratios.isnull().any():
                eth_index = np.where(all_assets == 'ETH')[0][0]
                initial_optimal_weights = self.mvo_sortino(historical_returns.values, sortino_ratios, eth_index)
                print('Initial optimal weights', initial_optimal_weights)
    
                if initial_optimal_weights is not None:
                    # Set the initial composition to the initial optimal weights
                    initial_composition = initial_optimal_weights
                else:
                    print("No initial optimal weights found, using initial composition from data")
                    initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
            else:
                print("Sortino ratios contain NaN, using initial composition from data")
                initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
        else:
            print("No returns available for initial historical period, using initial composition from data")
            initial_composition = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values
    
        current_composition = initial_composition.copy()
        start_index = data.index.get_loc(self.start_date)
        print('start index', start_index)
    
        for start in range(start_index, len(data)):
            end = start + 1
            period_data = data[start:end]
            if (start - start_index) % rebalancing_frequency == 0 and start != start_index:
                historical_returns = np.log(data[:start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].pct_change().dropna() + 1)
                print(f"historical_returns index start {historical_returns.index.min()} through {historical_returns.index.max()}")
    
                if historical_returns.shape[0] == 0 or historical_returns.shape[1] == 0:
                    print(f"No returns available for historical period up to {start}")
                    continue
                print(f"historical_returns index start {historical_returns.index.min()} through {historical_returns.index.max()}")
    
                sortino_ratios = historical_returns.apply(self.calculate_sortino_ratio)
                print("sortino ratios", sortino_ratios)
    
                if sortino_ratios.isnull().any():
                    print(f"Sortino ratios contain NaN for historical period up to {start}")
                    continue
    
                optimal_weights = self.mvo_sortino(historical_returns.values, sortino_ratios, eth_index)
                print('optimal weights', optimal_weights)
    
                # Ensure optimal_weights is not None
                if optimal_weights is None:
                    print(f"No optimal weights found for historical period up to {start}")
                    continue
    
                current_composition = np.round(current_composition, decimals=10)
                optimal_weights = np.round(optimal_weights, decimals=10)
    
                weight_changes = np.abs(current_composition - optimal_weights)
                significant_changes = weight_changes >= self.threshold
    
                print(f"Period {start} to {end}:")
                print(f"  Current composition: {[f'{weight:.10f}' for weight in current_composition]}")
                print(f"  Optimal weights: {[f'{weight:.10f}' for weight in optimal_weights]}")
                print(f"  Weight changes: {[f'{change:.10f}' for change in weight_changes]}")
    
                if np.any(significant_changes):
                    current_composition[significant_changes] = optimal_weights[significant_changes]
                    current_composition /= current_composition.sum()
            else:
                current_prices = data.iloc[start][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                previous_prices = data.iloc[start-1][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values
                log_returns = np.log(current_prices / previous_prices)
                current_composition = (current_composition * np.exp(log_returns))
                current_composition /= current_composition.sum()
    
            rebalanced_data.loc[period_data.index, [f'COMPOSITION_{asset}' for asset in all_assets]] = current_composition
    
        rebalanced_data = rebalanced_data[rebalanced_data.index >= self.start_date].iloc[:-1]
    
        return rebalanced_data

# Initialize the PortfolioOptimizer class
optimizer = PortfolioOptimizer(eth_bound=0.1)  # Example ETH bound

# Example data and assets
data = combined_all_assets

all_assets = ['ETH', 'WBTC', 'WETH']
optimizer.start_date = '2023-03-22'
optimizer.end_date = '2023-04-05'

# Rebalance the portfolio
rebalanced_data = optimizer.rebalance(data, all_assets, 1)
print(rebalanced_data)
rebalanced_data.plot()
