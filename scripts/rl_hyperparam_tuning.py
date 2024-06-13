import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from scripts.vault_data_processing import test_data, features, temporals, targets
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment
from scripts.vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return, evaluate_multi_predictions, abbreviate_number
from models.vault_rl_agent import RlAgent

initial_data = test_data.copy()

# Vault action ranges
vault_action_ranges = {
    'stETH Vault_dai_ceiling': [-0.5, 0.5],
    'ETH Vault_dai_ceiling': [-1, 1],
    'BTC Vault_dai_ceiling': [-1, 1],
    'Altcoin Vault_dai_ceiling': [-0.25, 0.25],
    'Stablecoin Vault_dai_ceiling': [-0.2, 0.2],
    'LP Vault_dai_ceiling': [-0.15, 0.15],
    'RWA Vault_dai_ceiling': [0, 0],
    'PSM Vault_dai_ceiling': [-1, 1]
}

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

# Define the parameter grid for exploration rate, adjustment factor, and action space
param_grid = {
    'exploration_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9],
    'adjustment_scale': [0.5, 1, 5, 10, 15, 20],
    'action_space': [vault_action_ranges]  # You can expand this if you want to test different ranges
}

# Create a list of all parameter combinations
grid = list(ParameterGrid(param_grid))

def evaluate_parameters(params, simulator, start_date, end_date, target_weights, bounds):
    # Initialize the agent with the given parameters
    agent = RlAgent(
        action_space=params['action_space'],
        target_weights=target_weights,
        vault_action_ranges=vault_action_ranges,
        exploration_rate=params['exploration_rate'],
        adjustment_scale=params['adjustment_scale']
    )
    
    # Initialize the simulation environment
    environment = SimulationEnvironment(
        simulator=simulator,
        start_date=start_date,
        end_date=end_date,
        agent=agent,
        bounds=bounds
    )
    
    # Run the simulation
    state, reward, done, info = environment.run()
    
    # Return the final reward as the performance metric
    return reward

# Initialize your simulator with appropriate data
# simulator = RL_VaultSimulator(...)

# Define the start and end dates for the simulation
start_date = '2022-05-20'
end_date = '2024-03-20'

start_date = pd.to_datetime(start_date).tz_localize(None)
end_date = pd.to_datetime(end_date).tz_localize(None)


test_data.index = pd.to_datetime(test_data.index).tz_localize(None)

# Initialize the simulator
simulator = RL_VaultSimulator(
    data=test_data,
    initial_data=test_data,
    features=features,
    targets=targets,
    temporals=temporals,
    start_date=start_date,
    end_date=end_date
)

# Define the target weights based on the initial strategy pie chart provided
target_weights = {
    'ETH Vault_collateral_usd': 0.20,
    'PSM Vault_collateral_usd': 0.20,
    'BTC Vault_collateral_usd': 0.10,
    'stETH Vault_collateral_usd': 0.30,
    'Altcoin Vault_collateral_usd': 0.05,
    'Stablecoin Vault_collateral_usd': 0.05,
    'LP Vault_collateral_usd': 0.05,
    'RWA Vault_collateral_usd': 0.05
}

# Run the grid search
best_params = None
best_reward = -np.inf

for params in grid:
    reward = evaluate_parameters(params, simulator, start_date, end_date, target_weights, bounds)
    if reward > best_reward:
        best_reward = reward
        best_params = params


results = {
    "best_params": best_params,
    "best_reward": best_reward
}

# Convert to DataFrame
results_df = pd.DataFrame([results])

# Save to CSV
results_df.to_csv('data/csv/rl_grid_search_results.csv', index=False)


print(f"Best parameters: {best_params}")
print(f"Best reward: {best_reward}")
