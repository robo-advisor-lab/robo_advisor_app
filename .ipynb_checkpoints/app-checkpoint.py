import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib
import tensorflow as tf
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Treasury Advisor
from models.treasury_mvo_model import mvo_model
from models.treasury_rl_model import PortfolioEnv, train_rl_agent
from scripts.treasury_utils import calculate_sortino_ratio, calculate_sortino_ratio, calculate_treynor_ratio, calculate_cagr, calculate_beta, normalize
from scripts.treasury_data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free

# Vault Advisor
from models.vault_mvo_agent import MVOAgent
from models.vault_rl_agent import RlAgent
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment
from scripts.vault_data_processing import test_data, targets, features, temporals, dai_ceilings, vault_names, key_mapping
from scripts.vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space 

# Set random seeds
random_seed = st.sidebar.selectbox('Select Random Seed', [20, 42, 100, 200])
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

## Treasury Advisor

def treasury_advisor(advisor_type, rebalance_frequency):
    if advisor_type == 'Reinforcement Learning':
        rl_rebalancing_frequency = rebalance_frequency
        treasury_start_date = panama_dao_start_date
        actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(combined_all_assets, mvo_combined_assets, current_risk_free, rl_rebalancing_frequency, treasury_start_date)
        return actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df
    else:
        mvo_rebalancing_frequency = rebalance_frequency
        threshold = 0
        model = mvo_model(current_risk_free, panama_dao_start_date, threshold)
        rebalanced_data = model.rebalance(combined_all_assets, mvo_combined_assets, mvo_rebalancing_frequency)
        mvo_daily_portfolio_returns = model.calculate_daily_portfolio_returns(rebalanced_data, mvo_combined_assets)
        mvo_daily_portfolio_returns = mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= panama_dao_start_date]
        mvo_cumulative_return = model.calculate_cumulative_return(mvo_daily_portfolio_returns)
        base_return = mvo_cumulative_return.reset_index()
        base_return.columns = ['DAY', 'base_cumulative_return']
        first_value = base_return['base_cumulative_return'].iloc[0]
        base_return['PanamaDAO_treasury_return'] = 100 + (100 * (base_return['base_cumulative_return'] - first_value))
        mvo_normalized_returns = base_return[['DAY', 'PanamaDAO_treasury_return']]
        mvo_normalized_returns.set_index('DAY', inplace=True)
        return rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns

## Vault Model
simulation_data = test_data
simulation_data.index = pd.to_datetime(simulation_data.index)
simulation_data.index = simulation_data.index.tz_localize(None)

def vault_advisor(advisor_type):
    portfolio_mvo_weights, portfolio_returns, portfolio_composition, total_portfolio_value = mvo(simulation_data)
    initial_weights = dict(portfolio_composition.loc['2022-05-20'].to_dict())
    optimized_weight_dict = dict(zip(vault_names, portfolio_mvo_weights))
    vault_action_ranges = {
        'stETH Vault_dai_ceiling': [-0.5, 0.5],
        'ETH Vault_dai_ceiling': [-0.5, 0.5],
        'BTC Vault_dai_ceiling': [-0.5, 0.5],
        'Altcoin Vault_dai_ceiling': [-0.25, 0.25],
        'Stablecoin Vault_dai_ceiling': [-0.25, 0.25],
        'LP Vault_dai_ceiling': [0, 0],
        'RWA Vault_dai_ceiling': [0, 0],
        'PSM Vault_dai_ceiling': [-0.5, 0.5]
    }
    action_space = generate_action_space(vault_action_ranges)
    start_date = '2022-05-20'
    end_date = '2024-03-20'
    if advisor_type == 'Reinforcement Learning':
        rfl_agent = RlAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
        initial_action_rl = rfl_agent.initial_strategy(initial_weights)
        rl_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
        rl_simulator.train_model()
        rl_environment = SimulationEnvironment(rl_simulator, start_date, end_date, rfl_agent)
        rl_environment.reset()
        state, reward, done, info = rl_environment.run()
        rl_action_df = rl_environment.generate_action_dataframe()
        rl_sim_results = rl_simulator.results
        return rl_sim_results, rl_action_df
    else:
        mevo_agent = MVOAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
        initial_action_mvo = mevo_agent.initial_strategy(initial_weights)
        mvo_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
        mvo_simulator.train_model()
        mvo_environment = SimulationEnvironment(mvo_simulator, start_date, end_date, mevo_agent)
        mvo_environment.reset()
        state, reward, done, info = mvo_environment.run()
        mvo_action_df = mvo_environment.generate_action_dataframe()
        mvo_sim_results = mvo_simulator.results
        return mvo_sim_results, mvo_action_df

# Load data

# Simulation function
def run_simulation(agent_class, advisor_type):
    if agent_class == "Treasury":
        if advisor_type == "Reinforcement Learning":
            actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df = treasury_advisor(advisor_type)
            st.session_state['treasury_rl'] = (actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df)
        elif advisor_type == "Mean-Variance Optimization":
            rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns = treasury_advisor(advisor_type)
            st.session_state['treasury_mvo'] = (rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns)
    elif agent_class == "Vault":
        results, actions = vault_advisor(advisor_type)
        st.session_state['vault'] = (results, actions)

num_runs = st.sidebar.number_input('Number of Runs', min_value=1, max_value=20, value=10)
agent_class = st.sidebar.selectbox('Select Advisor Type', ['Vault', 'Treasury'])
advisor_type = st.sidebar.selectbox('Select Algo Type', ['Reinforcement Learning', 'Mean-Variance Optimization'])
# Add rebalance frequency input only for Treasury Advisor
if agent_class == "Treasury":
    rebalance_frequency = st.sidebar.number_input('Rebalance Frequency', min_value=1, max_value=30, value=5)
else:
    rebalance_frequency = None
run_simulation_button = st.sidebar.button('Run Simulation')

tabs = st.tabs(["Vault Robo Advisor", "Treasury Robo Advisor"])

if run_simulation_button:
    run_simulation(agent_class, advisor_type)
    

if 'vault' in st.session_state:
    with tabs[0]:
        st.header("Vault Robo Advisor")
        vault_sim_results, vault_action_df = st.session_state['vault']
        st.line_chart(vault_sim_results)

        st.sidebar.write("Vault Advisor simulation completed. Check the respective Vault advisor tab for results.")

if 'treasury_rl' in st.session_state or 'treasury_mvo' in st.session_state:
    with tabs[1]:
        st.header("Treasury Robo Advisor")
        if advisor_type == 'Reinforcement Learning' and 'treasury_rl' in st.session_state:
            actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df = st.session_state['treasury_rl']
            st.line_chart(rl_normalized_returns)
        elif advisor_type == 'Mean-Variance Optimization' and 'treasury_mvo' in st.session_state:
            rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns = st.session_state['treasury_mvo']
            st.line_chart(mvo_normalized_returns)

        st.sidebar.write("Treasury Advisor simulation completed. Check the respective Treasury advisor tab for results.")
