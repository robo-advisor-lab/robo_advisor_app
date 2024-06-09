import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.colors as pc
import plotly.subplots as sp
from datetime import datetime, timedelta


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
from scripts.treasury_utils import calculate_sortino_ratio, calculate_treynor_ratio, calculate_cagr, calculate_beta, normalize, calculate_log_returns
from scripts.treasury_data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free, historical_normalized_returns, historical_cumulative_return, historical_returns as treasury_historical_portfolio_returns, pivot_data_filtered, pivot_assets, index_start, indicies, tbill, combined_assets, panamadao_returns, calculate_historical_returns

# Vault Advisor
from models.vault_mvo_agent import MVOAgent
from models.vault_rl_agent import RlAgent
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment
from scripts.vault_data_processing import test_data, targets, features, temporals, dai_ceilings, vault_names, key_mapping
from scripts.vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return, evaluate_multi_predictions, abbreviate_number
import random
import torch


# Set random seeds
random_seed = st.sidebar.selectbox('Select Random Seed', [20, 42, 100, 200], index = 2)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



st.set_option('deprecation.showPyplotGlobalUse', False)
## Treasury Advisor
#treasury_historical_portfolio_returns.set_index('DAY', inplace=True)
#st.write('historical returns')
#st.write(treasury_historical_portfolio_returns.isnull().sum().sum())
#treasury_historical_portfolio_returns = treasury_historical_portfolio_returns.fillna(0)

#st.write(treasury_historical_portfolio_returns)

#st.write(treasury_historical_portfolio_returns)




def treasury_advisor(rebalance_frequency, selected_assets, eth_bound, start_date, end_date):
    st.sidebar.write("Starting RL agent training...")
    filtered_assets = selected_assets
    rl_rebalancing_frequency = rebalance_frequency
    treasury_start_date = start_date 
    treasury_end_date = end_date 

    if not validate_date_range(treasury_starting_date, treasury_ending_date):
        st.stop()
    

    filtered_data_treasury = combined_all_assets[combined_all_assets.index  <= treasury_end_date]
    # RL Model
    actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(combined_all_assets, filtered_assets, eth_bound, current_risk_free, rl_rebalancing_frequency, treasury_start_date, treasury_end_date)
    st.sidebar.write("RL agent training completed.")
    st.session_state['treasury_rl'] = (actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df)
    #st.session_state['treasury_historical_returns'] = filtered_historical_portfolio_returns

    # MVO Model
    mvo_rebalancing_frequency = rebalance_frequency
    threshold = 0
    model = mvo_model(eth_bound, current_risk_free, treasury_start_date, treasury_end_date)
    st.sidebar.write(f"Rebalancing MVO model starting {treasury_start_date}...")
    rebalanced_data = model.rebalance(filtered_data_treasury, filtered_assets, mvo_rebalancing_frequency)
    st.sidebar.write("MVO model rebalancing completed.")
    mvo_daily_portfolio_returns = model.calculate_daily_portfolio_returns(rebalanced_data, mvo_combined_assets)
    mvo_daily_portfolio_returns = mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= panama_dao_start_date]
    mvo_cumulative_return = model.calculate_cumulative_return(mvo_daily_portfolio_returns)
    base_return = mvo_cumulative_return.reset_index()
    base_return.columns = ['DAY', 'base_cumulative_return']
    first_value = base_return['base_cumulative_return'].iloc[0]
    base_return['PanamaDAO_treasury_return'] = 100 + (100 * (base_return['base_cumulative_return'] - first_value))
    mvo_normalized_returns = base_return[['DAY', 'PanamaDAO_treasury_return']]
    mvo_normalized_returns.set_index('DAY', inplace=True)
    st.session_state['treasury_mvo'] = (rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns)

## Vault Model Historical
simulation_data = test_data
simulation_data.index = pd.to_datetime(simulation_data.index)
simulation_data.index = simulation_data.index.tz_localize(None)
vault_historical = test_data[targets]
start_date = '2022-05-20'
end_date = '2024-03-20'
start_date_dt = pd.to_datetime(start_date)
historical_cutoff = start_date_dt - pd.DateOffset(days=1)
vault_historical_data = vault_historical[vault_historical.index <= historical_cutoff]
test_data.index = pd.to_datetime(test_data.index).tz_localize(None)
vault_historical_data = vault_historical_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')

historical_ceilings = test_data[dai_ceilings]
ceiling_historical_cutoff = historical_cutoff.tz_localize(None)
historical_ceilings_for_sim =  historical_ceilings[historical_ceilings.index <= ceiling_historical_cutoff]
historical_sim = test_data[test_data.index <= end_date]
historical_sim = historical_sim[targets].merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
#historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = #historical_sortino(historical_portfolio_returns,historical_portfolio_composition)
#historical_returns = calc_cumulative_return(historical_portfolio_daily_returns)

#hist_comparison = test_data[(test_data.index >= start_date) & (test_data.index <= end_date)]
#st.write(hist_comparison[targets].index.min(), hist_comparison[targets].index.max())


def vault_advisor(start_date, end_date, bounds):
    if not validate_date_range(start_date, end_date):
        st.stop()
    
    if start_date is not type(str):
        start_date = str(start_date)
    if end_date is not type(str):
        end_date = str(end_date)
        
    portfolio_mvo_weights, portfolio_returns, portfolio_composition, total_portfolio_value = mvo(simulation_data, st.session_state['vault_bounds'])
    
        
    initial_weights = dict(portfolio_composition.loc[start_date].to_dict())
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
    
    #start_date = '2022-05-20'
    #end_date = '2024-03-20'

    # RL Model
    st.sidebar.write("Starting RL agent training...")
    rfl_agent = RlAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
    initial_action_rl = rfl_agent.initial_strategy(initial_weights)
    rl_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
    rl_simulator.train_model()
    rl_environment = SimulationEnvironment(rl_simulator, start_date, end_date, rfl_agent,bounds)
    rl_environment.reset()
    state, reward, done, info = rl_environment.run()
    rl_action_df = rl_environment.generate_action_dataframe()
    rl_action_df.set_index('date', inplace=True)
    rl_sim_results = rl_simulator.results
    rl_vaults = rl_simulator
    st.session_state['vault_rl'] = (rl_sim_results, rl_action_df, rl_vaults)

    # MVO Model
    st.sidebar.write("Starting MVO agent training...")
    mevo_agent = MVOAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
    initial_action_mvo = mevo_agent.initial_strategy(initial_weights)
    mvo_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
    mvo_simulator.train_model()
    mvo_environment = SimulationEnvironment(mvo_simulator, start_date, end_date, mevo_agent,bounds)
    mvo_environment.reset()
    state, reward, done, info = mvo_environment.run()
    mvo_action_df = mvo_environment.generate_action_dataframe()
    mvo_action_df.set_index('date', inplace=True)
    mvo_sim_results = mvo_simulator.results
    mvo_vaults = mvo_simulator
    st.session_state['vault_mvo'] = (mvo_sim_results, mvo_action_df, mvo_vaults)

# Load data

def validate_dates(start_date, end_date, rebalance_frequency):
    delta = (end_date - start_date).days
    if delta < rebalance_frequency:
        st.sidebar.error(f"The difference between start and end dates should be at least {rebalance_frequency} days.")
        return False
    return True

def validate_date_range(start_date, end_date):
    if start_date >= end_date:
        st.sidebar.error("End date must be after the start date.")
        return False
    return True

def get_user_bounds(vaults):
        bounds = []
        for vault in vaults:
            st.write(f"Set bounds for {vault}:")
            lower_bound = st.number_input(f"Lower bound for {vault} (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            upper_bound = st.number_input(f"Upper bound for {vault} (%)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            bounds.append((lower_bound, upper_bound))
        return bounds

vaults = ['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 
          'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 
          'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']

# Simulation function
def run_simulation(agent_class, eth_bound=None, rebalance_frequency=None, start_date=None, end_date=None, user_bounds=None):
    set_random_seed(random_seed)
    if agent_class == "Treasury":
        st.sidebar.write("Running Treasury Advisor...")

        if not validate_dates(treasury_starting_date, treasury_ending_date, rebalance_frequency):
            return
        treasury_advisor(rebalance_frequency,selected_assets, eth_bound, treasury_starting_date, treasury_ending_date)
        st.session_state['rebalance_frequency'] = rebalance_frequency
        filtered_historical_portfolio_returns = treasury_historical_portfolio_returns[(treasury_historical_portfolio_returns.index >= treasury_starting_date) & (treasury_historical_portfolio_returns.index <= treasury_ending_date)]
        #historical_treasury_sortino = calculate_sortino_ratio(filtered_historical_portfolio_returns['weighted_daily_return'].values, current_risk_free)
        st.session_state['treasury_historical_returns'] = filtered_historical_portfolio_returns
        #st.write(filtered_historical_portfolio_returns)
        st.session_state['initial_amount'] = initial_amount
        st.session_state['treasury_start_date'] = treasury_starting_date
        st.session_state['selected_assets'] = selected_assets
        st.session_state['eth_bound'] = eth_bound
    elif agent_class == "Vault":
        st.sidebar.write("Running Vault Advisor...")
        #st.sidebar.write(st.session_state['vault_bounds'])
        st.session_state['run_vault_bounds'] = st.session_state['vault_bounds']
        vault_advisor(vault_starting_date, vault_ending_date, st.session_state['vault_bounds'])
        pandas_vault_starting_date = pd.to_datetime(vault_starting_date)
        pandas_vault_ending_date = pd.to_datetime(vault_ending_date)
        filtered_historical_sim = historical_sim[historical_sim.index <= pandas_vault_ending_date]
        historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(filtered_historical_sim, st.session_state['vault_bounds'])
        #vault_filtered_historical_portfolio_returns= historical_portfolio_returns[(historical_portfolio_returns.index >= vault_starting_date) & (historical_portfolio_returns.index <= vault_ending_date)]
        historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)
        historical_returns = calc_cumulative_return(historical_portfolio_daily_returns)
        hist_comparison = test_data[(test_data.index >= pandas_vault_starting_date) & (test_data.index <= pandas_vault_ending_date)]
        st.session_state['vault_historical'] = (historical_total_portfolio_value, historical_portfolio_daily_returns, historical_returns, historical_sortino_ratio, historical_portfolio_composition, hist_comparison)
        
        

#num_runs = st.sidebar.number_input('Number of Runs', min_value=1, max_value=20, value=10)
agent_class = st.sidebar.selectbox('Select Advisor Type', ['Vault', 'Treasury'])
#st.sidebar.write(panama_dao_start_date)
treasury_min_start_date = panama_dao_start_date.date()
#st.write(f'treasury starting date min {treasury_min_start_date}')
#st.dataframe(combined_all_assets)
treasury_min_end_date = combined_all_assets.index.max().date()
treasury_min_end_date = treasury_min_end_date - timedelta(days=1)

#st.write(f'treasury end max {treasury_min_end_date}')


vault_min_start_date = pd.to_datetime('2022-05-20').date()

#st.write(f'vault starting date min {vault_min_start_date}')

vault_min_end_date = simulation_data.index.max().date()

#st.write(f'vault ending date min {vault_min_end_date}')




# Add rebalance frequency input only for Treasury Advisor
if agent_class == "Treasury":
    rebalance_frequency = st.sidebar.number_input('Rebalance Frequency (days)', min_value=1, max_value=30, value=7)
    initial_amount = st.sidebar.number_input('Initial Amount (USD)', value=100)
    selected_assets = st.sidebar.multiselect(
        'Select assets to include in the Treasury Advisor:',
        combined_assets,
        default=mvo_combined_assets  
    )
    if 'ETH' not in selected_assets:
        st.sidebar.write(f"Adding ETH to portfolio...")
        selected_assets.append('ETH')
    
    eth_bound = st.sidebar.number_input('ETH bound', min_value=0.0, max_value=1.0, value=0.2)
    

    
    treasury_starting_date = st.sidebar.date_input('Start Date', value=treasury_min_start_date, min_value=treasury_min_start_date, max_value=treasury_min_end_date- timedelta(days=1))
    treasury_ending_date = st.sidebar.date_input('End Date', value=treasury_min_end_date, min_value = treasury_min_start_date + timedelta(days=7), max_value=treasury_min_end_date)
    
    
    treasury_starting_date = datetime.combine(treasury_starting_date, datetime.min.time())
    treasury_ending_date = datetime.combine(treasury_ending_date, datetime.min.time())
    treasury_ending_date = treasury_ending_date + timedelta(days=1)
    bounds = st.session_state['vault_bounds']

    #st.write(treasury_starting_date)
    
    #filtered_historical_portfolio_returns = treasury_historical_portfolio_returns[(treasury_historical_portfolio_returns.index >= treasury_starting_date) & (treasury_historical_portfolio_returns.index <= treasury_ending_date)]
    #st.dataframe(treasury_historical_portfolio_returns)
    #st.dataframe(filtered_historical_portfolio_returns)
    #st.write('historical portfolio')
    #st.write(filtered_historical_portfolio_returns)
    
    
    
else:
    rebalance_frequency = 30
    initial_amount = 100
    selected_assets = mvo_combined_assets if 'treasury_rl' not in st.session_state or 'treasury_mvo' not in st.session_state else st.session_state['selected_assets']
    eth_bound = 0
    vault_starting_date = st.sidebar.date_input('Start Date', value=vault_min_start_date, min_value=vault_min_start_date, max_value = vault_min_end_date- timedelta(days=7))
    vault_ending_date = st.sidebar.date_input('End Date', value=vault_min_end_date, min_value=vault_min_start_date + timedelta(days=7), max_value=vault_min_end_date)
    def get_user_bounds(vaults):
        bounds = {}
        for vault in vaults:
            lower_bound = st.sidebar.number_input(f"Lower bound for {vault} ", min_value=0.05, max_value=0.3, value=0.05, step=0.01)
            upper_bound = st.sidebar.number_input(f"Upper bound for {vault} ", min_value=0.2, max_value=1.0, value=0.2, step=0.01)
            bounds[vault] = (lower_bound, upper_bound)
        return bounds

    vaults = ['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 
              'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 
              'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']
    bounds = get_user_bounds(vaults)
    st.session_state['vault_bounds'] = bounds
    
    
    
    #st.session_state['eth_bound'] = eth_bound
    #st.session_state['selected_assets'] = selected_assets
    #st.session_state['vault_starting_date'] = vault_starting_date


    
run_simulation_button = st.sidebar.button('Run Simulation')

tabs = st.tabs(["Home","Vault Robo Advisor", "Treasury Robo Advisor"])

if run_simulation_button:
    st.sidebar.write("Running the simulation, please wait...")
    
    run_simulation(agent_class, eth_bound, rebalance_frequency, st.session_state['vault_bounds'])
    st.sidebar.write("Simulation completed.")
    
with tabs[0]:
    st.title("Robo Advisor App")

    st.markdown("""
    ## Welcome to the Robo Advisor App
    
    This application is designed to help you manage and optimize your portfolio using advanced financial models and machine learning techniques. The app provides two main advisors:
    
    1. **Vault Robo Advisor**: Manages individual vaults, optimizing their asset composition and rebalancing based on predefined strategies.  Trained on and backtested to MakerDAO data.
    2. **Treasury Robo Advisor**: Manages the overall treasury, optimizing the asset allocation and rebalancing to maximize returns while minimizing risk.  Backtested on PanamaDAO data.  
    
    ### How to Use
    
    1. **Select Advisor Type**: Choose between "Vault" and "Treasury" advisor using the sidebar.
    2. **Set Parameters**: Depending on the selected advisor, configure the relevant parameters such as:
       - **Rebalance Frequency**: For Treasury Advisor, set how often the portfolio should be rebalanced (in days).
       - **Initial Amount**: Set the initial amount in USD for the Treasury Advisor.
       - **Select Assets**: Choose the assets to include in the portfolio.
       - **ETH Bound**: For Treasury Advisor. set the minimum bound for ETH in the portfolio.
       - **Vault Bounds**: For Vault Advisor, set custom lower and upper bounds for each vault.
       - **Start and End Dates**: Select the time period for the simulation.
    3. **Run Simulation**: Click the "Run Simulation" button to start the optimization process.
    
    ### Outputs
    
    After running the simulation, the app provides detailed results and visualizations for both advisors, including:
    
    - **Portfolio Composition**: Shows the distribution of assets over time.
    - **Cumulative Returns**: Compares the historical, RL (Reinforcement Learning), and MVO (Mean-Variance Optimization) returns.
    - **Performance Metrics**: Displays key metrics such as Sortino ratios, cumulative returns, and portfolio values.
    
    ### Resources

    - For more information on the indicies:
        - DeFi and Non-DeFi Treasury indicies can be found at this [flipside dashboard](https://flipsidecrypto.xyz/Brandyn/dao-treasury-returns-Y2F8vX).
        - [Sirloin Index](https://dune.com/queries/2233092) can be found here.  Accompanying dashboard for DAO Benchmarks [here](https://dune.com/steakhouse/howmuchubench). 
    
    - [GitHub Repository for Vault Advisor](https://github.com/BrandynHamilton/mkrdao_port_mgmt)
    - [GitHub Repository for Treasury Advisor](https://github.com/BrandynHamilton/treasury_advisor)
    """)

    
    


with tabs[1]:
    st.header("Vault Robo Advisor")
    st.markdown("**Backtesting to MakerDAO Historical Data**")
    
    if 'vault_rl' in st.session_state or 'vault_mvo' in st.session_state:
        st.write('Chosen Vault Bounds:')
        st.write(st.session_state['run_vault_bounds'] if 'run_vault_bounds' in st.session_state else st.session_state['vault_bounds'])
        if 'vault_rl' in st.session_state:
            rl_sim_results, rl_action_df, rl_vaults = st.session_state['vault_rl']
            st.write(f"Simulation from {rl_sim_results.index.min()} through {rl_sim_results.index.max()}")
            vault_sim_days = rl_sim_results.index.max() - rl_sim_results.index.min()
            st.write(f'Total Duration: {vault_sim_days}')
            st.write('Rebalancing Frequency: 24 Days')
            with st.expander('RL Agent Actions'):
                #cycle_action = rl_action_df['action'] * 24
                action_dict = rl_action_df['action'].dropna().to_dict()
                # Function to multiply values by 24
                def multiply_actions(actions, factor=24):
                    return {key: value / 100 * factor for key, value in actions.items()}
                
                # Apply the multiplication to each set of actions in the dictionary
                updated_action_dict = {key: multiply_actions(value) for key, value in action_dict.items()}
                
                # Function to format values as percentages
                def format_as_percentage(actions):
                    return {key: f"{value:.2%}" for key, value in actions.items()}
                
                # Apply the formatting to each set of actions in the dictionary
                formatted_action_dict = {key: format_as_percentage(value) for key, value in updated_action_dict.items()}
            
            # Display the DataFrame in Streamlit
                st.dataframe(formatted_action_dict, use_container_width=True)
            #rl_action_df.set_index('date', inplace=True)
            rl_combined_data = pd.concat([vault_historical_data.drop(columns=['RWA Vault_collateral_usd']), rl_sim_results])
            rl_combined_data.index = rl_combined_data.index.tz_localize(None)
            rl_sim_w_RWA = rl_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
            rl_sim_w_RWA.sort_index(inplace=True)
            rl_portfolio_mvo_weights, rl_portfolio_returns, rl_portfolio_composition, rl_total_portfolio_value = mvo(rl_sim_w_RWA, bounds)
            rl_sim_portfolio_daily_returns,  rl_sim_downside_returns, rl_sim_excess_returns, rl_sim_sortino_ratio = historical_sortino(rl_portfolio_returns,rl_portfolio_composition)
            rl_optimized_returns = calc_cumulative_return(rl_sim_portfolio_daily_returns)
            rl_optimized_returns = rl_optimized_returns[~rl_optimized_returns.index.duplicated(keep='first')]
            
            






            rl_sortino_timeseries = rl_action_df['current sortino ratio'].dropna()
            
            
        if 'vault_mvo' in st.session_state:
            mvo_sim_results, mvo_action_df, mvo_vaults = st.session_state['vault_mvo']
            with st.expander('MVO Agent Actions'):
                mvo_action_dict = mvo_action_df['action'].dropna().to_dict()
                updated_mvo_action_dict = {key: multiply_actions(value) for key, value in mvo_action_dict.items()}
                formatted_mvo_action_dict = {key: format_as_percentage(value) for key, value in updated_mvo_action_dict.items()}



                #mvo_action_df['action'] = mvo_action_df['action']
                st.dataframe(formatted_mvo_action_dict, use_container_width=True)
            #mvo_action_df.set_index('date', inplace=True)
            mvo_combined_data = pd.concat([vault_historical_data.drop(columns=['RWA Vault_collateral_usd']), mvo_sim_results])
            mvo_combined_data.index = mvo_combined_data.index.tz_localize(None)
            mvo_sim_w_RWA = mvo_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
            mvo_sim_w_RWA.sort_index(inplace=True)
            #Can filter for start date of sim 
            mvo_portfolio_mvo_weights, mvo_portfolio_returns, mvo_portfolio_composition, mvo_total_portfolio_value = mvo(mvo_sim_w_RWA, st.session_state['vault_bounds'])
            mvo_sim_portfolio_daily_returns,  mvo_sim_downside_returns, mvo_sim_excess_returns, mvo_sim_sortino_ratio = historical_sortino(mvo_portfolio_returns,mvo_portfolio_composition)
            mvo_optimized_returns = calc_cumulative_return(mvo_sim_portfolio_daily_returns)
            mvo_optimized_returns = mvo_optimized_returns[~mvo_optimized_returns.index.duplicated(keep='first')]






            mvo_sortino_timeseries = mvo_action_df['current sortino ratio'].dropna()
            
            
            #st.sidebar.write("Vault Advisor simulation completed. Check the respective Vault advisor tab for results.")

        st.divider()
        historical_total_portfolio_value, historical_portfolio_daily_returns, historical_returns, historical_sortino_ratio, historical_portfolio_composition, hist_comparison = st.session_state['vault_historical']
        #st.dataframe(mvo_sim_portfolio_daily_returns)
        #st.dataframe(rl_sim_portfolio_daily_returns)
        #st.dataframe(historical_portfolio_daily_returns)
        #st.dataframe(historical_returns)
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.metric(label="RL current portfolio value", value=f"${abbreviate_number(rl_total_portfolio_value.iloc[-1])}")
            st.metric(label="RL sortino ratio", value=f"{rl_sim_sortino_ratio:.2f}")
        
        with vcol2:
            st.metric(label="MVO current portfolio value", value=f"${abbreviate_number(mvo_total_portfolio_value.iloc[-1])}")
            st.metric(label="MVO sortino ratio", value=f"{mvo_sim_sortino_ratio:.2f}")

         
        #st.write(f"{historical_returns.index.min()} - {historical_returns.index.max()}")
        
        with vcol3:
            st.metric(label="Historical current portfolio value", value=f"${abbreviate_number(historical_total_portfolio_value.iloc[-1])}")
            st.metric(label="Historical sortino ratio", value=f"{historical_sortino_ratio:.2f}")
        vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault']
        st.subheader('Reinforcement Learning Results')
        
        with st.expander("RL Vault Results"):
            for vault in vault_names:
                rl_fig = rl_vaults.plot_vault_data(vault)
                st.plotly_chart(rl_fig)

        st.line_chart(rl_sim_results)
        

        
            #rl_metrics = evaluate_predictions(rl_sim_results, vault_historical_data)
        
            #mvo_metrics = evaluate_predictions(mvo_sim_results, vault_historical_data)

        

        st.subheader('Mean-Variance Optimization Results')
        
        with st.expander("MVO Vault Results"):
            for vault in vault_names:
                mvo_fig = mvo_vaults.plot_vault_data(vault)
                st.plotly_chart(mvo_fig)
        st.line_chart(mvo_sim_results)
            
        

        st.subheader('Historical Results')
        
        with st.expander("Robo Advisor Vs Historical"):
            metrics, figures = evaluate_multi_predictions(rl_sim_results, mvo_sim_results, vault_historical)
            for vault, fig in figures.items():
                st.plotly_chart(fig)

            

            st.write(metrics)
        st.line_chart(hist_comparison[targets])

        

        trace1 = go.Scatter(
            x=rl_optimized_returns.index,
            y=rl_optimized_returns.values,
            mode='lines',
            name='RL Robo Advisor Optimized Returns',
            line=dict(color='blue')
        )
        
        trace2 = go.Scatter(
            x=mvo_optimized_returns.index,
            y=mvo_optimized_returns.values,
            mode='lines',
            name='MVO Robo Advisor Optimized Returns',
            line=dict(color='green')
        )
        
        trace3 = go.Scatter(
            x=historical_returns.index,
            y=historical_returns.values,
            mode='lines',
            name='Historical Returns',
            line=dict(color='red')
        )
        
        # Create the layout
        layout = go.Layout(
            title='Optimized vs Historical Returns',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Returns'),
            legend=dict(x=0.1, y=0.9)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        
        # Render the figure in Streamlit
        st.plotly_chart(fig)
        
        trace1 = go.Scatter(
            x=rl_total_portfolio_value.index,
            y=rl_total_portfolio_value.values,
            mode='lines',
            name='RL Robo Advisor',
            line=dict(color='blue')
        )
        
        trace2 = go.Scatter(
            x=mvo_total_portfolio_value.index,
            y=mvo_total_portfolio_value.values,
            mode='lines',
            name='MVO Robo Advisor Portfolio',
            line=dict(color='green')
        )
        
        trace3 = go.Scatter(
            x=historical_total_portfolio_value.index,
            y=historical_total_portfolio_value.values,
            mode='lines',
            name='Historical Portfolio',
            line=dict(color='red')
        )
        #st.write(f"{historical_returns.index.min()} - {historical_returns.index.max()}")
        # Create the layout
        layout = go.Layout(
            title='Robo Advisor vs Historical TVL',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value'),
            legend=dict(x=0.1, y=0.9)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        
        # Render the figure
        st.plotly_chart(fig)

        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=('RL Robo Advisor', 'MVO Robo Advisor', 'Historical'))

        # RL Robo Advisor Pie Chart
        fig.add_trace(go.Pie(labels=rl_portfolio_composition.columns, 
                             values=rl_portfolio_composition.iloc[-1], 
                             name="RL Robo Advisor"), 
                      1, 1)
        
        # MVO Robo Advisor Pie Chart
        fig.add_trace(go.Pie(labels=mvo_portfolio_composition.columns, 
                             values=mvo_portfolio_composition.iloc[-1], 
                             name="MVO Robo Advisor"), 
                      1, 2)
        
        # Historical Portfolio Pie Chart
        fig.add_trace(go.Pie(labels=historical_portfolio_composition.columns, 
                             values=historical_portfolio_composition.iloc[-1], 
                             name="Historical"), 
                      1, 3)
        
        # Update layout
        fig.update_layout(title_text=f'Portfolio Compositions as of {historical_portfolio_composition.index.max()}')
        
        # Render the figure in Streamlit
        st.plotly_chart(fig)
        
        # Print portfolio values
        # Print portfolio values
        

        
        trace1 = go.Scatter(
            x=mvo_sortino_timeseries.index,
            y=mvo_sortino_timeseries.values,
            mode='lines',
            name='MVO Robo Advisor Sortino Ratios',
            line=dict(color='green')
        )
        
        trace2 = go.Scatter(
            x=rl_sortino_timeseries.index,
            y=rl_sortino_timeseries.values,
            mode='lines',
            name='RL Robo Advisor Sortino Ratios',
            line=dict(color='red')
        )
        
        # Create the layout
        layout = go.Layout(
            title='MVO vs RL Sortino Ratios',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value'),
            legend=dict(x=0.1, y=0.9)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        
        # Render the figure
        st.plotly_chart(fig)
        



with tabs[2]:
    st.header("Treasury Robo Advisor")
    st.markdown("**Backtesting to PanamaDAO Historical Data**")
    #st.write('Selected Assets:', selected_assets)
    #st.write('ETH Bound', eth_bound)
    if 'treasury_rl' in st.session_state or 'treasury_mvo' in st.session_state:
        selected_assets = st.session_state['selected_assets'] 
        

        #st.write(filtered_historical_portfolio_returns)
        filtered_historical_returns = st.session_state['treasury_historical_returns']['weighted_daily_return'].iloc[:-1]
        filtered_historical_returns = pd.concat([pd.Series([0], index=[filtered_historical_returns.index.min()]), filtered_historical_returns])
        filtered_historical_returns = filtered_historical_returns[~filtered_historical_returns.index.duplicated(keep='first')]
    
        #st.write(filtered_historical_returns)
       
        filtered_historical_sortino = calculate_sortino_ratio(filtered_historical_returns, current_risk_free)
        #st.write('filtered historical_sortino')
        #st.write(filtered_historical_sortino)
        filtered_cumulative_return = np.exp(np.log1p(filtered_historical_returns).cumsum()) - 1
        #st.write('filtered cumulative return')
        #st.write(filtered_cumulative_return)
        
        filtered_historical_normalized = normalize(filtered_cumulative_return, filtered_cumulative_return.index.min())
        #st.write('normalized_historical_filtered')
        #st.dataframe(filtered_historical_normalized)
        
        if 'treasury_rl' in st.session_state:
            actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = st.session_state['treasury_rl']
            
            st.write(f"Simulation from {rl_portfolio_returns.index.min()} through {rl_portfolio_returns.index.max()}")
            tresaury_sim_days = rl_portfolio_returns.index.max() - rl_portfolio_returns.index.min()
            st.write(f'Total Duration: {tresaury_sim_days}')
            #st.write('rl_port returns')
            #st.write(rl_portfolio_returns['Portfolio Return'])
            
            rl_sortino = calculate_sortino_ratio(rl_portfolio_returns['Portfolio Return'].values, current_risk_free)
            #st.write('rl sortino')
            #st.write(rl_sortino)
    
            #st.write('Reinforcement Learning Results')
            #st.line_chart(rl_normalized_returns)
        if 'treasury_mvo' in st.session_state:
            rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns = st.session_state['treasury_mvo']
            #st.write("rebalanced data", rebalanced_data)
            #st.write("mvo portfolio returns", mvo_daily_portfolio_returns)
            #st.write("historical portfolio returns", filtered_historical_returns)
            #st.write("rl returns", rl_portfolio_returns)
            #st.write("rl comp", composition_df)
            
            
            mvo_sortino = calculate_sortino_ratio(mvo_daily_portfolio_returns.values, current_risk_free)
            #mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= treasury_starting_date]
            #st.dataframe(mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= treasury_starting_date])
    
    
            #st.write('Mean-Variance Optimization Results')
            #st.line_chart(mvo_normalized_returns)
            #st.sidebar.write("Treasury Advisor simulation completed. Check the respective Treasury advisor tab for results.")
    
        #st.write('rl normalized')
        #st.write(rl_normalized_returns)
        #st.write('rl daily')
        #st.write(rl_portfolio_returns)
        #st.write('rl actions')
        #st.write(actions_df)
    
        # Convert the returns to strings and print
        #st.write(filtered_cumulative_return)
        historical_return = float(filtered_cumulative_return.iloc[-1]) * 100
        rl_return = float(rl_cumulative_return.iloc[-1]) * 100
        mvo_return = float(mvo_cumulative_return.iloc[-1]) * 100
        mvo_normalized_latest = mvo_normalized_returns["PanamaDAO_treasury_return"].iloc[-1]
        rl_normalized_latest = rl_normalized_returns["PanamaDAO_treasury_return"].iloc[-1]
        historical_normalized_latest = filtered_historical_normalized["treasury_return"].iloc[-1]    
        st.write(f"Initial amount: ${st.session_state['initial_amount']:,.2f}")
        st.write(f"Rebalancing Frequency: {st.session_state['rebalance_frequency']} days")
        st.write(f"ETH Minimum Bound: {st.session_state['eth_bound'] * 100:.2f}%")
        st.write(f"Selected Assets: {st.session_state['selected_assets']}")
        st.divider()
        #st.write('historical')
        #st.write(st.session_state['treasury_historical_returns'])
        #st.write('rl daily')
        #st.dataframe(rl_portfolio_returns)
        #st.write('RL cum')
        #st.dataframe(rl_cumulative_return)
        #st.write('MVO daily')
        #st.dataframe(mvo_daily_portfolio_returns)
        #st.write('MVO cum')
        #st.dataframe(mvo_cumulative_return)
    
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="RL Adjusted Return", value=f"${st.session_state['initial_amount'] * rl_normalized_latest / 100:,.0f}")
            #st.write('actual normalized', rl_normalized_latest)
            st.metric(label="RL Cumulative Return", value=f"{rl_return:.2f}%")
            st.metric(label="RL Sortino", value=f"{rl_sortino:.2f}")
            
            
        with col2:
            st.metric(label="MVO Adjusted Return", value=f"${st.session_state['initial_amount'] * mvo_normalized_latest / 100:,.0f}")
            st.metric(label="MVO Cumulative Return", value=f"{mvo_return:.2f}%")
            st.metric(label="MVO Sortino", value=f"{mvo_sortino:.2f}")
    
        with col3:
            #st.write('placeholder')
            st.metric(label="Historical Adjusted Return", value=f"${st.session_state['initial_amount'] * historical_normalized_latest / 100:,.0f}")
            st.metric(label="Historical Cumulative Return", value=f"{historical_return:.2f}%")
            st.metric(label="Historical Sortino", value=f"{filtered_historical_sortino:.2f}")
    
        
            
       
        trace1 = go.Scatter(
            x=filtered_historical_normalized.index,
            y=filtered_historical_normalized['treasury_return'],
            mode='lines',
            name='Historical Normalized Returns',
            line=dict(color='blue')
        )
       
        
        trace2 = go.Scatter(
            x=mvo_normalized_returns.index,
            y=mvo_normalized_returns['PanamaDAO_treasury_return'],
            mode='lines',
            name='MVO Normalized Returns',
            line=dict(color='red')
        )
        
        trace3 = go.Scatter(
            x=rl_normalized_returns.index,
            y=rl_normalized_returns['PanamaDAO_treasury_return'],
            mode='lines',
            name='RL Normalized Returns',
            line=dict(color='green')
        )
        
        # Create the layout
        layout = go.Layout(
            title='Comparison of Historical and Robo Advisor Normalized Returns',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Normalized Returns'),
            legend=dict(x=0.1, y=0.9)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        
        # Render the figure
        st.plotly_chart(fig)
        
        
        
    
    
    
        
    
        #filtered_cumulative_return = filtered_cumulative_return.rename(columns={"weighted_daily_return":"'cumulative_return'"})
        index_historical_normalized = normalize(filtered_cumulative_return, index_start)
        #st.write(filtered_cumulative_return)
        
        index_rl_normalized = normalize(rl_cumulative_return, index_start)
        index_mvo_normalized = normalize(mvo_cumulative_return, index_start)
        
        
        # In[125]:
        
        
        index_mvo_normalized.rename(columns={'treasury_return':'MVO Treasury Robo Advisor Return'}, inplace=True)
        
        
        # In[126]:
        
        
        index_rl_normalized.rename(columns={'treasury_return':'RL Treasury Robo Advisor Return'}, inplace=True)
        
        
        # In[127]:
        
        
        index_historical_normalized.rename(columns={'treasury_return':'Historical PanamaDAO Treasury Return'}, inplace=True)
        indicies.index = pd.to_datetime(indicies.index)
        combined_indicies = indicies.merge(index_historical_normalized, left_index=True, right_index=True, how='inner')
        combined_indicies = combined_indicies.merge(index_mvo_normalized, left_index=True, right_index=True, how='inner')
        combined_indicies = combined_indicies.merge(index_rl_normalized, left_index=True, right_index=True, how='inner')
        #st.write(combined_indicies.index.min())
        #st.write(combined_indicies.index.max())
        combined_indicies.drop(columns=['WBTC_DAILY_RETURN','WETH_DAILY_RETURN','WBTC_NORMALIZED_PRICE','WETH_NORMALIZED_PRICE'], inplace=True)
        # Define a color palette
        color_palette = pc.qualitative.Plotly
        
        # Create an empty list to hold the traces
        traces = []
        
        # Loop through each column in combined_indicies and create a trace for it
        for i, column in enumerate(combined_indicies.columns):
            trace = go.Scatter(
                x=combined_indicies.index,
                y=combined_indicies[column],
                mode='lines',
                name=column,
                line=dict(color=color_palette[i % len(color_palette)])
            )
            traces.append(trace)
        
        # Create the layout
        layout = go.Layout(
            title='Normalized Comparison of DAO Indicies to Robo Advisor Returns',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value'),
            legend=dict(x=0.1, y=0.9)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=traces, layout=layout)
        if not combined_indicies.empty:
            st.plotly_chart(fig)
    
        def clean_composition_values(values, threshold=1e-5):
            cleaned_values = []
            for value in values:
                if abs(value) < threshold or value < 0:
                    cleaned_values.append(0)
                else:
                    cleaned_values.append(value)
            return cleaned_values
        
        composition_columns = [f'COMPOSITION_{asset}' for asset in selected_assets]
        #st.write('composition columns', composition_columns)
        mvo_comp = rebalanced_data[composition_columns]
        mvo_comp.columns = mvo_comp.columns.str.replace('COMPOSITION_', '', regex=False)
        #st.dataframe(mvo_comp)
        latest_comp_mvo = mvo_comp.iloc[-1]
        #latest_comp_mvo.index = latest_comp_mvo.index.str.replace('COMPOSITION_', '', regex=False)
        #st.table(latest_comp_mvo)
    
        composition_df.columns = composition_df.columns.str.replace('COMPOSITION_', '', regex=False)
        
        latest_comp_rl = composition_df.iloc[-1]
        #st.table(latest_comp_rl)
        
        comp_columns = [col for col in pivot_data_filtered.columns if col.startswith('COMPOSITION_')]
        
        latest_historical_comp = pivot_data_filtered[comp_columns][pivot_data_filtered.index == composition_df.index.max()].iloc[-1]
        #st.write(latest_historical_comp)
        latest_historical_comp.index = latest_historical_comp.index.str.replace('COMPOSITION_', '', regex=False)
        #st.table(latest_historical_comp)
        comp_start = composition_df.index.min().date() - timedelta(days=1)
        comp_start = pd.to_datetime(comp_start)
        #st.write(comp_start)
    
        
        
        # List of all assets
        
         
        # Generate a color palette
        # Generate a color palette
        color_palette = pc.qualitative.Plotly
    
        # Create a color mapping
        color_mapping = {asset: color_palette[i % len(color_palette)] for i, asset in enumerate(st.session_state['selected_assets'])}
    
        
        def create_composition_pie(labels, values, name):
            # Clean the values
            cleaned_values = clean_composition_values(values)
            
            # Create a list of colors based on the provided labels
            colors = [color_mapping.get(label, '#d3d3d3') for label in labels]  # Default to light gray if label not found
        
            return go.Pie(
                labels=labels,
                values=cleaned_values,
                name=name,
                hole=0.3,
                marker=dict(colors=colors)
            )
    
        
        def mvo_composition():
            return create_composition_pie(
                labels=latest_comp_mvo.index,
                values=latest_comp_mvo.values,
                name='MVO Portfolio Composition'
            )
        
        def historical_composition():
            return create_composition_pie(
                labels=latest_historical_comp.index,
                values=latest_historical_comp.values,
                name='Historical Portfolio Composition'
            )
        
        def rl_composition():
            return create_composition_pie(
                labels=latest_comp_rl.index,
                values=latest_comp_rl.values,
                name='RL Portfolio Composition'
            )
        
        # Create subplots
        fig = sp.make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=['MVO', 'Historical', 'RL']
        )
        
        # Add traces to the subplots
        fig.add_trace(mvo_composition(), row=1, col=1)
        fig.add_trace(historical_composition(), row=1, col=2)
        fig.add_trace(rl_composition(), row=1, col=3)
        
        # Update layout
        fig.update_layout(
            title_text=f'Portfolio Compositions as of {rl_portfolio_returns.index.max()}',
            title_x=0.5,  # Center the main title
            title_y=0.95,  # Position the main title slightly lower
            title_xanchor='center',  # Anchor the title at the center
            title_yanchor='top'  # Anchor the title at the top
        )
        
        st.plotly_chart(fig)
    
    
        mvo_comp = rebalanced_data[composition_columns]
        mvo_comp.index = pd.to_datetime(mvo_comp.index)
        #st.write(mvo_comp)
        mvo_comp = mvo_comp[mvo_comp.index > comp_start]
        mvo_comp.columns = mvo_comp.columns.str.replace('COMPOSITION_', '', regex=False)
    
    
        traces = []
    
        # Define the color palette
        color_palette = pc.qualitative.Plotly
        
        # Loop through each column in composition_df and create a trace for it
        for i, column in enumerate(composition_df.columns):
            trace = go.Bar(
                x=composition_df.index,
                y=composition_df[column],
                name=column,
                marker=dict(color=color_palette[i % len(color_palette)])
            )
            traces.append(trace)
        
        # Create the layout
        layout = go.Layout(
            title='RL Portfolio Composition Over Time',
            barmode='stack',
            xaxis=dict(
                title='Date',
                tickmode='auto',
                nticks=20,
                tickangle=-45
            ),
            yaxis=dict(title='Composition'),
            legend=dict(x=1.05, y=1)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=traces, layout=layout)
        
        #st.write(composition_df)
        #st.write(actions_df)
        st.plotly_chart(fig)
    
        # Create an empty list to hold the traces
        traces = []
        
        # Define the color palette
        color_palette = pc.qualitative.Plotly
        
        # Loop through each column in composition_df and create a trace for it
        for i, column in enumerate(mvo_comp.columns):
            trace = go.Bar(
                x=mvo_comp.index,
                y=mvo_comp[column],
                name=column,
                marker=dict(color=color_palette[i % len(color_palette)])
            )
            traces.append(trace)
    
        # Create the layout
        layout = go.Layout(
            title='MVO Portfolio Composition Over Time',
            barmode='stack',
            xaxis=dict(
                title='Date',
                tickmode='auto',
                nticks=20,
                tickangle=-45
            ),
            yaxis=dict(title='Composition'),
            legend=dict(x=1.05, y=1)
        )
        
        # Combine the data and layout into a figure
        fig = go.Figure(data=traces, layout=layout)
        
        # Render the figure
        st.plotly_chart(fig)
        def create_interactive_sml(risk_free_rate, market_risk_premium, rl_beta, mvo_beta, historical_beta, defi_beta, non_defi_beta, rl_return, mvo_return, historical_return, defi_return, non_defi_return):
    # Collect all beta values and filter out None
            beta_values = [beta for beta in [rl_beta, mvo_beta, historical_beta, defi_beta, non_defi_beta] if beta is not None]
            
            # Determine the maximum beta value to set the range
            max_beta = max(beta_values) if beta_values else 6
            min_beta = min(beta_values) if min(beta_values) < 0 else 0
            beta_range = np.linspace(min_beta, np.absolute(max_beta) * 1.1, 100)  # Extend the range slightly beyond the max beta
        
            expected_returns = risk_free_rate + beta_range * market_risk_premium
        
            # Create the SML line
            sml_line = go.Scatter(x=beta_range, y=expected_returns, mode='lines', name='SML')
        
            # Add MakerDAO token as points for expected (based on selected market risk premium) and actual returns
            rl_expected = go.Scatter(
                x=[rl_beta],
                y=[risk_free_rate + rl_beta * market_risk_premium],  # Use the selected market risk premium
                mode='markers', 
                marker=dict(color='red', size=10),
                name=f'RL Expected'
            )
            
            rl_actual = go.Scatter(
                x=[rl_beta],
                y=[rl_return],
                mode='markers', 
                marker=dict(color='pink', size=10),
                name=f'RL Actual'
            )
        
            # Add LidoDAO token as points for expected and actual returns
            mvo_expected = go.Scatter(
                x=[mvo_beta],
                y=[risk_free_rate + mvo_beta * market_risk_premium],  # Use the selected market risk premium
                mode='markers', 
                marker=dict(color='blue', size=10),
                name=f'MVO Expected'
            )
            
            mvo_actual = go.Scatter(
                x=[mvo_beta],
                y=[mvo_return],
                mode='markers', 
                marker=dict(color='lightblue', size=10),
                name=f'MVO Actual'
            )
        
            # Convert rpl_beta from array to scalar if necessary
            #rpl_beta_value = rpl_beta[0] if isinstance(rpl_beta, np.ndarray) else rpl_beta
        
            # Add Rocket Pool token as points for expected and actual returns
            historical_expected = go.Scatter(
                x=[historical_beta],
                y=[risk_free_rate + historical_beta * market_risk_premium],  # Use the selected market risk premium
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Historical Expected'
            )
            
            historical_actual = go.Scatter(
                x=[historical_beta],
                y=[historical_return],
                mode='markers',
                marker=dict(color='yellow', size=10),
                name='Historical Actual'
            )
        
            data = [sml_line, rl_expected, rl_actual, mvo_expected, mvo_actual, historical_expected, historical_actual]
        
            if defi_beta is not None and defi_return is not None:
                defi_expected = go.Scatter(
                    x=[defi_beta],
                    y=[risk_free_rate + defi_beta * market_risk_premium],  # Use the selected market risk premium
                    mode='markers',
                    marker=dict(color='purple', size=10),
                    name='Defi Treasury Index Expected'
                )
                
                defi_actual = go.Scatter(
                    x=[defi_beta],
                    y=[defi_return],
                    mode='markers',
                    marker=dict(color='navy', size=10),
                    name='Defi Treasury Index Actual'
                )
                data.extend([defi_expected, defi_actual])
        
            if non_defi_beta is not None and non_defi_return is not None:
                non_defi_expected = go.Scatter(
                    x=[non_defi_beta],
                    y=[risk_free_rate + non_defi_beta * market_risk_premium],  # Use the selected market risk premium
                    mode='markers',
                    marker=dict(color='grey', size=10),
                    name='Non-Defi Treasury Index Expected'
                )
                
                non_defi_actual = go.Scatter(
                    x=[non_defi_beta],
                    y=[non_defi_return],
                    mode='markers',
                    marker=dict(color='tan', size=10),
                    name='Non-Defi Treasury Index Actual'
                )
                data.extend([non_defi_expected, non_defi_actual])
        
            # Add Risk-Free Rate line
            risk_free_line = go.Scatter(
                x=[min(beta_range), max(beta_range)], 
                y=[risk_free_rate, risk_free_rate], 
                mode='lines', 
                line=dict(dash='dash', color='green'),
                name='Risk-Free Rate'
            )
        
            data.append(risk_free_line)
        
            # Layout settings
            layout = go.Layout(
                title=f'SML',
                xaxis=dict(title='Beta (Systematic Risk)'),
                yaxis=dict(title='Return'),
                showlegend=True
            )
        
            # Combine all the plots
            fig = go.Figure(data=data, layout=layout)
            return fig, market_risk_premium
    
        if not combined_indicies.empty and 'SIRLOIN_INDEX' in combined_indicies.columns and 'RL Treasury Robo Advisor Return' in combined_indicies.columns:
            rl_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'RL Treasury Robo Advisor Return')
            rl_cagr = calculate_cagr(combined_indicies['RL Treasury Robo Advisor Return'])
            mvo_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'MVO Treasury Robo Advisor Return')
            mvo_cagr = calculate_cagr(combined_indicies['MVO Treasury Robo Advisor Return'])
            historical_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'Historical PanamaDAO Treasury Return')
            historical_cagr = calculate_cagr(combined_indicies['Historical PanamaDAO Treasury Return'])
            
    
            defi_index_history = combined_indicies['DEFI_TREASURY_INDEX']
            defi_index_cagr = calculate_cagr(defi_index_history)
            defi_index_cumulative_risk_premium = defi_index_cagr - current_risk_free
            non_defi_index_cagr = calculate_cagr(combined_indicies['NON_DEFI_TREASURY_INDEX'])
            
            defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'DEFI_TREASURY_INDEX')
            non_defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'NON_DEFI_TREASURY_INDEX')
            #st.metric(label="Defi Index CAGR", value=f"{defi_index_cagr * 100:.2f}%")
            #st.metric(label="Defi Index Cumulative Risk Premium", value=f"{defi_index_cumulative_risk_premium * 100:.2f}%")
            
            sirloin_history = combined_indicies['SIRLOIN_INDEX']
            benchmark_cagr = calculate_cagr(sirloin_history)
            cumulative_risk_premium = benchmark_cagr - current_risk_free
            index = "Sirloin Index"
            fig, market_risk_premium = create_interactive_sml(
                current_risk_free,
                cumulative_risk_premium,
                rl_beta,
                mvo_beta,
                historical_beta,
                defi_index_beta,
                non_defi_index_beta,
                rl_cagr,
                mvo_cagr,
                historical_cagr,
                defi_index_cagr,
                non_defi_index_cagr
    
            )
        else:
            prices = pivot_assets.copy()
            prices.set_index('DAY', inplace=True)
            dpi_history = prices['DAILY_PRICE_DPI']
            #st.dataframe(dpi_history)
            dpi_returns = calculate_log_returns(dpi_history)
            dpi_history_cum = np.exp(np.log1p(dpi_returns).cumsum()) - 1
            dpi_normalized = normalize(dpi_history_cum, st.session_state['treasury_start_date'])
            dpi_normalized.rename(columns={"treasury_return":"Defi Pulse Index"}, inplace=True)
            benchmark_cagr = calculate_cagr(dpi_history)
            
            cumulative_risk_premium = benchmark_cagr - current_risk_free
            index = "Defi Pulse Index"
    
            dpi_combined_indicies = dpi_normalized.merge(mvo_normalized_returns["PanamaDAO_treasury_return"].to_frame("MVO Treasury Robo Advisor Return"), left_index=True, right_index=True, how='inner')
            dpi_combined_indicies = dpi_combined_indicies.merge(rl_normalized_returns["PanamaDAO_treasury_return"].to_frame("RL Treasury Robo Advisor Return"), left_index=True, right_index=True, how='inner')
            dpi_combined_indicies = dpi_combined_indicies.merge(filtered_historical_normalized["treasury_return"].to_frame("Historical PanamaDAO Treasury Return"), left_index=True, right_index=True, how='inner')
            #st.dataframe(dpi_combined_indicies)
            
            rl_beta = calculate_beta(dpi_combined_indicies, 'Defi Pulse Index', 'RL Treasury Robo Advisor Return')
            rl_cagr = calculate_cagr(dpi_combined_indicies['RL Treasury Robo Advisor Return'])
            mvo_beta = calculate_beta(dpi_combined_indicies, 'Defi Pulse Index', 'MVO Treasury Robo Advisor Return')
            mvo_cagr = calculate_cagr(dpi_combined_indicies['MVO Treasury Robo Advisor Return'])
            historical_beta = calculate_beta(dpi_combined_indicies, 'Defi Pulse Index', 'Historical PanamaDAO Treasury Return')
            historical_cagr = calculate_cagr(dpi_combined_indicies['Historical PanamaDAO Treasury Return'])
            fig, market_risk_premium = create_interactive_sml(
                current_risk_free,
                cumulative_risk_premium,
                rl_beta,
                mvo_beta,
                historical_beta,
                None,  # No DEFI_TREASURY_INDEX beta
                None,  # No NON_DEFI_TREASURY_INDEX beta
                rl_cagr,
                mvo_cagr,
                historical_cagr,
                None,  # No DEFI_TREASURY_INDEX CAGR
                None  # No NON_DEFI_TREASURY_INDEX CAGR
    )
            print("Insufficient data for SIRLOIN_INDEX, using Defi Pulse Index data instead.")
        
    
        
            
            
            
    
        
        
        # In[146]:
        
        
        
        
        
        # In[147]:
        
            
        
        # In[148]:
        
        
       
        #st.metric(label="DPI CAGR", value=f"{dpi_cagr * 100:.2f}%")
        #st.metric(label="DPI Cumulative Risk Premium", value=f"{dpi_cumulative_risk_premium * 100:.2f}%")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.metric(label="RL Beta", value=f"{rl_beta:.2f}")
            st.metric(label="RL CAGR", value=f"{rl_cagr * 100:.2f}%")
        with tcol2:
            st.metric(label="MVO Beta", value=f"{mvo_beta:.2f}")
            st.metric(label="MVO CAGR", value=f"{mvo_cagr * 100:.2f}%")
        with tcol3:
            st.metric(label="Historical Beta", value=f"{historical_beta:.2f}")
            st.metric(label="Historical CAGR", value=f"{historical_cagr * 100:.2f}%")
        with st.expander(f"{index} as Benchmark"):
            st.metric(label=f"{index} CAGR", value=f"{benchmark_cagr * 100:.2f}%")
            #st.write(tbill)
            st.metric(label=f"Risk Free Rate (3 Month Tbill) as of {tbill['DATE'].iloc[-1]}:" , value=f"{current_risk_free * 100:.2f}%")
            st.metric(label=f"{index} Cumulative Risk Premium", value=f"{cumulative_risk_premium * 100:.2f}%")
    
    
        
    
        
    
    
    # In[161]:
    
        
    
    # In[162]:
    
    
        
        
        
        # In[163]:
        
        
        st.plotly_chart(fig)
