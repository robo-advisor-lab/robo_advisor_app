import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.colors as pc
import plotly.subplots as sp
import datetime


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
from scripts.treasury_utils import calculate_sortino_ratio, calculate_treynor_ratio, calculate_cagr, calculate_beta, normalize
from scripts.treasury_data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free, historical_normalized_returns, historical_cumulative_return, historical_returns as treasury_historical_portfolio_returns, pivot_data_filtered, pivot_assets, index_start, indicies, tbill, combined_assets

# Vault Advisor
from models.vault_mvo_agent import MVOAgent
from models.vault_rl_agent import RlAgent
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment
from scripts.vault_data_processing import test_data, targets, features, temporals, dai_ceilings, vault_names, key_mapping
from scripts.vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return, evaluate_multi_predictions, abbreviate_number

# Set random seeds
random_seed = st.sidebar.selectbox('Select Random Seed', [20, 42, 100, 200])
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


st.set_option('deprecation.showPyplotGlobalUse', False)
## Treasury Advisor
#st.dataframe(treasury_historical_portfolio_returns)



def treasury_advisor(rebalance_frequency, selected_assets, start_date, end_date):
    filtered_assets = selected_assets
    rl_rebalancing_frequency = rebalance_frequency
    treasury_start_date = start_date
    treasury_end_date = end_date
    

    st.write(f"data set index start: {combined_all_assets.index.min()} through {combined_all_assets.index.min()}")
    # RL Model
    actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(combined_all_assets, filtered_assets, current_risk_free, rl_rebalancing_frequency, treasury_start_date, treasury_end_date)
    st.session_state['treasury_rl'] = (actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df)
    #st.session_state['treasury_historical_returns'] = filtered_historical_portfolio_returns

    # MVO Model
    mvo_rebalancing_frequency = rebalance_frequency
    threshold = 0
    model = mvo_model(current_risk_free, treasury_start_date, treasury_end_date)
    rebalanced_data = model.rebalance(combined_all_assets, filtered_assets, mvo_rebalancing_frequency)
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
historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_sim)
historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)
historical_returns = calc_cumulative_return(historical_portfolio_daily_returns)

hist_comparison = test_data[(test_data.index >= start_date) & (test_data.index <= end_date)]
#st.write(hist_comparison[targets].index.min(), hist_comparison[targets].index.max())


def vault_advisor():
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

    # RL Model
    rfl_agent = RlAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
    initial_action_rl = rfl_agent.initial_strategy(initial_weights)
    rl_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
    rl_simulator.train_model()
    rl_environment = SimulationEnvironment(rl_simulator, start_date, end_date, rfl_agent)
    rl_environment.reset()
    state, reward, done, info = rl_environment.run()
    rl_action_df = rl_environment.generate_action_dataframe()
    rl_action_df.set_index('date', inplace=True)
    rl_sim_results = rl_simulator.results
    rl_vaults = rl_simulator
    st.session_state['vault_rl'] = (rl_sim_results, rl_action_df, rl_vaults)

    # MVO Model
    mevo_agent = MVOAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
    initial_action_mvo = mevo_agent.initial_strategy(initial_weights)
    mvo_simulator = RL_VaultSimulator(simulation_data, test_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
    mvo_simulator.train_model()
    mvo_environment = SimulationEnvironment(mvo_simulator, start_date, end_date, mevo_agent)
    mvo_environment.reset()
    state, reward, done, info = mvo_environment.run()
    mvo_action_df = mvo_environment.generate_action_dataframe()
    mvo_action_df.set_index('date', inplace=True)
    mvo_sim_results = mvo_simulator.results
    mvo_vaults = mvo_simulator
    st.session_state['vault_mvo'] = (mvo_sim_results, mvo_action_df, mvo_vaults)

# Load data

# Simulation function
def run_simulation(agent_class, rebalance_frequency=None, start_date=None, end_date=None):
    if agent_class == "Treasury":
        treasury_advisor(rebalance_frequency, selected_assets, treasury_starting_date, treasury_ending_date)
        st.session_state['rebalance_frequency'] = rebalance_frequency
        st.session_state['treasury_historical_returns'] = filtered_historical_portfolio_returns
    elif agent_class == "Vault":
        vault_advisor()

#num_runs = st.sidebar.number_input('Number of Runs', min_value=1, max_value=20, value=10)
agent_class = st.sidebar.selectbox('Select Advisor Type', ['Vault', 'Treasury'])
#st.sidebar.write(panama_dao_start_date)
min_start_date = panama_dao_start_date.date()
#st.dataframe(combined_all_assets)
min_end_date = combined_all_assets.index.max().date()


treasury_starting_date = st.sidebar.date_input('Start Date', value=min_start_date, min_value=min_start_date)
treasury_ending_date = st.sidebar.date_input('End Date', value=min_end_date, max_value=min_end_date)


treasury_starting_date = datetime.datetime.combine(treasury_starting_date, datetime.datetime.min.time())
treasury_ending_date = datetime.datetime.combine(treasury_ending_date, datetime.datetime.min.time())

filtered_historical_portfolio_returns = treasury_historical_portfolio_returns[(treasury_historical_portfolio_returns.index >= treasury_starting_date) & (treasury_historical_portfolio_returns.index <= treasury_ending_date)]
#st.dataframe(treasury_historical_portfolio_returns)
#st.dataframe(filtered_historical_portfolio_returns)
historical_treasury_sortino = calculate_sortino_ratio(filtered_historical_portfolio_returns.values, current_risk_free)

# Add rebalance frequency input only for Treasury Advisor
if agent_class == "Treasury":
    rebalance_frequency = st.sidebar.number_input('Rebalance Frequency (days)', min_value=1, max_value=30, value=5)
    initial_amount = st.sidebar.number_input('Initial Amount (USD)', value=100)
    selected_assets = st.sidebar.multiselect(
        'Select assets to include in the Treasury Advisor:',
        combined_assets,
        default=mvo_combined_assets  
    )
    st.session_state['initial_amount'] = initial_amount
    
else:
    rebalance_frequency = 5
    initial_amount = 100
    selected_assets = mvo_combined_assets
run_simulation_button = st.sidebar.button('Run Simulation')

tabs = st.tabs(["Vault Robo Advisor", "Treasury Robo Advisor"])

if run_simulation_button:
    run_simulation(agent_class, rebalance_frequency)
    

if 'vault_rl' in st.session_state or 'vault_mvo' in st.session_state:
    with tabs[0]:
        st.header("Vault Robo Advisor")
        
        if 'vault_rl' in st.session_state:
            rl_sim_results, rl_action_df, rl_vaults = st.session_state['vault_rl']
            st.write(f"Simulation from {rl_sim_results.index.min()} through {rl_sim_results.index.max()}")
            #rl_action_df.set_index('date', inplace=True)
            rl_combined_data = pd.concat([vault_historical_data.drop(columns=['RWA Vault_collateral_usd']), rl_sim_results])
            rl_combined_data.index = rl_combined_data.index.tz_localize(None)
            rl_sim_w_RWA = rl_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
            rl_sim_w_RWA.sort_index(inplace=True)
            rl_portfolio_mvo_weights, rl_portfolio_returns, rl_portfolio_composition, rl_total_portfolio_value = mvo(rl_sim_w_RWA)
            rl_sim_portfolio_daily_returns,  rl_sim_downside_returns, rl_sim_excess_returns, rl_sim_sortino_ratio = historical_sortino(rl_portfolio_returns,rl_portfolio_composition)
            rl_optimized_returns = calc_cumulative_return(rl_sim_portfolio_daily_returns)
            rl_optimized_returns = rl_optimized_returns[~rl_optimized_returns.index.duplicated(keep='first')]
            st.divider()
            






            rl_sortino_timeseries = rl_action_df['current sortino ratio'].dropna()
            
            
        if 'vault_mvo' in st.session_state:
            mvo_sim_results, mvo_action_df, mvo_vaults = st.session_state['vault_mvo']
            #mvo_action_df.set_index('date', inplace=True)
            mvo_combined_data = pd.concat([vault_historical_data.drop(columns=['RWA Vault_collateral_usd']), mvo_sim_results])
            mvo_combined_data.index = mvo_combined_data.index.tz_localize(None)
            mvo_sim_w_RWA = mvo_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
            mvo_sim_w_RWA.sort_index(inplace=True)
            mvo_portfolio_mvo_weights, mvo_portfolio_returns, mvo_portfolio_composition, mvo_total_portfolio_value = mvo(mvo_sim_w_RWA)
            mvo_sim_portfolio_daily_returns,  mvo_sim_downside_returns, mvo_sim_excess_returns, mvo_sim_sortino_ratio = historical_sortino(mvo_portfolio_returns,mvo_portfolio_composition)
            mvo_optimized_returns = calc_cumulative_return(mvo_sim_portfolio_daily_returns)
            mvo_optimized_returns = mvo_optimized_returns[~mvo_optimized_returns.index.duplicated(keep='first')]






            mvo_sortino_timeseries = mvo_action_df['current sortino ratio'].dropna()
            
            
            st.sidebar.write("Vault Advisor simulation completed. Check the respective Vault advisor tab for results.")

        

        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.metric(label="RL current portfolio value", value=f"${abbreviate_number(rl_total_portfolio_value.iloc[-1])}")
            st.metric(label="RL sortino ratio", value=f"{rl_sim_sortino_ratio:.2f}")
        
        with vcol2:
            st.metric(label="MVO current portfolio value", value=f"${abbreviate_number(mvo_total_portfolio_value.iloc[-1])}")
            st.metric(label="MVO sortino ratio", value=f"{mvo_sim_sortino_ratio:.2f}")
        
        with vcol3:
            st.metric(label="Historical current portfolio value", value=f"${abbreviate_number(historical_total_portfolio_value.iloc[-1])}")
            st.metric(label="Historical sortino ratio", value=f"{historical_sortino_ratio:.2f}")
        vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault']
        st.write('Reinforcement Learning Results')
        st.line_chart(rl_sim_results)
        with st.expander("RL Vault Results"):
            for vault in vault_names:
                rl_fig = rl_vaults.plot_vault_data(vault)
                st.plotly_chart(rl_fig)
        

        
            #rl_metrics = evaluate_predictions(rl_sim_results, vault_historical_data)
        
            #mvo_metrics = evaluate_predictions(mvo_sim_results, vault_historical_data)

        

        st.write('Mean-Variance Optimization Results')
        st.line_chart(mvo_sim_results)
        with st.expander("MVO Vault Results"):
            for vault in vault_names:
                mvo_fig = mvo_vaults.plot_vault_data(vault)
                st.plotly_chart(mvo_fig)
            
        

        st.write('Historical Results')
        st.line_chart(hist_comparison[targets])
        with st.expander("Robo Advisor Vs Historical"):
            metrics, figures = evaluate_multi_predictions(rl_sim_results, mvo_sim_results, vault_historical)
            for vault, fig in figures.items():
                st.plotly_chart(fig)

            st.write(metrics)

        

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
        fig.update_layout(title_text='Portfolio Compositions')
        
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
        


if 'treasury_rl' in st.session_state or 'treasury_mvo' in st.session_state:
    with tabs[1]:
        st.header("Treasury Robo Advisor")
        filtered_historical_returns = st.session_state['treasury_historical_returns']
        filtered_historical_sortino = calculate_sortino_ratio(filtered_historical_returns, current_risk_free)
        filtered_cum_log_return = filtered_historical_returns.cumsum() 
        filtered_cumulative_return = np.exp(filtered_cum_log_return) - 1
        filtered_historical_normalized = normalize(filtered_cumulative_return, filtered_cumulative_return.index.min())
        #st.dataframe(filtered_historical_normalized)
        
        if 'treasury_rl' in st.session_state:
            actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df = st.session_state['treasury_rl']
            st.write(f"Simulation from {rl_portfolio_returns.index.min()} through {rl_portfolio_returns.index.max()}")
            rl_sortino = calculate_sortino_ratio(rl_portfolio_returns['Portfolio Return'].values, current_risk_free)

            #st.write('Reinforcement Learning Results')
            #st.line_chart(rl_normalized_returns)
        if 'treasury_mvo' in st.session_state:
            rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns = st.session_state['treasury_mvo']
            mvo_sortino = calculate_sortino_ratio(mvo_daily_portfolio_returns.values, current_risk_free)

            #st.write('Mean-Variance Optimization Results')
            #st.line_chart(mvo_normalized_returns)
            st.sidebar.write("Treasury Advisor simulation completed. Check the respective Treasury advisor tab for results.")

        # Convert the returns to strings and print
        historical_return = float(filtered_cumulative_return.iloc[-1]) * 100
        rl_return = float(rl_cumulative_return.iloc[-1]) * 100
        mvo_return = float(mvo_cumulative_return.iloc[-1]) * 100
        mvo_normalized_latest = mvo_normalized_returns["PanamaDAO_treasury_return"].iloc[-1]
        rl_normalized_latest = rl_normalized_returns["PanamaDAO_treasury_return"].iloc[-1]
        historical_normalized_latest = filtered_historical_normalized["treasury_return"].iloc[-1]    
        st.write(f"Initial amount: ${st.session_state['initial_amount']:,.2f}")
        st.write(f"Rebalancing Frequency: {st.session_state['rebalance_frequency']} days")
        st.divider()

        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="RL Adjusted Return", value=f"${initial_amount * rl_normalized_latest / 100:,.0f}")
            #st.write('actual normalized', rl_normalized_latest)
            st.metric(label="RL Cumulative Return", value=f"{rl_return:.2f}%")
            st.metric(label="RL Sortino", value=f"{rl_sortino:.2f}")
            
            
        with col2:
            st.metric(label="MVO Adjusted Return", value=f"${initial_amount * mvo_normalized_latest / 100:,.0f}")
            st.metric(label="MVO Cumulative Return", value=f"{mvo_return:.2f}%")
            st.metric(label="MVO Sortino", value=f"{mvo_sortino:.2f}")

        with col3:
            st.metric(label="Historical Adjusted Return", value=f"${initial_amount * historical_normalized_latest / 100:,.0f}")
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
        
        
        



        

        
        index_historical_normalized = normalize(filtered_cumulative_return, index_start)
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
        
        # Render the figure
        st.plotly_chart(fig)
        
        composition_columns = [f'COMPOSITION_{asset}' for asset in selected_assets]
        mvo_comp = rebalanced_data[composition_columns]
        mvo_comp.columns = mvo_comp.columns.str.replace('COMPOSITION_', '', regex=False)
        latest_comp_mvo = mvo_comp.iloc[-1]

        composition_df.columns = composition_df.columns.str.replace('COMPOSITION_', '', regex=False)
        
        latest_comp_rl = composition_df.iloc[-1]
        
        comp_columns = [col for col in pivot_data_filtered.columns if col.startswith('COMPOSITION_')]
        
        latest_historical_comp = pivot_data_filtered[comp_columns].iloc[-2]
        #st.write(latest_historical_comp)
        latest_historical_comp.index = latest_historical_comp.index.str.replace('COMPOSITION_', '', regex=False)
        comp_start = pivot_data_filtered.index.min()

        
        
        # List of all assets
        
        
        # Generate a color palette
        colors = pc.qualitative.Alphabet[:len(selected_assets)]
        
        # Create a color mapping
        color_mapping = {asset: color for asset, color in zip(selected_assets, colors)}
        
        def create_composition_pie(labels, values, name):
            # Create a list of colors based on the provided labels
            colors = [color_mapping[label] for label in labels]
            return go.Pie(
                labels=labels,
                values=values,
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
        
        # Render the figure
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
            
            
            

        
        
        # In[146]:
        
        
        prices = pivot_assets.copy()
        prices.set_index('DAY', inplace=True)
        dpi_history = prices['DAILY_PRICE_DPI']
        
        
        # In[147]:
        
            
        
        # In[148]:
        
        
        dpi_cagr = calculate_cagr(dpi_history)
            
        dpi_cumulative_risk_premium = dpi_cagr - current_risk_free
        #st.metric(label="DPI CAGR", value=f"{dpi_cagr * 100:.2f}%")
        #st.metric(label="DPI Cumulative Risk Premium", value=f"{dpi_cumulative_risk_premium * 100:.2f}%")
        
        defi_index_history = combined_indicies['DEFI_TREASURY_INDEX']
        defi_index_cagr = calculate_cagr(defi_index_history)
        defi_index_cumulative_risk_premium = defi_index_cagr - current_risk_free
        #st.metric(label="Defi Index CAGR", value=f"{defi_index_cagr * 100:.2f}%")
        #st.metric(label="Defi Index Cumulative Risk Premium", value=f"{defi_index_cumulative_risk_premium * 100:.2f}%")
        
        sirloin_history = combined_indicies['SIRLOIN_INDEX']
        sirloin_cagr = calculate_cagr(sirloin_history)
        sirloin_cumulative_risk_premium = sirloin_cagr - current_risk_free
        st.write('Sirlion Index as Benchmark')
        st.metric(label="Sirloin CAGR", value=f"{sirloin_cagr * 100:.2f}%")
        #st.write(tbill)
        st.metric(label=f"Risk Free Rate (3 Month Tbill) as of {tbill['DATE'].iloc[-1]}:" , value=f"{current_risk_free * 100:.2f}%")
        st.metric(label="Sirloin Cumulative Risk Premium", value=f"{sirloin_cumulative_risk_premium * 100:.2f}%")


        defi_index_cagr = calculate_cagr(combined_indicies['DEFI_TREASURY_INDEX'])
        non_defi_index_cagr = calculate_cagr(combined_indicies['NON_DEFI_TREASURY_INDEX'])
        
        defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'DEFI_TREASURY_INDEX')
        non_defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'NON_DEFI_TREASURY_INDEX')

        def create_interactive_sml(risk_free_rate, market_risk_premium, rl_beta, mvo_beta, historical_beta, defi_beta, non_defi_beta, rl_return, mvo_return, historical_return, defi_return, non_defi_return):
            betas = np.linspace(0, 6, 100)
            expected_returns = risk_free_rate + betas * market_risk_premium
        
            # Create the SML line
            sml_line = go.Scatter(x=betas, y=expected_returns, mode='lines', name=f'SML')
        
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
            
            
        
            # Add Risk-Free Rate line
            risk_free_line = go.Scatter(
                x=[0, max(betas)], 
                y=[risk_free_rate, risk_free_rate], 
                mode='lines', 
                line=dict(dash='dash', color='green'),
                name='Risk-Free Rate'
            )
        
            # Layout settings
            layout = go.Layout(
                title=f'SML',
                xaxis=dict(title='Beta (Systematic Risk)'),
                yaxis=dict(title='Return'),
                showlegend=True
            )
        
            # Combine all the plots
            fig = go.Figure(data=[sml_line, rl_expected, rl_actual, mvo_expected, mvo_actual, historical_expected, historical_actual, 
                                  defi_expected, defi_actual, non_defi_expected,non_defi_actual, risk_free_line], layout=layout)
            return fig, market_risk_premium
    
    
    # In[161]:
    
        
    
    # In[162]:
    
    
        fig, market_risk_premium = create_interactive_sml(
            current_risk_free,
            sirloin_cumulative_risk_premium,
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
        
        
        # In[163]:
        
        
        st.plotly_chart(fig)



