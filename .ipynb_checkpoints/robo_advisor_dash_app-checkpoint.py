import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import tensorflow as tf
import random

# Treasury Advisor imports
from models.treasury_mvo_model import mvo_model
from models.treasury_rl_model import train_rl_agent
from scripts.treasury_utils import calculate_sortino_ratio, calculate_treynor_ratio, calculate_cagr, calculate_beta, normalize
from scripts.treasury_data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free, historical_normalized_returns, historical_cumulative_return, historical_returns as historical_portfolio_returns, pivot_data_filtered, pivot_assets, index_start, indicies

# Vault Advisor imports
from models.vault_mvo_agent import MVOAgent
from models.vault_rl_agent import RlAgent
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment
from scripts.vault_data_processing import test_data, test_data as simulation_data, targets, features, temporals, dai_ceilings, vault_names, key_mapping
from scripts.vault_utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space, calc_cumulative_return

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Robo Advisor Dashboard"

# Set random seeds
def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Layout
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Robo Advisor Dashboard", className="text-center"), className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Random Seed"),
                dcc.Dropdown(id='random-seed', options=[{'label': str(i), 'value': i} for i in [20, 42, 100, 200]], value=20)
            ], width=3),
            dbc.Col([
                dbc.Label("Select Advisor Type"),
                dcc.Dropdown(id='advisor-type', options=[{'label': i, 'value': i} for i in ['Vault', 'Treasury']], value='Vault')
            ], width=3),
            dbc.Col([
                dbc.Label("Rebalance Frequency"),
                dcc.Input(id='rebalance-frequency', type='number', value=5, min=1, max=30)
            ], width=3),
            dbc.Col([
                dbc.Button("Run Simulation", id='run-simulation', color='primary', className="mt-4")
            ], width=3)
        ]),
        html.Hr(),
        dbc.Tabs([
            dbc.Tab(label="Vault Robo Advisor", tab_id="vault"),
            dbc.Tab(label="Treasury Robo Advisor", tab_id="treasury")
        ], id="tabs", active_tab="vault"),
        html.Div(id="tab-content")
    ])
])

# Callbacks
@app.callback(
    [Output('tab-content', 'children')],
    [Input('run-simulation', 'n_clicks')],
    [State('random-seed', 'value'), State('advisor-type', 'value'), State('rebalance-frequency', 'value')]
)
def update_simulation(n_clicks, random_seed, advisor_type, rebalance_frequency):
    if n_clicks is None:
        return [html.Div()]

    set_random_seed(random_seed)

    if advisor_type == "Treasury":
        return [treasury_advisor_content(rebalance_frequency)]
    elif advisor_type == "Vault":
        return [vault_advisor_content()]

def treasury_advisor_content(rebalance_frequency):
    # Run Treasury Advisor models
    treasury_advisor(rebalance_frequency)

    actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df = st.session_state['treasury_rl']
    rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns = st.session_state['treasury_mvo']

    # Plot results
    fig = make_comparison_plot(historical_normalized_returns, rl_normalized_returns, mvo_normalized_returns)
    comp_fig = make_composition_pie_chart(composition_df, rebalanced_data, pivot_data_filtered)

    return dbc.Container([
        dbc.Row(dbc.Col(dcc.Graph(figure=fig), width=12)),
        dbc.Row(dbc.Col(dcc.Graph(figure=comp_fig), width=12))
    ])

def vault_advisor_content():
    # Run Vault Advisor models
    vault_advisor()

    rl_sim_results, rl_action_df = st.session_state['vault_rl']
    mvo_sim_results, mvo_action_df = st.session_state['vault_mvo']

    # Plot results
    fig = make_comparison_plot(historical_returns, rl_sim_results, mvo_sim_results)
    comp_fig = make_composition_pie_chart(rl_action_df, mvo_action_df, historical_returns)

    return dbc.Container([
        dbc.Row(dbc.Col(dcc.Graph(figure=fig), width=12)),
        dbc.Row(dbc.Col(dcc.Graph(figure=comp_fig), width=12))
    ])

def make_comparison_plot(historical, rl, mvo):
    trace1 = go.Scatter(
        x=rl.index, y=rl.values, mode='lines', name='RL Robo Advisor Optimized Returns', line=dict(color='blue')
    )
    trace2 = go.Scatter(
        x=mvo.index, y=mvo.values, mode='lines', name='MVO Robo Advisor Optimized Returns', line=dict(color='green')
    )
    trace3 = go.Scatter(
        x=historical.index, y=historical.values, mode='lines', name='Historical Returns', line=dict(color='red')
    )

    layout = go.Layout(
        title='Optimized vs Historical Returns',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Returns'),
        legend=dict(x=0.1, y=0.9)
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    return fig

def make_composition_pie_chart(rl_data, mvo_data, historical_data):
    fig = sp.make_subplots(
        rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=['RL Robo Advisor', 'MVO Robo Advisor', 'Historical']
    )

    fig.add_trace(go.Pie(labels=rl_data.columns, values=rl_data.iloc[-1], name="RL Robo Advisor"), 1, 1)
    fig.add_trace(go.Pie(labels=mvo_data.columns, values=mvo_data.iloc[-1], name="MVO Robo Advisor"), 1, 2)
    fig.add_trace(go.Pie(labels=historical_data.columns, values=historical_data.iloc[-1], name="Historical"), 1, 3)

    fig.update_layout(title_text='Portfolio Compositions')
    return fig

# Treasury Advisor functions
def treasury_advisor(rebalance_frequency):
    rl_rebalancing_frequency = rebalance_frequency
    treasury_start_date = panama_dao_start_date

    # RL Model
    actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(
        combined_all_assets, mvo_combined_assets, current_risk_free, rl_rebalancing_frequency, treasury_start_date
    )
    st.session_state['treasury_rl'] = (actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df)

    # MVO Model
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
    st.session_state['treasury_mvo'] = (rebalanced_data, mvo_daily_portfolio_returns, mvo_cumulative_return, mvo_normalized_returns)

## Vault Advisor functions
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
    st.session_state['vault_rl'] = (rl_sim_results, rl_action_df)

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
    st.session_state['vault_mvo'] = (mvo_sim_results, mvo_action_df)

if __name__ == '__main__':
    app.run_server(debug=True)
