import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
import random

from scripts.treasury_utils import calculate_sortino_ratio, mvo, calculate_log_returns

def normalize_portfolio(portfolio):
    total = np.sum(portfolio)
    if total == 0:
        # If the total is zero, avoid division by zero and return an equally distributed portfolio
        return np.ones_like(portfolio) / len(portfolio)
    return portfolio / total

class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_composition, all_assets, eth_bound, risk_free, rebalancing_frequency=7, start_date=None, end_date=None):
        super(PortfolioEnv, self).__init__()

        self.data = data
        self.risk_free = risk_free
        self.all_assets = np.array(all_assets)
        self.num_assets = len(all_assets)
        self.initial_composition = np.nan_to_num(initial_composition)  # Ensure no NaNs in initial composition
        self.rebalancing_frequency = rebalancing_frequency
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.eth_index = np.where(self.all_assets == 'ETH')[0][0] if 'ETH' in self.all_assets else None
        self.eth_bound = eth_bound

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.num_assets,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        self.actions_log = []
        self.returns_log = []
        self.composition_log = []
        self.previous_action = np.ones(self.num_assets) / self.num_assets  # Set initial action
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        if self.start_date:
            pos = self.data.index.searchsorted(self.start_date)
            if pos >= len(self.data.index):
                raise ValueError(f"Start date {self.start_date} is out of the range of the data index.")
            self.current_step = pos
            self.start_step = pos
        else:
            self.current_step = 0
            self.start_step = 0
    
        if self.end_date:
            pos = self.data.index.searchsorted(self.end_date)
            if pos >= len(self.data.index):
                raise ValueError(f"End date {self.end_date} is out of the range of the data index.")
            self.end_step = pos
        else:
            self.end_step = len(self.data) - 1
    
        self.done = False
        self.portfolio = self.initial_composition
        self.actions_log = []
        self.returns_log = []
        self.composition_log = [(self.data.index[self.current_step], self.portfolio.copy())]
        self.returns_log.append((self.data.index[self.current_step], 0))
    
        # Perform the initial action
        action = self.previous_action
        self.portfolio = normalize_portfolio(action)
        self.actions_log.append((self.data.index[self.current_step], action))
        return self._next_observation()


        
    def _apply_eth_bound(self):
        if self.portfolio[self.eth_index] < self.eth_bound:
            diff = self.eth_bound - self.portfolio[self.eth_index]
            remaining_allocation = 1 - self.eth_bound
    
            self.portfolio[self.eth_index] = self.eth_bound
            other_weights = np.delete(self.portfolio, self.eth_index)
            other_weights = other_weights / np.sum(other_weights) * remaining_allocation
            self.portfolio = np.insert(other_weights, self.eth_index, self.eth_bound)
            self.portfolio = normalize_portfolio(self.portfolio)

            
    def _next_observation(self):
        prices = self.data.iloc[self.current_step][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].fillna(0).values
        obs = np.concatenate([self.portfolio, prices])
        obs = np.nan_to_num(obs)
        if np.any(np.isnan(obs)):
            print("NaN values in observation:", obs)
            raise ValueError("Observation contains NaN values")
        return obs

    def step(self, action):
        if self.done or self.current_step >= self.end_step:
            self.done = True
            return self._next_observation(), 0, self.done, {}
    
        current_prices = self.data.iloc[self.current_step][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].fillna(0).values
        if self.current_step > 0:
            prev_prices = self.data.iloc[self.current_step - 1][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].values
            prev_prices = np.nan_to_num(prev_prices, nan=1.0)  # Replace NaNs with 1.0 to avoid invalid log returns
        else:
            prev_prices = current_prices
    
        log_returns = np.log(current_prices / prev_prices)
        new_portfolio_values = self.portfolio * np.exp(log_returns)
        sum_new_portfolio_values = np.sum(new_portfolio_values)
        self.portfolio = new_portfolio_values / sum_new_portfolio_values if sum_new_portfolio_values != 0 else new_portfolio_values
        self.portfolio = normalize_portfolio(new_portfolio_values)
    
        # Debugging: Print current step and rebalancing frequency
        print(f"Current Step: {self.current_step}, Rebalancing Frequency: {self.rebalancing_frequency}")
    
        # Rebalance the portfolio at the specified frequency or on the first step
        if self.current_step == self.start_step or (self.current_step - self.start_step) % self.rebalancing_frequency == 0:
            print('Rebalancing Portfolio')
            action = np.clip(action, 0, 1)
            action_sum = np.sum(action)
            action = action / action_sum if action_sum != 0 else action
            self.portfolio = normalize_portfolio(action)
            
            # Ensure ETH allocation meets the minimum bound if specified
            if self.portfolio[self.eth_index] < self.eth_bound:
                diff = self.eth_bound - self.portfolio[self.eth_index]
                remaining_allocation = 1 - self.eth_bound
                self.portfolio[self.eth_index] = self.eth_bound
                other_weights = np.delete(self.portfolio, self.eth_index)
                other_weights = other_weights / np.sum(other_weights) * remaining_allocation
                self.portfolio = np.insert(other_weights, self.eth_index, self.eth_bound)
                self.portfolio = normalize_portfolio(self.portfolio)
    
            self.previous_action = self.portfolio
            self.actions_log.append((self.data.index[self.current_step], self.portfolio))
    
        self.composition_log.append((self.data.index[self.current_step], self.portfolio.copy()))
    
        portfolio_return = np.sum(log_returns * self.portfolio)
        self.returns_log.append((self.data.index[self.current_step], portfolio_return))
    
        if len(self.returns_log) > 0:
            sortino_ratio = calculate_sortino_ratio([r for _, r in self.returns_log], self.risk_free)
        else:
            sortino_ratio = 0
    
        optimized_weights, _, composition, _ = mvo(self.data.iloc[:self.current_step + 1], self.all_assets, self.risk_free)
        current_weights = composition.iloc[-1].values if optimized_weights is not None else None
        max_distance = sum(abs(1 - value) for value in optimized_weights) if optimized_weights is not None else 0
        distance_penalty = sum(abs(current_weights[i] - optimized_weights[i]) for i in range(len(optimized_weights))) / max_distance if max_distance != 0 else 0
    
        reward = portfolio_return + 2 * sortino_ratio - distance_penalty
        obs = self._next_observation()
        self.current_step += 1
    
        return obs, reward, self.done, {}


def train_rl_agent(data, all_assets, eth_bound, risk_free, rebalancing_frequency=7, start_date=None, end_date=None, seed=None):
    data_start_date = data.index.min()
    initial_composition = data.loc[data.index >= start_date][[f'COMPOSITION_{asset}' for asset in all_assets]].to_numpy()[0]
    initial_composition = np.nan_to_num(initial_composition)  # Ensure no NaNs in initial composition
    data = data.loc[(data.index >= data_start_date) & (data.index <= end_date)]
    env = PortfolioEnv(data=data, initial_composition=initial_composition, all_assets=all_assets, eth_bound=eth_bound, risk_free=risk_free, rebalancing_frequency=rebalancing_frequency, start_date=start_date, end_date=end_date)

    if seed is not None:
        env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    model = PPO('MlpPolicy', env, verbose=1)

    total_timesteps = len(data)
    training_duration_factor = 1000
    total_timesteps *= training_duration_factor
    obs = env.reset()
    obs = np.nan_to_num(obs)
    for _ in range(total_timesteps):
        if np.any(np.isnan(obs)):
            print("Observation with NaNs:", obs)
            print("Observation index:", env.current_step)
            print("Current prices:", env.data.iloc[env.current_step][[f'DAILY_PRICE_{asset}' for asset in all_assets]].values)
            print("Portfolio:", env.portfolio)

        action, _ = model.predict(obs)
        print('action', action)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum == 0:
            action_sum = 1e-8
        action = action / action_sum

        obs, rewards, done, info = env.step(action)
        if done:
            break

    actions_log = env.actions_log
    returns_log = env.returns_log
    composition_log = env.composition_log

    actions_df = pd.DataFrame([x[1] for x in actions_log], columns=[f'WEIGHT_{asset}' for asset in all_assets])
    actions_df.index = [x[0] for x in actions_log]

    composition_df = pd.DataFrame([x[1] for x in composition_log], columns=[f'COMPOSITION_{asset}' for asset in all_assets])
    composition_df.index = [x[0] for x in composition_log]

    returns_df = pd.DataFrame([x[1] for x in returns_log], columns=['Portfolio Return'])
    returns_df.index = [x[0] for x in returns_log]

    prices_df = data.loc[data.index >= start_date, [f'DAILY_PRICE_{asset}' for asset in all_assets]]
    log_price_returns = calculate_log_returns(prices_df)

    # Ensure composition_df and log_price_returns have the same index
    common_index = composition_df.index.intersection(log_price_returns.index)
    composition_df = composition_df.loc[common_index]
    log_price_returns = log_price_returns.loc[common_index]

    portfolio_values = (composition_df.values * log_price_returns.values).sum(axis=1)
    returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
    rl_portfolio_returns = returns_df
    rl_cumulative_return = (1 + rl_portfolio_returns).cumprod() - 1

    cumulative_return_df = rl_cumulative_return.reset_index()
    cumulative_return_df.columns = ['DAY', 'cumulative_return']

    base_return = cumulative_return_df.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})

    combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')

    first_value = combined['base_cumulative_return'].iloc[0]
    combined['PanamaDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))

    rl_normalized_returns = combined[['DAY', 'PanamaDAO_treasury_return']]
    rl_normalized_returns.set_index('DAY', inplace=True)

    return actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df
