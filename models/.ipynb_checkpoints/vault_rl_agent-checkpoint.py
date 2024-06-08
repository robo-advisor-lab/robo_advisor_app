import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta

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
"""
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"""
class RlAgent:
    def __init__(self, action_space, target_weights, vault_action_ranges, learning_rate=0.01, discount_factor=0.95,
                 exploration_rate=0.9, exploration_decay=0.95, min_exploration_rate=0.01, initial_strategy_period=0,
                 adjustment_scale=10):
        self.q_table = {}
        self.actions = action_space
        self.target_weights = target_weights
        self.vault_action_ranges = vault_action_ranges
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.initial_strategy_period = initial_strategy_period
        self.current_cycle = 0
        self.adjustment_scale = adjustment_scale
        self.dai_ceilings = None

    def random_action(self, state):
        """
        Generate a random action for exploration.
        """
        action = {}
        for vault, range_options in self.vault_action_ranges.items():
            min_val, max_val = range_options
            action[vault] = np.random.uniform(min_val, max_val)
        return action

    def get_state_representation(self, state):
        return tuple(state.values())

    def calculate_adjustment(self, current, target, vault):
        print('vault', vault)
        simplified_vault_name = vault.replace('_collateral_usd', '_dai_ceiling')
        print('vault', simplified_vault_name)
        print('target weights in calc', target)
        difference = target - current
        desired_adjustment = difference * self.adjustment_scale
        print('weight difference', difference)
        print('Desired adjustment', desired_adjustment)
        print('adjustment scale', self.adjustment_scale)
        legal_adjustments = self.vault_action_ranges.get(simplified_vault_name, [])
        print('legal adjustments', legal_adjustments)
        if legal_adjustments:
            min_val, max_val = legal_adjustments
            closest_adjustment = max(min(desired_adjustment, max_val), min_val)
            print('closest adjustment', closest_adjustment)
            return closest_adjustment
        else:
            print(f"No action range defined for {simplified_vault_name}")
            return 0

    def initial_strategy(self, state):
        print("Calculating initial strategy...")
        action_dict = {}
        print('items in state', state.items())
        for vault, current_weight in state.items():
            print('vault', vault)
            simplified_vault_name = vault.replace('_collateral_usd', '_dai_ceiling')
            print('simplified vault name initial strategy', simplified_vault_name)
            target_weight = self.target_weights.get(vault, 0)
            adjustment = self.calculate_adjustment(current_weight, target_weight, vault)
            if adjustment != 0:  # Only add adjustments if they are non-zero
                action_dict[simplified_vault_name] = adjustment
            print(f"Vault: {vault},'simplified vault initial strategy {simplified_vault_name}, Current: {current_weight}, Target: {target_weight}, Adjustment: {adjustment}")
        self.initial_actions = action_dict  # Save the computed actions
        print("Initial Actions: ", action_dict)
        return action_dict

    def calculate_dynamic_strategy(self, current_weights):
        action_dict = {}
        print("Calculating dynamic strategy based on optimized weights...")
        for vault, current_weight in current_weights.items():
            target_weight = self.target_weights.get(vault, 0)
            adjustment = self.calculate_adjustment(current_weight, target_weight, vault)
            adjusted_vault_name = vault.replace('_collateral_usd', '_dai_ceiling')
            if adjustment != 0:
                action_dict[adjusted_vault_name] = adjustment
            print(f"Vault: {vault}, Current: {current_weight}, Target: {target_weight}, Adjustment: {adjustment}")
        return action_dict

    def agent_policy(self, state, dai_ceilings):
        self.current_cycle += 1
        self.dai_ceilings = dai_ceilings
        state_key = self.get_state_representation(state)
        print(f"Cycle Number: {self.current_cycle}, Epsilon: {self.epsilon}")

        if self.current_cycle <= self.initial_strategy_period:
            action = self.random_action(state)
            reason = 'exploration (random selection for initial strategy period)'
        elif np.random.rand() < self.epsilon:
            action = self.random_action(state)
            reason = 'exploration (random selection for wider state exploration)'
        else:
            action = self.calculate_dynamic_strategy(state)
            reason = 'exploitation (based on learned values aiming for optimal performance)'

        if self.current_cycle > self.initial_strategy_period:
            self.epsilon *= (self.epsilon_decay ** 3)  # 2 or 3
        elif self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.epsilon = max(self.epsilon, self.epsilon_min)  # Ensure epsilon doesn't go below the minimum

        print('State:', state)
        print(f"Chosen Action: {action}, Reason: {reason}")
        return action

    def exploit_best_known_action(self, state_key):
        if state_key in self.q_table and not self.q_table[state_key].all() == 0:
            action_index = np.argmax(self.q_table[state_key])
            action = self.actions[action_index]
        else:
            action = self.calculate_dynamic_strategy(self.current_state_weights)
        return action

    def update_q_table(self, old_state, action_index, reward, new_state):
        old_state_key = self.get_state_representation(old_state)
        new_state_key = self.get_state_representation(new_state)
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = np.zeros(len(self.actions))
        old_value = self.q_table[old_state_key][action_index]
        next_max = np.max(self.q_table[new_state_key])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[old_state_key][action_index] = new_value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f'Epsilon after decay: {self.epsilon}')
