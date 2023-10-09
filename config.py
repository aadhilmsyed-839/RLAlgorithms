#############################
# PROGRAM IMPORTS
#############################

# Import Reinforcement Learning Algoirthms
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import ARS, MaskablePPO, QRDQN, RecurrentPPO, TQC, TRPO

# Import Logger and Recorder
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder

# Import OpenAI Gym & Other Important Libraries
import gymnasium as gym
import time
from typing import Type

# Import Data Analysis & Visualization Libraies
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
# NON-CHANGEABLE PARAMETERS
#############################

test_num = 1                       # Tracks the Test # of Current Iteration
results_dict = {}                  # Stores the Result for each Algorithm
it = 0                             # Iterator for the colors list

# Colors for creating data visuals
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#############################
# CHANGEABLE PARAMETERS
#############################

env_name    = "LunarLander-v2"     # Name of the Environment to Run
train_steps = 500_000              # Number of Timesteps for Learning

# RL Algorithms for Testing
rl_algs = [A2C, DQN, PPO, TRPO]
