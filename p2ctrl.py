import serial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 1000
MAX_TIMESTEPS = 1000
HIDDEN_SIZE = 128
LR = 0.001

#DQN: This class defines the architecture of the DQN with MLP and implements the training process.

#ReplayMemory: This class is responsible for storing and retrieving experiences, which are tuples containing the state, action, reward, next state, and done flag.

#Agent: This class acts as the interface between the DQN and the environment. It receives observations from the environment, selects actions using the DQN, and updates the DQN using experiences from the replay memory.

#Environment: This class simulates the gas chamber system and provides the state and reward information to the agent.

#Pod: This class encapsulates the details of the chamber system, such as the equations that govern its behavior.

# Plotter: This class provides a visual representation of the training process, displaying the training progress and performance metrics over time.
