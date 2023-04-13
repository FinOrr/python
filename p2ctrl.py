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
class DQN_MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN_MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#ReplayMemory: This class is responsible for storing and retrieving experiences, which are tuples containing the state, action, reward, next state, and done flag.

#Agent: This class acts as the interface between the DQN and the environment. It receives observations from the environment, selects actions using the DQN, and updates the DQN using experiences from the replay memory.
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, batch_size=32, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN_MLP(input_size, output_size).to(self.device)
        self.target_net = DQN_MLP(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = random.randrange(self.policy_net.output_size)
        self.steps_done += 1
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                                           for s in batch.next_state if s is not None])
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
#Environment: This class simulates the gas chamber system and provides the state and reward information to the agent.

#Pod: This class encapsulates the details of the chamber system, such as the equations that govern its behavior.

# Plotter: This class provides a visual representation of the training process, displaying the training progress and performance metrics over time.
