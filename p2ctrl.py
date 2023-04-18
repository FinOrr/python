import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def train(model, target_model, replay_memory, batch_size, gamma, optimizer, loss_fn, device):
    if len(replay_memory.memory) < batch_size:
        return
    transitions = replay_memory.sample(batch_size)
    batch = np.array(transitions, dtype=object).transpose()
    states = torch.cat(batch[0])
    actions = torch.cat(torch.tensor(batch[1]).unsqueeze(1))
    rewards = torch.cat(torch.tensor(batch[2]).unsqueeze(1))
    next_states = torch.cat(batch[3])
    dones = torch.cat(torch.tensor(batch[4]).unsqueeze(1))
    
    # Compute Q values for current states
    q_values = model(states.float())
    q_values = q_values.gather(1, actions)

    # Compute Q values for next states with target network
    target_q_values = target_model(next_states.float()).detach()
    max_target_q_values = target_q_values.max(1)[0].unsqueeze(1)
    expected_q_values = rewards.float() + gamma * max_target_q_values * (1 - dones.float())

    # Compute loss and update model
    loss = loss_fn(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate(model, validation_set, device):
    total_reward = 0
    for state, action, reward, next_state, done in validation_set:
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.bool, device=device)

        q_values = model(state.unsqueeze(0))
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze()

        total_reward += reward
    return total_reward

def build_network(input_dim, output_dim, hidden_dim):
    model = DQNN(input_dim, output_dim, hidden_dim)
    target_model = DQNN(input_dim, output_dim, hidden_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    return model, target_model

def compile_network(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    return optimizer, loss_fn

def preprocess(state):
    # Scale each sensor reading to the range [0, 1]
    state_scaled = (state - 8000) / (120000 - 8000)

    # Convert to numpy array and reshape to (1, num_sensors)
    state_np = np.array(state_scaled).reshape(1, -1)

    return state_np

