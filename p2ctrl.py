import serial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Push a transition to the memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, input_size, output_size, learning_rate, gamma):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

    def select_action(self, state):
        sample = np.random.rand()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(self.output_size)]], device=self.device, dtype=torch.long)

    def learn(self, batch_size):
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def get_state(target_pressure, sensor_0, sensor_1, sensor_2):
    """Return the current state of the system as a NumPy array."""
    state = np.array([target_pressure, sensor_0, sensor_1, sensor_2])
    return state

def take_action(action):
    """Send a control signal to the system based on the chosen action."""
    # Map the action to a control signal
    # Send the control signal to the system
    # This will be a FSM state, ignore for now
    pass


if __name__ == '__main__':
    # Connect to the serial port
    ser = serial.Serial('COM3', 9600)
    
    # Define the input and output sizes
    input_size = 4
    output_size = 8

    # Set the learning rate and discount factor
    learning_rate = 0.001
    gamma = 0.99

    # Initialize the agent
    agent = Agent(input_size, output_size, learning_rate, gamma)

    # Set the target pressure
    target_pressure = 50.0

    # Set the maximum number of episodes and steps per episode
    num_episodes = 1000
    max_steps = 1000

    # Initialize the memory and episode rewards
    memory = ReplayMemory(10000)
    episode_rewards = []

    # Start the training loop
    for episode in range(num_episodes):
        # Reset the environment
        state = get_state(ser, input_size)
        episode_reward = 0
        for step in range(max_steps):
            # Select an action
            action = agent.select_action(state)

            # Send the action to the system and get the next state, reward, and done flag
            next_state, reward, done = take_action(ser, action, state, target_pressure)

            # Add the transition to the memory
            memory.push(state, action, next_state, reward)

            # Update the state and episode reward
            state = next_state
            episode_reward += reward

            # Learn from the memory
            agent.learn(32)

            # Update the target network
            if step % 10 == 0:
                agent.update_target_net()

            # Check if the episode is done
            if done:
                break

        # Append the episode reward
        episode_rewards.append(episode_reward)

        # Print the episode information
        print("Episode: %d, Steps: %d, Reward: %.2f, Epsilon: %.2f" % (episode, step, episode_reward, agent.epsilon_end))

    # Close the serial port
    ser.close()

    # Plot the episode rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
